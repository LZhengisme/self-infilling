import inspect
import json
import os
import warnings
from collections import defaultdict
from accelerate import PartialState  # Can also be Accelerator or AcceleratorState
from accelerate.utils import set_seed
import torch
import math
import numpy as np
from multiprocessing.sharedctypes import Value
from functools import partial
from tqdm import tqdm
from transformers import (
    StoppingCriteriaList,
    LogitsProcessorList
)
from lm_eval import tasks

from lm_eval.generation_pipelines.synthesis_utils import (
    vanilla_completion,
    vanilla_infilling,
)

from lm_eval.generation_pipelines.self_infill_utils import (
    SelfInfillingLogitsProcessor,
    SelfInfillEndOfFunctionCriteria,
    self_infill_completion,
)

from lm_eval.utils import (
    EndOfFunctionCriteria,
    TooLongFunctionCriteria,
    build_fim_sentinel_dict
)

_WARNING = """
################################################################################
                                  !!!WARNING!!!
################################################################################
The "code_eval"/"apps_metric" you are about to use, execute untrusted 
model-generated code in Python.
Although it is highly unlikely that model-generated code will do something
overtly malicious in response to this test suite, model-generated code may act
destructively due to a lack of model capability or alignment.
Users are strongly encouraged to sandbox this evaluation suite so that it
does not perform destructive actions on their host or network. For more
information on how OpenAI sandboxes its code, see the paper "Evaluating Large
Language Models Trained on Code" (https://arxiv.org/abs/2107.03374).
Once you have read this disclaimer and taken appropriate precautions, set the argument 
"allow_code_execution" to True.
################################################################################\
"""


class Evaluator:
    def __init__(self, accelerator, model, tokenizer, args):
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

        self.metric_output_path = args.metric_output_path

        # code evaluation permission
        self.allow_code_execution = args.allow_code_execution

    def _build_prompts(
        self, 
        task_name, 
        task,
        dataset,
        n_tasks,
        args
    ):
        if self.accelerator.is_main_process:
            print(f"number of problems for this task is {n_tasks}")
        assert args.n_samples % args.batch_size == 0

        n_copies = math.ceil(args.n_samples / args.batch_size)

        prompts = []
        for sample in range(n_tasks):
            prompt = task.get_prompt(dataset[sample])
            prompts.append(prompt)

        if n_copies == 1 and n_tasks % self.accelerator.state.num_processes != 0:
            warnings.warn(
                "n_tasks isn't proportional to num devices. "
                "In this case, an additional duplicate for some task will be passed for generation, "
                "which is removed at the end of generation."
            )
        input_prompts = []
        for sample in range(n_tasks):
            for _ in range(n_copies):
                if isinstance(prompts[sample], dict):
                    sample_dict = {"task_id": sample}
                    for k, v in prompts[sample].items():
                        sample_dict[k] = v
                elif isinstance(prompts[sample], str):
                    sample_dict = {
                        "prompt": prompts[sample],
                        "task_id": sample
                    }
                input_prompts.append(sample_dict)

        return input_prompts

    def _prepare_model(self, model, args):
        is_loaded_in_8bit = getattr(model, "is_loaded_in_8bit", False)
        is_loaded_in_4bit = getattr(model, "is_loaded_in_4bit", False)
        is_wrapped = is_loaded_in_8bit or is_loaded_in_4bit
        if args.max_memory_per_gpu is not None:
            pass
        elif not is_loaded_in_8bit and not is_loaded_in_4bit:
            # we only wrap data loader to avoid extra memory occupation
            model = model.to(self.accelerator.device)
        else:
            # model.to() is not supported for 8bit and 4bit models
            model = self.accelerator.prepare(model)

        _model = self.accelerator.unwrap_model(model) if is_wrapped else model
        return _model

    def _build_gen_tok_kwargs(self, task, args):
        fim_sentinel_dict = build_fim_sentinel_dict(self.tokenizer)
        # Setup generation settings
        gen_kwargs = {
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_length": args.max_length_generation,
            "max_new_tokens": args.max_new_tokens_generation,
        }
        tokenizer_kwargs = {
            "max_length": args.max_length_generation,
            "return_token_type_ids": False,
        }
        if task.stop_words:
            print(">>>>>>> Add EOS and EOT tokens into stop_words")
            task.stop_words.append(self.tokenizer.eos_token)
            task.stop_words.append(fim_sentinel_dict["fim_ending"])
        
        stopping_criteria = []
        # Check if the task has a custom check_fn method for the stopping criteria
        if hasattr(task, "check_fn"):
            stopping_criteria.append(
                EndOfFunctionCriteria(0, task.stop_words, self.tokenizer, task.check_fn)
            )
        elif task.stop_words:
            check_fn = None
            stopping_criteria.append(
                EndOfFunctionCriteria(0, task.stop_words, self.tokenizer, check_fn)
            )
        if hasattr(task, "max_length_multiplier") and task.max_length_multiplier:
            stopping_criteria.append(
                TooLongFunctionCriteria(0, task.max_length_multiplier)
            )
        
        if stopping_criteria:
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList(stopping_criteria)
        return gen_kwargs, tokenizer_kwargs

    def generate_text(self, task_name):
        set_seed(self.args.seed, device_specific=True)

        task = tasks.get_task(task_name)
        dataset = task.get_dataset()

        ###################################
        # set up task attributes
        # for codellama, do not strip() since \n is encoded into a single token,
        # but for starcoder models, we need to strip prompt.
        if (
            (
                "starcoder" in self.tokenizer.name_or_path or
                "santacoder" in self.tokenizer.name_or_path
            ) and 
            (
                "humaneval" in task_name or 
                "mbpp" in task_name
            )
        ):
            task.strip_prompt = True
        else:
            # DS1000 and GSM8k does not apply strip() because
            # otherwise the model might continue to complete 
            # the start indicator.
            task.strip_prompt = False

        n_tasks = self.args.limit if self.args.limit else len(dataset)
        references = [task.get_reference(dataset[i]) for i in range(n_tasks)]

        ######################################################################
        # Build input prompts
        input_prompts = self._build_prompts(task_name, task, dataset, n_tasks, self.args)

        ######################################################################
        # Prepare and wrap models
        _model = self._prepare_model(self.model, self.args)

        ######################################################################
        # Setup generation and tokenization
        gen_kwargs, tokenizer_kwargs = self._build_gen_tok_kwargs(task, self.args)

        
        if self.args.use_self_infill:
            # NOTE: the custom logit processor is appended to the default processor list
            # NOTE: temperature scaling as well as top-k/p are performed after the logits_processor.

            assert self.args.self_infill_suffix_split_mode in ["vanilla", "extended", "half"]
            assert isinstance(self.args.self_infill_max_iters, int)
            assert self.args.self_infill_tau > 0 and self.args.self_infill_tau < 1
            if "cot-gsm8k" in task_name:
                default_suffix_prompt = task.suffix_prompt
            else:
                default_suffix_prompt = "return"
            
            # overwrite stopping_criteria
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
                [
                    SelfInfillEndOfFunctionCriteria(
                        0,
                        task.stop_words,
                        self.tokenizer,
                        check_fn=None
                    )
                ]
            )
            gen_kwargs["logits_processor"] = LogitsProcessorList(
                [
                    SelfInfillingLogitsProcessor(
                        0, 
                        task.stop_words,
                        self.tokenizer,
                        check_fn=None,
                        tau=self.args.self_infill_tau,
                        random_interrupt=self.args.self_infill_random_interruption,
                        suffix_prompt=default_suffix_prompt
                    )
                ]
            )
            code_generator_fn = partial(self_infill_completion,
                model=_model,
                tokenizer=self.tokenizer,
                task=task,
                batch_size=self.args.batch_size,
                gen_kwargs=gen_kwargs,
                tokenizer_kwargs=tokenizer_kwargs,
                default_suffix_prompt=default_suffix_prompt,
                suffix_split_mode=self.args.self_infill_suffix_split_mode,
                max_iters=self.args.self_infill_max_iters,
            )
        else:
            if "insertion" in task_name:
                # for DS-1000 insertion mode
                code_generator_fn = partial(vanilla_infilling,
                    model=_model,
                    task=task,
                    tokenizer=self.tokenizer,
                    batch_size=self.args.batch_size,
                    gen_kwargs=gen_kwargs,
                    tokenizer_kwargs=tokenizer_kwargs
                )
            else:
                code_generator_fn = partial(vanilla_completion,
                    model=_model,
                    task=task,
                    tokenizer=self.tokenizer,
                    batch_size=self.args.batch_size,
                    gen_kwargs=gen_kwargs,
                    tokenizer_kwargs=tokenizer_kwargs
                )

        distributed_state = PartialState()
        gathered_gen_code_dict = defaultdict(list)  # dict of list of generated tokens
        with distributed_state.split_between_processes(input_prompts, apply_padding=True) as prompts_per_process:
            device = self.accelerator.device
            num_instances = len(prompts_per_process)
            for input_dict in tqdm(
                prompts_per_process,
                total=num_instances,
            ):  
                generated_code = code_generator_fn(input_dict=input_dict, device=device)
                generated_code_task_dict = {
                    input_dict["task_id"] : generated_code
                }
                # collect all generated_strs
                # move gather inside the loop to avoid socket timeout
                if torch.distributed.is_initialized():
                    output_gen_code_list = [None for _ in range(PartialState().num_processes)]
                    torch.distributed.all_gather_object(output_gen_code_list, generated_code_task_dict)
                    for d in output_gen_code_list:
                        for k, v in d.items():
                            gathered_gen_code_dict[k].extend(v)
                else:
                    for k, v in generated_code_task_dict.items():
                        gathered_gen_code_dict[k].extend(v)

        generations = [[] for _ in range(n_tasks)]
        for task_id, generated_code in gathered_gen_code_dict.items():
            for gen_code in generated_code:
                if self.args.postprocess:
                    trimmed_gen_code = task.postprocess_generation(gen_code, int(task_id))
                    if (
                        "humaneval" in task_name and
                        "codellama" in self.tokenizer.name_or_path
                    ):
                        # sometimes CodeLlama models generate 3 spaces as indentation
                        # post-process the code to fix the inconsistency
                        tmp = ""
                        for line in trimmed_gen_code.splitlines():
                            lspace = len(line) - len(line.lstrip())
                            if lspace == 3:
                                tmp += " "
                            tmp += line + "\n"
                        trimmed_gen_code = tmp
                    elif (
                        "mbpp" in task_name or 
                        "ds1000" in task_name or
                        "gsm" in task_name
                    ):
                        trimmed_gen_code = task.extract_completion(trimmed_gen_code, int(task_id))
                    generations[task_id].append(trimmed_gen_code)
                else:
                    warnings.warn(
                        "model output is not postprocessed, this might lower evaluation scores"
                    )
                    generations[task_id].append(gen_code)

            if len(generations[task_id]) > self.args.n_samples:
                generations[task_id] = generations[task_id][: self.args.n_samples]
                warnings.warn(
                    f"Number of tasks wasn't proportional to number of devices, we removed extra predictions to only keep nsamples={self.args.n_samples}"
                )

        return generations, references

    def evaluate(self, task_name, generations, references):
        task = tasks.get_task(task_name)
        if task.requires_execution and not self.allow_code_execution:
            raise ValueError(_WARNING)

        if self.accelerator.is_main_process:
            # make sure tokenizer plays nice with multiprocessing
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            if self.allow_code_execution and task.requires_execution:
                os.environ["HF_ALLOW_CODE_EVAL"] = "1"
            print("Evaluating generations...")
            run_stats = self.args.load_generations_path.replace("generations", "run_stats")
            os.environ["RUN_STATS_SAVE_PATH"] = run_stats
            results = task.process_results(generations, references)
            return results

    def save_generations(self, generations, references=None):
        if self.accelerator.is_main_process:
            with open(self.args.save_generations_path, "w") as fp:
                json.dump(generations, fp)
                print(f"generations were saved at {self.args.save_generations_path}")
            if self.args.save_references and references:
                with open(os.path.dirname(self.args.save_references_path) + "/references.json", "w") as fp:
                    json.dump(references, fp)
                    print("references were saved")

    def load_generations(self, task_name):
        task = tasks.get_task(task_name)
        dataset = task.get_dataset()
        # if args.limit is None, use all samples
        n_tasks = self.args.limit if self.args.limit else len(dataset)
        references = [task.get_reference(dataset[i]) for i in range(n_tasks)]

        if self.args.check_references:
            if "get_solution" in inspect.signature(task.get_reference).parameters:
                solutions = [[task.get_reference(dataset[i], get_solution=True)] for i in range(n_tasks)]
            else:
                solutions = [[ref] for ref in references]
            return solutions, references
        else:
            assert self.args.load_generations_path
            # load generated code
            with open(self.args.load_generations_path) as fp:
                generations_by_task = json.load(fp)
                generations = generations_by_task[task_name]
                num_tasks = len(generations_by_task.keys())
                if self.accelerator.is_main_process:
                    print(
                        f"generations loaded, {num_tasks} tasks in total\n"
                        f"{task_name} Task selected\n"
                        f"{n_tasks} problems selected from {len(generations)} with {len(generations[0])} candidates"
                    )
            generations = generations[:n_tasks]

        ret_generations = []
        for l in generations:
            if len(l) > self.args.n_samples:
                ret_generations.append(l[: self.args.n_samples])
                warnings.warn(
                    f"Number of tasks wasn't proportional to number of devices, we removed extra predictions to only keep nsamples={self.args.n_samples}"
                )
            else:
                ret_generations.append(l)
        return ret_generations, references
