import torch
from transformers.generation import LogitsProcessor
from transformers import StoppingCriteria
import re
import copy
from lm_eval.utils import (
    build_fim_sentinel_dict,
    self_infill_split,
    check_degenerate,
    check_repeat,
    compact_lines
)

class SelfInfillEndOfFunctionCriteria(StoppingCriteria):
    """
        Custom `StoppingCriteria` which checks if all generated functions in the batch are completed.
    """
    def __init__(self, start_length, eof_strings, tokenizer, check_fn=None):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer
        if check_fn is None:
            check_fn = lambda decoded_generation: any(
                [stop_string in decoded_generation for stop_string in self.eof_strings]
            )
        fim_sentinel_dict = build_fim_sentinel_dict(tokenizer, suffix_first=False)
        self.fim_prefix_id = fim_sentinel_dict["fim_prefix_id"]
        self.fim_middle_id = fim_sentinel_dict["fim_middle_id"]
        self.fim_suffix_id = fim_sentinel_dict["fim_suffix_id"]
        self.fim_prefix = fim_sentinel_dict["fim_prefix"]
        self.fim_middle = fim_sentinel_dict["fim_middle"]
        self.fim_suffix = fim_sentinel_dict["fim_suffix"]
        self.check_fn = check_fn
        self.prefixes_for_check_fn = None

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if ALL generated sequences contain any of the end-of-function strings."""
        # check whether there is <prefix> present in input_ids
        has_prefix_id_sequences = torch.any(
            input_ids == self.fim_prefix_id, dim=-1
        )
        # check whether there is <suffix> present in input_ids
        has_suffix_id_sequences = torch.any(
            input_ids == self.fim_suffix_id, dim=-1
        )
        # check whether there is <middle> present in input_ids
        has_middle_id_sequences = torch.any(
            input_ids == self.fim_middle_id, dim=-1
        )

        decoded_generations = self.tokenizer.batch_decode(input_ids[:, self.start_length :])
        if not any(has_prefix_id_sequences):
            # no sentinel tokens found: the left-to-right generation mode
            # simply check whether all generations are completed.
            if self.prefixes_for_check_fn:
                decoded_generations = [
                        prefix_for_check_fn + decoded_generation
                        for prefix_for_check_fn, decoded_generation 
                        in zip(self.prefixes_for_check_fn, decoded_generations)
                    ]
            return all(
                [
                    self.check_fn(decoded_generation) 
                    for decoded_generation in decoded_generations
                ]
            )
        else:
            stop_gen = []
            has_prefix_id_sequences_list = has_prefix_id_sequences.cpu().tolist()
            has_suffix_id_sequences_list = has_suffix_id_sequences.cpu().tolist()
            has_middle_id_sequences_list = has_middle_id_sequences.cpu().tolist()
            for i, decoded_generation in enumerate(decoded_generations):
                has_prefix_id = has_prefix_id_sequences_list[i]
                has_suffix_id = has_suffix_id_sequences_list[i]
                has_middle_id = has_middle_id_sequences_list[i]
                if has_prefix_id & has_suffix_id & (~has_middle_id):
                    # still generating the suffix.
                    # the termination of the suffix generation will be 
                    # handled by LogitProcessor.
                    stop_gen.append(False)
                elif has_prefix_id & has_suffix_id & has_middle_id:
                    # generating the infilled middle
                    # only pass middle to check_fn 
                    split_by_middle = decoded_generation.split(self.fim_middle)
                    if len(split_by_middle) == 1:
                        # <MID> comes before start_length
                        decoded_generation = split_by_middle[0]
                    else:
                        decoded_generation = split_by_middle[1]
                    stop_gen.append(self.check_fn(decoded_generation))
                else:
                    # generating the prefix
                    # pass generated tokens so far to check_fn 
                    stop_gen.append(self.check_fn(decoded_generation))
            return all(stop_gen)

class SelfInfillingLogitsProcessor(LogitsProcessor):
    """
        Custom `LogitsProcessor` which processes next_logit to implement self-infilling.
    """
    def __init__(
            self, 
            start_length, 
            eof_strings, 
            tokenizer, 
            check_fn=None,
            tau=0.1,
            random_interrupt=False,
            suffix_prompt="return"
        ):
        fim_sentinel_dict = build_fim_sentinel_dict(tokenizer, suffix_first=False)
        self.fim_prefix_id = fim_sentinel_dict["fim_prefix_id"]
        self.fim_middle_id = fim_sentinel_dict["fim_middle_id"]
        self.fim_suffix_id = fim_sentinel_dict["fim_suffix_id"]
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer
        if check_fn is None:
            check_fn = lambda decoded_generation: any(
                [stop_string in decoded_generation for stop_string in self.eof_strings]
            )
        self.check_fn = check_fn
        self.tau = tau
        self.random_interrupt = random_interrupt
        # we record the suffix prompt tokens before generate the suffix
        self.reset_suffix_prompt(suffix_prompt)
        self.prefixes_for_check_fn = None

    def reset_suffix_prompt(self, suffix_prompt):
        """
            Given an input suffix_prompt (str),
            this function records a corresponding list of tokens
        """
        self.suffix_prompt = suffix_prompt
        if "codellama" in self.tokenizer.name_or_path:
            suffix_prompt_tokens = self.tokenizer(
                "{}{}".format(self.tokenizer.suffix_token, self.suffix_prompt)
            ).input_ids
            prompt_start = suffix_prompt_tokens.index(self.fim_suffix_id)
            suffix_prompt_tokens = suffix_prompt_tokens[prompt_start+1:]
        elif "starcoder" in self.tokenizer.name_or_path:
            suffix_prompt_tokens = self.tokenizer(
                "<fim_suffix>{}".format(self.suffix_prompt)
            ).input_ids
            prompt_start = suffix_prompt_tokens.index(self.fim_suffix_id)
            suffix_prompt_tokens = suffix_prompt_tokens[prompt_start+1:]
        self.suffix_prompt_token = suffix_prompt_tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # check whether there is <fim_prefix> present in input_ids
        has_prefix_id_sequences = torch.any(
            input_ids == self.fim_prefix_id, dim=-1
        )
        if not any(has_prefix_id_sequences):
            # infilling mode is not-invoked. Early exit here.
            return scores
        
        # check whether there is <fim_suffix> present in input_ids
        has_suffix_id_sequences = torch.any(
            input_ids == self.fim_suffix_id, dim=-1
        )
        # check whether there is <fim_middle> present in input_ids
        has_middle_id_sequences = torch.any(
            input_ids == self.fim_middle_id, dim=-1
        )

        has_prefix_suffix_id_sequences = (
            has_prefix_id_sequences &
            has_suffix_id_sequences
        )

        only_has_prefix_id_sequences = (
            has_prefix_id_sequences &
            (~has_suffix_id_sequences) &
            (~has_middle_id_sequences)
        )

        only_has_prefix_suffix_id_sequences = (
            has_prefix_suffix_id_sequences &
            (~has_middle_id_sequences)
        )

        has_prefix_suffix_middle_id_sequences = (
            has_prefix_suffix_id_sequences &
            has_middle_id_sequences
        )

        ############################################
        decoded_generations = self.tokenizer.batch_decode(input_ids[:, self.start_length :])
        if self.prefixes_for_check_fn:
            decoded_generations = [
                    prefix_for_check_fn + decoded_generation
                    for prefix_for_check_fn, decoded_generation 
                    in zip(self.prefixes_for_check_fn, decoded_generations)
                ]


        # heuristic to check whether the suffix
        # is currently being generated.
        not_suffixed_yet_mask = torch.tensor(
            [(not self.suffix_prompt) or self.suffix_prompt not in decoded_generation.split("\n")[-1] for decoded_generation in decoded_generations], 
            dtype=torch.bool, 
            device=input_ids.device
        )

        # check whether the generated code so far is finished
        # (i.e., hits one of stop words)
        finish_mask = torch.tensor(
            [self.check_fn(decoded_generation) for decoded_generation in decoded_generations], 
            dtype=torch.bool, 
            device=input_ids.device
        )
        
        for i in [self.fim_prefix_id, self.fim_middle_id, self.fim_suffix_id]:
            # no <fim_prefix> found. Simply fall back to regular decoding
            # and do not allow the model to generate any FIM-related sentinels
            scores[~has_prefix_id_sequences, i] = -float("inf")

            # if <fim_prefix>, <fim_suffix>, and <fim_middle> are all present,
            # mask all sentinels out
            # because we only want to complete the middle part.
            scores[has_prefix_suffix_middle_id_sequences, i] = -float("inf")

            # if only <fim_prefix> is present,
            # optinally mask all sentinels out (by uncommenting the following line)
            # because sometimes we want the model to 
            # continue predicting next tokens until it is unconfident, 
            # at which point <fim_suffix> is emitted.
            scores[only_has_prefix_id_sequences, i] = -float("inf")

        for i in [self.fim_prefix_id, self.fim_suffix_id]:
            # if only <fim_prefix> and <fim_suffix> are present,
            # mask <fim_prefix> and <fim_suffix> out
            scores[only_has_prefix_suffix_id_sequences, i] = -float("inf")

        # if the entropy of distribution over the next token is large,
        # generate a suffix to create an infilling pattern
        max_prob = torch.softmax(scores, dim=-1).amax(dim=-1)
        _mask = max_prob < self.tau # [B]
        if self.random_interrupt:
            # re-scale the prob such that
            # prob > self.tau => sample_prob = 0.0
            # prob = self.tau => sample_prob = 0.5
            # prob = 0.0 => sample_prob = 1.0
            calibrated_prob = (
                _mask.float() * (1 - (0.5 * max_prob / self.tau))
            )
            uncertain_mask = torch.bernoulli(calibrated_prob).bool()
        else:
            uncertain_mask = _mask # [B]

        ###########################################
        # emit an <SUF> token if 
        # (1) the current sequence only has <PRE>     AND
        # (2) the model becomes uncertain             AND
        # (3) the `suffix` has not been generated yet AND
        # (4) the generation is not finished yet
        force_suffix_mask = (
            only_has_prefix_id_sequences &
            uncertain_mask &
            not_suffixed_yet_mask &
            (~finish_mask)
        )
        # produce a point mass over <fim_suffix>
        scores[force_suffix_mask, :] = -float("inf")
        scores[force_suffix_mask, self.fim_suffix_id] = 0

        ############################################
        # start generating suffix prompt
        input_ids_len = input_ids.shape[-1]
        backward_steps = min(input_ids_len, len(self.suffix_prompt_token))
        for i in range(backward_steps):
            last_token_being_fim_suffix = (
                (input_ids[:, -1-i] == self.fim_suffix_id)
            )
            force_suffix_prompt_mask = (
                only_has_prefix_suffix_id_sequences &
                last_token_being_fim_suffix
            )
            scores[force_suffix_prompt_mask, :] = -float("inf")
            scores[force_suffix_prompt_mask, self.suffix_prompt_token[i]] = 0

        ############################################
        # Generate <MID> if 
        # 1) there is only prefix and suffix sentinels AND
        # 2) the suffix is finished
        force_middle_mask = (
            only_has_prefix_suffix_id_sequences &
            finish_mask
        )
        # produce a point mass over <fim_middle>
        scores[force_middle_mask, :] = -float("inf")
        scores[force_middle_mask, self.fim_middle_id] = 0
        return scores


def rstrip_prompt(prompt, tokenizer):
    # this function removes extra trailing space
    # to facilitate tokenization
    if ("starcoder" in tokenizer.name_or_path):
        prompt = prompt.rstrip()
    elif ("codellama" in tokenizer.name_or_path):
        # Codellama tokenizer tokenize newline+following space 
        # `\n   ` as [`\n`, `   `]. So keep the last newline
        stripped_prompt = prompt.rstrip()
        trailing_spaces = prompt[len(stripped_prompt):]
        # remove possible indentation after the last \n
        if "\n" in trailing_spaces:
            trailing_spaces = re.sub(r'(\n[ \t]+)(?=\n*$)', '\n', trailing_spaces)
        else:
            # either trailing_space = empty or does not contain \n
            trailing_spaces = ""
        prompt = stripped_prompt + trailing_spaces
    return prompt

def build_l2r_prompt(
    result_dict,
    default_suffix_prompt,
    task,
    task_id,
    prompt,
):
    """
        This function constructs the input for
        left-to-right generation:
        @input:  prefix+infill
        @output: suffix
    """

    prefix, infill, suffix = result_dict["split"]
    # clean up prefix+middle
    # and extract the **generated** content
    gen_middle = task.postprocess_generation(
        prefix + infill,
        int(task_id)
    )[len(prompt):]
    if result_dict["fim_suffix_present"] and result_dict["fim_middle_present"]:
        if not result_dict["fim_ending_present"]:
            # infilled but not finished successfully.
            # the generated middle possibly overrides the suffix
            # and completes the code without joining suffix.

            # check whether the generation contains suffix_prompt
            last_sp_pos = gen_middle.rfind(default_suffix_prompt)
            if last_sp_pos == -1:
                # if not found, this means the generated middle does not
                # follow the constraint, and thus we clear the context. 
                extracted_middle = ""
            else:
                # otherwise, we extract the middle from the complete program
                # for subsequent left-to-right decoding.
                extracted_middle = gen_middle[:last_sp_pos]
        else:
            # successfully infilled.
            extracted_middle = gen_middle

    elif not result_dict["fim_suffix_present"] and not result_dict["fim_middle_present"]:
        # the model completed the program without invoking self-infilling.
        # extract the middle that ends with the last occurrence of suffix prompt
        last_sp_pos = gen_middle.rfind(default_suffix_prompt)

        if last_sp_pos == -1:
            extracted_middle = ""
        else:
            last_sp_pos = last_sp_pos
            extracted_middle = gen_middle[:last_sp_pos]
    
    elif result_dict["fim_suffix_present"] and not result_dict["fim_middle_present"]:
        # the suffix does not stop generating.
        extracted_middle = gen_middle
    
    elif not result_dict["fim_suffix_present"] and result_dict["fim_middle_present"]:
        # this case would not happen, 
        # since the probability of generating <fim_middle> is masked to 0 
        # if there is not <fim_suffix> yet
        extracted_middle = ""
    
    return extracted_middle


def build_si_prompt(
    l2r_gen_code, 
    suffix,
    _revert_to_si_code, 
    default_suffix_prompt,
    prompt,
    fim_sentinel_dict,
    suffix_split_mode="default",
):
    """
        This function constructs the input for
        self-infilling generation:
        @input:  suffix
        @output: suffix_completion + prefix + middle
    """
    #############################################################
    ### 1. process the suffix extracted from left-to-right generation
    completion = l2r_gen_code[len(prompt):]
    # split generated code into lines of code (merge consecutive newlines)
    gen_code_lines = compact_lines(completion.split("\n"))
    if not suffix.strip():
        if not _revert_to_si_code:
            # if the l2r generated suffix is empty, rebuild from completion
            last_sp_pos = completion.find(default_suffix_prompt)
            if last_sp_pos == -1:
                # l2r not generating a proper suffix. reset to empty suffix
                suffix = ""
            else:
                # the l2r mode generates a non-empty suffix.
                if suffix_split_mode == "vanilla":
                    suffix = completion[last_sp_pos:]
                elif suffix_split_mode in ["extended", "half"]:
                    # split the completion and use the latter half as the suffix
                    suffix = "\n".join(
                        gen_code_lines[int(0.5 * len(gen_code_lines)):]
                    )
                else:
                    raise ValueError("Invalid mode")
        else:
            # if the previously self-infilled suffix is empty, reset to default suffix prompt
            suffix = ""
    else:
        # suffixes are non-empty
        if suffix_split_mode == "vanilla":
            suffix = suffix
        elif suffix_split_mode in ["extended", "half"]:
            suffix = "\n".join(
                gen_code_lines[int(0.5 * len(gen_code_lines)):]
            )
        else:
            raise ValueError("Invalid mode")

    #############################################################
    ### 2. unpack the suffix into (suffix_prompt, suffix_completion)
    last_sp_pos = suffix.rfind(default_suffix_prompt)
    if last_sp_pos == -1:
        # the suffix does not contain default suffix_prompt.
        # reset it to default_suffix_prompt
        suffix_prompt = default_suffix_prompt
    else:
        if suffix_split_mode in ["vanilla", "extended"]:
            last_sp_pos = last_sp_pos + len(default_suffix_prompt)
            suffix_prompt = suffix[:last_sp_pos]
        elif suffix_split_mode == "half":
            # note that we append a <MID> sentinel to indicate that the
            # suffix context is complete
            suffix_prompt = suffix + fim_sentinel_dict["fim_middle"]
        else:
            raise ValueError("Invalid mode")
        
        # if the resulting suffix prompt is still empty,
        # reset to default_suffix_prompt
        if not suffix_prompt.strip():
            suffix_prompt = default_suffix_prompt
    return suffix_prompt

@torch.inference_mode()
def self_infill_completion(
    input_dict,
    model,
    tokenizer,
    task,
    device,
    batch_size,
    gen_kwargs,
    tokenizer_kwargs,
    default_suffix_prompt,
    max_iters=1,
    suffix_split_mode="vanilla",
):
    cur_gen_kwargs = copy.deepcopy(gen_kwargs)
    cur_gen_kwargs.update({"num_return_sequences": 1})

    prompt = input_dict["prompt"]
    task_id = input_dict["task_id"]
    if "default_suffix_prompt" in input_dict:
        default_suffix_prompt = input_dict["default_suffix_prompt"]
        cur_gen_kwargs["logits_processor"][0].reset_suffix_prompt(default_suffix_prompt)

    fim_sentinel_dict = build_fim_sentinel_dict(tokenizer, suffix_first=False)
    fim_prefix = fim_sentinel_dict["fim_prefix"]
    fim_suffix = fim_sentinel_dict["fim_suffix"]
    fim_middle = fim_sentinel_dict["fim_middle"]

    if hasattr(task, "_gsm_type") and task._gsm_type == "cot":
        # used in non-code tasks like CoT-GSM8k
        def is_degenerate(code):
            return (
                check_repeat(code) or
                (
                    default_suffix_prompt not in code and
                    default_suffix_prompt not in task.stop_words
                )
            )
    else:
        def is_degenerate(code):
            return (
                check_degenerate(code) or 
                check_repeat(code) or
                default_suffix_prompt not in code
            )
    
    si_prompts = [f"{fim_prefix}{prompt}" for _ in range(batch_size)]

    no_looping = False
    if max_iters == 0:
        # single self-infilling call
        no_looping = True
        max_iters = 1
    
    for cur_iter in range(max_iters):
        ##########################################
        # 1. setup self-infilling
        si_inputs = tokenizer(
            si_prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            **tokenizer_kwargs,
        ).to(device)

        si_input_len = si_inputs.attention_mask.sum(dim=-1).max().item()
        prefixes_for_check_fn = [si_prompt[len(f"{fim_prefix}{prompt}"):] for si_prompt in si_prompts]
        cur_gen_kwargs["logits_processor"][0].start_length = si_input_len
        cur_gen_kwargs["stopping_criteria"][0].start_length = si_input_len
        cur_gen_kwargs["logits_processor"][0].prefixes_for_check_fn = prefixes_for_check_fn
        cur_gen_kwargs["stopping_criteria"][0].prefixes_for_check_fn = None

        si_generated_tokens = model.generate(
            input_ids=si_inputs.input_ids, 
            attention_mask=si_inputs.attention_mask,
            **cur_gen_kwargs,
        )
        si_generated_code = tokenizer.batch_decode(
            si_generated_tokens,
            skip_special_tokens=False, 
            clean_up_tokenization_spaces=False
        )

        proc_si_generated_code = []
        prev_si_suffixes = []

        l2r_prompts = []

        for si_gen_code in si_generated_code:

            ##############################################
            # 2. parse and post-process self-infill generation
            result_dict = self_infill_split(tokenizer, si_gen_code, fim_sentinel_dict)

            prefix, infill, suffix = result_dict["split"]

            prev_si_suffixes.append(suffix)

            # clean up extra tokens after stop tokens and
            # store previous generation for fallback later on
            si_gen_code = prefix + infill + suffix
            si_gen_code = task.postprocess_generation(
                si_gen_code,
                int(task_id)
            )
            proc_si_generated_code.append(si_gen_code)
            
            ##############################################
            # 3. generate prompts for left-to-right suffix generation
            new_prefix_middle = build_l2r_prompt(
                result_dict,
                default_suffix_prompt,
                task,
                task_id,
                prompt,
            )
            new_prefix_middle = rstrip_prompt(new_prefix_middle, tokenizer)
            l2r_prompt = f"{prompt}{new_prefix_middle}"
            l2r_prompts.append(l2r_prompt)

        si_generated_code = proc_si_generated_code
        if no_looping:
            return si_generated_code
        
        ##############################################
        # 4. Setup left-to-right generation
        l2r_inputs = tokenizer(
            l2r_prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            **tokenizer_kwargs
        ).to(device)

        l2r_input_len = l2r_inputs.attention_mask.sum(dim=-1).max().item()
        prefixes_for_check_fn = [l2r_prompt[len(prompt):] for l2r_prompt in l2r_prompts]
        cur_gen_kwargs["logits_processor"][0].start_length = l2r_input_len
        cur_gen_kwargs["stopping_criteria"][0].start_length = l2r_input_len
        cur_gen_kwargs["logits_processor"][0].prefixes_for_check_fn = None
        cur_gen_kwargs["stopping_criteria"][0].prefixes_for_check_fn = prefixes_for_check_fn

        l2r_generated_tokens = model.generate(
            input_ids=l2r_inputs.input_ids, 
            attention_mask=l2r_inputs.attention_mask,
            **cur_gen_kwargs,
        )

        l2r_generated_code = tokenizer.batch_decode(
            l2r_generated_tokens,
            # set to True as l2r does not involve sentinels
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        ##############################################
        # 5. post-process left-to-right generated code
        # and parse the output suffix.
        # Fall back to previous self-infill generation when necessary
        l2r_generated_code = [
            task.postprocess_generation(
                l2r_gen_code,
                int(task_id)
            )
            for l2r_gen_code in l2r_generated_code
        ]

        revert_to_si_code = []
        l2r_suffixes = []
        _generated_code = []
        for (
            l2r_gen_code,
            l2r_prompt,
            si_gen_code,
            si_suffix,
        ) in zip(
            l2r_generated_code,
            l2r_prompts,
            si_generated_code,
            prev_si_suffixes
        ):
            cur_completion = l2r_gen_code[len(prompt):]
            cur_degenerate = is_degenerate(cur_completion)
            revert_to_si_code.append(cur_degenerate)
            if cur_degenerate:
                l2r_suffixes.append(si_suffix)
                _generated_code.append(si_gen_code)
            else:
                l2r_suffixes.append(l2r_gen_code[len(l2r_prompt):])
                _generated_code.append(l2r_gen_code)
                
        l2r_generated_code = _generated_code
        
        ##############################################
        # 6. either exit looping after the last iteration,
        #    or construct new self-infill prompts 
        if cur_iter == max_iters - 1:
            return l2r_generated_code
        else:
            si_prompts = []
            for (
                l2r_gen_code, 
                suffix, 
                _revert_to_si_code,
            ) in zip(
                l2r_generated_code, 
                l2r_suffixes, 
                revert_to_si_code,
            ):
                suffix_prompt = build_si_prompt(
                    l2r_gen_code, 
                    suffix, 
                    _revert_to_si_code, 
                    default_suffix_prompt,
                    prompt,
                    fim_sentinel_dict,
                    suffix_split_mode=suffix_split_mode,
                )
                si_prompt = f"{fim_prefix}{prompt}{fim_suffix}{suffix_prompt}"
                si_prompts.append(si_prompt)

