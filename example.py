import torch
import argparse
import sys
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    StoppingCriteriaList,
    LogitsProcessorList
)
from lm_eval.generation_pipelines.self_infill_utils import (
    SelfInfillingLogitsProcessor,
    SelfInfillEndOfFunctionCriteria,
)

from lm_eval.utils import (
    build_fim_sentinel_dict,
    self_infill_split
)

def setup_generation_config(
        tokenizer, 
        suffix_prompt, 
        stop_words,
        args,
    ):
    """
    Setup configuration for generate.
    """
    gen_kwargs = {
        "do_sample": True,
        "temperature": 0.2,
        "top_p": 0.95,
        "max_new_tokens": 128,
        "stopping_criteria": StoppingCriteriaList([
            SelfInfillEndOfFunctionCriteria(0, stop_words, tokenizer)
        ]),
        "logits_processor": LogitsProcessorList([
            SelfInfillingLogitsProcessor(
                0, stop_words, tokenizer, 
                tau=args.self_infill_tau,
                suffix_prompt=suffix_prompt
            )
        ])
    }
    tokenizer_kwargs = {
        "return_token_type_ids": False,
    }
    return gen_kwargs, tokenizer_kwargs

def main(args):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, 
        trust_remote_code=True, 
        use_fast=False
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        trust_remote_code=True, 
        device_map = "auto"
    ).to(args.device)

    # Read provided prompt from a file
    if args.prompt_file:
        try:
            with open(args.prompt_file, "r") as f:
                prompt = f.read()
        except IOError:
            print(f"Error: Could not read file {args.prompt_file}")
            sys.exit(1)
    else:
        print("No user prompt provided. Exiting.")
        sys.exit(1)

    # Read provided suffix prompt from the specified file
    if args.suffix_prompt_file:
        try:
            with open(args.suffix_prompt_file, "r") as f:
                suffix_prompt = f.read().strip()
        except IOError:
            print(f"Error: Could not read file {args.suffix_prompt_file}")
            sys.exit(1)
    else:
        print("No user prompt provided. Defaults to empty.")
        suffix_prompt = ""


    # setup infilling sentinel tokens
    fim_sentinel_dict = build_fim_sentinel_dict(tokenizer)
    fim_prefix = fim_sentinel_dict["fim_prefix"]

    si_prompts = f"{fim_prefix}{prompt}"

    # setup stop words
    stop_words = [
        "</code>",
        tokenizer.eos_token, 
        fim_sentinel_dict["fim_ending"]
    ]
    gen_kwargs, tokenizer_kwargs = setup_generation_config(
        tokenizer, 
        suffix_prompt, 
        stop_words,
        args
    )

    # Tokenize inputs
    si_inputs = tokenizer(
        si_prompts, 
        padding=True, 
        truncation=True, 
        return_tensors="pt", 
        **tokenizer_kwargs
    ).to(args.device)

    # Generate code
    si_generated_tokens = model.generate(
        input_ids=si_inputs.input_ids, 
        attention_mask=si_inputs.attention_mask,
        **gen_kwargs,
    )
    si_generated_code = tokenizer.batch_decode(
        si_generated_tokens,
        skip_special_tokens=False, 
        clean_up_tokenization_spaces=False
    )[0]

    print(">>> Raw self-infilled code:\n", si_generated_code)

    # Parse self-infilled generation to code
    result_dict = self_infill_split(tokenizer, si_generated_code, fim_sentinel_dict)
    prefix, infill, suffix = result_dict["split"]
    code = prefix + infill + suffix

    # the self-infilling execution status can be inspected from result_dict
    # e.g., result_dict["fim_ending_present"] indicates whether 
    # the infilled middle successfully connects to the prefix and the suffix.
    if result_dict["fim_suffix_present"]:
        print("[self-infill] self-infilling interruption invoked")
        if result_dict["fim_middle_present"] and result_dict["fim_ending_present"]:
            print("[self-infill] successfully self-infilled")
        elif result_dict["fim_middle_present"] and not result_dict["fim_ending_present"]:
            print("[self-infill] self-infilling fails to join the suffix")
        elif not result_dict["fim_middle_present"]:
            print("[self-infill] self-infilling fails to produce a suffix")
        else:
            print("[self-infill] should not happen here")
    else:
        print("[self-infill] NOT invoking self-infilling")

    print("\n\n>>> Final code:\n", code)

if __name__ == "__main__":
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Self-infilling code generation')
    parser.add_argument('--model', type=str, default="codellama/CodeLlama-7b-hf", help='HF Model name or path')
    parser.add_argument('--device', type=str, default="cuda", help='Device to use (cuda or cpu)')
    parser.add_argument('--self-infill-tau', type=float, default=0.25, help='threshold tau for self-infilling interruption')
    parser.add_argument('--prompt-file', type=str, default=None, help='File path for the user prompt')
    parser.add_argument('--suffix-prompt-file', type=str, default=None, help='File path for the custom suffix prompt')
    args = parser.parse_args()
    main(args)