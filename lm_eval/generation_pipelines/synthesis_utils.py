import math
import numpy as np
from multiprocessing.sharedctypes import Value
import torch
from tqdm import tqdm

from lm_eval.utils import (
    build_fim_sentinel_dict,
    self_infill_split,
)

def vanilla_completion(
    input_dict,
    model,
    tokenizer,
    task,
    device,
    batch_size,
    gen_kwargs,
    tokenizer_kwargs,
):
    task_id = input_dict["task_id"]
    prompt = input_dict["prompt"]
    with torch.no_grad():
        outputs = tokenizer(
            prompt,
            padding=True,
            truncation=True,
            return_tensors="pt",
            **tokenizer_kwargs
        )
        batch = {
            "ids": outputs.input_ids.to(device),
            "task_id": task_id,
            "input_len": outputs.attention_mask.sum(),
        }
        max_input_len = batch["input_len"].item()
        if task.stop_words:
            if gen_kwargs.get("stopping_criteria", None) is not None:
                # Set the start_length after which to check for stopping to be the longest input ignoring padding
                gen_kwargs["stopping_criteria"][0].start_length = max_input_len
        if hasattr(task, "max_length_multiplier") and task.max_length_multiplier:
            idx = 1 if task.stop_words else 0
            gen_kwargs["stopping_criteria"][idx].input_length = max_input_len
        inputs = batch["ids"][:, : batch["input_len"]]
        generated_tokens = model.generate(
            input_ids=inputs,
            num_return_sequences=batch_size,
            **gen_kwargs,
        )
        generated_tokens = generated_tokens.cpu().numpy()
        generated_code = []
        for generated_token in generated_tokens:
            if generated_token[0] == tokenizer.bos_token_id:
                generated_token = generated_token[1:]

            generated_code.append(tokenizer.decode(
                generated_token, skip_special_tokens=False, clean_up_tokenization_spaces=False
            ))
    return generated_code

def vanilla_infilling(
    input_dict,
    model,
    tokenizer,
    task,
    device,
    batch_size,
    gen_kwargs,
    tokenizer_kwargs,
):
    suffix_first = False
    task_id = input_dict["task_id"]
    prefix = input_dict["prefix"]
    suffix = input_dict["suffix"]
    fim_sentinel_dict = build_fim_sentinel_dict(tokenizer, suffix_first=suffix_first)
    fim_prefix = fim_sentinel_dict["fim_prefix"]
    fim_suffix = fim_sentinel_dict["fim_suffix"]
    fim_middle = fim_sentinel_dict["fim_middle"]
    if suffix_first:
        prompt = f"{fim_prefix}{fim_suffix}{suffix}{fim_middle}{prefix}"
    else:
        prompt = f"{fim_prefix}{prefix}{fim_suffix}{suffix}{fim_middle}"
    with torch.no_grad():
        outputs = tokenizer(
            prompt,
            padding=True,
            truncation=True,
            return_tensors="pt",
            **tokenizer_kwargs
        )
        batch = {
            "ids": outputs.input_ids.to(device),
            "task_id": task_id,
            "input_len": outputs.attention_mask.sum(),
        }
        max_input_len = batch["input_len"].item()
        if task.stop_words:
            if gen_kwargs.get("stopping_criteria", None) is not None:
                gen_kwargs["stopping_criteria"][0].start_length = max_input_len
        if hasattr(task, "max_length_multiplier") and task.max_length_multiplier:
            idx = 1 if task.stop_words else 0
            gen_kwargs["stopping_criteria"][idx].input_length = max_input_len
        inputs = batch["ids"][:, : batch["input_len"]]
        generated_tokens = model.generate(
            input_ids=inputs,
            num_return_sequences=batch_size,
            **gen_kwargs,
        )
        generated_tokens = generated_tokens.cpu().numpy()
        generated_code = []
        for generated_token in generated_tokens:
            if generated_token[0] == tokenizer.bos_token_id:
                generated_token = generated_token[1:]
            gen_code = tokenizer.decode(
                generated_token, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            result_dict = self_infill_split(tokenizer, gen_code, fim_sentinel_dict)
            prefix, infill, suffix =  result_dict["split"]
            generated_code.append(infill)
    return generated_code
