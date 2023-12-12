"""Program Synthesis with Large Language Models
https://arxiv.org/abs/2108.07732

The benchmark consists of around 1,000 crowd-sourced Python programming problems, 
designed to be solvable by entry level programmers, covering programming fundamentals, 
standard library functionality, and so on. Each problem consists of a task description, 
code solution and 3 automated test cases. As described in the paper, a subset of the data
has been hand-verified by the authors.

Homepage:: https://github.com/google-research/google-research/tree/master/mbpp
"""

import re
import os
import json
from evaluate import load

from lm_eval.base import Task

_CITATION = """
@article{austin2021program,
  title={Program Synthesis with Large Language Models},
  author={Austin, Jacob and Odena, Augustus and Nye, Maxwell and Bosma, Maarten and Michalewski, Henryk and Dohan, David and Jiang, Ellen and Cai, Carrie and Terry, Michael and Le, Quoc and others},
  journal={arXiv preprint arXiv:2108.07732},
  year={2021}
}
"""

def create_all_tasks():
    """Creates a dictionary of tasks from a list of levels
    :return: {task_name: task}
        e.g. {multiple-py: Task, multiple-java: Task}
    """
    return {"mbpp-commentformat": create_task("comment"), "mbpp": create_task("base")}


def create_task(prompt_format):
    class MBPP(GeneralMBPP):
        def __init__(self):
            super().__init__(prompt_format)

    return MBPP

EXAMPLARS = [
    {
        "task_id": 2,
        "text": "Write a function to find the similar elements from the given two tuple lists.",
        "code": "def similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res) ",
        "test_list": [
            "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
            "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
            "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)",
        ],
        "test_setup_code": "",
        "challenge_test_list": [],
    },
    {
        "task_id": 3,
        "text": "Write a python function to identify non-prime numbers.",
        "code": "import math\r\ndef is_not_prime(n):\r\n    result = False\r\n    for i in range(2,int(math.sqrt(n)) + 1):\r\n        if n % i == 0:\r\n            result = True\r\n    return result",
        "test_list": [
            "assert is_not_prime(2) == False",
            "assert is_not_prime(10) == True",
            "assert is_not_prime(35) == True",
        ],
        "test_setup_code": "",
        "challenge_test_list": [],
    },
    {
        "task_id": 4,
        "text": "Write a function to find the largest integers from a given list of numbers using heap queue algorithm.",
        "code": "import heapq as hq\r\ndef heap_queue_largest(nums,n):\r\n  largest_nums = hq.nlargest(n, nums)\r\n  return largest_nums",
        "test_list": [
            "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] ",
            "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] ",
            "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]",
        ],
        "test_setup_code": "",
        "challenge_test_list": [],
    },
]

BASE_INSTRUCTION = """
You are an expert Python programmer, and here is your task: {instruction} Your code should pass these tests:\n\n{test_cases}
[BEGIN]
{code}"""

COMMENT_INSTRUCTION = '"""\n{instruction} Your code should pass these tests:\n\n{test_cases}\n"""\n{code}'

class GeneralMBPP(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "mbpp"

    def __init__(self, prompt_format):
        super().__init__(
            stop_words=["\nclass", "\nassert", '\n"""', "\nprint", "\nif", "\n<|/", "\n```"],
            requires_execution=True,
        )
        self.strip_prompt = False
        self.prompt_format = prompt_format
        assert self.prompt_format in ["comment", "base"]
        if self.prompt_format == "comment":
            self.inst_template = COMMENT_INSTRUCTION
            self.end_word = "\n\n"
        elif self.prompt_format == "base":
            self.inst_template = BASE_INSTRUCTION
            self.stop_words.append("\n[DONE]")
            self.stop_words.append("\nYou")
            self.end_word = "\n[DONE]"

        self.examplar_prompt = ""
        for i, d in enumerate(EXAMPLARS):
            self.examplar_prompt += (
                self.inst_template.format(
                    instruction=d["text"], 
                    test_cases="\n".join(d["test_list"]), 
                    code=d["code"]
                ) + self.end_word
            )

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        dataset = self.dataset["test"]
        # the wrong split of mbpp can be loaded with old datasets cache
        assert (
            len(dataset) == 500
        ), "please ensure you have the latest version of MBPP dataset, try deleting its old cache"
        return dataset

    def get_prompt(self, doc):
        """Builds the 3-shot prompt for the LM to generate from.
        MBPP prompt is built following to InCoder (Fried et al.) approach
        prompt = docstring that includes one test
        """
        prompt = self.examplar_prompt + self.inst_template.format(
            instruction=doc["text"], 
            test_cases="\n".join(doc["test_list"]),
            code=""
        )
        if self.strip_prompt:
            return prompt.strip()
        else:
            return prompt

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return "\n".join(doc["test_list"])

    @staticmethod
    def _stop_at_stop_token(decoded_string, stop_tokens):
        """
        Produces the prefix of decoded_string that ends at the first occurrence of
        a stop_token.
        WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
        itself.
        """
        min_stop_index = len(decoded_string)
        for stop_token in stop_tokens:
            stop_index = decoded_string.find(stop_token)
            if stop_index != -1 and stop_index < min_stop_index:
                min_stop_index = stop_index
        return decoded_string[:min_stop_index]

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
        """
        prompt = self.get_prompt(self.dataset["test"][idx])
        generation = generation[len(prompt) :]
        return prompt + self._stop_at_stop_token(generation, self.stop_words)

    def extract_completion(self, generation, idx):
        prompt = self.get_prompt(self.dataset["test"][idx])
        extracted_generation = generation[len(prompt) :]
        return extracted_generation

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        code_metric = load("code_eval")
        results, logs = code_metric.compute(
            references=references,
            predictions=generations,
            num_workers=int(os.getenv("HF_CODE_EVAL_NUM_PROC", "1")),
        )

        with open(os.getenv("RUN_STATS_SAVE_PATH", "logs_humaneval.json"), "w") as f:
            json.dump(logs, f, indent=4, ensure_ascii=False)

        return results
