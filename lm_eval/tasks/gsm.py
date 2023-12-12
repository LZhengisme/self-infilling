"""PAL: Program-aided Language Models
https://arxiv.org/abs/2211.10435

GSM-8k: Training Verifiers to Solve Math Word Problems
https://arxiv.org/abs/2110.14168

In PaL, Large Language Model solves reasoning problems that involve complex arithmetic and procedural tasks by generating 
reasoning chains of text and code.This offloads the execution of the code to a program runtime, in our case, a Python interpreter.

This task implements PAL methodology to evaluate GSM-8k and GSM-Hard benchmarks.
"""

import json
import os
import re
from enum import Enum
from typing import Union
import warnings
from collections import Counter, defaultdict
from evaluate import load

from lm_eval.base import Task
from lm_eval.tasks.custom_metrics.pal_metric.pal_code_exec import compute

_CITATION = """
@article{gao2022pal,
  title={PAL: Program-aided Language Models},
  author={Gao, Luyu and Madaan, Aman and Zhou, Shuyan and Alon, Uri and Liu, Pengfei and Yang, Yiming and Callan, Jamie and Neubig, Graham},
  journal={arXiv preprint arXiv:2211.10435},
  year={2022}
}

@article{cobbe2021gsm8k,
  title={Training Verifiers to Solve Math Word Problems},
  author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and Hesse, Christopher and Schulman, John},
  journal={arXiv preprint arXiv:2110.14168},
  year={2021}
}
"""
# Number of few shot examples to consider
NUM_SHOTS = 8

PAL_EXAMPLARS = {
    "questions": ["Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
                  "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
                  "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
                  "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
                  "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
                  "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
                  "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
                  "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?"],
    "solutions": ["    money_initial = 23\n    bagels = 5\n    bagel_cost = 3\n    money_spent = bagels * bagel_cost\n    money_left = money_initial - money_spent\n    result = money_left\n    return result",
                "    golf_balls_initial = 58\n    golf_balls_lost_tuesday = 23\n    golf_balls_lost_wednesday = 2\n    golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday\n    result = golf_balls_left\n    return result",
                "    computers_initial = 9\n    computers_per_day = 5\n    num_days = 4  # 4 days between monday and thursday\n    computers_added = computers_per_day * num_days\n    computers_total = computers_initial + computers_added\n    result = computers_total\n    return result",
                "    toys_initial = 5\n    mom_toys = 2\n    dad_toys = 2\n    total_received = mom_toys + dad_toys\n    total_toys = toys_initial + total_received\n    result = total_toys\n    return result",
                "    jason_lollipops_initial = 20\n    jason_lollipops_after = 12\n    denny_lollipops = jason_lollipops_initial - jason_lollipops_after\n    result = denny_lollipops\n    return result",
                "    leah_chocolates = 32\n    sister_chocolates = 42\n    total_chocolates = leah_chocolates + sister_chocolates\n    chocolates_eaten = 35\n    chocolates_left = total_chocolates - chocolates_eaten\n    result = chocolates_left\n    return result",
                "    cars_initial = 3\n    cars_arrived = 2\n    total_cars = cars_initial + cars_arrived\n    result = total_cars\n    return result",
                "    trees_initial = 15\n    trees_after = 21\n    trees_added = trees_after - trees_initial\n    result = trees_added\n    return result"]
}

COT_EXAMPLARS = {
    "questions": ["Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
                  "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
                  "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
                  "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
                  "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
                  "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
                  "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
                  "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?"],
    "solutions": ["Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.",
                "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.",
                "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.",
                "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.",
                "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.",
                "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.",
                "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.",
                "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.",
    ],
    "short_answers": [
        "8",
        "33",
        "29",
        "9",
        "8",
        "39",
        "5",
        "6"
    ]
}

def create_all_tasks():
    # for GSM-8k tasks, majority voting will be used if multiple samples are available
    task_dict = {
        f"pal-gsm8k": PALGsm8k,
        f"cot-gsm8k": CoTGsm8k,
    }
    return task_dict

class PALGsm8k(Task):

    DATASET_PATH = "gsm8k"
    DATASET_NAME = "main"
    SPLIT = "test"

    def __init__(self):
        stop_words = ["\n\n\n", "\nQ:"]
        requires_execution = True
        super().__init__(stop_words, requires_execution)
        self._gsm_type = "pal"

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        if self.SPLIT:
            return self.dataset[self.SPLIT]
        return self.dataset

    @staticmethod
    def few_shot_prompt(text):
        """Two shot prompt format as source & target language documentation"""
        prompt = ""
        for question, solution in zip(
            PAL_EXAMPLARS["questions"][:NUM_SHOTS], PAL_EXAMPLARS["solutions"][:NUM_SHOTS]
        ):
            prompt += f'''Q: {question}\n\n# solution in Python:\n\n\ndef solution():\n    """{question}"""\n{solution}\n\n\n\n\n\n'''
        prompt += f"""Q: {text}\n\n# solution in Python:\n\n\n"""
        return prompt

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        prompt = self.few_shot_prompt(doc["question"])
        return prompt

    @staticmethod
    def parse_target(txt):
        def _is_num(txt):
            try:
                txt = txt.replace(",", "")
                float(txt)
            except ValueError:
                return False
            return True

        txt = txt.strip()
        if _is_num(txt):
            txt = txt.replace(",", "")
            try:
                num = int(txt)
            except ValueError:
                num = float(txt)
            return num
        return txt

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        _answer_delim = "#### "
        target = doc["answer"].split(_answer_delim)[-1]
        return self.parse_target(target)

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
        prompt = self.get_prompt(self.dataset[self.SPLIT][idx])
        generation = generation[len(prompt) :]
        return prompt + self._stop_at_stop_token(generation, self.stop_words)

    def extract_completion(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for this task)
        """
        output = generation.split("# solution in Python:", NUM_SHOTS + 1)[-1].strip()
        if "Q:" in output:
            output = output.split("Q:")[0]
        output += "\nprint(solution())"
        return output

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(float)
            list of references
        """
        results = compute(
            references=references,
            predictions=generations,
        )
        return results

class CoTGsm8k(Task):

    DATASET_PATH = "gsm8k"
    DATASET_NAME = "main"
    SPLIT = "test"

    def __init__(self):
        stop_words = ["\n\n", "\nQ:"]
        requires_execution = True
        super().__init__(stop_words, requires_execution)
        self.suffix_prompt = "\nTherefore, the answer is"
        self._gsm_type = "cot"

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        if self.SPLIT:
            return self.dataset[self.SPLIT]
        return self.dataset

    def few_shot_prompt(self, text):
        """Two shot prompt format as source & target language documentation"""
        prompt = ""
        for question, solution, answer in zip(
            COT_EXAMPLARS["questions"][:NUM_SHOTS], 
            COT_EXAMPLARS["solutions"][:NUM_SHOTS],
            COT_EXAMPLARS["short_answers"][:NUM_SHOTS],
        ):
            prompt += f"Q: {question}\nA: {solution} {self.suffix_prompt} {answer}.\n\n"
        prompt += f"""Q: {text}\nA:"""
        return prompt

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        prompt = self.few_shot_prompt(doc["question"])
        return prompt

    @staticmethod
    def parse_target(txt):
        def _is_num(txt):
            try:
                txt = txt.replace(",", "")
                float(txt)
            except ValueError:
                return False
            return True

        txt = txt.strip()
        if _is_num(txt):
            txt = txt.replace(",", "")
            try:
                num = int(txt)
            except ValueError:
                num = float(txt)
            return num
        return txt

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        _answer_delim = "#### "
        target = doc["answer"].split(_answer_delim)[-1]
        return self.parse_target(target)

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
        prompt = self.get_prompt(self.dataset[self.SPLIT][idx])
        generation = generation[len(prompt) :]
        return prompt + self._stop_at_stop_token(generation, self.stop_words)

    def extract_completion(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for this task)
        """
        prompt = self.get_prompt(self.dataset[self.SPLIT][idx])
        generation = generation[len(prompt) :]
        return generation

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(float)
            list of references
        """
        scores = []
        # Number of code generated that failed execution.
        errored = 0
        for task_id, (gens, ref) in enumerate(zip(generations, references)):
            answers = []
            ref = float(ref)
            for gen in gens:
                ans = self.answer_cleansing(gen)
                try:
                    answers.append(float(ans))
                except ValueError as e:
                    errors = True
            if answers:
                if len(answers) > 1:
                    counter = Counter(answers)
                    answers = [counter.most_common()[0][0]]
                answer = answers[0]
                try:
                    score = 1 if abs(answer - ref) < 1e-3 else 0
                except ValueError as e:
                    errored += 1
                    score = 0
            else:
                answer = None
                score = 0
                errored += 1
            scores.append(score)
        return {"accuracy": sum(scores) / len(scores), "num_failed_execution": errored}


    def answer_cleansing(self, gen):

        print("pred_before : " + gen)
        
        gens = gen.split(self.suffix_prompt)
        answer_flag = True if len(gens) > 1 else False 
        pred = gens[-1]

        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]

        # If there is no candidate in list, null is set.
        if len(pred) == 0:
            pred = ""
        else:
            if answer_flag:
                # choose the first element in list ...
                pred = pred[0]
            else:
                # choose the last element in list ...
                pred = pred[-1]
        
        # (For arithmetic tasks) if a word ends with period, it will be omitted ...
        if pred != "":
            if pred[-1] == ".":
                pred = pred[:-1]
        
        print("pred_after : " + pred)
        
        return pred


