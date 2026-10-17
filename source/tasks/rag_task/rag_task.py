"""HotpotQA distractor questions with their original answers and supporting sentences."""

from copy import deepcopy
from functools import lru_cache
import json
import os
from pathlib import Path
import random

from tasks.arbigraph_tools import comparator, validate_limit


DATA_PATH = Path(os.environ.get(
    "ARBIGRAPH_HOTPOTQA_PATH",
    Path(__file__).with_name("hotpot_train_v1.1.json"),
))
MAX_BANK_DRAWS = 400
COMPARATOR = "hotpotqa"
RAG_POOL_SIZE = 200
RAG_POOL_SEED = 0
same_answer = comparator(COMPARATOR)


class RAGTask:
    def __init__(self, path):
        with Path(path).open(encoding="utf-8") as handle:
            self.samples = json.load(handle)
        if not isinstance(self.samples, list) or not self.samples:
            raise ValueError("Expected a nonempty list of labeled HotpotQA samples.")

    @staticmethod
    def solution_generator(sample):
        """Return a source example's original gold answer and supporting facts."""
        return {"answer": sample["answer"], "sp": deepcopy(sample["supporting_facts"])}

    def generate_inputs(self, num_inputs, correct_index=None):
        """Draw complete examples with distinct outputs under HotpotQA joint EM."""
        validate_limit(num_inputs, "num_inputs")
        bank = []

        if correct_index is not None:
            if not 0 <= correct_index < len(self.samples):
                raise ValueError(
                    f"HotpotQA sample index {correct_index} is outside the "
                    f"{len(self.samples)} questions available."
                )
            sample = self.samples[correct_index]
            bank.append({
                "input": {"question": sample["question"], "context": deepcopy(sample["context"])},
                "output": self.solution_generator(sample),
            })

        for _ in range(num_inputs - len(bank)):
            for _ in range(MAX_BANK_DRAWS):
                sample = random.choice(self.samples)
                output = self.solution_generator(sample)
                if any(same_answer(output, previous["output"]) for previous in bank):
                    continue
                bank.append({
                    "input": {"question": sample["question"], "context": deepcopy(sample["context"])},
                    "output": output,
                })
                break
            else:
                raise ValueError(f"Could not draw {num_inputs} HotpotQA examples with distinct answers and supporting facts.")
        return bank

    @staticmethod
    def prompt_generator(task_number):
        return (
            f"Answer the question in task_{task_number}_input using the provided context.\n"
            f'Output the result as task_{task_number}_out = '
            '{"result": {"answer": "<short answer>", "sp": [["<title>", <zero_based_sentence_index>], ...]}}, '
            'where "sp" is a list of supporting facts, each represented as a pair: article title and the '
            'zero-based index of the sentence in the context.\n'
        )


@lru_cache(maxsize=1)
def load_task(path):
    return RAGTask(path)


def load_pool():
    instance = load_task(DATA_PATH)
    indices = list(range(len(instance.samples)))
    random.Random(RAG_POOL_SEED).shuffle(indices)
    selected_indices = []
    seen_questions = set()
    for sample_index in indices:
        question = instance.samples[sample_index]["question"]
        if question in seen_questions:
            continue
        seen_questions.add(question)
        selected_indices.append(sample_index)
        if len(selected_indices) == RAG_POOL_SIZE:
            break

    return [
        {"category": "rag", "task_id": task_id,
         "task_name": f"hotpotqa_distractor_{instance.samples[sample_index]['_id']}",
         "input_type": "object", "output_type": "object", "comparator": COMPARATOR,
         "task_implementation": instance, "sample_index": sample_index}
        for task_id, sample_index in enumerate(selected_indices)
    ]


def make_stage(task, task_number, input_names, _input, get_input_attempts_per_task=5, with_bank=False):
    instance = task["task_implementation"]
    bank = instance.generate_inputs(
        get_input_attempts_per_task + 1 if (_input is not None or with_bank) else 1,
        correct_index=task["sample_index"],
    )
    correct, *wrong_inputs = bank
    return (correct["output"], instance.prompt_generator(task_number), [],
            correct["input"], wrong_inputs)
