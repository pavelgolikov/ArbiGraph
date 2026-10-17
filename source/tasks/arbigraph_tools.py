"""ArbiGraph's agent tools.

get_input serves stored task inputs based on the supplied dependency answers.
submit_answers is the tool the model uses to submit the final answer to the graded task.
"""

from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from grade_results import comparator


DEFAULT_INPUT_ATTEMPTS = 5


GET_INPUT_TOOL = {
    "name": "get_input",
    "description": (
        "Use this tool to request the input for a specific task by providing its global task index and a "
        "list of its input dependencies' values. This tool returns the object to use as task's input.\n"
        'For a root task with no dependencies, use input_dependencies=[].\n'
        'Single-dependency example: if task_3 depends on task_1_out = {"result": 7}, '
        'call this tool with arguments {"task_index": 3, "input_dependencies": [{"result": 7}]}.\n'
        'Multi-dependency example: if task_4 depends on task_1_out = {"result": 7} '
        'and task_2_out = {"result": [2, 5]}, call this tool with arguments '
        '{"task_index": 4, "input_dependencies": [{"result": 7}, {"result": [2, 5]}]}.\n'
        'User prompt will indicate the order of the arguments to pass to this tool.'
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "task_index": {"type": "integer", "minimum": 1},
            "input_dependencies": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"result": {}},
                    "required": ["result"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["task_index", "input_dependencies"],
        "additionalProperties": False,
    },
}


SUBMIT_ANSWERS_TOOL = {
    "name": "submit_answers",
    "description": (
        "Use this tool to submit the final result of the task indicated by the prompt. "
        "Pass a list holding exactly one object, that task's result. A list of any "
        "other length, or an object not of the form {\"result\": ...}, is rejected as a whole. Only an "
        "answer submitted with this tool is graded.\n"
        'Example: if task 3 is the graded task and task_3_out = {"result": 4}, call this tool with '
        'arguments {"answers": [{"result": 4}]}.'
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "answers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"result": {}},
                    "required": ["result"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["answers"],
        "additionalProperties": False,
    },
}


def validate_limit(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


class InputTool:
    """One model attempt's prebuilt inputs and its per-task request counters.

    Each task record has parents (global indices), input, output, and wrong_inputs.
    Answers are stored without the {"result": ...} envelope. JSON-loaded index
    keys are accepted. Creating another instance resets the counters.
    """

    @classmethod
    def from_records(cls, records, **limits):
        """Rebuild the tool lookup from node records in topological order."""
        task_numbers = {record["node_id"]: index for index, record in enumerate(records, 1)}
        return cls({
            task_numbers[record["node_id"]]: {
                "parents": [task_numbers[parent] for parent in record["parent_ids"]],
                "input": record["input_value"],
                "output": record["output_value"],
                "wrong_inputs": record["wrong_inputs"],
                "comparator": record.get("comparator"),
            }
            for record in records
        }, **limits)

    def __init__(self, tasks, attempts_per_task=DEFAULT_INPUT_ATTEMPTS):
        validate_limit(attempts_per_task, "get_input_attempts_per_task")
        self.attempts_per_task = attempts_per_task
        self.tasks = {int(index): deepcopy(task) for index, task in tasks.items()}
        for task in self.tasks.values():
            task["same_answer"] = comparator(task.get("comparator"))
            if task["parents"]:
                bank = task["wrong_inputs"]
                if len(bank) != attempts_per_task:
                    raise ValueError(f"Each wrong-input bank must contain {attempts_per_task} entries.")
                answers = [task["output"]]
                for pair in bank:
                    if any(task["same_answer"](pair["output"], answer) for answer in answers):
                        raise ValueError("Bank answers must differ from ground truth and each other.")
                    answers.append(pair["output"])
        self.counts = dict.fromkeys(self.tasks, 0)
        # Whether a child's request has ever reported this task's output correctly.
        # None until some child names it, which never happens for a childless node.
        self.reported = dict.fromkeys(self.tasks)

    def reported_correct(self):
        """Report, per task that a child has named, whether its output was stated right.

        A child cannot ask for its input without saying what it believes its parents
        produced, so these requests are an exact record of the values the model carried
        forward, and unlike the text of an answer they do not depend on the model
        choosing to write anything down.
        """
        return {str(index): value for index, value in self.reported.items() if value is not None}

    def get_input(self, task_index=None, input_dependencies=None):
        """Return an input, or an error when the request names no known task.

        Once the task is known the call is a submission and costs one of its
        attempts. It is right only if it has the documented shape and carries
        every parent's true output; a call that is merely misshapen is not
        interpreted, it is simply wrong, and is answered from the bank like any
        other wrong answer.
        """
        if type(task_index) is not int or task_index not in self.tasks:
            return "Error: unknown task index."
        if self.counts[task_index] >= self.attempts_per_task:
            return "Error: input request limit reached for this task."

        task = self.tasks[task_index]
        parents = task["parents"]
        attempt = self.counts[task_index]
        self.counts[task_index] += 1
        # Shape and values are one test: zip would silently accept a short list.
        shaped = isinstance(input_dependencies, list) and len(input_dependencies) == len(parents)
        matches = [
            isinstance(value, dict)
            and set(value) == {"result"}
            and self.tasks[parent]["same_answer"](value["result"], self.tasks[parent]["output"])
            for parent, value in zip(parents, input_dependencies)
        ] if shaped else []
        # A well-shaped request names each parent, so what it says about each one is
        # recorded. A misshapen request names no parent in particular and is left
        # unattributed, exactly as it is left uninterpreted when serving the input.
        if shaped:
            for parent, match in zip(parents, matches):
                self.reported[parent] = bool(self.reported[parent]) or match
        correct = shaped and all(matches)
        # A task with no parents states no upstream value, so it has no bank.
        selected = task if correct or not parents else task["wrong_inputs"][attempt]
        return deepcopy(selected["input"])
