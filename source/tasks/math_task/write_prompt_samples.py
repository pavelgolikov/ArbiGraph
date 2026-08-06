"""Write one prompt sample for each math task."""

from __future__ import annotations

import random
import sys
from pathlib import Path


sys.dont_write_bytecode = True

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tasks.math_task.math_task import MATH_TASKS  # noqa: E402


OUTPUT_PATH = Path(__file__).with_name("math_task_prompt_samples.txt")
SCALAR_MAX_MAG = 100
LIST_LEN_MAX = 10
LIST_INPUT = [12, -7, 34, 0, -45, 19, 8, -23, 56, -11]
SCALAR_INPUT = 17


# Return sample input names and values for the current math task interface.
def task_inputs(task_number: int, input_type: str) -> tuple[list[str], int | list[int]]:
    if input_type == "scalar":
        return [f"task_{task_number}_input"], SCALAR_INPUT
    if input_type == "list":
        return [f"task_{task_number}_input"], LIST_INPUT
    raise ValueError(f"Cannot generate prompt sample for input_type={input_type!r}.")


# Instantiate every registered math task and write only its prompt.
def main() -> None:
    random.seed(0)
    prompts = []
    for task_number, task_cls in enumerate(MATH_TASKS, start=1):
        input_names, input_value = task_inputs(task_number, task_cls.input_type)
        task = task_cls(
            task_number,
            input_names,
            input_value,
            SCALAR_MAX_MAG,
            LIST_LEN_MAX,
        )
        prompts.append(task.prompt.rstrip())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n\n".join(prompts) + "\n", encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
