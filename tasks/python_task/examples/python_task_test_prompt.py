"""
Test script: generates an example prompt for each Python trace task in
leetcode_candidate_algos.txt.

Instantiates each task with random inputs and prints the prompt the model
sees, as if the task were task_3 in a chain receiving input from task_2.
"""
import sys
import os
import random

# Allow running as a standalone script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tasks.python_task.python_task import (
    PythonTraceTask,
    load_candidate_algos,
    generate_random_value,
)

random.seed(42)

SCALAR_MAX_MAG = 100
LIST_LEN_MAX = 10
TASK_IND = 3  # example task index in a chain
CHAIN_NAME = "task_2_out"
TIMEOUT = 5

ALGOS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "leetcode_candidate_algos.txt")


def rand_static_val(type_hint):
    """Generate random static values within the shared bounds."""
    return generate_random_value(
        type_hint,
        list_length=LIST_LEN_MAX,
        val_min=-SCALAR_MAX_MAG,
        val_max=SCALAR_MAX_MAG,
    )


def print_task(func_index, func_info, task):
    """Print a task's prompt with header/footer."""
    input_kind = "list" if func_info['has_list_in'] else "scalar"
    output_kind = "list" if func_info['has_list_out'] else "scalar"
    sep = "-" * 70
    print(f"\n{'=' * 70}")
    print(f"  [{func_index:3d}] {func_info['name']}  ({input_kind}->{output_kind})")
    print(f"  Output: {task.out}")
    print(f"{'=' * 70}")
    print(task.prompt)
    print(sep)


def test_all_prompts():
    all_funcs = load_candidate_algos(ALGOS_FILE)

    print("=" * 70)
    print(f"  PYTHON TASK PROMPT TEST  |  scalar_max_mag={SCALAR_MAX_MAG}"
          f"  list_len_max={LIST_LEN_MAX}")
    print(f"  Total candidate algorithms: {len(all_funcs)}")
    print("=" * 70)

    succeeded = 0
    failed = []

    for func_index, func_info in enumerate(all_funcs):
        input_kind = "list" if func_info['has_list_in'] else "scalar"

        # Try a few random inputs until one succeeds
        attempts = 0
        max_attempts = 10
        task = None
        while attempts < max_attempts:
            attempts += 1
            chained_value = generate_random_value(
                input_kind,
                list_length=LIST_LEN_MAX,
                val_min=-SCALAR_MAX_MAG,
                val_max=SCALAR_MAX_MAG,
            )
            inp = {"chained": {"chain_name": CHAIN_NAME, "value": chained_value}}
            try:
                task = PythonTraceTask(
                    task_ind=TASK_IND,
                    inp=inp,
                    scalar_max_mag=SCALAR_MAX_MAG,
                    list_len_max=LIST_LEN_MAX,
                    rand_static_val=rand_static_val,
                    algos_file=ALGOS_FILE,
                    timeout=TIMEOUT,
                    func_index=func_index,
                )
                break
            except Exception:
                task = None

        if task is not None:
            print_task(func_index, func_info, task)
            succeeded += 1
        else:
            failed.append((func_index, func_info['name']))
            print(f"\n{'!' * 70}")
            print(f"  [{func_index:3d}] {func_info['name']}  -- FAILED after"
                  f" {max_attempts} attempts")
            print(f"{'!' * 70}")

    print(f"\n\n{'=' * 70}")
    print(f"  SUMMARY: {succeeded}/{len(all_funcs)} tasks generated successfully")
    if failed:
        print(f"  Failed ({len(failed)}):")
        for idx, name in failed:
            print(f"    [{idx:3d}] {name}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    test_all_prompts()
