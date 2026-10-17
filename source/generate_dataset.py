"""Generate datasets from current-format DAG topology JSON."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import shutil
from typing import Any

from fill_dag import fill_dag, load_pool, resolve_pinned_pool, write_prompt
from format_json import format_json
from parse_dag import parse_custom_dag, render_custom_dag_svg
from tqdm import tqdm


# Find the category of the graph's target node.
def _target_category(dag_data: dict[str, Any]) -> str:
    target_id = dag_data["target"]["id"]
    for node in dag_data["nodes"]:
        if node["id"] == target_id:
            return node["category"]
    raise ValueError(f"Unknown target node id {target_id!r}.")


# Return the task with the requested native task id.
def _task_by_id(pool: list[dict[str, Any]], task_id: int) -> dict[str, Any]:
    for task in pool:
        if task["task_id"] == task_id:
            return task
    raise ValueError(f"Unknown target native_task_id {task_id}.")


# Copy the DAG data and pin it to one concrete target task id.
def _dag_for_target(dag_data: dict[str, Any], task_id: int) -> dict[str, Any]:
    concrete = copy.deepcopy(dag_data)
    concrete["target"]["native_task_id"] = str(task_id)
    return concrete


def _node_outputs(dag_sample: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "node_id": record["node_id"],
            "task_number": record["task_number"],
            "native_task_id": record["task_id"],
            "native_task_name": record["task_name"],
            "output_value": record["output_value"],
            "comparator": record["comparator"],
            "parent_ids": record["parent_ids"],
            "input_value": record["input_value"],
            "wrong_inputs": record["wrong_inputs"],
        }
        for record in dag_sample["nodes"]
    ]


# Create one sample from one concrete target DAG.
def _make_sample(
    dag_data: dict[str, Any], sample_idx: int, seed: int, get_input_attempts_per_task: int,
    fill_options: dict[str, Any],
) -> dict[str, Any]:
    dag_sample = fill_dag(
        parse_custom_dag(dag_data), seed=seed, sample_idx=sample_idx,
        get_input_attempts_per_task=get_input_attempts_per_task, **fill_options,
    )
    target_record = next(record for record in dag_sample["nodes"] if record["node_id"] == dag_sample["target_node"])
    return {
        "target_native_task_id": target_record["task_id"],
        "target_node_id": target_record["node_id"],
        "sample_idx": sample_idx,
        "prompt": write_prompt(dag_sample),
        "node_outputs": _node_outputs(dag_sample),
    }


# Generate a complete sample group for one concrete target task.
def _target_samples(
    dag_data: dict[str, Any],
    task: dict[str, Any],
    num_samples_per_task: int,
    rng: random.Random,
    progress: Any,
    get_input_attempts_per_task: int,
    fill_options: dict[str, Any],
) -> list[dict[str, Any]]:
    samples = []
    concrete_dag = _dag_for_target(dag_data, task["task_id"])

    for sample_idx in range(num_samples_per_task):
        samples.append(_make_sample(
            concrete_dag, sample_idx, rng.randrange(2**31), get_input_attempts_per_task, fill_options,
        ))
        progress.update(1)

    return samples


# Generate a dataset from one current-format DAG topology.
def generate_dataset(
    dag_data: dict[str, Any],
    num_samples_per_task: int,
    seed: int = 0,
    get_input_attempts_per_task: int = 5,
    pool_task_ids: list[int] | None = None,
    pin_index: int | None = None,
    distinct_tasks: bool = False,
    target_task_ids: list[int] | None = None,
    shuffle_task_labels: bool = False,
) -> dict[str, Any]:
    rng = random.Random(seed)
    fill_options = {
        "pool_task_ids": pool_task_ids,
        "pin_index": pin_index,
        "distinct_tasks": distinct_tasks,
        "shuffle_task_labels": shuffle_task_labels,
    }
    target_category = _target_category(dag_data)
    target_pool = load_pool(target_category)
    native_task_id = dag_data["target"]["native_task_id"]

    if pool_task_ids is not None:
        allowed = set(pool_task_ids)
        target_pool = [task for task in target_pool if task["task_id"] in allowed]

    # Each task takes the first instantiation that can fill a node with a parent. A task
    # with no such instantiation belongs in no graph, so it leaves the pool rather than
    # being calibrated and then never used.
    pin_indices = None
    if pin_index is not None:
        resolved = resolve_pinned_pool(
            target_category, target_pool, pin_index, get_input_attempts_per_task,
        )
        dropped = [task["task_id"] for task in target_pool if task["task_id"] not in resolved]
        if dropped:
            print(f"Dropped {len(dropped)} {target_category} tasks with no instantiation that "
                  f"fills a node with a parent: {dropped}")
        moved = {task_id: index for task_id, index in resolved.items() if index != pin_index}
        if moved:
            print(f"Moved {len(moved)} {target_category} tasks past their first instantiation: {moved}")
        target_pool = [task for task in target_pool if task["task_id"] in resolved]
        pool_task_ids = sorted(resolved)
        fill_options["pool_task_ids"] = pool_task_ids
        pin_indices = {str(task_id): resolved[task_id] for task_id in sorted(resolved)}

    if native_task_id == "all":
        target_tasks = list(target_pool)
        # Narrowing the sweep shrinks a run without touching the pool the nodes draw
        # from, so a short trial keeps the graphs it produces full size.
        if target_task_ids is not None:
            wanted = set(target_task_ids)
            target_tasks = [task for task in target_tasks if task["task_id"] in wanted]
            if not target_tasks:
                raise ValueError("No task in the pool is left to take the target role.")
    else:
        target_task = _task_by_id(target_pool, int(native_task_id))
        target_tasks = [target_task]

    samples = []
    total_samples = len(target_tasks) * num_samples_per_task

    with tqdm(total=total_samples, desc="Generating samples", unit="sample", disable=None) as progress:
        for task in target_tasks:
            try:
                samples.extend(_target_samples(
                    dag_data, task, num_samples_per_task, rng, progress,
                    get_input_attempts_per_task, fill_options,
                ))
            except Exception as exc:
                raise RuntimeError(f"Could not generate target task {task['task_id']}: {exc}") from exc

    summary = {
        "mode": "dag",
        "target_node": dag_data["target"]["id"],
        "target_category": target_category,
        "target_native_task_id": native_task_id,
        "num_samples_per_task": num_samples_per_task,
        "num_samples": len(samples),
        "seed": seed,
        "pool_task_ids": None if pool_task_ids is None else sorted(pool_task_ids),
        "pin_index": pin_index,
        "pin_indices": pin_indices,
        "distinct_tasks": distinct_tasks,
        "shuffle_task_labels": shuffle_task_labels,
        "target_task_ids": None if target_task_ids is None else sorted(target_task_ids),
    }

    return {
        "summary": summary,
        "samples": samples,
    }


# Read a list of native task ids from a comma-separated argument or a JSON file.
def _task_id_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    if os.path.isfile(value):
        with open(value, "r", encoding="utf-8") as handle:
            return [int(task_id) for task_id in json.load(handle)]
    return [int(part) for part in value.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate a benchmark dataset from DAG topology JSON.")
    parser.add_argument("--graph", required=True, help="Input DAG topology JSON file.")
    parser.add_argument("--output_dir", required=True, help="Output directory to write. Created if missing.",)
    parser.add_argument("--num_samples_per_task", type=int, default=1, help="Number of samples to generate for each target task.",)
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--get_input_attempts_per_task", type=int, default=5,
        help="Number of alternative inputs to generate for each dependent task.",
    )
    parser.add_argument(
        "--pool_task_ids", default=None,
        help="Restrict every node to these native task ids: a comma-separated list, or the path "
             "to a JSON file holding a list of ids.",
    )
    parser.add_argument(
        "--pin_index", type=int, default=None,
        help="Give each task one fixed instantiation, the same static data and input value "
             "wherever the task appears and in every sample. Selects which instantiation.",
    )
    parser.add_argument(
        "--distinct_tasks", action="store_true",
        help="Fill each graph from the pool without replacement, so no task repeats in a graph.",
    )
    parser.add_argument(
        "--target_task_ids", default=None,
        help="Restrict which tasks take the target role, given the same way as --pool_task_ids. "
             "The pool the other nodes draw from is unchanged, so a short trial run still "
             "produces full-size graphs. Only meaningful when the topology targets \"all\".",
    )
    parser.add_argument(
        "--shuffle_task_labels", action="store_true",
        help="Randomly relabel every task per sample, then print task blocks in numerical order.",
    )
    args = parser.parse_args(argv)

    pool_task_ids = _task_id_list(args.pool_task_ids)
    target_task_ids = _task_id_list(args.target_task_ids)

    with open(args.graph, "r", encoding="utf-8") as handle:
        dag_data = json.load(handle)

    graph = parse_custom_dag(dag_data)
    dataset = generate_dataset(
        dag_data, args.num_samples_per_task, seed=args.seed,
        get_input_attempts_per_task=args.get_input_attempts_per_task,
        pool_task_ids=pool_task_ids, pin_index=args.pin_index,
        distinct_tasks=args.distinct_tasks, target_task_ids=target_task_ids,
        shuffle_task_labels=args.shuffle_task_labels,
    )

    output_name = os.path.basename(os.path.normpath(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)
    graph_path = os.path.join(args.output_dir, f"{output_name}_input_graph.json")
    svg_path = os.path.join(args.output_dir, f"{output_name}_topology.svg")
    dataset_path = os.path.join(args.output_dir, f"{output_name}_dataset.json")

    shutil.copyfile(args.graph, graph_path)
    task_numbers = None
    annotation = None
    if args.shuffle_task_labels and dataset["samples"]:
        task_numbers = {
            record["node_id"]: record["task_number"]
            for record in dataset["samples"][0]["node_outputs"]
        }
        annotation = "Visible task labels for Prompt 1 (sample_idx=0)"
    render_custom_dag_svg(
        graph, svg_path, task_numbers=task_numbers, annotation=annotation,
    )
    with open(dataset_path, "w", encoding="utf-8") as handle:
        handle.write(format_json(dataset))

    print(f"Wrote {len(dataset['samples'])} samples to {args.output_dir}")


if __name__ == "__main__":
    main()
