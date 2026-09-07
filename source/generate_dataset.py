"""Generate datasets from current-format DAG topology JSON."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import shutil
from typing import Any

from fill_dag import fill_dag, load_pool, write_prompt
from parse_dag import format_json, parse_custom_dag, render_custom_dag_svg
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
            "native_task_id": record["task_id"],
            "native_task_name": record["task_name"],
            "output_value": record["output_value"],
        }
        for record in dag_sample["nodes"]
    ]


# Create one sample from one concrete target DAG.
def _make_sample(dag_data: dict[str, Any], sample_idx: int, seed: int) -> dict[str, Any]:
    dag_sample = fill_dag(parse_custom_dag(dag_data), seed=seed, sample_idx=sample_idx)
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
) -> list[dict[str, Any]]:
    samples = []
    concrete_dag = _dag_for_target(dag_data, task["task_id"])

    for sample_idx in range(num_samples_per_task):
        samples.append(_make_sample(concrete_dag, sample_idx, rng.randrange(2**31)))
        progress.update(1)

    return samples


# Generate a dataset from one current-format DAG topology.
def generate_dataset(
    dag_data: dict[str, Any],
    num_samples_per_task: int,
    seed: int = 0,
    strict: bool = True,
) -> dict[str, Any]:
    rng = random.Random(seed)
    target_category = _target_category(dag_data)
    target_pool = load_pool(target_category)
    native_task_id = dag_data["target"]["native_task_id"]

    if native_task_id == "all":
        target_tasks = list(target_pool)
    else:
        target_task = _task_by_id(target_pool, int(native_task_id))
        target_tasks = [target_task]

    samples = []
    skipped_targets = []
    total_samples = len(target_tasks) * num_samples_per_task

    with tqdm(total=total_samples, desc="Generating samples", unit="sample", disable=None) as progress:
        for task in target_tasks:
            try:
                samples.extend(_target_samples(dag_data, task, num_samples_per_task, rng, progress))
            except Exception as exc:
                if strict:
                    raise RuntimeError(f"Could not generate target task {task['task_id']}: {exc}") from exc
                skipped_targets.append({
                    "task_id": task["task_id"],
                    "task_name": task["task_name"],
                    "reason": str(exc),
                })

    summary = {
        "mode": "dag",
        "target_node": dag_data["target"]["id"],
        "target_category": target_category,
        "target_native_task_id": native_task_id,
        "num_samples_per_task": num_samples_per_task,
        "num_samples": len(samples),
        "seed": seed,
        "strict": strict,
    }
    if not strict:
        summary["skipped_targets"] = skipped_targets

    return {
        "summary": summary,
        "samples": samples,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate a benchmark dataset from DAG topology JSON.")
    parser.add_argument("--graph", required=True, help="Input DAG topology JSON file.")
    parser.add_argument("--output_dir", required=True, help="Output directory to write. Created if missing.",)
    parser.add_argument("--num_samples_per_task", type=int, default=1, help="Number of samples to generate for each target task.",)
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--non_strict", action="store_false", dest="strict", help="Skip target tasks that fail to generate instead of stopping at the first failure.",)
    args = parser.parse_args(argv)

    with open(args.graph, "r", encoding="utf-8") as handle:
        dag_data = json.load(handle)

    graph = parse_custom_dag(dag_data)
    dataset = generate_dataset(
        dag_data,
        args.num_samples_per_task,
        seed=args.seed,
        strict=args.strict,
    )

    output_name = os.path.basename(os.path.normpath(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)
    graph_path = os.path.join(args.output_dir, f"{output_name}_input_graph.json")
    svg_path = os.path.join(args.output_dir, f"{output_name}_topology.svg")
    dataset_path = os.path.join(args.output_dir, f"{output_name}_dataset.json")

    shutil.copyfile(args.graph, graph_path)
    render_custom_dag_svg(graph, svg_path)
    with open(dataset_path, "w", encoding="utf-8") as handle:
        handle.write(format_json(dataset))

    print(f"Wrote {len(dataset['samples'])} samples to {args.output_dir}")


if __name__ == "__main__":
    main()
