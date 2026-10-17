"""Make matched peninsula and island variants of the existing 18-task braid.

Only two graph edges, their node parent lists, and the corresponding get_input
instructions change. Every task, pinned input, output, wrong-input bank, label,
and sample assignment is copied from the already evaluated full braid.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "generate"))

from dataset_to_text import format_prompts  # noqa: E402
from format_json import format_json  # noqa: E402
from parse_dag import parse_custom_dag, render_custom_dag_svg  # noqa: E402

MODELS = ("gemma-4-31B-it", "GPT-OSS-120B", "Qwen3.8-27B")
CATEGORIES = ("math", "python")
SOURCE = "eighteen_braid_target"
CONDITIONS = {
    "peninsula": (("task_17", "task_18"),),
    "island": (("task_17", "task_18"), ("task_1", "task_3")),
}
OPTIONAL = {"task_3", "task_10", "task_11", "task_12", "task_13", "task_14", "task_15", "task_17"}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def changed_block(block: str, task_number: int, parents: list[int]) -> str:
    expected_start = f"Task {task_number}:\nTo get task_{task_number}_input, call the get_input tool with "
    if not block.startswith(expected_start):
        raise ValueError(f"Unexpected task {task_number} prompt")
    old_line, separator, body = block.partition("\n")
    if not separator or old_line != f"Task {task_number}:":
        raise ValueError(f"Unexpected task {task_number} heading")
    _, separator, task_body = body.partition("\n")
    if not separator:
        raise ValueError(f"Missing task {task_number} body")
    if parents:
        names = ", ".join(f"task_{parent}_out" for parent in parents)
        instruction = (f"To get task_{task_number}_input, call the get_input tool with "
                       f"task_index={task_number} and input_dependencies=[{names}], "
                       "substituting the actual values.")
    else:
        instruction = (f"To get task_{task_number}_input, call the get_input tool with "
                       f"task_index={task_number} and input_dependencies=[].")
    return f"Task {task_number}:\n{instruction}\n{task_body}"


def transform_sample(source: dict, condition: str) -> dict:
    sample = copy.deepcopy(source)
    if sample["target_node_id"] != "task_18" or len(sample["node_outputs"]) != 18:
        raise ValueError("Expected an 18-node braid targeting task_18")
    blocks = re.split(r"\n\n(?=Task \d+:\n)", sample["prompt"].rstrip("\n"))
    if len(blocks) != 19 or "\n\n".join(blocks) + "\n" != sample["prompt"]:
        raise ValueError("Could not split source prompt without changing it")
    visible = {record["node_id"]: index for index, record in enumerate(sample["node_outputs"], 1)}
    if set(visible) != {f"task_{i}" for i in range(1, 19)}:
        raise ValueError("Unexpected source node IDs")
    expected = {"task_18": ["task_16", "task_17"], "task_3": ["task_1"]}
    for node_id in ("task_18", "task_3"):
        record = sample["node_outputs"][visible[node_id] - 1]
        if record["parent_ids"] != expected[node_id]:
            raise ValueError(f"Unexpected source parents for {node_id}")
        if node_id == "task_3" and condition == "peninsula":
            continue
        record["parent_ids"] = ["task_16"] if node_id == "task_18" else []
        number = visible[node_id]
        parent_numbers = [visible[parent] for parent in record["parent_ids"]]
        blocks[number] = changed_block(blocks[number], number, parent_numbers)
    sample["prompt"] = "\n\n".join(blocks) + "\n"
    return sample


def build(model: str, category: str, condition: str) -> Path:
    source_dir = ROOT / "generate" / "generated" / "gap" / model / category / SOURCE
    source_path = source_dir / f"{SOURCE}_dataset.json"
    source_graph_path = source_dir / f"{SOURCE}_input_graph.json"
    source = read_json(source_path)
    graph = read_json(source_graph_path)
    if len(graph["nodes"]) != 18 or graph["target"]["id"] != "task_18":
        raise ValueError("Unexpected source graph")
    source_edges = {(edge["source"], edge["destination"]) for edge in graph["edges"]}
    cuts = set(CONDITIONS[condition])
    if not cuts <= source_edges:
        raise ValueError(f"Missing source edges: {cuts - source_edges}")
    graph["edges"] = [edge for edge in graph["edges"]
                      if (edge["source"], edge["destination"]) not in cuts]
    dag = parse_custom_dag(graph)
    ancestors = nx.ancestors(dag, "task_18")
    if set(dag) - ancestors - {"task_18"} != OPTIONAL:
        raise ValueError("The disconnected branch differs from the intended eight nodes")
    components = list(nx.weakly_connected_components(dag))
    if condition == "island" and len(components) != 2:
        raise ValueError("Island variant must have two components")
    if condition == "peninsula" and len(components) != 1:
        raise ValueError("Peninsula variant must have one component")

    samples = [transform_sample(sample, condition) for sample in source["samples"]]
    if len({(sample["target_native_task_id"], sample["sample_idx"]) for sample in samples}) != len(samples):
        raise ValueError("Source dataset contains duplicate sample keys")
    for original, changed in zip(source["samples"], samples, strict=True):
        for old_record, new_record in zip(original["node_outputs"], changed["node_outputs"], strict=True):
            old_copy, new_copy = copy.deepcopy(old_record), copy.deepcopy(new_record)
            old_copy.pop("parent_ids")
            new_copy.pop("parent_ids")
            if old_copy != new_copy:
                raise ValueError("A pinned task changed during graph transformation")
        expected_parents = {node: [] for node in dag}
        for edge in graph["edges"]:
            expected_parents[edge["destination"]].append(edge["source"])
        for record in changed["node_outputs"]:
            if record["parent_ids"] != expected_parents[record["node_id"]]:
                raise ValueError("Dataset parent lists disagree with the graph")

    summary = copy.deepcopy(source["summary"])
    summary["paired_full_braid_dataset"] = str(source_path.relative_to(ROOT))
    summary["branch_condition"] = condition
    summary["removed_edges"] = [{"source": a, "destination": b} for a, b in CONDITIONS[condition]]
    summary["optional_node_ids"] = sorted(OPTIONAL)
    name = f"eighteen_braid_{condition}_target"
    output_dir = ROOT / "generate" / "generated" / "branch_pruning" / model / category / name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}_dataset.json").write_text(
        format_json({"summary": summary, "samples": samples}), encoding="utf-8")
    (output_dir / f"{name}_input_graph.json").write_text(
        json.dumps(graph, indent=2) + "\n", encoding="utf-8")
    render_custom_dag_svg(dag, str(output_dir / f"{name}_topology.svg"))
    (output_dir / f"{name}_dataset.txt").write_text(
        format_prompts([sample["prompt"].rstrip() for sample in samples]), encoding="utf-8")
    print(f"{model}/{category}/{condition}: {len(samples)} matched graphs -> {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS[:2])
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=CATEGORIES)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=tuple(CONDITIONS))
    args = parser.parse_args()
    for model in args.models:
        for category in args.categories:
            for condition in args.conditions:
                build(model, category, condition)


if __name__ == "__main__":
    main()
