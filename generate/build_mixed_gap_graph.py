"""Pair mixed graphs with existing pinned, pure-math gap graphs.

The topology, math target, and nine retained math nodes are copied exactly.
The other nine nodes take pinned Python, RLVE, and RAG problems from their
calibrated pools. No graph outcome enters construction.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "generate"))

from dataset_to_text import format_prompts  # noqa: E402
from fill_dag import load_pool, make_stage, pin_key  # noqa: E402
from format_json import format_json  # noqa: E402
from parse_dag import parse_custom_dag, render_custom_dag_svg  # noqa: E402


MODEL_NAMES = ("Qwen3.8-27B", "GPT-OSS-120B", "gemma-4-31B-it")
TOPOLOGIES = (
    "eighteen_chain_target",
    "eighteen_braid_target",
    "eighteen_fan_out_fan_in",
)
REPLACEMENTS = {
    2: "python", 4: "python", 6: "python", 8: "python", 10: "python",
    12: "rlve", 14: "rlve", 16: "rlve", 17: "rag",
}
RETAINED_MATH = tuple(number for number in range(1, 19)
                      if number not in REPLACEMENTS)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def pinned_records(path: Path):
    """Return one authoritative solo record per task from a generated dataset."""
    records = {}
    for sample in read_json(path)["samples"]:
        task_id = int(sample["target_native_task_id"])
        record = sample["node_outputs"][0]
        if task_id in records and records[task_id] != record:
            raise ValueError(f"Unpinned solo problem {task_id} in {path}")
        records[task_id] = record
    return records


def prompt_blocks(prompt: str):
    blocks = re.split(r"\n\n(?=Task \d+:\n)", prompt.rstrip("\n"))
    numbers = [int(re.match(r"Task (\d+):\n", block).group(1))
               for block in blocks[1:]]
    if numbers != list(range(1, 19)) or "\n\n".join(blocks) + "\n" != prompt:
        raise ValueError("Could not split the pure prompt into 18 exact task blocks")
    return blocks[0], blocks[1:]


def replacement(category, task_id, number, source_record, input_names,
                parent_outputs, tasks, solo_reference):
    task = tasks[category][task_id]
    parent_ids = source_record["parent_ids"]
    parent_input = parent_outputs[0] if len(parent_outputs) == 1 else parent_outputs
    random.seed(pin_key(category, task_id, 0))
    output, task_prompt, _static, input_value, wrong_inputs = make_stage(
        category, task, number, input_names, parent_input,
        get_input_attempts_per_task=5, with_bank=True,
    )
    record = {
        "node_id": source_record["node_id"],
        "native_task_id": task_id,
        "native_task_name": task["task_name"],
        "output_value": output,
        "comparator": task.get("comparator", "numeric"),
        "parent_ids": parent_ids,
        "input_value": input_value,
        "wrong_inputs": wrong_inputs,
    }
    if "task_number" in source_record:
        record["task_number"] = source_record["task_number"]
    reference = solo_reference[category][task_id]
    for field in ("native_task_id", "native_task_name", "output_value",
                  "comparator", "input_value", "wrong_inputs"):
        if record[field] != reference[field]:
            raise ValueError(f"{category} {task_id} changed pinned {field}")

    input_prompt = (
        f"To get task_{number}_input, call the get_input tool with "
        f"task_index={number} and input_dependencies=[{', '.join(input_names)}], "
        "substituting the actual values.\n"
    )
    block = f"Task {number}:\n" + input_prompt + task_prompt
    return record, block.strip()


def build(model: str, name: str):
    data_root = ROOT / "generate" / "generated"
    mixed_root = data_root / "mixed_gap"
    gap_root = data_root / "gap" / model
    pure_path = gap_root / "math" / name / f"{name}_dataset.json"
    pure_graph_path = gap_root / "math" / name / f"{name}_input_graph.json"
    pure = read_json(pure_path)
    graph = read_json(pure_graph_path)
    if len(graph["nodes"]) != 18 or graph["target"]["id"] != "task_18":
        raise ValueError(f"Unexpected pure topology for {model}/{name}")
    if [node["id"] for node in graph["nodes"]] != [f"task_{i}" for i in range(1, 19)]:
        raise ValueError(f"Unexpected node order for {model}/{name}")
    for number, category in REPLACEMENTS.items():
        graph["nodes"][number - 1]["category"] = category

    selected = {category: read_json(mixed_root / f"selected_{category}_ids.json")
                for category in ("rag", "rlve")}
    python_data = gap_root / "python" / "single_target_baseline" / "single_target_baseline_dataset.json"
    python_ids = read_json(python_data)["summary"]["pool_task_ids"]
    if len(python_ids) < 5 or len(selected["rlve"]) != 5 or len(selected["rag"]) != 3:
        raise ValueError("Unexpected selected-pool size")
    solo_reference = {
        "python": pinned_records(python_data),
        "rag": pinned_records(mixed_root / "rag_calibration" / "rag_calibration_dataset.json"),
        "rlve": pinned_records(mixed_root / "rlve_calibration" / "rlve_calibration_dataset.json"),
    }
    tasks = {category: {task["task_id"]: task for task in load_pool(category)}
             for category in solo_reference}

    samples = []
    for row_index, source in enumerate(pure["samples"]):
        target_id = int(source["target_native_task_id"])
        sample_idx = int(source["sample_idx"])
        intro, source_blocks = prompt_blocks(source["prompt"])
        rng = random.Random(f"arbigraph-mixed-gap:{model}:{target_id}:{sample_idx}")
        python_choices = rng.sample(python_ids, 5)
        rlve_choices = [selected["rlve"][(row_index + offset) % 5]
                        for offset in range(3)]
        rag_choice = selected["rag"][sample_idx % 3]
        choices = dict(zip((2, 4, 6, 8, 10), python_choices))
        choices.update(zip((12, 14, 16), rlve_choices))
        choices[17] = rag_choice

        source_records = {record["node_id"]: record for record in source["node_outputs"]}
        if set(source_records) != {f"task_{i}" for i in range(1, 19)}:
            raise ValueError(f"Unexpected records in {model}/{name}")
        visible_numbers = {record["node_id"]: number
                           for number, record in enumerate(source["node_outputs"], 1)}
        records_by_node, records, blocks = {}, [], []
        for number, source_record in enumerate(source["node_outputs"], 1):
            node_id = source_record["node_id"]
            node_number = int(node_id.removeprefix("task_"))
            if node_number in REPLACEMENTS:
                parent_ids = source_record["parent_ids"]
                if any(parent_id not in records_by_node for parent_id in parent_ids):
                    raise ValueError(f"Parent order is not topological in {model}/{name}")
                record, block = replacement(
                    REPLACEMENTS[node_number], choices[node_number], number,
                    source_record,
                    [f"task_{visible_numbers[parent_id]}_out" for parent_id in parent_ids],
                    [records_by_node[parent_id]["output_value"] for parent_id in parent_ids],
                    tasks, solo_reference,
                )
            else:
                record = copy.deepcopy(source_record)
                block = source_blocks[number - 1]
            records_by_node[node_id] = record
            records.append(record)
            blocks.append(block)

        if any(records_by_node[f"task_{number}"] != source_records[f"task_{number}"]
               for number in RETAINED_MATH):
            raise ValueError(f"Retained math node changed for {model}/{target_id}/{sample_idx}")
        sample = {
            "target_native_task_id": target_id,
            "target_node_id": "task_18",
            "sample_idx": sample_idx,
            "prompt": "\n\n".join([intro, *blocks]) + "\n",
            "node_outputs": records,
        }
        samples.append(sample)

    summary = copy.deepcopy(pure["summary"])
    summary["pool_task_ids"] = None
    summary["pin_indices"] = None
    summary["paired_pure_dataset"] = str(pure_path.relative_to(ROOT))
    summary["category_pool_task_ids"] = {
        "math": pure["summary"]["pool_task_ids"],
        "python": python_ids,
        "rlve": selected["rlve"],
        "rag": selected["rag"],
    }
    summary["replacement_categories_by_node_id"] = {
        f"task_{number}": category for number, category in REPLACEMENTS.items()
    }
    dataset = {"summary": summary, "samples": samples}

    output_dir = mixed_root / model / "mixed" / name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}_dataset.json").write_text(format_json(dataset), encoding="utf-8")
    (output_dir / f"{name}_input_graph.json").write_text(
        json.dumps(graph, indent=2) + "\n", encoding="utf-8")
    render_custom_dag_svg(parse_custom_dag(graph), str(output_dir / f"{name}_topology.svg"))
    (output_dir / f"{name}_dataset.txt").write_text(
        format_prompts([sample["prompt"].rstrip() for sample in samples]),
        encoding="utf-8",
    )
    print(f"{model}: wrote {len(samples)} paired mixed graphs to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES)
    parser.add_argument("--topology", choices=TOPOLOGIES)
    args = parser.parse_args()
    for model in (args.model,) if args.model else MODEL_NAMES:
        for topology in (args.topology,) if args.topology else TOPOLOGIES:
            build(model, topology)


if __name__ == "__main__":
    main()
