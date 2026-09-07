from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


CATEGORIES = ("math", "python", "gsm")
TOPOLOGIES = (
    "single_target_baseline",
    "three_independent_target",
    "three_chain_target",
    "six_independent_target",
    "six_chain_target",
    "four_way_fan_in_target",
    "two_branch_recombine_target",
)
EXPECTED_ROWS = {
    "math": 640,
    "python": 1504,
    "gsm": 656,
}

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SOURCE_DATA_ROOT = REPO_ROOT / "source" / "generated" / "hf_datasets"
SOURCE_TOPOLOGY_ROOT = REPO_ROOT / "source" / "topologies"
BUILD_ROOT = SCRIPT_DIR / "build" / "arbigraph"
TEMPLATE_PATH = SCRIPT_DIR / "dataset_card_template.md"


def config_name(category: str, topology: str) -> str:
    return f"{category}_{topology}"


def source_dataset_path(category: str, topology: str) -> Path:
    return SOURCE_DATA_ROOT / category / topology / f"{topology}_dataset.json"


def source_topology_path(category: str, topology: str) -> Path:
    return SOURCE_TOPOLOGY_ROOT / category / "hf" / f"{topology}.json"


def source_topology_svg_path(category: str, topology: str) -> Path:
    return SOURCE_DATA_ROOT / category / topology / f"{topology}_topology.svg"


def json_string(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def find_target_record(sample: dict[str, Any]) -> dict[str, Any]:
    target_node_id = sample["target_node_id"]
    for record in sample["node_outputs"]:
        if record["node_id"] == target_node_id:
            return record
    raise ValueError(f"Target node {target_node_id!r} is missing from node_outputs.")


def row_from_sample(category: str, topology: str, sample: dict[str, Any]) -> dict[str, Any]:
    target_record = find_target_record(sample)
    target_native_task_id = sample["target_native_task_id"]
    sample_idx = sample["sample_idx"]

    return {
        "id": f"{category}/{topology}/{target_native_task_id}/{sample_idx}",
        "category": category,
        "topology": topology,
        "target_node_id": sample["target_node_id"],
        "target_native_task_id": target_native_task_id,
        "target_native_task_name": target_record["native_task_name"],
        "sample_idx": sample_idx,
        "prompt": sample["prompt"],
        "target_output_json": json_string(target_record["output_value"]),
        "node_outputs_json": json_string(sample["node_outputs"]),
    }


def read_source_dataset(category: str, topology: str) -> dict[str, Any]:
    path = source_dataset_path(category, topology)
    if not path.exists():
        raise FileNotFoundError(f"Missing generated dataset: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(category: str, topology: str) -> int:
    dataset = read_source_dataset(category, topology)
    rows = [row_from_sample(category, topology, sample) for sample in dataset["samples"]]

    expected = EXPECTED_ROWS[category]
    if len(rows) != expected:
        raise ValueError(f"{category}/{topology} has {len(rows)} rows, expected {expected}.")

    output_path = BUILD_ROOT / "data" / category / f"{topology}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def copy_topology(category: str, topology: str) -> None:
    source_path = source_topology_path(category, topology)
    if not source_path.exists():
        raise FileNotFoundError(f"Missing topology file: {source_path}")

    output_path = BUILD_ROOT / "topologies" / category / f"{topology}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, output_path)

    source_svg_path = source_topology_svg_path(category, topology)
    if not source_svg_path.exists():
        raise FileNotFoundError(f"Missing topology SVG file: {source_svg_path}")

    output_svg_path = BUILD_ROOT / "topologies" / category / f"{topology}.svg"
    shutil.copyfile(source_svg_path, output_svg_path)


def build_configs_yaml() -> str:
    lines = ["configs:"]
    for category in CATEGORIES:
        for topology in TOPOLOGIES:
            lines.extend([
                f"- config_name: {config_name(category, topology)}",
                "  data_files:",
                "  - split: test",
                f"    path: data/{category}/{topology}.jsonl",
            ])
    return "\n".join(lines)


def build_config_table(row_counts: dict[tuple[str, str], int]) -> str:
    lines = [
        "| Config | Category | Topology | Split | Rows |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for category in CATEGORIES:
        for topology in TOPOLOGIES:
            lines.append(
                f"| `{config_name(category, topology)}` | `{category}` | `{topology}` | `test` | "
                f"{row_counts[(category, topology)]} |"
            )
    return "\n".join(lines)


def write_dataset_card(row_counts: dict[tuple[str, str], int]) -> None:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    readme = template.replace("{{CONFIGS_YAML}}", build_configs_yaml())
    readme = readme.replace("{{CONFIG_TABLE}}", build_config_table(row_counts))
    (BUILD_ROOT / "README.md").write_text(readme, encoding="utf-8")
    preview = readme.split("---", 2)[2].lstrip()
    (SCRIPT_DIR / "build" / "README_preview.md").write_text(preview, encoding="utf-8")


def clean_build_root() -> None:
    if BUILD_ROOT.exists():
        shutil.rmtree(BUILD_ROOT)
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    clean_build_root()
    row_counts = {}

    for category in CATEGORIES:
        for topology in TOPOLOGIES:
            row_counts[(category, topology)] = write_jsonl(category, topology)
            copy_topology(category, topology)

    write_dataset_card(row_counts)
    print(f"Wrote Hugging Face release folder to {BUILD_ROOT}")


if __name__ == "__main__":
    main()
