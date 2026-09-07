from __future__ import annotations

import json
import os
from pathlib import Path

from old.hf_release.build_hf_release import BUILD_ROOT, CATEGORIES, EXPECTED_ROWS, TOPOLOGIES, config_name


EXPECTED_KEYS = {
    "id",
    "category",
    "topology",
    "target_node_id",
    "target_native_task_id",
    "target_native_task_name",
    "sample_idx",
    "prompt",
    "target_output_json",
    "node_outputs_json",
}


def iter_configs() -> list[tuple[str, str, str]]:
    return [
        (config_name(category, topology), category, topology)
        for category in CATEGORIES
        for topology in TOPOLOGIES
    ]


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def validate_readme(config_names: set[str]) -> None:
    readme_path = BUILD_ROOT / "README.md"
    if not readme_path.exists():
        raise FileNotFoundError(f"Missing dataset card: {readme_path}")

    text = readme_path.read_text(encoding="utf-8")
    for name in config_names:
        if f"config_name: {name}" not in text:
            raise ValueError(f"README.md is missing config {name}.")
    if "config_name: default" in text:
        raise ValueError("README.md must not define a combined default config.")


def validate_no_unwanted_files() -> None:
    unwanted_suffixes = ("_dataset.json", "_dataset.txt", "_topology.svg")
    for path in BUILD_ROOT.rglob("*"):
        if path.is_dir():
            continue
        if any(path.name.endswith(suffix) for suffix in unwanted_suffixes):
            raise ValueError(f"Unexpected generated source artifact in HF build: {path}")
        if "__pycache__" in path.parts or ".pytest_cache" in path.parts:
            raise ValueError(f"Unexpected cache file in HF build: {path}")


def validate_jsonl_file(category: str, topology: str) -> None:
    path = BUILD_ROOT / "data" / category / f"{topology}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing JSONL data file: {path}")

    rows = read_jsonl(path)
    expected_rows = EXPECTED_ROWS[category]
    if len(rows) != expected_rows:
        raise ValueError(f"{path} has {len(rows)} rows, expected {expected_rows}.")

    for index, row in enumerate(rows):
        if set(row) != EXPECTED_KEYS:
            raise ValueError(f"{path}:{index + 1} has unexpected keys: {sorted(row)}")
        if row["category"] != category:
            raise ValueError(f"{path}:{index + 1} has category {row['category']!r}.")
        if row["topology"] != topology:
            raise ValueError(f"{path}:{index + 1} has topology {row['topology']!r}.")
        if "\\boxed" in row["prompt"] or "boxed" in row["prompt"]:
            raise ValueError(f"{path}:{index + 1} contains old boxed output format text.")
        target_output = json.loads(row["target_output_json"])
        node_outputs = json.loads(row["node_outputs_json"])
        if not isinstance(node_outputs, list):
            raise ValueError(f"{path}:{index + 1} node_outputs_json is not a JSON list.")
        if target_output is None:
            raise ValueError(f"{path}:{index + 1} target_output_json unexpectedly decoded to null.")


def validate_topology_file(category: str, topology: str) -> None:
    path = BUILD_ROOT / "topologies" / category / f"{topology}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing topology file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("target", {}).get("native_task_id") != "all":
        raise ValueError(f"{path} should keep native_task_id as all.")

    svg_path = BUILD_ROOT / "topologies" / category / f"{topology}.svg"
    if not svg_path.exists():
        raise FileNotFoundError(f"Missing topology SVG file: {svg_path}")
    svg_text = svg_path.read_text(encoding="utf-8")
    if "<svg " not in svg_text or "</svg>" not in svg_text:
        raise ValueError(f"{svg_path} does not look like a complete SVG file.")


def validate_with_datasets(config_names: list[str]) -> None:
    cache_dir = BUILD_ROOT.parent / "datasets_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir))

    try:
        from datasets import get_dataset_config_names, load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install HF validation dependencies with: pip install -r hf_release/requirements.txt"
        ) from exc

    discovered = set(get_dataset_config_names(str(BUILD_ROOT)))
    expected = set(config_names)
    if discovered != expected:
        raise ValueError(f"HF config mismatch. Found {sorted(discovered)}, expected {sorted(expected)}.")

    for name, category, _topology in iter_configs():
        dataset = load_dataset(str(BUILD_ROOT), name, split="test", cache_dir=str(cache_dir))
        expected_rows = EXPECTED_ROWS[category]
        if len(dataset) != expected_rows:
            raise ValueError(f"HF load for {name} returned {len(dataset)} rows, expected {expected_rows}.")


def main() -> None:
    if not BUILD_ROOT.exists():
        raise FileNotFoundError(f"Missing build folder: {BUILD_ROOT}")

    configs = iter_configs()
    config_names = [name for name, _category, _topology in configs]
    validate_readme(set(config_names))
    validate_no_unwanted_files()

    for _name, category, topology in configs:
        validate_jsonl_file(category, topology)
        validate_topology_file(category, topology)

    validate_with_datasets(config_names)
    print(f"Validated {len(config_names)} Hugging Face dataset configs in {BUILD_ROOT}")


if __name__ == "__main__":
    main()
