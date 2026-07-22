"""Convert a benchmark dataset JSON file to a prompt-oriented text file."""

import argparse
import json
import os
from typing import Any, Dict


SEPARATOR = "=" * 70
DIVIDER = "-" * 70


def load_dataset(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data.get("samples"), list):
        raise ValueError("Input JSON must contain a top-level 'samples' list")
    return data


def format_dataset(data: Dict[str, Any]) -> str:
    summary = data.get("summary", {})
    samples = data["samples"]
    mode_label = str(summary.get("mode", "benchmark")).replace("_", " ").upper()

    lines = [
        SEPARATOR,
        f"  {mode_label} DATASET  |  samples={len(samples)}",
        SEPARATOR,
        "",
    ]

    for source_index, sample in enumerate(samples):
        if "prompt" not in sample:
            raise ValueError(f"Sample {source_index} does not contain a 'prompt' field")

        task_name = sample.get("task_name", "UnknownTask")
        sample_idx = sample.get("sample_idx")
        lines.extend(
            [
                SEPARATOR,
                f"  Sample {source_index}  |  {task_name}  |  sample_idx={sample_idx}",
                f"  Output: {sample.get('ground_truth')}",
                SEPARATOR,
                str(sample["prompt"]).rstrip(),
                "",
                DIVIDER,
                "",
            ]
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a benchmark dataset JSON file to a prompt-oriented text file."
    )
    parser.add_argument("--input", required=True, help="Input dataset JSON file")
    parser.add_argument( "--output", default="", help="Output text file. Defaults to same name with .txt extension.")
    args = parser.parse_args()

    output_path = args.output or os.path.splitext(args.input)[0] + ".txt"
    data = load_dataset(args.input)
    text = format_dataset(data)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Converted {len(data['samples'])} samples to {output_path}")


if __name__ == "__main__":
    main()
