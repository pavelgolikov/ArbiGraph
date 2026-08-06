"""Extract prompt strings from a benchmark dataset JSON file."""

import argparse
import json
import os
from typing import Any


SEPARATOR = "=" * 70


def load_prompts(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data.get("samples"), list):
        raise ValueError("Input JSON must contain a top-level 'samples' list")

    prompts = []
    for sample_index, sample in enumerate(data["samples"]):
        if not isinstance(sample, dict):
            raise ValueError(f"Sample {sample_index} must be a JSON object")

        prompt = sample.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError(f"Sample {sample_index} does not contain a string 'prompt' field")

        prompts.append(prompt.rstrip())

    return prompts


def format_prompts(prompts: list[str]) -> str:
    lines = []
    for prompt_index, prompt in enumerate(prompts, start=1):
        lines.extend(
            [
                SEPARATOR,
                f"Prompt {prompt_index}",
                SEPARATOR,
                prompt,
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract prompt strings from a benchmark dataset JSON file."
    )
    parser.add_argument("--input", required=True, help="Input dataset JSON file")
    parser.add_argument("--output", default="", help="Output text file. Defaults to same name with .txt extension.")
    args = parser.parse_args()

    output_path = args.output or os.path.splitext(args.input)[0] + ".txt"
    prompts = load_prompts(args.input)
    text = format_prompts(prompts)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Wrote {len(prompts)} prompts to {output_path}")


if __name__ == "__main__":
    main()
