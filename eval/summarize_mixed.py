"""Summarize completed local six-node mixed-category evaluations as one CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


TOPOLOGY = "six_mixed_recombine_target"
EXPECTED_SAMPLES = 160
MODELS = (
    "Qwen3.8-27B",
    "GPT-OSS-120B",
    "Nemotron-3-Nano-Omni-30B-A3B",
    "Qwen3.6-35B-A3B",
)
NODES = (
    "math_root",
    "rag_upper",
    "python_upper",
    "python_lower",
    "math_lower",
    "rlve_target",
)


def wilson(successes: int, total: int) -> tuple[float, float]:
    z = 1.96
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - half_width, center + half_width


def summarize(path: Path, model: str) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data["samples"]
    graded = [sample for sample in samples if sample.get("grading")]
    if len(samples) != EXPECTED_SAMPLES or len(graded) != EXPECTED_SAMPLES:
        raise ValueError(
            f"{path}: expected {EXPECTED_SAMPLES} fully graded samples, "
            f"found {len(graded)}/{len(samples)}"
        )
    if any(sample.get("target_node_id") != "rlve_target" for sample in graded):
        raise ValueError(f"{path}: unexpected target node")
    if any(set(sample["grading"]["node_correct"]) != set(NODES) for sample in graded):
        raise ValueError(f"{path}: unexpected node set")

    target_correct = sum(bool(sample["grading"]["target_correct"]) for sample in graded)
    low, high = wilson(target_correct, len(graded))
    row: dict[str, object] = {
        "model": model,
        "samples": len(graded),
        "target_correct": target_correct,
        "target_accuracy": target_correct / len(graded),
        "target_wilson_low": low,
        "target_wilson_high": high,
        "mean_token_length": sum(sample.get("token_length") or 0 for sample in graded) / len(graded),
        "context_length_exceeded": sum(
            sample.get("agent_status") == "context_length_exceeded" for sample in graded
        ),
    }
    for node_id in NODES:
        row[f"{node_id}_accuracy"] = sum(
            bool(sample["grading"]["node_correct"][node_id]) for sample in graded
        ) / len(graded)

    # These HotpotQA metrics score the model's written RAG answer. Intermediate
    # node correctness above can instead come from a child's reported dependency.
    for metric in ("em", "sp_em", "joint_em"):
        row[f"rag_written_{metric}"] = sum(
            sample["grading"]["hotpotqa"]["rag_upper"][metric] for sample in graded
        ) / len(graded)
    return row


def main() -> None:
    default_root = Path(__file__).resolve().parent / "results" / "mixed"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", type=Path, default=default_root)
    parser.add_argument("--output", type=Path, default=default_root / "summary.csv")
    parser.add_argument("--models", nargs="+", default=MODELS)
    args = parser.parse_args()

    rows = []
    for model in args.models:
        path = args.results_root / model / f"agent_{TOPOLOGY}_{model}.json"
        if not path.is_file():
            print(f"Missing {path}; skipping {model}")
            continue
        rows.append(summarize(path, model))
    if not rows:
        raise SystemExit("No completed mixed-category results found.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} model summaries to {args.output}")


if __name__ == "__main__":
    main()
