"""Choose which tasks fill the nodes of a pinned-instantiation experiment.

A task earns a place in the pool by three tests, all decided before any graph is
run and none of them reading a topology result:

1. The model solves it on its own. Its single-target baseline rollouts, already
   on disk, must all be correct, or at least `--min_correct` of them.
2. Its pinned instantiation can fill a node that has a parent. Such a node
   carries a bank of alternative inputs whose answers must all differ, and some
   instantiations cannot supply one.
3. That pinned instantiation is the one the single-target baseline generated, so
   the calibration measures a problem the model has already been run on. Building
   the bank can force a different draw, which puts the task outside the baseline.

Failing the first test means a graph holding the task fails for a reason that has
nothing to do with composition. Failing either of the others means the task would
be calibrated and then never appear in a graph, or would appear as a different
problem from the calibrated one.

An optional fourth test, `--max_mean_tokens`, drops a task whose single-target
rollouts generate more than that many tokens on average. A model with a short
context window cannot hold many such tasks in one graph, and a graph that
overflows grades as a failure for a reason that is not composition either. The
test reads token counts, never outcomes, so it cannot favour tasks the model
happens to solve.

Another optional test, `--within`, keeps only the tasks another pool also holds,
such as another model's. Every task pins the same problem for every model, so
the two models' experiments then run on identical problems. That pool is fixed
before this model's gap runs, so the test reads none of the results they measure.

The output is one JSON list of native task ids per category, ready to pass to
generate_dataset.py as --pool_task_ids.

    python generate/screen_task_pools.py \
        --results eval/results/exp1/Qwen3.8-27B \
        --output_dir generate/pools/Qwen3.8-27B

GPT-OSS-120B has a 131k context, and a few tasks reason long enough that
eighteen of them overflow it. Its pools add the token test, and its python pool
keeps only the tasks Qwen3.8-27B's python pool holds:

    python generate/screen_task_pools.py \
        --results eval/results/exp1/GPT-OSS-120B \
        --baseline 'agent_single_target_baseline_target_only_harmony_*.json' \
        --output_dir generate/pools/GPT-OSS-120B \
        --max_mean_tokens 12000 \
        --categories math
    python generate/screen_task_pools.py \
        --results eval/results/exp1/GPT-OSS-120B \
        --baseline 'agent_single_target_baseline_target_only_harmony_*.json' \
        --output_dir generate/pools/GPT-OSS-120B \
        --max_mean_tokens 12000 \
        --categories python \
        --within generate/pools/Qwen3.8-27B

gemma-4-31B-it has Qwen3.8-27B's 262k context and needs no token test. Its python
pool keeps only the tasks GPT-OSS-120B's python pool holds, which Qwen3.8-27B's
also holds, so all three models run their python experiments on shared problems:

    python generate/screen_task_pools.py \
        --results eval/results/exp1/gemma-4-31B-it \
        --output_dir generate/pools/gemma-4-31B-it \
        --categories math
    python generate/screen_task_pools.py \
        --results eval/results/exp1/gemma-4-31B-it \
        --output_dir generate/pools/gemma-4-31B-it \
        --categories python \
        --within generate/pools/GPT-OSS-120B
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

from fill_dag import load_pool, make_stage, pin_key, resolve_pin


BASELINE = "agent_single_target_baseline_*.json"


# Count each task's correct single-target rollouts from a finished baseline run,
# with the mean number of tokens those rollouts generated.
def baseline_scores(results_path: Path) -> dict[int, tuple[int, int, float]]:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    outcomes: dict[int, list[bool]] = collections.defaultdict(list)
    tokens: dict[int, list[int]] = collections.defaultdict(list)
    for sample in results["samples"]:
        grading = sample.get("grading") or {}
        if "target_correct" in grading:
            task_id = int(sample["target_native_task_id"])
            outcomes[task_id].append(bool(grading["target_correct"]))
            tokens[task_id].append(sample.get("token_length") or 0)
    return {
        task_id: (sum(runs), len(runs), sum(tokens[task_id]) / len(tokens[task_id]))
        for task_id, runs in outcomes.items()
    }


# Build one instantiation and report its input and answer, or None when it fails.
def instantiate(category: str, task: dict[str, Any], index: int, attempts: int, with_bank: bool):
    random.seed(pin_key(category, task["task_id"], index))
    try:
        output, _prompt, _static, value, _bank = make_stage(
            category, task, 1, ["task_1_input"], None, attempts, with_bank,
        )
    except Exception:
        return None
    return json.dumps([value, output], sort_keys=True)


# Apply the tests to one category and report what each one rejected.
def screen(category: str, results_path: Path, args) -> tuple[list[int], dict[str, list[int]]]:
    scores = baseline_scores(results_path)
    pool = {task["task_id"]: task for task in load_pool(category)}
    within = None
    if args.within is not None:
        within = set(json.loads((args.within / f"{category}.json").read_text(encoding="utf-8")))
    kept, dropped = [], {"unsolved": [], "too_long": [], "outside": [], "no_bank": [], "off_baseline": []}

    for task_id in sorted(scores):
        correct, total, mean_tokens = scores[task_id]
        if correct < min(args.min_correct, total):
            dropped["unsolved"].append(task_id)
            continue
        if args.max_mean_tokens is not None and mean_tokens > args.max_mean_tokens:
            dropped["too_long"].append(task_id)
            continue
        if within is not None and task_id not in within:
            dropped["outside"].append(task_id)
            continue
        task = pool[task_id]
        index = resolve_pin(category, task, args.pin_index, args.get_input_attempts_per_task)
        if index is None:
            dropped["no_bank"].append(task_id)
            continue
        banked = instantiate(category, task, index, args.get_input_attempts_per_task, True)
        plain = instantiate(category, task, index, args.get_input_attempts_per_task, False)
        if banked != plain:
            dropped["off_baseline"].append(task_id)
            continue
        kept.append(task_id)

    return kept, dropped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True, type=Path,
                        help="Directory holding <category>/agent_single_target_baseline_*.json.")
    parser.add_argument("--output_dir", required=True, type=Path, help="Where to write <category>.json.")
    parser.add_argument("--categories", nargs="+", default=["math", "python"])
    parser.add_argument("--baseline", default=BASELINE,
                        help="Glob naming the one baseline results file per category, when a "
                             "directory holds runs under several tags.")
    parser.add_argument("--min_correct", type=int, default=8,
                        help="Correct single-target rollouts a task needs to stay in the pool.")
    parser.add_argument("--pin_index", type=int, default=0,
                        help="Which baseline sample_idx supplies each task's pinned instantiation.")
    parser.add_argument("--get_input_attempts_per_task", type=int, default=5,
                        help="Bank size a node with a parent must carry. Must match generation.")
    parser.add_argument("--max_mean_tokens", type=float, default=None,
                        help="Drop a task whose single-target rollouts generate more tokens than this on "
                             "average. Off by default.")
    parser.add_argument("--within", type=Path, default=None,
                        help="Directory of another pool's <category>.json files; keep only tasks it also "
                             "holds. Off by default.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for category in args.categories:
        matches = sorted((args.results / category).glob(args.baseline))
        if len(matches) != 1:
            raise SystemExit(f"Expected one {args.baseline} under {args.results / category}, found {len(matches)}.")
        kept, dropped = screen(category, matches[0], args)
        total = len(kept) + sum(len(ids) for ids in dropped.values())

        path = args.output_dir / f"{category}.json"
        path.write_text(json.dumps(kept) + "\n", encoding="utf-8")
        print(f"{category}: kept {len(kept)} of {total} tasks -> {path}")
        reasons = {
            "unsolved": f"fewer than {args.min_correct} correct single-target rollouts",
            "too_long": f"more than {args.max_mean_tokens:g} generated tokens per rollout on average"
                        if args.max_mean_tokens is not None else "",
            "outside": f"not in {args.within / f'{category}.json'}" if args.within is not None else "",
            "no_bank": "no instantiation can fill a node with a parent",
            "off_baseline": "the bank forces an instantiation the baseline never generated",
        }
        for reason, ids in dropped.items():
            if ids:
                print(f"   dropped {len(ids):3d} ({reasons[reason]}): {ids}")


if __name__ == "__main__":
    main()
