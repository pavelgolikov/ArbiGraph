"""Measure reward variation and agent behavior in the RLVR pilot."""

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def metrics(groups):
    """Summarize groups of calculator-enabled agent trajectories."""

    rollouts = [rollout for group in groups for rollout in group]
    rewards_by_group = [
        [int(rollout["grading"]["target_correct"]) for rollout in group]
        for group in groups
    ]
    completion_lengths = sorted(rollout["token_length"] for rollout in rollouts)
    group_variances = [
        statistics.pvariance(rewards)
        for rewards in rewards_by_group
    ]
    return {
        "num_prompts": len(groups),
        "num_rollouts": len(rollouts),
        "pass_rate": statistics.fmean(
            rollout["grading"]["target_correct"] for rollout in rollouts
        ),
        "mixed_group_fraction": statistics.fmean(
            min(rewards) < max(rewards) for rewards in rewards_by_group
        ),
        "all_zero_group_fraction": statistics.fmean(
            max(rewards) == 0 for rewards in rewards_by_group
        ),
        "all_one_group_fraction": statistics.fmean(
            min(rewards) == 1 for rewards in rewards_by_group
        ),
        "mean_group_reward_variance": statistics.fmean(group_variances),
        "agent_success_rate": statistics.fmean(
            rollout["agent_status"] == "success" for rollout in rollouts
        ),
        "tool_use_rate": statistics.fmean(
            rollout["agent_tool_calls"] > 0 for rollout in rollouts
        ),
        "mean_tool_calls": statistics.fmean(
            rollout["agent_tool_calls"] for rollout in rollouts
        ),
        "mean_turns": statistics.fmean(
            rollout["agent_turns"] for rollout in rollouts
        ),
        "all_node_answer_rate": sum(
            sum(rollout["grading"]["node_answered"].values())
            for rollout in rollouts
        ) / sum(rollout["grading"]["num_nodes"] for rollout in rollouts),
        "all_nodes_correct_rate": statistics.fmean(
            rollout["grading"]["all_nodes_correct"] for rollout in rollouts
        ),
        "agent_status_counts": dict(sorted(Counter(
            rollout["agent_status"] for rollout in rollouts
        ).items())),
        "mean_completion_tokens": statistics.fmean(completion_lengths),
        "p95_completion_tokens": completion_lengths[
            math.ceil(0.95 * len(completion_lengths)) - 1
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        parser.error(f"{args.output} exists; pass --overwrite to replace it")
    with args.input.open(encoding="utf-8") as handle:
        results = json.load(handle)

    rollouts_per_prompt = results["summary"]["rollouts_per_prompt"]
    grouped = defaultdict(list)
    for sample in results["samples"]:
        if not sample.get("grading"):
            raise ValueError("Run eval/grade_results.py before this analysis.")
        key = (
            sample["condition"],
            sample["target_native_task_id"],
            sample["sample_idx"],
        )
        grouped[key].append(sample)

    by_condition = defaultdict(list)
    by_target = defaultdict(lambda: defaultdict(list))
    for (condition, target_task_id, _), group in sorted(grouped.items()):
        if len(group) != rollouts_per_prompt:
            raise ValueError(
                f"Expected {rollouts_per_prompt} trajectories per prompt; got {len(group)}."
            )
        if sorted(sample["rollout_idx"] for sample in group) != list(
            range(rollouts_per_prompt)
        ):
            raise ValueError("A rollout group has missing or duplicate rollout indices.")
        by_condition[condition].append(group)
        by_target[target_task_id][condition].append(group)

    analysis = {
        "summary": results["summary"],
        "by_condition": {
            condition: metrics(groups)
            for condition, groups in sorted(by_condition.items())
        },
        "by_target_task": {
            str(task_id): {
                condition: metrics(groups)
                for condition, groups in sorted(conditions.items())
            }
            for task_id, conditions in sorted(by_target.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")

    print(
        f"{'condition':<18} {'pass':>8} {'mixed':>8} {'all-zero':>10} "
        f"{'tools':>8} {'turns':>8} {'p95 tok':>9}"
    )
    for condition, values in analysis["by_condition"].items():
        print(
            f"{condition:<18} {values['pass_rate']:>8.1%} "
            f"{values['mixed_group_fraction']:>8.1%} "
            f"{values['all_zero_group_fraction']:>10.1%} "
            f"{values['mean_tool_calls']:>8.1f} "
            f"{values['mean_turns']:>8.1f} "
            f"{values['p95_completion_tokens']:>9}"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
