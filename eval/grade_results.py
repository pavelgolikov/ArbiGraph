"""Grade a finished agent run.

run_agent_calc.py produces answers; this grades them. It reads a results JSON,
compares every node's parsed answer against the ground truth already stored in
that sample's node_outputs, and writes a "grading" section back into the file:
one per sample, one aggregate under summary.

Pass --graph <topology JSON> to also get parent-conditioned metrics; without it
everything that does not need edges is still computed.
"""

import argparse
import json
import os
import re
import sys
import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _matches_integer_exactly(prediction: object, ground_truth: int) -> bool:
    """Accept integer-valued numeric literals without rounding or truncation."""
    if isinstance(prediction, bool):
        return False
    if isinstance(prediction, int):
        return prediction == ground_truth
    if isinstance(prediction, float):
        return (
            math.isfinite(prediction)
            and prediction.is_integer()
            and int(prediction) == ground_truth
        )
    return False


def _integer_components_are_exact(prediction: object, ground_truth: object) -> bool | None:
    """Validate parseable literals wherever the ground truth requires integers."""
    if isinstance(ground_truth, int) and not isinstance(ground_truth, bool):
        return _matches_integer_exactly(prediction, ground_truth)

    if isinstance(ground_truth, list):
        if not isinstance(prediction, list) or len(prediction) != len(ground_truth):
            return False

        checked_integer = False
        for pred_item, truth_item in zip(prediction, ground_truth):
            result = _integer_components_are_exact(pred_item, truth_item)
            if result is False:
                return False
            if result is True:
                checked_integer = True
        return True if checked_integer else None

    return None


def grade_value(prediction: object, ground_truth: object) -> bool:
    integer_check = _integer_components_are_exact(prediction, ground_truth)
    if integer_check is False:
        return False
    if integer_check is True:
        return True

    if isinstance(ground_truth, list):
        if not isinstance(prediction, list) or len(prediction) != len(ground_truth):
            return False
        return all(grade_value(p, g) for p, g in zip(prediction, ground_truth))

    if isinstance(ground_truth, int) and not isinstance(ground_truth, bool):
        return _matches_integer_exactly(prediction, ground_truth)

    if isinstance(ground_truth, float):
        try:
            return math.isclose(float(prediction), float(ground_truth), abs_tol=1e-3)
        except (ValueError, TypeError):
            return False

    if isinstance(prediction, float):
        try:
            return math.isclose(float(prediction), float(ground_truth), abs_tol=1e-3)
        except (ValueError, TypeError):
            return False

    return prediction == ground_truth


def save_json(data, path):
    """Match run_agent_calc.py's formatting so regrading does not reflow the file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    json_str = json.dumps(data, indent=2)
    json_str = re.sub(r'\[\s+([^\[\]\{\}]*?)\s+\]', lambda m: '[' + re.sub(r'\s+', ' ', m.group(1)) + ']', json_str)
    json_str = re.sub(r'\[\s+\]', '[]', json_str)
    with open(path, "w") as f:
        f.write(json_str)


def load_parents(graph_path: str) -> Dict[str, List[str]]:
    with open(graph_path) as f:
        graph = json.load(f)
    parents: Dict[str, List[str]] = {node["id"]: [] for node in graph["nodes"]}
    for edge in graph.get("edges", []):
        parents[edge["destination"]].append(edge["source"])
    return parents


def grade_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Grade every node in one sample.

    A node is `correct` only if an answer was parsed and it matches. `answered`
    separates a wrong answer from one the agent never produced, which is the
    difference between a reasoning failure and a context-management failure.
    """
    # node_outputs is in topological order and the prompt numbers tasks by that
    # position, so the answer key is the node's position, NOT its node_id. The two
    # diverge whenever topological order differs from the ids in the topology JSON
    # (two_branch_recombine: node task_4 is asked for as task_3_out).
    answers = sample.get("task_answers") or {}
    node_correct: Dict[str, bool] = {}
    node_answered: Dict[str, bool] = {}
    answer_turn: Dict[str, Optional[int]] = {}
    node_position: Dict[str, int] = {}

    for position, record in enumerate(sample["node_outputs"], start=1):
        node_id = record["node_id"]
        node_position[node_id] = position
        entry = answers.get(f"task_{position}_out")
        node_answered[node_id] = entry is not None
        answer_turn[node_id] = None if entry is None else entry.get("turn")
        node_correct[node_id] = (
            False if entry is None else bool(grade_value(entry["answer"], record["output_value"]))
        )

    order = [record["node_id"] for record in sample["node_outputs"]]
    first_incorrect = next((node_id for node_id in order if not node_correct[node_id]), None)
    target = sample["target_node_id"]

    return {
        "node_correct": node_correct,
        "node_position": node_position,
        "node_answered": node_answered,
        "answer_turn": answer_turn,
        "target_correct": node_correct[target],
        "all_nodes_correct": all(node_correct.values()),
        "num_nodes_correct": sum(node_correct.values()),
        "num_nodes": len(order),
        "first_incorrect_node": first_incorrect,
    }


def aggregate(samples: List[Dict[str, Any]], parents: Optional[Dict[str, List[str]]]) -> Dict[str, Any]:
    graded = [s for s in samples if s.get("grading")]
    n = len(graded)
    if n == 0:
        return {"num_graded": 0}

    position_correct: Dict[str, List[bool]] = defaultdict(list)
    template_correct: Dict[str, List[bool]] = defaultdict(list)
    status_counts: Counter = Counter()
    first_incorrect_counts: Counter = Counter()
    child_given_parents: Dict[str, List[bool]] = defaultdict(list)
    node_answered: Dict[str, List[bool]] = defaultdict(list)

    for sample in graded:
        grading = sample["grading"]
        status_counts[sample.get("agent_status")] += 1
        first_incorrect_counts[grading["first_incorrect_node"] or "none"] += 1
        template_correct[str(sample["target_native_task_id"])].append(grading["target_correct"])
        for node_id, ok in grading["node_correct"].items():
            slot = f"task_{grading['node_position'][node_id]}"
            position_correct[slot].append(ok)
            node_answered[slot].append(grading["node_answered"][node_id])
        if parents:
            for node_id, ok in grading["node_correct"].items():
                node_parents = parents.get(node_id) or []
                if node_parents and all(grading["node_correct"].get(p) for p in node_parents):
                    child_given_parents[node_id].append(ok)

    def rate(values: List[bool]) -> float:
        return sum(values) / len(values)

    template_rates = [rate(v) for v in template_correct.values()]

    summary = {
        "num_graded": n,
        "target_accuracy": rate([s["grading"]["target_correct"] for s in graded]),
        "all_node_exact_accuracy": rate([s["grading"]["all_nodes_correct"] for s in graded]),
        "average_node_accuracy": sum(
            s["grading"]["num_nodes_correct"] / s["grading"]["num_nodes"] for s in graded
        ) / n,
        # plan 5.5: the unit of generalization is the task template, not the sample
        "macro_target_accuracy_over_templates": sum(template_rates) / len(template_rates),
        "num_templates": len(template_rates),
        "node_accuracy_by_position": {
            k: rate(v) for k, v in sorted(position_correct.items(), key=lambda kv: int(kv[0].split("_")[1]))
        },
        "node_answer_rate_by_position": {
            k: rate(v) for k, v in sorted(node_answered.items(), key=lambda kv: int(kv[0].split("_")[1]))
        },
        "first_incorrect_node_counts": dict(sorted(first_incorrect_counts.items())),
        "agent_status_counts": dict(sorted(status_counts.items(), key=lambda kv: str(kv[0]))),
        "target_accuracy_by_template": {k: rate(v) for k, v in sorted(template_correct.items())},
    }
    if parents:
        summary["conditional_child_accuracy_given_correct_parents"] = {
            k: {"accuracy": rate(v), "n": len(v)} for k, v in sorted(child_given_parents.items())
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Grade an agent results JSON in place.")
    parser.add_argument("--input", required=True, help="Results JSON written by run_agent_calc.py")
    parser.add_argument("--output", default="", help="Write here instead of updating --input in place")
    parser.add_argument("--graph", default="", help="Topology JSON, for parent-conditioned metrics")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    parents = load_parents(args.graph) if args.graph else None

    ungraded = 0
    for sample in data["samples"]:
        if not sample.get("task_answers") and sample.get("agent_status") is None:
            sample["grading"] = None
            ungraded += 1
            continue
        sample["grading"] = grade_sample(sample)

    data["summary"]["grading"] = aggregate(data["samples"], parents)

    output_path = args.output or args.input
    save_json(data, output_path)

    grading = data["summary"]["grading"]
    print(f"Graded {grading['num_graded']} samples -> {output_path}")
    if ungraded:
        print(f"Skipped {ungraded} samples with no agent output.")
    if grading["num_graded"]:
        print(f"Target Accuracy:         {grading['target_accuracy']:.2%}")
        print(f"Macro Target Accuracy:   {grading['macro_target_accuracy_over_templates']:.2%} "
              f"over {grading['num_templates']} templates")
        print(f"All-Node Exact Accuracy: {grading['all_node_exact_accuracy']:.2%}")
        print(f"Average Node Accuracy:   {grading['average_node_accuracy']:.2%}")
        print(f"By position:             {', '.join(f'{k}={v:.2%}' for k, v in grading['node_accuracy_by_position'].items())}")


if __name__ == "__main__":
    main()
