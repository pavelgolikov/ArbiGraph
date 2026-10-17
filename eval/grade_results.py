"""Grade a finished agent run.

run_agent.py produces answers; this grades them. It reads a results JSON,
compares every node's parsed answer against the ground truth already stored in
that sample's node_outputs, and writes a "grading" section back into the file:
one per sample, one aggregate under summary.

Pass --graph <topology JSON> to also get parent-conditioned metrics; without it
everything that does not need edges is still computed.
"""

import argparse
import importlib.util
import json
import os
import sys
import math
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

HOTPOTQA_METRICS = (
    "em", "f1", "prec", "recall",
    "sp_em", "sp_f1", "sp_prec", "sp_recall",
    "joint_em", "joint_f1", "joint_prec", "joint_recall",
)


@lru_cache(maxsize=1)
def _load_hotpotqa_evaluator():
    """Import the official evaluator from the hotpotqa checkout beside this repository."""
    path = Path(__file__).resolve().parents[2] / "hotpotqa" / "hotpot_evaluate_v1.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"HotpotQA evaluator missing: {path}\n"
            f"Run: git clone https://github.com/hotpotqa/hotpot {path.parent}"
        )
    spec = importlib.util.spec_from_file_location("hotpot_evaluate_v1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_hotpotqa(prediction, ground_truth):
    """Score one node in memory using the official HotpotQA evaluator."""
    evaluator = _load_hotpotqa_evaluator()
    metrics = dict.fromkeys(HOTPOTQA_METRICS, 0.0)
    # Missing or malformed components receive zero, as absent predictions do upstream.
    if isinstance(prediction, dict):
        if isinstance(prediction.get("answer"), str):
            evaluator.update_answer(metrics, prediction["answer"], ground_truth["answer"])
        support = prediction.get("sp")
        if isinstance(support, list) and all(
            isinstance(fact, (list, tuple)) and len(fact) == 2
            and isinstance(fact[0], str) and type(fact[1]) is int
            for fact in support
        ):
            evaluator.update_sp(metrics, support, ground_truth["sp"])

    # Upstream defines this joint combination inside its file-based eval() wrapper.
    metrics["joint_em"] = metrics["em"] * metrics["sp_em"]
    metrics["joint_prec"] = metrics["prec"] * metrics["sp_prec"]
    metrics["joint_recall"] = metrics["recall"] * metrics["sp_recall"]
    joint_prec, joint_recall = metrics["joint_prec"], metrics["joint_recall"]
    if joint_prec + joint_recall > 0:
        metrics["joint_f1"] = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
    return metrics


def joint_exact_match(prediction, ground_truth):
    return score_hotpotqa(prediction, ground_truth)["joint_em"] == 1.0


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


# A task declares how its own answers are compared, by name, because node records
# round-trip through JSON. "numeric" suits integer, float, and list outputs; a
# task whose answer is not a number registers its own rule here.
COMPARATORS = {"numeric": grade_value, "hotpotqa": joint_exact_match}
DEFAULT_COMPARATOR = "numeric"


def comparator(name=None):
    """Return the answer comparator a task declared, or the numeric default."""
    if name is None:
        name = DEFAULT_COMPARATOR
    try:
        return COMPARATORS[name]
    except KeyError:
        raise ValueError(f"Unknown comparator {name!r}.") from None


def save_json(data, path):
    """Match run_agent.py's formatting so regrading does not reflow the file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    json_str = json.dumps(data, indent=2)
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
    # node_outputs is in visible task-number order, so the answer key is the
    # node's serialized position, NOT its node_id. The two diverge whenever labels
    # are shuffled or topology IDs differ from the prompt numbering.
    answers = sample.get("submitted_answers") or {}
    # Only the graded task's submission counts towards the target. Every other node is
    # scored from the output the model wrote as text, which is a diagnostic.
    written = sample.get("task_answers") or {}
    # What a child's get_input call said about each parent. A child cannot ask for its
    # input without stating what it believes its parents produced, so this is an exact
    # record of the value the model carried forward, and it does not depend on the model
    # writing that value down anywhere. It therefore outranks the text.
    reported = sample.get("reported_correct") or {}
    target_node_id = sample["target_node_id"]
    node_correct: Dict[str, bool] = {}
    node_answered: Dict[str, bool] = {}
    answer_turn: Dict[str, Optional[int]] = {}
    node_position: Dict[str, int] = {}
    hotpotqa = {}

    for position, record in enumerate(sample["node_outputs"], start=1):
        node_id = record["node_id"]
        node_position[node_id] = position
        entry = answers.get(f"task_{position}_out")
        if entry is None and node_id != target_node_id:
            entry = written.get(f"task_{position}_out")
        node_answered[node_id] = entry is not None
        answer_turn[node_id] = None if entry is None else entry.get("turn")
        if record.get("comparator") == "hotpotqa":
            metrics = score_hotpotqa(None if entry is None else entry["answer"], record["output_value"])
            hotpotqa[node_id] = metrics
            node_correct[node_id] = metrics["joint_em"] == 1.0
        else:
            same_answer = comparator(record.get("comparator"))
            node_correct[node_id] = (
                False if entry is None else bool(same_answer(entry["answer"], record["output_value"]))
            )
        # Only the target is scored from its submission. Every other node is scored from
        # what its children reported, falling back to the text when it has no children.
        observed = reported.get(str(position))
        if observed is not None and node_id != target_node_id:
            node_correct[node_id] = observed
            node_answered[node_id] = True

    order = [record["node_id"] for record in sample["node_outputs"]]
    first_incorrect = next((node_id for node_id in order if not node_correct[node_id]), None)
    target = target_node_id

    grading = {
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
    if hotpotqa:
        grading["hotpotqa"] = hotpotqa
    return grading


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
    # Average complete per-example scores, including zeroes for unanswered RAG nodes.
    # Targets are also reported separately when a graph contains other RAG nodes.
    all_hotpotqa = [metrics for sample in graded
                   for metrics in sample["grading"].get("hotpotqa", {}).values()]
    target_hotpotqa = [sample["grading"]["hotpotqa"][sample["target_node_id"]] for sample in graded
                      if sample["target_node_id"] in sample["grading"].get("hotpotqa", {})]
    for key, values in (("hotpotqa", all_hotpotqa), ("hotpotqa_target", target_hotpotqa)):
        if values:
            summary[key] = {"num_graded": len(values), **{
                name: sum(metrics[name] for metrics in values) / len(values) for name in HOTPOTQA_METRICS
            }}
    return summary


def main():
    parser = argparse.ArgumentParser(description="Grade an agent results JSON in place.")
    parser.add_argument("--input", required=True, help="Results JSON written by run_agent.py")
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
        for key in ("hotpotqa", "hotpotqa_target"):
            if key in grading:
                metrics = grading[key]
                print(f"{key} ({metrics['num_graded']} nodes): "
                      f"answer EM/F1={metrics['em']:.2%}/{metrics['f1']:.2%}, "
                      f"support EM/F1={metrics['sp_em']:.2%}/{metrics['sp_f1']:.2%}, "
                      f"joint EM/F1={metrics['joint_em']:.2%}/{metrics['joint_f1']:.2%}")


if __name__ == "__main__":
    main()
