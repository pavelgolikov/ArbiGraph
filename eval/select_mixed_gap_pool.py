"""Select pinned RAG and RLVE tasks from the completed eight-repeat prescreen.

Only prescreen outcomes and the pre-existing math/python calibration lengths enter
this choice. The subsequent calibration samples remain held out.
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "generate" / "generated" / "mixed_gap"
RESULT_ROOT = ROOT / "eval" / "results"
MODELS = ("Qwen3.8-27B", "GPT-OSS-120B", "gemma-4-31B-it")
REPEATS = 8


def read_prescreen(category, candidate_ids):
    by_task = defaultdict(dict)
    for model in MODELS:
        path = (RESULT_ROOT / "mixed_gap" / "prescreen" / model / category /
                f"agent_{category}_prescreen_{model}.json")
        samples = json.loads(path.read_text(encoding="utf-8"))["samples"]
        for sample in samples:
            task_id = int(sample["target_native_task_id"])
            if task_id not in candidate_ids:
                raise ValueError(f"Unexpected {category} task {task_id} in {path}")
            if "grading" not in sample:
                raise ValueError(f"Ungraded {category} task {task_id} in {path}")
            by_task[task_id].setdefault(model, []).append(sample)

    if set(by_task) != set(candidate_ids):
        raise ValueError(f"Missing {category} candidates: {set(candidate_ids) - set(by_task)}")
    details = {}
    for task_id, by_model in by_task.items():
        if any(len(by_model.get(model, [])) != REPEATS for model in MODELS):
            raise ValueError(f"Expected {REPEATS} repeats per model for {category} {task_id}")
        details[task_id] = {
            "correct_of_8": {
                model: sum(bool(s["grading"]["target_correct"])
                           for s in by_model[model]) for model in MODELS
            },
            "mean_generated_tokens": statistics.mean(
                s["token_length"] for model in MODELS for s in by_model[model]
            ),
        }
    return details


def baseline_token_reference():
    medians = []
    for model in MODELS:
        for category in ("math", "python"):
            values = []
            folder = RESULT_ROOT / "gap" / model / category
            for path in sorted(folder.glob("agent_single_target_baseline*.json")):
                for sample in json.loads(path.read_text(encoding="utf-8"))["samples"]:
                    if sample.get("grading") and sample.get("token_length") is not None:
                        values.append(sample["token_length"])
            if not values:
                raise ValueError(f"No graded gap calibration for {model}/{category}")
            medians.append(statistics.median(values))
    return statistics.median(medians)


def main():
    manifest = json.loads((DATA_ROOT / "candidates.json").read_text(encoding="utf-8"))
    details = {category: read_prescreen(category, set(manifest[category]))
               for category in ("rag", "rlve")}
    for task_id, row in details["rag"].items():
        row["context_chars"] = manifest["rag_context_chars"][str(task_id)]

    perfect = {category: [task_id for task_id, row in details[category].items()
                          if all(row["correct_of_8"][model] == REPEATS for model in MODELS)]
               for category in ("rag", "rlve")}
    if len(perfect["rag"]) < 3:
        raise ValueError("Fewer than three RAG tasks passed 8/8 on all models")
    rag = sorted(perfect["rag"],
                 key=lambda task_id: (-details["rag"][task_id]["mean_generated_tokens"],
                                      task_id))[:3]

    reference = baseline_token_reference()
    rlve_eligible = [task_id for task_id in perfect["rlve"]
                     if details["rlve"][task_id]["mean_generated_tokens"] >= 1800]
    if len(rlve_eligible) < 5:
        raise ValueError("Fewer than five RLVE tasks passed the accuracy and length rules")
    rlve = sorted(rlve_eligible,
                  key=lambda task_id: (abs(details["rlve"][task_id]["mean_generated_tokens"]
                                           - reference), task_id))[:5]

    selection = {
        "models": MODELS,
        "prescreen_repeats_per_task": REPEATS,
        "selection_rule": {
            "accuracy": "8/8 correct on each of the three models",
            "rag": "Three highest mean generated-token counts among accuracy passers",
            "rlve": "Five nearest the median of six math/python solo-token medians, among accuracy passers with at least 1800 mean generated tokens",
        },
        "math_python_reference_median_generated_tokens": reference,
        "selected": {"rag": rag, "rlve": rlve},
        "prescreen": {category: {str(task_id): row for task_id, row in sorted(rows.items())}
                      for category, rows in details.items()},
    }
    (DATA_ROOT / "selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    for category, ids in (("rag", rag), ("rlve", rlve)):
        (DATA_ROOT / f"selected_{category}_ids.json").write_text(
            json.dumps(ids, indent=2) + "\n", encoding="utf-8")
        print(f"{category}: {ids}")
    print(f"Math/python reference median: {reference:.1f} generated tokens")


if __name__ == "__main__":
    main()
