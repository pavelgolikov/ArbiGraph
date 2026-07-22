"""Build forgetting benchmarks by prepending unrelated baseline tasks.

Each source sample remains the evaluated target, but its prompt is preceded by
randomly selected distractor tasks. The model is instructed to solve every
task, while accuracy is measured on the target placed last.
"""

import datetime
import json
import os
import random
import re
from typing import Any, Dict, List


# Copy only source metadata that belongs to the evaluated target. Prompt,
# ground truth, and forgetting metadata are rebuilt below.
TARGET_SAMPLE_FIELDS = ("task_id", "task_name", "sample_idx", "chain_input", "extra_inputs")

FORGETTING_PROMPT_PREAMBLE = (
    "Solve all tasks below in order and show your work for each one. "
    "Each Task N block is independent: all names, symbols, and definitions "
    "introduced in a block are local to that block and do not carry over to "
    "another block. Put each answer in its own \\boxed{} block with the exact "
    "requested output name inside the box, for example "
    "\\boxed{task_1_out = ...}."
)

TASK_SCOPED_NAME_RE = re.compile(
    r"task_1(?!\d)|"
    r"(?<![A-Za-z0-9_])"
    r"(?:list|val|points|M1|M2|P|S|mat_1|matrix_1|poly_1|poly_2|seq2)_1(?!\d)"
)


def save_json(data: Dict[str, Any], path: str) -> None:
    """Write JSON using the compact-list formatting used by other datasets."""
    json_str = json.dumps(data, indent=2)

    # Keep arrays of simple scalar values on one line while leaving nested
    # objects and arrays in normal indented JSON form.
    json_str = re.sub(
        r"\[\s+([^\[\]\{\}]*?)\s+\]",
        lambda m: "[" + re.sub(r"\s+", " ", m.group(1)) + "]",
        json_str,
    )
    json_str = re.sub(r"\[\s+\]", "[]", json_str)
    with open(path, "w") as f:
        f.write(json_str)


def load_json(path: str) -> Dict[str, Any]:
    """Load a dataset or completed result file from disk."""
    with open(path, "r") as f:
        return json.load(f)


def renumber_prompt(prompt: str, task_number: int) -> str:
    """Rewrite a standalone Task 1 prompt for its position in a task chain.

    Both visible task headings and task-scoped variable names are renumbered.
    Names are matched explicitly because a generic ``_1`` replacement would
    also corrupt mathematical notation such as ``H_1``.
    """
    task_suffix = f"_{task_number:d}"
    prompt = re.sub(r"\bTask 1\b", f"Task {task_number:d}", prompt)

    def replace_task_index(match: re.Match) -> str:
        return re.sub(r"_1$", task_suffix, match.group(0))

    return TASK_SCOPED_NAME_RE.sub(replace_task_index, prompt)


def _display_path(path: str, repo_root: str) -> str:
    """Prefer a portable repo-relative source path when one is available."""
    abs_path = os.path.abspath(path)
    try:
        rel_path = os.path.relpath(abs_path, repo_root)
    except ValueError:
        # Different drives on Windows cannot be represented by a relative path.
        return abs_path
    if rel_path.startswith(".."):
        # Preserve the absolute path when the source lives outside the repo.
        return abs_path
    return rel_path


def _sample_metadata(sample: Dict[str, Any], source_index: int, task_number: int) -> Dict[str, Any]:
    """Record enough provenance to reconstruct a selected distractor."""
    metadata = {
        "source_index": source_index,
        "task_number": task_number,
        "task_id": sample.get("task_id"),
        "task_name": sample.get("task_name"),
        "sample_idx": sample.get("sample_idx"),
    }

    # Some dataset modes expose the value entering the standalone task. Keep it
    # when present because it is useful for auditing generated chains.
    if "chain_input" in sample:
        metadata["chain_input"] = sample["chain_input"]
    return metadata


def _clean_target_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Copy only metadata that identifies the evaluated target sample."""
    return {key: sample[key] for key in TARGET_SAMPLE_FIELDS if key in sample}


def _choose_distractors(
    samples: List[Dict[str, Any]],
    target_sample: Dict[str, Any],
    num_distractors: int,
    rng: random.Random,
) -> List[int]:
    """Choose distractors with distinct task names.

    Distractors cannot share the target's task ID or task name, and no two
    distractors can share a task name.
    """
    target_task_id = target_sample.get("task_id")
    target_task_name = target_sample.get("task_name")
    candidates_by_name: Dict[Any, List[int]] = {}

    for index, sample in enumerate(samples):
        if sample.get("task_id") == target_task_id:
            continue
        if target_task_name is not None and sample.get("task_name") == target_task_name:
            continue

        # Task names are expected in benchmark datasets. Fall back to task_id
        # so older or custom datasets still receive template-level uniqueness.
        task_name = sample.get("task_name")
        template_key = task_name if task_name is not None else ("task_id", sample.get("task_id"))
        candidates_by_name.setdefault(template_key, []).append(index)

    # Fail loudly instead of silently reducing the requested context load.
    if len(candidates_by_name) < num_distractors:
        raise ValueError(
            "Not enough distinct distractor task names for "
            f"task_id={target_task_id!r}, task_name={target_task_name!r}: "
            f"need {num_distractors}, found {len(candidates_by_name)}"
        )

    selected_names = rng.sample(list(candidates_by_name), num_distractors)
    return [rng.choice(candidates_by_name[name]) for name in selected_names]


def generate_forgetting_dataset(
    input_path: str,
    mode: str,
    num_distractors: int = 3,
    seed: int = 0,
    repo_root: str = "",
) -> Dict[str, Any]:
    """Create one forgetting-chain sample for every source sample.

    Args:
        input_path: Baseline dataset or result JSON used as the source pool.
        mode: Dataset family name included in the generated summary.
        num_distractors: Number of unrelated tasks placed before each target.
        seed: Seed controlling distractor selection.
        repo_root: Root used to make ``source_dataset`` portable when possible.

    Returns:
        A dataset with the same number of samples and ground truths as the
        source, but with each prompt expanded into a distractor-plus-target
        chain.
    """
    if num_distractors < 1:
        raise ValueError("--num_distractors must be at least 1")

    # Normalize inputs once so metadata and random sampling remain reproducible.
    repo_root = os.path.abspath(repo_root or os.getcwd())
    source_data = load_json(input_path)
    source_summary = source_data.get("summary", {})
    source_samples = source_data.get("samples", [])
    rng = random.Random(seed)

    forgetting_samples = []
    # Distractors occupy tasks 1..N; the original sample is always task N+1.
    target_task_number = num_distractors + 1

    for target_source_index, target_sample in enumerate(source_samples):
        distractor_indices = _choose_distractors(
            source_samples,
            target_sample,
            num_distractors,
            rng,
        )

        # Preserve sampled distractor order and append the evaluated target.
        ordered_indices = distractor_indices + [target_source_index]

        # Every baseline prompt is authored as Task 1, so each prompt and its
        # variables must be renamed before concatenation to avoid collisions.
        prompt_parts = [
            renumber_prompt(source_samples[source_index]["prompt"], task_number)
            for task_number, source_index in enumerate(ordered_indices, start=1)
        ]

        # Keep the target's dataset identity and ground truth. Distractors only
        # contribute prompt text and provenance; they are not independently
        # scored in this benchmark.
        forgetting_sample = _clean_target_sample(target_sample)
        task_prompt = "\n\n".join(part.rstrip() for part in prompt_parts)
        forgetting_sample["prompt"] = f"{FORGETTING_PROMPT_PREAMBLE}\n\n{task_prompt}\n"
        forgetting_sample["ground_truth"] = target_sample["ground_truth"]

        # Store source indices and task positions to make every generated chain
        # auditable without duplicating entire distractor records.
        forgetting_sample["forgetting"] = {
            "target_source_index": target_source_index,
            "target_task_number": target_task_number,
            "num_distractors": num_distractors,
            "distractors": [
                _sample_metadata(source_samples[source_index], source_index, task_number)
                for task_number, source_index in enumerate(distractor_indices, start=1)
            ],
        }
        forgetting_samples.append(forgetting_sample)

    # Preserve source-level cardinality metadata while adding the parameters
    # needed to reproduce this forgetting transformation.
    return {
        "summary": {
            "creation_time": datetime.datetime.now().isoformat(),
            "mode": f"{mode}_forgetting",
            "source_dataset": _display_path(input_path, repo_root),
            "source_mode": source_summary.get("mode"),
            "num_tasks": source_summary.get("num_tasks"),
            "num_samples_per_task": source_summary.get("num_samples_per_task"),
            "source_num_samples": len(source_samples),
            "num_samples": len(forgetting_samples),
            "seed": seed,
            "num_distractors": num_distractors,
            "distractor_exclusion": "same_task_id_or_task_name",
            "distractor_uniqueness": "task_name",
            "target_position": "last",
            "target_task_number": target_task_number,
        },
        "samples": forgetting_samples,
    }
