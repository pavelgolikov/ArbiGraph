"""Replay persisted agent histories to audit model-context-limit pressure.

By default this script selects the latest completed result for every
model/dataset pair under ``results/forgetting``. Use ``--write`` to save only
the ``max_model_len_cutoff_percentage`` metric in each selected result summary.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from run_agent_calc import (
    SYSTEM_PROMPT,
    TOOL_SCHEMA,
    build_final_answer_repair_prompt,
    get_initial_calculator_repair_prompt,
    missing_task_outputs,
    parse_tool_calls,
    run_calculator_tool,
    save_json,
    update_task_answers,
)


DEFAULT_FALLBACK_TOKENIZER = "Qwen/Qwen3.5-27B"


def _turn_items(turn_outputs: dict[str, str]) -> list[tuple[str, str]]:
    return sorted(turn_outputs.items(), key=lambda item: int(item[0]))


def _is_complete_result(data: dict[str, Any]) -> bool:
    samples = data.get("samples", [])
    return bool(samples) and all(
        "correct" in sample and sample.get("agent_status") is not None for sample in samples
    ) and data.get("summary", {}).get("overall_accuracy") is not None


def _latest_completed_results(result_root: Path) -> list[Path]:
    latest: dict[tuple[str, str], tuple[float, Path]] = {}
    for path in result_root.glob("**/repair/*.json"):
        data = json.loads(path.read_text())
        if not _is_complete_result(data):
            continue

        relative = path.relative_to(result_root)
        if len(relative.parts) < 4:
            continue
        key = relative.parts[0], relative.parts[1]
        candidate = (path.stat().st_mtime, path)
        if key not in latest or candidate[0] > latest[key][0]:
            latest[key] = candidate

    return [candidate[1] for _, candidate in sorted(latest.items())]


def _load_tokenizer(model: str, fallback_model: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=True)
        return tokenizer, model, True
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(
            fallback_model,
            trust_remote_code=True,
            local_files_only=True,
        )
        return tokenizer, fallback_model, False


def _replay_final_messages(
    sample: dict[str, Any],
    system_prompt: str,
    generation_params: dict[str, Any],
) -> tuple[list[dict[str, str]], str]:
    """Return the message list immediately before the final stored generation."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": sample["prompt"]},
    ]
    expected_outputs = sample["expected_task_outputs"]
    task_answers: dict[str, dict[str, Any]] = {}
    require_initial_calculator_call = generation_params.get("require_initial_calculator_call", False)
    initial_calculator_call_seen = not require_initial_calculator_call
    initial_repair_attempts_used = 0
    final_repair_attempts_used = 0
    initial_repair_attempts = generation_params.get("initial_tool_repair_attempts", 1)
    final_repair_attempts = generation_params.get("final_answer_repair_attempts", 3)
    final_messages: list[dict[str, str]] | None = None
    final_output = ""

    for raw_turn, out_text in _turn_items(sample["turn_outputs"]):
        final_messages = list(messages)
        final_output = out_text
        update_task_answers(task_answers, out_text, expected_outputs, int(raw_turn))

        tool_calls = parse_tool_calls(out_text)
        has_calculator_call = any(call.get("name") == "calculator" for call in tool_calls)

        if not initial_calculator_call_seen and not has_calculator_call:
            if initial_repair_attempts_used < initial_repair_attempts:
                messages.append({"role": "assistant", "content": out_text})
                messages.append(
                    {
                        "role": "user",
                        "content": get_initial_calculator_repair_prompt(out_text, tool_calls),
                    }
                )
                initial_repair_attempts_used += 1
                continue
            initial_calculator_call_seen = True
            tool_calls = []

        if not tool_calls:
            if not missing_task_outputs(expected_outputs, task_answers):
                break
            if final_repair_attempts_used < final_repair_attempts:
                messages.append({"role": "assistant", "content": out_text})
                messages.append(
                    {
                        "role": "user",
                        "content": build_final_answer_repair_prompt(expected_outputs, task_answers),
                    }
                )
                final_repair_attempts_used += 1
                continue
            break

        messages.append({"role": "assistant", "content": out_text})
        for call in tool_calls:
            name = call.get("name") or "unknown"
            if name == "calculator":
                try:
                    result = run_calculator_tool(call.get("arguments", {}))
                except Exception as exc:  # Preserve the evaluator's error text.
                    result = f"Failed to parse arguments: {exc}"
            else:
                result = f"Unknown tool: {name}"
            messages.append({"role": "tool", "name": name, "content": str(result)})

        if has_calculator_call:
            initial_calculator_call_seen = True

    if final_messages is None:
        raise ValueError("Result sample has no stored assistant turns.")
    return final_messages, final_output


def _has_unclosed_tool_call(text: str) -> bool:
    return (
        text.count("<tool_call>") > text.count("</tool_call>")
        or text.count("<function=") > text.count("</function>")
    )


def analyze_result(
    data: dict[str, Any],
    tokenizer,
    tokenizer_model: str,
    tokenizer_is_exact: bool,
) -> dict[str, Any]:
    summary = data["summary"]
    generation_params = summary["generation_params"]
    max_model_len = generation_params["max_model_len"]
    max_generation_tokens = generation_params["max_tokens"]
    system_prompt = summary.get("system_prompt") or SYSTEM_PROMPT
    samples = data["samples"]

    max_prompt_tokens = -1
    max_request_tokens = -1
    max_prompt_sample: dict[str, Any] | None = None
    budget_exceeded = []
    suspected_cutoffs = []

    for row_index, sample in enumerate(samples):
        messages, final_output = _replay_final_messages(sample, system_prompt, generation_params)
        rendered_prompt = tokenizer.apply_chat_template(
            messages,
            tools=TOOL_SCHEMA,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_tokens = len(tokenizer.encode(rendered_prompt))
        request_tokens = prompt_tokens + max_generation_tokens
        final_output_tokens = len(tokenizer.encode(final_output))
        remaining_tokens = max_model_len - prompt_tokens

        sample_reference = {
            "row_index": row_index,
            "task_id": sample.get("task_id"),
            "task_name": sample.get("task_name"),
            "sample_idx": sample.get("sample_idx"),
            "prompt_tokens": prompt_tokens,
            "request_tokens_with_max_generation": request_tokens,
            "remaining_tokens_before_final_generation": remaining_tokens,
            "final_output_tokens": final_output_tokens,
            "agent_turns": sample.get("agent_turns"),
            "agent_status": sample.get("agent_status"),
        }

        if prompt_tokens > max_prompt_tokens:
            max_prompt_tokens = prompt_tokens
            max_request_tokens = request_tokens
            max_prompt_sample = sample_reference

        if request_tokens > max_model_len:
            budget_exceeded.append(sample_reference)
            if (
                remaining_tokens >= 0
                and final_output_tokens >= remaining_tokens
                and _has_unclosed_tool_call(final_output)
            ):
                suspected_cutoffs.append(sample_reference)

    sample_count = len(samples)
    return {
        "analysis_version": 1,
        "analysis_completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "analysis_method": "Replay stored agent turns through the chat template and measure the final request, the maximum context because message history is append-only.",
        "analysis_tokenizer_model": tokenizer_model,
        "analysis_tokenizer_matches_result_model": tokenizer_is_exact,
        "max_model_len": max_model_len,
        "max_generation_tokens": max_generation_tokens,
        "sample_count": sample_count,
        "max_replayed_prompt_tokens": max_prompt_tokens,
        "max_replayed_request_tokens_with_max_generation": max_request_tokens,
        "max_replayed_prompt_sample": max_prompt_sample,
        "request_budget_exceeded_count": len(budget_exceeded),
        "request_budget_exceeded_rate": len(budget_exceeded) / sample_count,
        "request_budget_exceeded_samples": budget_exceeded,
        "suspected_generation_cutoff_count": len(suspected_cutoffs),
        "suspected_generation_cutoff_rate": len(suspected_cutoffs) / sample_count,
        "suspected_generation_cutoff_samples": suspected_cutoffs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, help="Explicit result JSON files to analyze.")
    parser.add_argument("--result-root", type=Path, default=Path("results/forgetting"))
    parser.add_argument("--fallback-tokenizer", default=DEFAULT_FALLBACK_TOKENIZER)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write max_model_len_cutoff_percentage into result summaries.",
    )
    args = parser.parse_args()

    paths = args.inputs or _latest_completed_results(args.result_root)
    if not paths:
        parser.error("No completed result files found.")

    for path in paths:
        data = json.loads(path.read_text())
        if not _is_complete_result(data):
            print(f"Skipping incomplete result: {path}")
            continue

        result_model = data["summary"]["generation_params"]["model"]
        tokenizer, tokenizer_model, tokenizer_is_exact = _load_tokenizer(
            result_model,
            args.fallback_tokenizer,
        )
        analysis = analyze_result(data, tokenizer, tokenizer_model, tokenizer_is_exact)
        print(
            f"{path}: {analysis['suspected_generation_cutoff_count']}/{analysis['sample_count']} "
            f"suspected cutoffs; max prompt {analysis['max_replayed_prompt_tokens']} tokens"
        )

        if args.write:
            data["summary"].pop("context_length_analysis", None)
            data["summary"]["max_model_len_cutoff_percentage"] = (
                100 * analysis["suspected_generation_cutoff_rate"]
            )
            save_json(data, path)


if __name__ == "__main__":
    main()
