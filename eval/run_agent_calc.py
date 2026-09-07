import argparse
import json
import os
import sys
import datetime
import re
import asyncio
import copy
import zlib
from collections import Counter
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tqdm.asyncio import tqdm as async_tqdm

from parse_results import (
    looks_like_tool_call,
    missing_task_outputs,
    normalize_tool_arguments,
    parse_tool_calls,
    update_task_answers,
)


def filename_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    component = component.strip("._-")
    if not component:
        raise ValueError(f"Invalid empty filename component from {value!r}")
    return component


def timestamped_output_path(output_dir: str, input_path: str, model: str, prefix: str) -> str:
    input_stem = filename_component(os.path.splitext(os.path.basename(input_path))[0])
    model_stem = filename_component(model.split("/")[-1])
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{input_stem}_{model_stem}_{timestamp}.json"
    return os.path.join(output_dir, filename)


def save_json(data, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    json_str = json.dumps(data, indent=2)
    # Collapse arrays of simple types (like numbers/strings) horizontally
    json_str = re.sub(r'\[\s+([^\[\]\{\}]*?)\s+\]', lambda m: '[' + re.sub(r'\s+', ' ', m.group(1)) + ']', json_str)
    json_str = re.sub(r'\[\s+\]', '[]', json_str)
    with open(path, "w") as f:
        f.write(json_str)


# ---------------------------------------------------------------------------
# Per-sample checkpoints
# ---------------------------------------------------------------------------
# Progress is checkpointed one file per sample rather than by rewriting the
# whole results file. A run over a few thousand samples takes tens of thousands
# of checkpoints, and re-serializing every sample's transcript each time costs
# far more than the generation it protects. The aggregate results file is
# written once at startup and once at the end; the sidecar directory carries
# everything in between and is merged back in on resume.
SAMPLE_STATE_FIELDS = (
    "token_length",
    "agent_turns",
    "agent_tool_calls",
    "turn_outputs",
    "task_answers",
    "agent_status",
    "agent_resume_state",
)

SAMPLE_CHECKPOINT_RE = re.compile(r"sample_(\d+)\.json")


def sample_checkpoint_dir(results_path: str) -> str:
    return os.path.splitext(os.path.abspath(results_path))[0] + "_partial"


def save_sample_state(checkpoint_dir: str, index: int, sample: Dict[str, Any]) -> None:
    """Write one sample's progress, replacing the file atomically."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"sample_{index:06d}.json")
    tmp_path = f"{path}.tmp"
    payload = {field: sample.get(field) for field in SAMPLE_STATE_FIELDS}
    with open(tmp_path, "w") as f:
        json.dump(payload, f)
    os.replace(tmp_path, path)


def load_sample_states(checkpoint_dir: str, samples: List[Dict[str, Any]]) -> int:
    """Merge sidecar checkpoints back into the sample list. Returns count restored."""
    if not os.path.isdir(checkpoint_dir):
        return 0
    restored = 0
    for name in sorted(os.listdir(checkpoint_dir)):
        match = SAMPLE_CHECKPOINT_RE.fullmatch(name)
        if not match:
            continue
        index = int(match.group(1))
        if index >= len(samples):
            print(f"Warning: checkpoint {name} has no matching sample; skipping.")
            continue
        try:
            with open(os.path.join(checkpoint_dir, name)) as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Warning: could not read checkpoint {name} ({exc}); sample will be rerun.")
            continue
        if not isinstance(payload, dict):
            continue
        samples[index].update(
            {field: value for field, value in payload.items() if field in SAMPLE_STATE_FIELDS}
        )
        restored += 1
    return restored


def clear_sample_states(checkpoint_dir: str) -> None:
    """Drop the sidecar directory once its contents are folded into the results file."""
    if not os.path.isdir(checkpoint_dir):
        return
    for name in os.listdir(checkpoint_dir):
        if SAMPLE_CHECKPOINT_RE.fullmatch(name) or name.endswith(".json.tmp"):
            os.remove(os.path.join(checkpoint_dir, name))
    try:
        os.rmdir(checkpoint_dir)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# System prompt for the calculator-agent loop.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a helpful reasoning assistant with access to a calculator tool. Think step by step. "
    "You MUST use the calculator tool EVERY TIME you need to perform arithmetic operations. "
    "When several arithmetic expressions are useful, submit them together in one calculator call. "
)

TOOL_CALL_MALFORMED_REPAIR_PROMPT = (
    "Your previous calculator tool call was incomplete, malformed, or not a calculator call. "
    "Retry now with exactly one short complete calculator tool call and nothing else. "
    "Use the calculator function with the expressions parameter. "
    "Include 1 to 3 short useful arithmetic expressions. "
    "Do not include prose, reasoning, Markdown, or the final answer. "
    "Close every tag you open."
)

# ---------------------------------------------------------------------------
# Repair prompts and loop-detection thresholds
# ---------------------------------------------------------------------------
FINAL_ANSWER_REPAIR_PROMPT = (
    "Your previous response did not include all of the requested task output assignments. "
    "Provide every requested task output that you have not yet given. "
    "If any previously provided output should be corrected, provide its corrected assignment; "
    "the latest version of each output will be used. "
    "Use more calculator calls if needed. Do not restart the solution."
)

CUTOFF_REPAIR_PROMPT = (
    "Your previous response was cut off by the token limit. "
    "Continue from exactly where you stopped. Do not restart or repeat earlier work. "
    "Be concise and move toward the requested task output assignments. "
    "Use calculator calls if needed."
)

# Floor on the context room left for a turn. Below this there is not enough space
# to emit even one task output assignment, so the episode is out of context.
MIN_USABLE_OUTPUT_TOKENS = 512
CUTOFF_LOOP_COMPRESSION_WINDOW = 3
CUTOFF_LOOP_COMPRESSION_RATIO = 12.0
CUTOFF_LOOP_NO_PROGRESS_CUTOFFS = 2


def last_turn_output(turn_outputs: Any) -> Optional[str]:
    """Return the latest stored assistant turn text.
    """
    if not isinstance(turn_outputs, dict) or not turn_outputs:
        return None

    def turn_order(item):
        """Build a comparable ordering key for stored turn-output entries."""
        key = item[0]
        try:
            return 0, int(key)
        except (TypeError, ValueError):
            return 1, str(key)

    return max(turn_outputs.items(), key=turn_order)[1]


def recent_turn_compression_ratio(turn_outputs: Dict[str, str], window: int) -> float:
    if len(turn_outputs) < window:
        return 0.0

    def turn_order(item):
        key = item[0]
        try:
            return 0, int(key)
        except (TypeError, ValueError):
            return 1, str(key)

    recent_text = "\n".join(
        (text or "")
        for _, text in sorted(turn_outputs.items(), key=turn_order)[-window:]
    )
    raw = recent_text.encode("utf-8", "replace")
    if not raw:
        return 0.0
    return len(raw) / max(1, len(zlib.compress(raw)))


# ---------------------------------------------------------------------------
# MCP tool helpers
# ---------------------------------------------------------------------------
async def call_mcp_tool(client: Any, name: str, arguments: Any) -> str:
    result = await client.call_tool(name, normalize_tool_arguments(arguments))
    texts = [
        str(content.text)
        for content in result.content
        if getattr(content, "text", None) is not None
    ]
    return "\n".join(texts)


async def _generate_text(engine, prompt, sampling_params, request_id):
    """Generate one completion and report why it stopped.
    """
    results_generator = engine.generate(prompt, sampling_params, request_id)

    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    if final_output is None or not final_output.outputs:
        return "", None
    completion = final_output.outputs[0]
    return completion.text, getattr(completion, "finish_reason", None)


def render_initial_model_prompt(tokenizer, tools, system_prompt: str, task_prompt: str,
                                chat_template_kwargs: Optional[Dict[str, Any]] = None) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_prompt}
    ]
    return tokenizer.apply_chat_template(messages, tools=tools, tokenize=False,
                                         add_generation_prompt=True, **(chat_template_kwargs or {}))


def confirm_run() -> bool:
    while True:
        try:
            answer = input("Run evaluation? yes/no: ").strip().lower()
        except EOFError:
            print()
            return False
        if answer == "yes":
            return True
        if answer == "no":
            return False
        print("Please answer yes or no.", flush=True)


def write_first_sample_example(tokenizer, tools, system_prompt: str, data: Dict[str, Any],
                              chat_template_kwargs: Optional[Dict[str, Any]] = None) -> str:
    first_sample = data["samples"][0]
    prompt = render_initial_model_prompt(tokenizer, tools, system_prompt, first_sample["prompt"], chat_template_kwargs)
    path = os.path.join(REPO_ROOT, "sample_to_view.txt")
    with open(path, "w") as f:
        f.write(prompt)
        if not prompt.endswith("\n"):
            f.write("\n")
    return path

# ---------------------------------------------------------------------------
# Agent loop (real vLLM)
# ---------------------------------------------------------------------------
async def run_agent_loop(
    engine,
    tokenizer,
    sampling_params,
    tools: List[Dict[str, Any]],
    mcp_client: Any,
    system_prompt: str,
    task_prompt: str,
    max_agent_turns: int,
    index: int,
    *,
    repair_sampling_params=None,
    max_model_len: int | None = None,
    tool_repair_attempts: int = 5,
    final_answer_repair_attempts: int = 20,
    cutoff_repair_attempts: int = 20,
    max_turns_without_new_answer: int = 200,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
    expected_outputs: List[str] | None = None,
    resume_state: Optional[Dict[str, Any]] = None,
    checkpoint_callback=None,
) -> Dict[str, Any]:
    """
    Run the agentic tool-calling loop for a single task using AsyncLLMEngine.
    Returns a dict with status, turn outputs, turns taken, total tool calls,
    and collected task answers.

    """
    if resume_state is None:
        resume_state = {}
    available_tool_names = {
        tool["function"]["name"]
        for tool in tools
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    }

    requested_outputs = resume_state.get("expected_task_outputs") or expected_outputs
    if not requested_outputs:
        raise ValueError("expected_outputs is required.")

    final_answer_repair_budget = final_answer_repair_attempts
    cutoff_repair_budget = cutoff_repair_attempts

    messages = resume_state.get("messages") or [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_prompt}
    ]

    total_tool_calls = resume_state.get("total_tool_calls", 0)
    turn_outputs = dict(resume_state.get("turn_outputs") or {})
    task_answers = dict(resume_state.get("task_answers") or {})
    no_progress_cutoffs = resume_state.get("no_progress_cutoffs", 0)
    tool_repair_attempts_used = resume_state.get("tool_repair_attempts_used", 0)
    final_answer_repair_attempts_used = resume_state.get("final_answer_repair_attempts_used", 0)
    cutoff_repair_attempts_used = resume_state.get("cutoff_repair_attempts_used", 0)
    next_sampling_mode = resume_state.get("next_sampling_mode", "normal")
    start_turn = resume_state.get("next_turn", len(turn_outputs))

    def final_status(missing_requested_answers=False):
        if missing_requested_answers:
            return "missing_requested_task_answers"
        return "success"

    def build_resume_state(next_turn):
        return {
            "messages": messages,
            "next_turn": next_turn,
            "total_tool_calls": total_tool_calls,
            "tool_repair_attempts_used": tool_repair_attempts_used,
            "final_answer_repair_attempts_used": final_answer_repair_attempts_used,
            "cutoff_repair_attempts_used": cutoff_repair_attempts_used,
            "no_progress_cutoffs": no_progress_cutoffs,
            "next_sampling_mode": next_sampling_mode,
            "expected_task_outputs": requested_outputs,
        }

    def agent_result(status, turns_taken, current_resume_state=None):
        return {
            "status": status,
            "turns_taken": turns_taken,
            "total_tool_calls": total_tool_calls,
            "turn_outputs": turn_outputs,
            "expected_task_outputs": requested_outputs,
            "task_answers": task_answers,
            "resume_state": current_resume_state,
        }

    async def checkpoint(turns_taken):
        if checkpoint_callback is None:
            return
        await checkpoint_callback(
            agent_result("in_progress", turns_taken, build_resume_state(turns_taken))
        )

    for turn in range(start_turn, max_agent_turns):
        full_prompt = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False,
                                                    add_generation_prompt=True, **(chat_template_kwargs or {}))
        
        request_id = f"req_{index}_turn_{turn}"
        current_sampling_params = (
            repair_sampling_params
            if next_sampling_mode == "repair" and repair_sampling_params is not None
            else sampling_params
        )
        next_sampling_mode = "normal"
        if max_model_len is not None:
            input_tokens = len(tokenizer.encode(full_prompt, add_special_tokens=False))
            remaining_tokens = max_model_len - input_tokens
            if remaining_tokens < MIN_USABLE_OUTPUT_TOKENS:
                turn_outputs[str(turn)] = ""
                return agent_result("context_length_exceeded", turn)
            if remaining_tokens < current_sampling_params.max_tokens:
                current_sampling_params = copy.copy(current_sampling_params)
                current_sampling_params.max_tokens = remaining_tokens
        try:
            out_text, finish_reason = await _generate_text(engine, full_prompt, current_sampling_params, request_id)
        except Exception as exc:
            # One sample must not take the run down with it.
            print(f"[sample {index}] generation failed on turn {turn}: {exc}", flush=True)
            turn_outputs[str(turn)] = ""
            return agent_result("generation_error", turn)
        turn_outputs[str(turn)] = out_text
        previous_answer_names = set(task_answers)
        update_task_answers(task_answers, out_text, requested_outputs, turn)
        new_answer_added = bool(set(task_answers) - previous_answer_names)

        tool_calls = parse_tool_calls(out_text)

        cutoff_detected = finish_reason == "length"
        missing_outputs = missing_task_outputs(requested_outputs, task_answers)

        if not missing_outputs:
            return agent_result(final_status(), turn + 1)

        last_progress_turn = max(
            (entry["turn"] for entry in task_answers.values()),
            default=-1,
        )
        if turn - last_progress_turn >= max_turns_without_new_answer:
            return agent_result("tool_loop_no_progress", turn + 1)

        if cutoff_detected and not new_answer_added and missing_outputs:
            compression_ratio = recent_turn_compression_ratio(
                turn_outputs,
                CUTOFF_LOOP_COMPRESSION_WINDOW,
            )
            if compression_ratio >= CUTOFF_LOOP_COMPRESSION_RATIO:
                no_progress_cutoffs += 1
            else:
                no_progress_cutoffs = 0
            if no_progress_cutoffs >= CUTOFF_LOOP_NO_PROGRESS_CUTOFFS:
                return agent_result("cutoff_no_progress", turn + 1)
        else:
            no_progress_cutoffs = 0

        if (
            cutoff_detected
            and not tool_calls
            and not looks_like_tool_call(out_text)
            and cutoff_repair_attempts_used < cutoff_repair_budget
        ):
            messages.append({"role": "assistant", "content": out_text})
            messages.append({"role": "user", "content": CUTOFF_REPAIR_PROMPT})
            cutoff_repair_attempts_used += 1
            next_sampling_mode = "repair"
            await checkpoint(turn + 1)
            continue

        if not tool_calls:
            if looks_like_tool_call(out_text):
                if tool_repair_attempts_used < tool_repair_attempts:
                    messages.append({"role": "assistant", "content": out_text})
                    messages.append({"role": "user", "content": TOOL_CALL_MALFORMED_REPAIR_PROMPT})
                    tool_repair_attempts_used += 1
                    next_sampling_mode = "repair"
                    await checkpoint(turn + 1)
                    continue

                return agent_result("malformed_tool_calls", turn + 1)

            # Outputs are known to still be missing here: the completion check
            # above returns before reaching this point once they are all in.
            if final_answer_repair_attempts_used < final_answer_repair_budget:
                messages.append({"role": "assistant", "content": out_text})
                messages.append({"role": "user", "content": FINAL_ANSWER_REPAIR_PROMPT})
                final_answer_repair_attempts_used += 1
                next_sampling_mode = "repair"
                await checkpoint(turn + 1)
                continue

            return agent_result(final_status(missing_requested_answers=True), turn + 1)

        # Store the assistant response before tool results.
        assistant_msg = {"role": "assistant", "content": out_text}
        messages.append(assistant_msg)

        # Execute each parsed tool call and append the result.
        for tc in tool_calls:
            total_tool_calls += 1
            name = tc.get("name") or "unknown"
            if name in available_tool_names:
                try:
                    result = await call_mcp_tool(mcp_client, name, tc.get("arguments", {}))
                except Exception as e:
                    result = f"Failed to parse arguments: {e}"
            else:
                result = f"Unknown tool: {name}"

            messages.append({
                "role": "tool",
                "name": name,
                "content": str(result)
            })

        await checkpoint(turn + 1)

    return agent_result("timeout", max_agent_turns)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run agent evaluation with calculator tool on context-management benchmark datasets.",
        allow_abbrev=False,
    )
    # CLI arguments for agent evaluation.
    parser.add_argument("--input", type=str, required=True, help="Input dataset or results JSON file")
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--output", type=str, default="", help="Exact output results JSON file path.")
    output_group.add_argument("--output-dir", type=str, default="", help="Directory for a timestamped output file.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0.0 for deterministic)")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k sampling parameter")
    parser.add_argument("--min_p", type=float, default=0.0, help="Min-p sampling parameter")
    parser.add_argument("--presence_penalty", type=float, default=0.0, help="Presence penalty")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty")
    parser.add_argument("--reasoning_effort", type=str, default="xhigh", help="Reasoning depth passed to the chat template (xhigh, medium, low)")
    parser.add_argument("--preserve_thinking", action=argparse.BooleanOptionalAction, default=True, help="Keep reasoning context from earlier turns (--no-preserve_thinking to drop it)")
    parser.add_argument("--max_tokens", type=int, default=32768, help="Max tokens per normal generation turn")
    parser.add_argument("--repair_turn_max_tokens", type=int, default=32768, help="Max tokens per repair generation turn")
    parser.add_argument("--max_model_len", type=int, default=262144, help="Max model sequence length for vLLM")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-27B", help="Model name/path")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs (tensor_parallel_size)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="vLLM gpu_memory_utilization")
    parser.add_argument("--max_cudagraph_capture_size", type=int, default=None, help="Limit vLLM CUDA graph capture batch size")
    parser.add_argument("--hf_overrides_json", type=str, default="", help="JSON object passed to vLLM hf_overrides")
    parser.add_argument("--disable_custom_all_reduce", action="store_true", help="Disable vLLM custom all-reduce kernels/fusions for multi-GPU startup stability")
    parser.add_argument("--disable_language_model_only", action="store_false", dest="language_model_only", default=True, help="Do not pass language_model_only=True to vLLM")
    parser.add_argument("--example", action="store_true", help="Write the first rendered model prompt to sample_to_view.txt and ask before running")
    # Agent-specific parameters
    parser.add_argument("--max_agent_turns", type=int, default=2000, help="Maximum number of agent turns per task")
    parser.add_argument("--tool_repair_attempts", type=int, default=5, help="Repair turns to try when the model emits a malformed tool call. Flat total per sample, NOT per task.")
    parser.add_argument("--final_answer_repair_attempts", type=int, default=20, help="Repair turns to try when requested task output assignments are missing. Flat total per sample, NOT per task.")
    parser.add_argument("--cutoff_repair_attempts", type=int, default=20, help="Repair turns to try after a generation cutoff. Flat total per sample, NOT per task.")
    parser.add_argument("--max_turns_without_new_answer", type=int, default=200, help="Abort a sample after this many consecutive turns produce no new task output. Flat budget per sample, NOT per task. Bounds tool-call loops, which no repair budget covers.")
    parser.add_argument("--max_concurrent_samples", type=int, default=128, help="Maximum samples in flight at once (0 for unlimited)")

    args = parser.parse_args()
    chat_template_kwargs = {
        "reasoning_effort": args.reasoning_effort,
        "preserve_thinking": args.preserve_thinking,
    }
    hf_overrides = None
    if args.hf_overrides_json:
        try:
            hf_overrides = json.loads(args.hf_overrides_json)
        except json.JSONDecodeError as exc:
            parser.error(f"--hf_overrides_json must be valid JSON: {exc}")
        if not isinstance(hf_overrides, dict):
            parser.error("--hf_overrides_json must decode to a JSON object")

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        sys.exit(1)

    with open(args.input, "r") as f:
        data = json.load(f)

    expected_model = args.model
    is_dataset = "generation_params" not in data.get("summary", {})
    system_prompt = SYSTEM_PROMPT

    if not is_dataset:
        if args.output_dir:
            parser.error("--output-dir is only for new dataset runs; use --output or continue in place")
        system_prompt = data.get("summary", {}).get("system_prompt") or system_prompt
        # Continuing from a partial results file – validate generation params match
        gen_params = data.get("summary", {}).get("generation_params", {})
        if gen_params.get("temperature") != args.temperature or \
           gen_params.get("top_p") != args.top_p or \
           gen_params.get("top_k") != args.top_k or \
           gen_params.get("min_p") != args.min_p or \
           gen_params.get("presence_penalty") != args.presence_penalty or \
           gen_params.get("repetition_penalty") != args.repetition_penalty or \
           gen_params.get("reasoning_effort") != args.reasoning_effort or \
           gen_params.get("preserve_thinking") != args.preserve_thinking or \
           gen_params.get("max_tokens") != args.max_tokens or \
           gen_params.get("repair_turn_max_tokens", gen_params.get("max_tokens")) != args.repair_turn_max_tokens or \
           gen_params.get("max_model_len") != args.max_model_len or \
           gen_params.get("model") != expected_model or \
           gen_params.get("max_agent_turns") != args.max_agent_turns or \
           gen_params.get("tool_repair_attempts", 5) != args.tool_repair_attempts or \
           gen_params.get("final_answer_repair_attempts") != args.final_answer_repair_attempts or \
           gen_params.get("cutoff_repair_attempts") != args.cutoff_repair_attempts or \
           gen_params.get("max_turns_without_new_answer") != args.max_turns_without_new_answer or \
           gen_params.get("hf_overrides") != hf_overrides:
            print(f"Error: Generation parameters do not match the continue file.")
            print(f"  File params: {gen_params}")
            print(f"  CLI params: temp={args.temperature}, top_p={args.top_p}, "
                  f"top_k={args.top_k}, min_p={args.min_p}, "
                  f"presence_penalty={args.presence_penalty}, repetition_penalty={args.repetition_penalty}, "
                  f"reasoning_effort={args.reasoning_effort}, preserve_thinking={args.preserve_thinking}, "
                  f"max_tokens={args.max_tokens}, repair_turn_max_tokens={args.repair_turn_max_tokens}, "
                  f"max_model_len={args.max_model_len}, "
                  f"model={expected_model}, max_agent_turns={args.max_agent_turns}, "
                  f"tool_repair_attempts={args.tool_repair_attempts}, "
                  f"final_answer_repair_attempts={args.final_answer_repair_attempts}, "
                  f"cutoff_repair_attempts={args.cutoff_repair_attempts}, "
                  f"max_turns_without_new_answer={args.max_turns_without_new_answer}, "
                  f"hf_overrides={hf_overrides}")
            sys.exit(1)

        json_path = args.output if args.output else args.input
        gen_params["repair_turn_max_tokens"] = args.repair_turn_max_tokens
        gen_params["language_model_only"] = args.language_model_only
        gen_params["max_cudagraph_capture_size"] = args.max_cudagraph_capture_size
        gen_params["hf_overrides"] = hf_overrides
        gen_params["disable_custom_all_reduce"] = args.disable_custom_all_reduce
        data["summary"]["start_times"].append(datetime.datetime.now().isoformat())
    else:
        # Create a new results structure from a dataset
        output_dir = args.output_dir if args.output_dir else REPO_ROOT
        json_path = args.output if args.output else timestamped_output_path(
            output_dir,
            args.input,
            expected_model,
            "agent",
        )

        summary = data["summary"].copy()
        summary.update({
            "start_times": [datetime.datetime.now().isoformat()],
            "generation_params": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "min_p": args.min_p,
                "presence_penalty": args.presence_penalty,
                "repetition_penalty": args.repetition_penalty,
                "reasoning_effort": args.reasoning_effort,
                "preserve_thinking": args.preserve_thinking,
                "max_tokens": args.max_tokens,
                "repair_turn_max_tokens": args.repair_turn_max_tokens,
                "max_model_len": args.max_model_len,
                "model": expected_model,
                "max_agent_turns": args.max_agent_turns,
                "tool_repair_attempts": args.tool_repair_attempts,
                "final_answer_repair_attempts": args.final_answer_repair_attempts,
                "cutoff_repair_attempts": args.cutoff_repair_attempts,
                "max_turns_without_new_answer": args.max_turns_without_new_answer,
                "language_model_only": args.language_model_only,
                "max_cudagraph_capture_size": args.max_cudagraph_capture_size,
                "hf_overrides": hf_overrides,
                "disable_custom_all_reduce": args.disable_custom_all_reduce
            },
            "system_prompt": system_prompt,
            "mode": "agent",
            "average_token_length": None,
            "average_tool_calls": None,
            "average_turns": None,
            "tool_use_rate": None,
            "agent_status_counts": None
        })
        new_data = {
            "summary": summary,
            "samples": []
        }
        for s in data["samples"]:
            sample = s.copy()
            if "token_length" not in sample:
                sample["token_length"] = None
            if "agent_turns" not in sample:
                sample["agent_turns"] = None
            if "agent_tool_calls" not in sample:
                sample["agent_tool_calls"] = None
            if "turn_outputs" not in sample:
                sample["turn_outputs"] = None
            if "task_answers" not in sample:
                sample["task_answers"] = None
            if "agent_status" not in sample:
                sample["agent_status"] = None
            if "agent_resume_state" not in sample:
                sample["agent_resume_state"] = None
            new_data["samples"].append(sample)
        data = new_data

    checkpoint_dir = sample_checkpoint_dir(json_path)
    restored = load_sample_states(checkpoint_dir, data["samples"])
    if restored:
        print(f"Restored {restored} samples from {checkpoint_dir}", flush=True)

    save_json(data, json_path)


    # Filter pending samples
    pending_indices = [
        i
        for i, sample in enumerate(data["samples"])
        if sample.get("agent_status") == "in_progress"
        or last_turn_output(sample.get("turn_outputs")) is None
    ]
    if pending_indices:
        print(f"Found {len(pending_indices)} pending samples.", flush=True)
    else:
        # Still fall through so the summary is (re)computed and the sidecar
        # checkpoints are folded into the results file.
        print("All samples already evaluated; refreshing summary.", flush=True)

    async def process_all_samples():
        from contextlib import asynccontextmanager
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm import SamplingParams
        from transformers import AutoTokenizer

        @asynccontextmanager
        async def mcp_client_session(server_params):
            async with stdio_client(server_params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    yield session

        print("Initializing Async vLLM Engine...", flush=True)
        engine_kwargs = {
            "model": args.model,
            "tensor_parallel_size": args.num_gpus,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "skip_mm_profiling": True,
            "gdn_prefill_backend": "triton",
            "trust_remote_code": True,
            "language_model_only": args.language_model_only,
            "disable_custom_all_reduce": args.disable_custom_all_reduce,
        }
        if hf_overrides is not None:
            engine_kwargs["hf_overrides"] = hf_overrides
        if args.max_cudagraph_capture_size is not None:
            engine_kwargs["max_cudagraph_capture_size"] = args.max_cudagraph_capture_size

        engine_args = AsyncEngineArgs(**engine_kwargs)
        print(
            "Creating Async vLLM Engine "
            f"(language_model_only={args.language_model_only}, "
            f"hf_overrides={'set' if hf_overrides is not None else 'unset'}, "
            f"max_cudagraph_capture_size={args.max_cudagraph_capture_size}, "
            f"disable_custom_all_reduce={args.disable_custom_all_reduce})...",
            flush=True
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        print("Async vLLM Engine created.", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        print("Tokenizer loaded.", flush=True)

        print("Running sequential Triton JIT compilation/warmup request...", flush=True)
        warmup_params = SamplingParams(temperature=0.0, max_tokens=1)
        async for _ in engine.generate("Warmup query.", warmup_params, "warmup_req"):
            pass
        print("Triton JIT compilation complete.", flush=True)

        common_sampling = dict(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            presence_penalty=args.presence_penalty,
            repetition_penalty=args.repetition_penalty,
        )
        sampling_params = SamplingParams(max_tokens=args.max_tokens, **common_sampling)
        repair_sampling_params = SamplingParams(max_tokens=args.repair_turn_max_tokens, **common_sampling)
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-B", os.path.join(REPO_ROOT, "calculator_mcp_server.py")],
            env=dict(os.environ),
        )
        async with mcp_client_session(server_params) as mcp_client:
            list_tools_result = await mcp_client.list_tools()
            tool_schemas = []
            for tool in list_tools_result.tools:
                tool_schemas.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema,
                    }
                })
            if not tool_schemas:
                raise ValueError("MCP server did not return any tools.")

            if args.example:
                sample_path = write_first_sample_example(tokenizer, tool_schemas, system_prompt, data, chat_template_kwargs)
                print(f"Sample prompt written to: {sample_path}", flush=True)
                if not confirm_run():
                    print("Evaluation aborted.", flush=True)
                    sys.exit(0)
            sample_semaphore = (
                asyncio.Semaphore(args.max_concurrent_samples)
                if args.max_concurrent_samples > 0
                else None
            )

            async def save_sample_checkpoint(index, result):
                sample = data["samples"][index]
                sample["agent_status"] = "in_progress"
                sample["agent_turns"] = result["turns_taken"]
                sample["agent_tool_calls"] = result["total_tool_calls"]
                sample["turn_outputs"] = result.get("turn_outputs", {})
                sample["task_answers"] = result.get("task_answers", {})
                sample["agent_resume_state"] = result.get("resume_state")
                save_sample_state(checkpoint_dir, index, sample)

            async def process_sample(index):
                if sample_semaphore is None:
                    return await run_sample(index)
                async with sample_semaphore:
                    return await run_sample(index)

            async def run_sample(index):
                sample = data["samples"][index]
                resume_state = None
                if sample.get("agent_status") == "in_progress":
                    resume_state = dict(sample.get("agent_resume_state") or {})
                    resume_state["turn_outputs"] = sample.get("turn_outputs") or {}
                    resume_state["task_answers"] = sample.get("task_answers") or {}
                expected_outputs = [
                    f"{record['node_id']}_out"
                    for record in sample["node_outputs"]
                ]
                result = await run_agent_loop(
                    engine=engine,
                    tokenizer=tokenizer,
                    sampling_params=sampling_params,
                    tools=tool_schemas,
                    mcp_client=mcp_client,
                    system_prompt=system_prompt,
                    task_prompt=sample["prompt"],
                    max_agent_turns=args.max_agent_turns,
                    index=index,
                    repair_sampling_params=repair_sampling_params,
                    max_model_len=args.max_model_len,
                    tool_repair_attempts=args.tool_repair_attempts,
                    final_answer_repair_attempts=args.final_answer_repair_attempts,
                    cutoff_repair_attempts=args.cutoff_repair_attempts,
                    max_turns_without_new_answer=args.max_turns_without_new_answer,
                    chat_template_kwargs=chat_template_kwargs,
                    expected_outputs=expected_outputs,
                    resume_state=resume_state,
                    checkpoint_callback=lambda partial_result: save_sample_checkpoint(index, partial_result),
                )

                token_count = 0
                for text in result.get("turn_outputs", {}).values():
                    if text:
                        token_count += len(tokenizer.encode(text))
                sample["token_length"] = token_count
                sample["agent_turns"] = result["turns_taken"]
                sample["agent_tool_calls"] = result["total_tool_calls"]
                sample["turn_outputs"] = result.get("turn_outputs", {})
                sample["task_answers"] = result.get("task_answers", {})
                sample["agent_status"] = result["status"]
                sample["agent_resume_state"] = None
                save_sample_state(checkpoint_dir, index, sample)
                return index

            tasks = [asyncio.create_task(process_sample(idx)) for idx in pending_indices]
            for future in async_tqdm.as_completed(tasks, total=len(tasks), desc="Agent evaluation"):
                await future

    if pending_indices:
        asyncio.run(process_all_samples())

    # ---------------------------------------------------------------------------
    # Update summary statistics
    # ---------------------------------------------------------------------------

    evaluated = [s for s in data["samples"] if s["token_length"] is not None]
    total_tokens = sum(s["token_length"] for s in evaluated)
    total_tool_calls = sum(s["agent_tool_calls"] for s in evaluated if s["agent_tool_calls"] is not None)
    total_turns = sum(s["agent_turns"] for s in evaluated if s["agent_turns"] is not None)
    status_counts = Counter(s.get("agent_status") for s in evaluated)
    agents_using_tools = sum(1 for s in evaluated if s.get("agent_tool_calls") and s["agent_tool_calls"] > 0)
    eval_count = len(evaluated)

    data["summary"]["average_token_length"] = total_tokens / eval_count if eval_count > 0 else 0
    data["summary"]["average_tool_calls"] = total_tool_calls / eval_count if eval_count > 0 else 0
    data["summary"]["average_turns"] = total_turns / eval_count if eval_count > 0 else 0
    data["summary"]["tool_use_rate"] = agents_using_tools / eval_count if eval_count > 0 else 0
    data["summary"]["agent_status_counts"] = {
        str(status): count
        for status, count in sorted(status_counts.items(), key=lambda kv: (-kv[1], str(kv[0])))
    }

    save_json(data, json_path)

    unfinished = sum(1 for s in data["samples"] if s.get("agent_status") == "in_progress")
    if unfinished:
        print(f"\n{unfinished} samples are still in progress; keeping {checkpoint_dir} for resume.")
    else:
        clear_sample_states(checkpoint_dir)

    print(f"\nFinished! Results saved to: {json_path}")
    print(f"Average Token Length: {data['summary']['average_token_length']:.1f}")
    print(f"Average Tool Calls: {data['summary']['average_tool_calls']:.1f}")
    print(f"Average Agent Turns: {data['summary']['average_turns']:.1f}")
    print(f"Tool Use Rate: {data['summary']['tool_use_rate']:.2%}")
    print("Agent status:")
    for status, count in data["summary"]["agent_status_counts"].items():
        share = count / eval_count if eval_count > 0 else 0
        print(f"  {status:34s} {count:6d}/{eval_count} ({share:.2%})")


if __name__ == "__main__":
    main()
