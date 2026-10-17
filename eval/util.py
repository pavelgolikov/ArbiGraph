"""Shared evaluation prompts, agent loop, checkpoints, and runner helpers."""

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

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
from tqdm.asyncio import tqdm as async_tqdm

from parse_results import (
    GEMMA_OPEN_THOUGHT,
    assistant_message_from_response,
    looks_like_tool_call,
    missing_task_outputs,
    normalize_tool_arguments,
    parse_tool_calls,
    update_task_answers,
)

sys.path.insert(0, os.path.join(os.path.dirname(EVAL_DIR), "source"))
from tasks.arbigraph_tools import GET_INPUT_TOOL, SUBMIT_ANSWERS_TOOL, InputTool


def save_json(data, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    json_str = json.dumps(data, indent=2)
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
    "submitted_answers",
    "reported_correct",
    "agent_status",
    "agent_resume_state",
)

SAMPLE_CHECKPOINT_RE = re.compile(r"sample_(\d+)\.json")


def sample_checkpoint_dir(results_path: str) -> str:
    return os.path.splitext(os.path.abspath(results_path))[0] + "_partial"


def load_run_data(input_path: str, output_path: str):
    """Resume an existing output only when it belongs to the supplied dataset."""
    with open(input_path) as f:
        data = json.load(f)
    if not os.path.exists(output_path):
        if os.path.exists(sample_checkpoint_dir(output_path)):
            raise ValueError("Checkpoints exist without a results file; cannot verify the dataset or generation parameters")
        return data

    with open(output_path) as f:
        results = json.load(f)
    if "generation_params" not in results.get("summary", {}):
        raise ValueError(f"Output exists but is not an evaluation results file: {output_path}")

    # These fields are added or updated by evaluation and grading.
    summary_fields = {
        "start_times", "generation_params", "system_prompt", "mode", "grading",
        "average_token_length", "average_tool_calls", "average_turns",
        "tool_use_rate", "agent_status_counts",
    }
    if ({k: v for k, v in data["summary"].items() if k not in summary_fields}
            != {k: v for k, v in results["summary"].items() if k not in summary_fields}):
        raise ValueError(f"Input dataset summary does not match the existing output: {output_path}")
    if len(data["samples"]) != len(results["samples"]):
        raise ValueError(f"Input sample count does not match the existing output: {output_path}")
    sample_fields = {*SAMPLE_STATE_FIELDS, "grading"}
    for index, (original, saved) in enumerate(zip(data["samples"], results["samples"])):
        if ({k: v for k, v in original.items() if k not in sample_fields}
                != {k: v for k, v in saved.items() if k not in sample_fields}):
            raise ValueError(f"Input sample {index} does not match the existing output: {output_path}")
    return results


def save_sample_state(checkpoint_dir: str, index: int, sample: Dict[str, Any]) -> None:
    """Write one sample's progress, replacing the file atomically."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"sample_{index:06d}.json")
    tmp_path = f"{path}.tmp"
    payload = {field: sample.get(field) for field in SAMPLE_STATE_FIELDS}
    with open(tmp_path, "w") as f:
        json.dump(payload, f)
    os.replace(tmp_path, path)


def load_sample_states(checkpoint_dir: str, samples: List[Dict[str, Any]], *, strict=False) -> int:
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
            if strict:
                raise ValueError(f"Checkpoint {name} has no matching sample")
            print(f"Warning: checkpoint {name} has no matching sample; skipping.")
            continue
        try:
            with open(os.path.join(checkpoint_dir, name)) as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            if strict:
                raise
            print(f"Warning: could not read checkpoint {name} ({exc}); sample will be rerun.")
            continue
        if strict and (not isinstance(payload, dict) or set(payload) != set(SAMPLE_STATE_FIELDS)):
            raise ValueError(f"Invalid sample checkpoint: {name}")
        if not isinstance(payload, dict):
            continue
        samples[index].update(
            {field: value for field, value in payload.items() if field in SAMPLE_STATE_FIELDS}
        )
        restored += 1
    return restored


def clear_sample_states(checkpoint_dir: str, *, strict=False) -> None:
    """Drop the sidecar directory once its contents are folded into the results file."""
    if not os.path.isdir(checkpoint_dir):
        return
    for name in os.listdir(checkpoint_dir):
        if SAMPLE_CHECKPOINT_RE.fullmatch(name) or name.endswith(".json.tmp"):
            os.remove(os.path.join(checkpoint_dir, name))
    try:
        os.rmdir(checkpoint_dir)
    except OSError:
        if strict:
            raise


# ---------------------------------------------------------------------------
# System prompt for the calculator-agent loop.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a helpful reasoning assistant with access to tools. Think step by step. "
    "You MUST use the calculator tool EVERY TIME you need to perform arithmetic operations. "
    "When several arithmetic expressions are useful, submit them together in one calculator call. "
)

TOOL_CALL_MALFORMED_REPAIR_PROMPT = (
    "Your previous tool call was incomplete or malformed. "
    "Retry now with exactly one short complete call to an available tool and nothing else. "
    "Use the tool's declared name and parameters. "
    "Do not include prose, reasoning, Markdown, or the final answer. "
    "Close every tag you open."
)

# ---------------------------------------------------------------------------
# Repair prompts and loop-detection thresholds
# ---------------------------------------------------------------------------
FINAL_ANSWER_REPAIR_PROMPT = (
    "You have not yet submitted your answer. "
    "Finish any remaining tasks, then call the submit_answers tool once with the result of the graded task. "
    "Use more tool calls if needed. Do not restart the solution."
)

CUTOFF_REPAIR_PROMPT = (
    "Your previous response was cut off by the token limit. "
    "Continue from exactly where you stopped. Do not restart or repeat earlier work. "
    "Be concise and move toward the requested task output assignments. "
    "Use tool calls if needed."
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
def mcp_server_args(samples: List[Dict[str, Any]]) -> List[str]:
    # All task inputs, including HotpotQA contexts, are materialized in the dataset.
    return ["-B", os.path.join(EVAL_DIR, "mcp_server.py")]


async def call_mcp_tool(client: Any, name: str, arguments: Any) -> str:
    """Call an MCP tool. Transport failures raise; tool failures come back as a message."""
    try:
        arguments = normalize_tool_arguments(arguments)
    except TypeError as exc:
        return f"Error: {exc}"
    result = await client.call_tool(name, arguments)
    texts = [
        str(content.text)
        for content in result.content
        if getattr(content, "text", None) is not None
    ]
    joined = "\n".join(texts)
    # isError marks a tool that ran and failed, as opposed to one that answered.
    return f"Error: {joined}" if result.isError else joined


def call_get_input_tool(input_tool, arguments):
    if not isinstance(arguments, dict):
        return "Error: arguments must be a JSON object."
    # Anything beyond the two documented keys is not the documented call, so it
    # is passed on as a request that states no dependencies, and is wrong.
    dependencies = arguments.get("input_dependencies")
    if set(arguments) - {"task_index", "input_dependencies"}:
        dependencies = None
    result = input_tool.get_input(arguments.get("task_index"), dependencies)
    return result if isinstance(result, str) else json.dumps(result)


def graded_output_name(sample: Dict[str, Any]) -> str:
    """Return the output name of the one task a sample is scored on.

    node_outputs is in visible task-number order, so the name follows the
    target's serialized position, not its node id.
    """
    for position, record in enumerate(sample["node_outputs"], start=1):
        if record["node_id"] == sample["target_node_id"]:
            return f"task_{position}_out"
    raise ValueError(f"Target node {sample['target_node_id']!r} is missing from node_outputs.")


def output_task_number(name: str) -> int:
    """The visible task number in an output name such as task_18_out."""
    match = re.fullmatch(r"task_(\d+)_out", name)
    if match is None:
        raise ValueError(f"Not a task output name: {name!r}")
    return int(match.group(1))


# A graded task cannot have been solved before its input was requested, so a
# submission made earlier holds another task's answer, or a guess. It is rejected
# like a malformed one and the episode goes on. Results record the rule, and a run
# never resumes under a different one.
SUBMISSION_RULE = "after_target_input"


def validate_submission(arguments, expected_outputs, input_tool=None):
    """Return a submit_answers call's answers by output name, or a rejection message.

    Position is the only link between an answer and its task, so a list of the
    wrong length cannot be matched to tasks and the whole submission is rejected.
    With the input tool, a submission is also rejected until every graded task's
    input has been requested.
    """
    rejected = "Error: submission rejected, no answers were recorded."
    answers = arguments.get("answers") if isinstance(arguments, dict) else None
    if not isinstance(answers, list):
        return f"{rejected} The answers argument must be a list holding one object per graded task."
    if len(answers) != len(expected_outputs):
        noun = "object" if len(expected_outputs) == 1 else "objects"
        return (f"{rejected} The answers list must contain {len(expected_outputs)} {noun}, "
                f"for {', '.join(expected_outputs)}, but it contained {len(answers)}.")
    malformed = [
        name for name, value in zip(expected_outputs, answers)
        if not isinstance(value, dict) or set(value) != {"result"}
    ]
    if malformed:
        objects = "object" if len(malformed) == 1 else "objects"
        return f"{rejected} The {objects} for {', '.join(malformed)} must have exactly one key, result."
    if input_tool is not None:
        unrequested = [output_task_number(name) for name in expected_outputs
                       if not input_tool.counts.get(output_task_number(name))]
        if unrequested and len(expected_outputs) == 1:
            task = unrequested[0]
            return (f"{rejected} Task {task} is the graded task, and its input has not been requested yet. "
                    f"Solve task {task}, then submit its result.")
        if unrequested:
            tasks = ", ".join(str(task) for task in unrequested)
            return (f"{rejected} These graded tasks have not had their inputs requested yet: {tasks}. "
                    f"Solve every graded task, then submit their results.")
    return {name: value["result"] for name, value in zip(expected_outputs, answers)}


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------
# vLLM's chat endpoint renders GPT-OSS prompts with the openai_harmony library
# and ignores the model's chat template, so an API run and a local run agree
# only if the harness renders GPT-OSS the same way. It does, with vLLM's own
# functions, on both stacks. Every other model uses its chat template.
HARMONY_DATE_RE = re.compile(r"Current date: (\d{4}-\d{2}-\d{2})")


def is_harmony_model(model: str) -> bool:
    return "gpt-oss" in model.lower()


def is_gemma4_model(model: str) -> bool:
    return re.search(r"gemma-?4", model.lower()) is not None


def keeps_special_tokens(model: str) -> bool:
    """Whether a model marks its tool calls with special tokens, which decoding must keep."""
    return is_harmony_model(model) or is_gemma4_model(model)


def prompt_renderer(model: str) -> str:
    """Recorded in generation_params so a run never resumes under the other renderer."""
    return "harmony" if is_harmony_model(model) else "chat_template"


def render_harmony_prompt(messages, tools, reasoning_effort, start_date=None) -> List[int]:
    """Token IDs of a GPT-OSS prompt, built exactly as vLLM's chat endpoint builds it.

    The steps are those of vLLM's _make_request_with_harmony: a system message
    holding the reasoning effort and the date, a developer message listing the
    tools, then the conversation. The system prompt becomes a second system
    message after the tools, earlier reasoning is read from each assistant
    message's `reasoning` field, and a tool result is named through its
    `tool_call_id`. vLLM dates the system message with the server's clock, so a
    prompt checked against a server's is rendered on the server's date.
    """
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.parser.harmony_utils import (
        get_developer_message,
        get_system_message,
        parse_chat_inputs_to_harmony_messages,
        render_for_completion,
    )
    from vllm.tokenizers.mistral import maybe_serialize_tool_calls

    request = ChatCompletionRequest(
        model="gpt-oss", messages=messages, tools=tools, tool_choice="none",
        reasoning_effort=reasoning_effort,
    )
    maybe_serialize_tool_calls(request)
    conversation = [get_system_message(
        reasoning_effort=request.reasoning_effort, start_date=start_date,
        with_custom_tools=bool(request.tools),
    )]
    if request.tools:
        conversation.append(get_developer_message(tools=request.tools))
    conversation.extend(parse_chat_inputs_to_harmony_messages(request.messages))
    return render_for_completion(conversation)


def harmony_start_date(tokenizer, token_ids: List[int]) -> Optional[str]:
    """The date a rendered Harmony prompt's system message carries."""
    match = HARMONY_DATE_RE.search(tokenizer.decode(token_ids, skip_special_tokens=False))
    return match.group(1) if match else None


def _template_messages(messages):
    """Messages as vLLM's chat endpoint hands them to a chat template.

    A chat-format tool call carries its arguments as a JSON string. vLLM parses it
    into an object first, and a template such as Gemma 4's accepts only the object.
    """
    if not any(message.get("tool_calls") for message in messages):
        return messages
    converted = copy.deepcopy(messages)
    for message in converted:
        for call in message.get("tool_calls") or ():
            arguments = call["function"].get("arguments")
            if not arguments:
                call["function"]["arguments"] = {}
            elif isinstance(arguments, str):
                call["function"]["arguments"] = json.loads(arguments)
    return converted


def render_prompt(tokenizer, messages, tools, chat_template_kwargs=None, *, harmony=False):
    """The prompt text, and for Harmony the exact token IDs the model is given."""
    if harmony:
        token_ids = render_harmony_prompt(messages, tools, (chat_template_kwargs or {}).get("reasoning_effort"))
        return tokenizer.decode(token_ids, skip_special_tokens=False), token_ids
    text = tokenizer.apply_chat_template(_template_messages(messages), tools=tools, tokenize=False,
                                         add_generation_prompt=True, **(chat_template_kwargs or {}))
    return text, None


def render_initial_model_prompt(tokenizer, tools, system_prompt: str, task_prompt: str,
                                chat_template_kwargs: Optional[Dict[str, Any]] = None,
                                *, harmony=False) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_prompt}
    ]
    return render_prompt(tokenizer, messages, tools, chat_template_kwargs, harmony=harmony)[0]


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
                              chat_template_kwargs: Optional[Dict[str, Any]] = None,
                              *, harmony=False) -> str:
    first_sample = data["samples"][0]
    prompt = render_initial_model_prompt(tokenizer, tools, system_prompt, first_sample["prompt"],
                                         chat_template_kwargs, harmony=harmony)
    path = os.path.join(EVAL_DIR, "sample_to_view.txt")
    with open(path, "w") as f:
        f.write(prompt)
        if not prompt.endswith("\n"):
            f.write("\n")
    return path


async def run_agent_loop(
    generate_text,
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
    graded_outputs: List[str] | None = None,
    resume_state: Optional[Dict[str, Any]] = None,
    input_tool: InputTool | None = None,
    checkpoint_callback=None,
    fail_fast=False,
    harmony=False,
) -> Dict[str, Any]:
    """
    Run the agentic tool-calling loop for a single task using the supplied generation function.
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
    # Every task's output is still written out as text, since a child cannot ask for its
    # input without stating its parents' values. Only the graded task's result is
    # submitted, and only that submission is scored.
    submit_outputs = resume_state.get("graded_task_outputs") or graded_outputs or requested_outputs

    final_answer_repair_budget = final_answer_repair_attempts
    cutoff_repair_budget = cutoff_repair_attempts

    messages = resume_state.get("messages") or [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_prompt}
    ]

    total_tool_calls = resume_state.get("total_tool_calls", 0)
    turn_outputs = dict(resume_state.get("turn_outputs") or {})
    task_answers = dict(resume_state.get("task_answers") or {})
    submitted_answers = None
    no_progress_cutoffs = resume_state.get("no_progress_cutoffs", 0)
    tool_repair_attempts_used = resume_state.get("tool_repair_attempts_used", 0)
    final_answer_repair_attempts_used = resume_state.get("final_answer_repair_attempts_used", 0)
    cutoff_repair_attempts_used = resume_state.get("cutoff_repair_attempts_used", 0)
    next_sampling_mode = resume_state.get("next_sampling_mode", "normal")
    start_turn = resume_state.get("next_turn", len(turn_outputs))
    if input_tool is not None and "get_input_counts" in resume_state:
        saved_counts = resume_state["get_input_counts"]
        for task_index in input_tool.counts:
            input_tool.counts[task_index] = saved_counts["attempts"][str(task_index)]
        for task_index, reported in (saved_counts.get("reported") or {}).items():
            input_tool.reported[int(task_index)] = reported

    def final_status(missing_requested_answers=False):
        if missing_requested_answers:
            if not missing_task_outputs(requested_outputs, task_answers):
                return "answers_not_submitted"
            return "missing_requested_task_answers"
        return "success"

    def build_resume_state(next_turn):
        state = {
            "messages": messages,
            "next_turn": next_turn,
            "total_tool_calls": total_tool_calls,
            "tool_repair_attempts_used": tool_repair_attempts_used,
            "final_answer_repair_attempts_used": final_answer_repair_attempts_used,
            "cutoff_repair_attempts_used": cutoff_repair_attempts_used,
            "no_progress_cutoffs": no_progress_cutoffs,
            "next_sampling_mode": next_sampling_mode,
            "expected_task_outputs": requested_outputs,
            "graded_task_outputs": submit_outputs,
        }
        if input_tool is not None:
            state["get_input_counts"] = {
                "attempts": {str(index): count for index, count in input_tool.counts.items()},
                "reported": input_tool.reported_correct(),
            }
        return state

    def agent_result(status, turns_taken, current_resume_state=None):
        return {
            "status": status,
            "turns_taken": turns_taken,
            "total_tool_calls": total_tool_calls,
            "turn_outputs": turn_outputs,
            "expected_task_outputs": requested_outputs,
            "graded_task_outputs": submit_outputs,
            "task_answers": task_answers,
            "submitted_answers": submitted_answers,
            "reported_correct": {} if input_tool is None else input_tool.reported_correct(),
            "resume_state": current_resume_state,
        }

    async def checkpoint(turns_taken):
        if checkpoint_callback is None:
            return
        await checkpoint_callback(
            agent_result("in_progress", turns_taken, build_resume_state(turns_taken))
        )

    for turn in range(start_turn, max_agent_turns):
        full_prompt, prompt_token_ids = render_prompt(
            tokenizer, messages, tools, chat_template_kwargs, harmony=harmony,
        )
        # A Harmony prompt goes to the model as the exact tokens rendered.
        prompt_kwargs = {"prompt_token_ids": prompt_token_ids} if harmony else {}

        request_id = f"req_{index}_turn_{turn}"
        current_sampling_params = (
            repair_sampling_params
            if next_sampling_mode == "repair" and repair_sampling_params is not None
            else sampling_params
        )
        next_sampling_mode = "normal"
        if max_model_len is not None:
            input_tokens = (len(prompt_token_ids) if harmony
                            else len(tokenizer.encode(full_prompt, add_special_tokens=False)))
            remaining_tokens = max_model_len - input_tokens
            if remaining_tokens < MIN_USABLE_OUTPUT_TOKENS:
                turn_outputs[str(turn)] = ""
                return agent_result("context_length_exceeded", turn)
            if remaining_tokens < current_sampling_params.max_tokens:
                current_sampling_params = copy.copy(current_sampling_params)
                current_sampling_params.max_tokens = remaining_tokens
        try:
            out_text, finish_reason = await generate_text(
                full_prompt, current_sampling_params, request_id=request_id,
                messages=messages, tools=tools, chat_template_kwargs=chat_template_kwargs,
                **prompt_kwargs,
            )
        except Exception as exc:
            if fail_fast:
                raise
            # One sample must not take the run down with it.
            print(f"[sample {index}] generation failed on turn {turn}: {exc}", flush=True)
            turn_outputs[str(turn)] = ""
            return agent_result("generation_error", turn)
        turn_outputs[str(turn)] = out_text
        # A Gemma 4 prompt can end by opening the thought channel, and the model's text
        # continues it; the turn is carried forward with that opening restored.
        opened_thought = not harmony and full_prompt.endswith(GEMMA_OPEN_THOUGHT)
        turn_text = GEMMA_OPEN_THOUGHT + out_text if opened_thought else out_text
        previous_answer_names = set(task_answers)
        update_task_answers(task_answers, out_text, requested_outputs, turn)
        new_answer_added = bool(set(task_answers) - previous_answer_names)

        tool_calls = parse_tool_calls(out_text)

        cutoff_detected = finish_reason == "length"

        # Only a successful submit_answers call ends the episode; text answers are a record.
        # Each submission is judged once, on the state before this turn's calls run, so a
        # get_input made in the same turn cannot vouch for it: the model had not seen
        # that input when it answered. Its reply below is this same verdict.
        verdicts = {
            position: validate_submission(tc.get("arguments"), submit_outputs, input_tool)
            for position, tc in enumerate(tool_calls)
            if tc.get("name") == "submit_answers"
        }
        accepted = [answers for answers in verdicts.values() if isinstance(answers, dict)]
        if accepted:
            submitted_answers = {
                name: {"answer": answer, "turn": turn} for name, answer in accepted[-1].items()
            }
            return agent_result(final_status(), turn + 1)

        last_progress_turn = max(
            (entry["turn"] for entry in task_answers.values()),
            default=-1,
        )
        if turn - last_progress_turn >= max_turns_without_new_answer:
            return agent_result("tool_loop_no_progress", turn + 1)

        if cutoff_detected and not new_answer_added:
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
            messages.append(assistant_message_from_response(turn_text, tool_calls))
            messages.append({"role": "user", "content": CUTOFF_REPAIR_PROMPT})
            cutoff_repair_attempts_used += 1
            next_sampling_mode = "repair"
            await checkpoint(turn + 1)
            continue

        if not tool_calls:
            if looks_like_tool_call(out_text):
                if tool_repair_attempts_used < tool_repair_attempts:
                    messages.append(assistant_message_from_response(turn_text, tool_calls))
                    messages.append({"role": "user", "content": TOOL_CALL_MALFORMED_REPAIR_PROMPT})
                    tool_repair_attempts_used += 1
                    next_sampling_mode = "repair"
                    await checkpoint(turn + 1)
                    continue

                return agent_result("malformed_tool_calls", turn + 1)

            # No submission has succeeded here: a successful submit_answers
            # call returns above before reaching this point.
            if final_answer_repair_attempts_used < final_answer_repair_budget:
                messages.append(assistant_message_from_response(turn_text, tool_calls))
                messages.append({"role": "user", "content": FINAL_ANSWER_REPAIR_PROMPT})
                final_answer_repair_attempts_used += 1
                next_sampling_mode = "repair"
                await checkpoint(turn + 1)
                continue

            return agent_result(final_status(missing_requested_answers=True), turn + 1)

        # Store the assistant response before tool results. Harmony names each
        # tool result by the id of the call it answers.
        call_ids = [f"call_{turn}_{position}" for position in range(len(tool_calls))]
        assistant_msg = assistant_message_from_response(turn_text, tool_calls, call_ids)
        messages.append(assistant_msg)

        # Execute each parsed tool call and append the result.
        for position, (call_id, tc) in enumerate(zip(call_ids, tool_calls)):
            total_tool_calls += 1
            name = tc.get("name") or "unknown"
            if name not in available_tool_names:
                result = f"Error: unknown tool {name!r}."
            else:
                try:
                    if name == "get_input":
                        result = call_get_input_tool(input_tool, tc.get("arguments", {}))
                    elif name == "submit_answers":
                        # A successful submission returns above, so this one was rejected.
                        result = verdicts[position]
                    else:
                        result = await call_mcp_tool(mcp_client, name, tc.get("arguments", {}))
                except Exception as e:
                    result = f"Error: {e}"

            tool_message = {
                "role": "tool",
                "name": name,
                "content": str(result)
            }
            if harmony or "tool_calls" in assistant_msg:
                tool_message["tool_call_id"] = call_id
            messages.append(tool_message)

        await checkpoint(turn + 1)

    return agent_result("timeout", max_agent_turns)


SAMPLING_FIELDS = (
    "temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty",
)


def generation_params(args):
    fields = SAMPLING_FIELDS + (
        "reasoning_effort", "preserve_thinking", "max_tokens", "repair_turn_max_tokens",
        "max_model_len", "model", "max_agent_turns", "tool_repair_attempts",
        "get_input_attempts_per_task",
        "final_answer_repair_attempts", "cutoff_repair_attempts", "max_turns_without_new_answer",
    )
    params = {name: getattr(args, name) for name in fields}
    params["prompt_renderer"] = prompt_renderer(args.model)
    params["submission_rule"] = SUBMISSION_RULE
    params["enable_thinking"] = getattr(args, "enable_thinking", None)
    return params


def make_sampling_params(args, sampling_class):
    common = {name: getattr(args, name) for name in SAMPLING_FIELDS}
    return (
        sampling_class(max_tokens=args.max_tokens, **common),
        sampling_class(max_tokens=args.repair_turn_max_tokens, **common),
    )


def new_results(data, params, system_prompt, *, preserve_sample_state=False):
    summary = data["summary"].copy()
    summary.update({
        "start_times": [datetime.datetime.now().isoformat()],
        "generation_params": params,
        "system_prompt": system_prompt,
        "mode": "agent",
        "average_token_length": None,
        "average_tool_calls": None,
        "average_turns": None,
        "tool_use_rate": None,
        "agent_status_counts": None,
    })
    samples = []
    for original in data["samples"]:
        sample = original.copy()
        sample.update({
            field: original.get(field) if preserve_sample_state else None
            for field in SAMPLE_STATE_FIELDS
        })
        samples.append(sample)
    return {"summary": summary, "samples": samples}


def prepare_samples(data, json_path, *, strict=False):
    checkpoint_dir = sample_checkpoint_dir(json_path)
    restored = load_sample_states(checkpoint_dir, data["samples"], strict=strict)
    if restored:
        print(f"Restored {restored} samples from {checkpoint_dir}", flush=True)
    save_json(data, json_path)
    pending_indices = [
        index for index, sample in enumerate(data["samples"])
        if sample.get("agent_status") == "in_progress"
        or last_turn_output(sample.get("turn_outputs")) is None
    ]
    if pending_indices:
        print(f"Found {len(pending_indices)} pending samples.", flush=True)
    else:
        print("All samples already evaluated; refreshing summary.", flush=True)
    return checkpoint_dir, pending_indices


async def process_samples(args, data, pending_indices, checkpoint_dir, server_args, *,
                          tokenizer, generate_text, sampling_params, repair_sampling_params,
                          system_prompt, strict=False):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    chat_template_kwargs = {
        "reasoning_effort": args.reasoning_effort,
        "preserve_thinking": args.preserve_thinking,
    }
    # Only a template that reads it, such as Gemma 4's, sees a difference; unset
    # leaves each template's own default.
    if getattr(args, "enable_thinking", None) is not None:
        chat_template_kwargs["enable_thinking"] = args.enable_thinking
    harmony = is_harmony_model(args.model)
    server_params = StdioServerParameters(command=sys.executable, args=server_args, env=dict(os.environ))
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as mcp_client:
            await mcp_client.initialize()
            listed_tools = await mcp_client.list_tools()
            tool_schemas = [{
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                },
            } for tool in listed_tools.tools]
            if not tool_schemas:
                raise ValueError("MCP server did not return any tools.")
            for tool in (GET_INPUT_TOOL, SUBMIT_ANSWERS_TOOL):
                tool_schemas.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["inputSchema"],
                    },
                })

            if args.example:
                path = write_first_sample_example(tokenizer, tool_schemas, system_prompt, data,
                                                  chat_template_kwargs, harmony=harmony)
                print(f"Sample prompt written to: {path}", flush=True)
                if not confirm_run():
                    print("Evaluation aborted.", flush=True)
                    sys.exit(0)

            async def save_result(index, result):
                sample = data["samples"][index]
                sample["agent_status"] = result["status"]
                sample["agent_turns"] = result["turns_taken"]
                sample["agent_tool_calls"] = result["total_tool_calls"]
                sample["turn_outputs"] = result["turn_outputs"]
                sample["task_answers"] = result["task_answers"]
                sample["submitted_answers"] = result["submitted_answers"]
                sample["reported_correct"] = result["reported_correct"]
                sample["agent_resume_state"] = result["resume_state"]
                if result["status"] != "in_progress":
                    sample["token_length"] = sum(
                        len(tokenizer.encode(text)) for text in result["turn_outputs"].values() if text
                    )
                save_sample_state(checkpoint_dir, index, sample)

            async def run_sample(index):
                sample = data["samples"][index]
                resume_state = None
                if sample.get("agent_status") == "in_progress":
                    if strict:
                        resume_state = dict(sample["agent_resume_state"])
                        resume_state["turn_outputs"] = sample["turn_outputs"]
                        resume_state["task_answers"] = sample["task_answers"]
                    else:
                        resume_state = dict(sample.get("agent_resume_state") or {})
                        resume_state["turn_outputs"] = sample.get("turn_outputs") or {}
                        resume_state["task_answers"] = sample.get("task_answers") or {}
                result = await run_agent_loop(
                    generate_text=generate_text,
                    index=index,
                    fail_fast=strict,
                    tokenizer=tokenizer,
                    sampling_params=sampling_params,
                    tools=tool_schemas,
                    mcp_client=mcp_client,
                    system_prompt=system_prompt,
                    task_prompt=sample["prompt"],
                    max_agent_turns=args.max_agent_turns,
                    repair_sampling_params=repair_sampling_params,
                    max_model_len=args.max_model_len,
                    tool_repair_attempts=args.tool_repair_attempts,
                    final_answer_repair_attempts=args.final_answer_repair_attempts,
                    cutoff_repair_attempts=args.cutoff_repair_attempts,
                    max_turns_without_new_answer=args.max_turns_without_new_answer,
                    chat_template_kwargs=chat_template_kwargs,
                    expected_outputs=[f"task_{position}_out" for position in range(1, len(sample["node_outputs"]) + 1)],
                    graded_outputs=[graded_output_name(sample)],
                    resume_state=resume_state,
                    input_tool=InputTool.from_records(
                        sample["node_outputs"],
                        attempts_per_task=args.get_input_attempts_per_task,
                    ),
                    checkpoint_callback=lambda result: save_result(index, result),
                    harmony=harmony,
                )
                await save_result(index, result)

            semaphore = asyncio.Semaphore(args.max_concurrent_samples) if args.max_concurrent_samples > 0 else None

            async def process_sample(index):
                if semaphore is None:
                    return await run_sample(index)
                async with semaphore:
                    return await run_sample(index)

            tasks = [asyncio.create_task(process_sample(index)) for index in pending_indices]
            for future in async_tqdm.as_completed(tasks, total=len(tasks), desc="Agent evaluation"):
                await future


def finish_results(data, json_path, checkpoint_dir, *, strict=False):
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
        clear_sample_states(checkpoint_dir, strict=strict)

    print(f"\nFinished! Results saved to: {json_path}")
    print(f"Average Token Length: {data['summary']['average_token_length']:.1f}")
    print(f"Average Tool Calls: {data['summary']['average_tool_calls']:.1f}")
    print(f"Average Agent Turns: {data['summary']['average_turns']:.1f}")
    print(f"Tool Use Rate: {data['summary']['tool_use_rate']:.2%}")
    print("Agent status:")
    for status, count in data["summary"]["agent_status_counts"].items():
        share = count / eval_count if eval_count > 0 else 0
        print(f"  {status:34s} {count:6d}/{eval_count} ({share:.2%})")
