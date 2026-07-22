import argparse
import json
import os
import sys
import datetime
import re
import ast
import asyncio
from typing import Any, Dict, List, Optional, Tuple

import getpass
username = getpass.getuser()
os.environ['TRITON_CACHE_DIR'] = f'/tmp/triton_cache_{username}'
os.environ['TORCH_EXTENSIONS_DIR'] = f'/tmp/torch_extensions_{username}'
os.environ['VLLM_CACHE_ROOT'] = f'/tmp/vllm_cache_{username}'
os.environ['VLLM_CONFIG_ROOT'] = f'/tmp/vllm_config_{username}'

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from grader import grade
from output_paths import timestamped_output_path
from tqdm.asyncio import tqdm as async_tqdm


def save_json(data, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    json_str = json.dumps(data, indent=2)
    # Collapse arrays of simple types (like numbers/strings) horizontally
    json_str = re.sub(r'\[\s+([^\[\]\{\}]*?)\s+\]', lambda m: '[' + re.sub(r'\s+', ' ', m.group(1)) + ']', json_str)
    json_str = re.sub(r'\[\s+\]', '[]', json_str)
    with open(path, "w") as f:
        f.write(json_str)


# ---------------------------------------------------------------------------
# System prompt – instructs model to use the calculator tool and box answers
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a helpful reasoning assistant with access to a calculator tool. "
    "You MUST use the calculator tool EVERY TIME you need to perform arithmetic operations. "
    "When several arithmetic expressions are useful, submit them together in one calculator call. "
    "Do not truncate any lists unless otherwise instructed to. "
    "Think step by step. Put each requested answer in its own \\boxed{} block with "
    "the exact output name inside the box, for example \\boxed{task_1_out = ...}. "
)

INITIAL_CALCULATOR_NO_TOOL_REPAIR_PROMPT = (
    "Your previous response did not include a calculator tool call. "
    "Output exactly one complete calculator tool call now and nothing else. "
    "Use the calculator function with the expressions parameter. "
    "Include 1 to 3 short useful arithmetic expressions. "
    "Do not include prose, reasoning, Markdown, or the final answer. "
    "Close every tag you open."
)

INITIAL_CALCULATOR_MALFORMED_TOOL_REPAIR_PROMPT = (
    "Your previous calculator tool call was incomplete, malformed, or not a calculator call. "
    "Retry now with exactly one short complete calculator tool call and nothing else. "
    "Use the calculator function with the expressions parameter. "
    "Include 1 to 3 short useful arithmetic expressions. "
    "Do not include prose, reasoning, Markdown, or the final answer. "
    "Close every tag you open."
)

# ---------------------------------------------------------------------------
# Multi-task answer collection and final-answer repair
# ---------------------------------------------------------------------------
# This section is the complete trajectory-level answer policy. Every assistant
# response is inspected, later answers replace earlier versions of the same
# task output, and repair is requested only while expected outputs are absent.
TASK_OUTPUT_RE = re.compile(
    r"(?<![A-Za-z0-9_])task_([1-9]\d*)_out\b",
    re.IGNORECASE,
)
BOXED_TASK_ANSWER_RE = re.compile(
    r"^\s*(?:\\text\{\s*)?task_([1-9]\d*)_out(?:\s*\})?"
    r"\s*(?:=|:|\bis\b)\s*(.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)

FINAL_ANSWER_REPAIR_PROMPT = (
    "Your previous response did not include all of the requested answers in the required "
    "\\boxed{task_N_out = ...} format. "
)

CUTOFF_REPAIR_PROMPT = (
    "Your previous response was cut off by the token limit. "
    "Continue from exactly where you stopped. Do not restart or repeat earlier work. "
    "Be concise and move toward the requested \\boxed{task_N_out = ...} answers. "
    "Use calculator calls if needed."
)

def expected_task_outputs(task_prompt: str) -> List[str]:
    """Read the requested task output keys from a benchmark prompt.

    The prompt is the source of truth for how many task answers the agent is
    expected to provide. The returned names are deduplicated and kept in
    first-seen order; for multi-task prompts, the final name in this list is
    the benchmark target used for grading.
    """
    outputs = []
    seen = set()
    for match in TASK_OUTPUT_RE.finditer(task_prompt):
        output_name = f"task_{match.group(1)}_out"
        if output_name not in seen:
            seen.add(output_name)
            outputs.append(output_name)
    return outputs


def _boxed_blocks(text: str) -> List[Tuple[int, int, str]]:
    """Parse complete ``\\boxed{...}`` blocks from response text.

    A small brace counter is used instead of a flat regex so boxed answers with
    nested braces are not truncated. Malformed open boxes are ignored because
    there is no reliable end boundary. Each returned tuple is
    ``(start_offset, end_offset, stripped_content)`` in source order.
    """
    blocks = []
    for match in re.finditer(r"\\boxed\{", text):
        content_start = match.end()
        cursor = content_start
        brace_depth = 1
        while cursor < len(text) and brace_depth:
            if text[cursor] == "{":
                brace_depth += 1
            elif text[cursor] == "}":
                brace_depth -= 1
            cursor += 1
        if brace_depth == 0:
            blocks.append((match.start(), cursor, text[content_start:cursor - 1].strip()))
    return blocks


def _strip_answer_wrappers(answer: str) -> str:
    """Normalize answer text without changing its semantic value.

    The parser removes common prose prefixes, lightweight Markdown/code/math
    wrappers, and trailing sentence punctuation. It intentionally does not
    coerce types or rewrite list/scalar contents, because grading should still
    receive the model's textual answer.
    """
    answer = answer.strip()
    answer = re.sub(
        r"^\s*(?:the\s+)?(?:answer|result|output)\s*(?:=|:|is\b)\s*",
        "",
        answer,
        flags=re.IGNORECASE,
    )
    answer = answer.strip()

    if answer.startswith("`") and answer.endswith("`"):
        answer = answer.strip("`").strip()
    if answer.startswith("$") and answer.endswith("$") and len(answer) >= 2:
        answer = answer[1:-1].strip()
    if answer.startswith(r"\(") and answer.endswith(r"\)"):
        answer = answer[2:-2].strip()

    return answer.rstrip().rstrip(";,.")


def _mask_tool_call_payloads(text: str) -> str:
    """Hide closed tool-call payloads before answer extraction.

    Calculator calls can contain task labels, intermediate results, or scratch
    values that should not be counted as final answers. The payloads are
    blanked out while preserving string length and newlines, so boxed-block
    parsing sees the same surrounding response structure.
    """
    patterns = (
        r"<tool_call>.*?</tool_call>",
        r"<function=[^>]+>.*?</function>",
    )
    masked = text
    for pattern in patterns:
        masked = re.sub(
            pattern,
            lambda match: "".join("\n" if char == "\n" else " " for char in match.group(0)),
            masked,
            flags=re.DOTALL,
        )
    return masked


def _looks_like_concrete_answer(answer: str) -> bool:
    """Filter out placeholders and label echoes before storing an answer.

    The repair logic should not treat empty text, ``unknown``-style placeholders,
    or bare variable names as completed task outputs. Common scalar literals
    such as ``true``, ``false``, and ``none`` are allowed.
    """
    normalized = answer.strip().lower()
    if not normalized:
        return False
    if normalized in {"...", "unknown", "pending", "not calculated", "not yet"}:
        return False
    if re.fullmatch(r"[a-z_][a-z0-9_]*", normalized) and normalized not in {"true", "false", "none"}:
        return False
    return True


def _task_answer_from_box(
    content: str,
    expected_outputs: List[str],
) -> Optional[Tuple[str, str]]:
    """Parse one explicitly labeled task answer from boxed content.

    The task label must be inside the box and must name one of the outputs
    requested by the prompt. An optional LaTeX ``\\text{}`` wrapper around the
    label is accepted, but surrounding prose and labels outside the box are
    deliberately ignored.
    """
    match = BOXED_TASK_ANSWER_RE.fullmatch(content)
    if match is None:
        return None

    output_name = f"task_{match.group(1)}_out"
    if output_name not in set(expected_outputs):
        return None

    answer = _strip_answer_wrappers(match.group(2))
    if not _looks_like_concrete_answer(answer):
        return None
    return output_name, answer


def answers_from_response(
    response_text: str,
    expected_outputs: List[str],
) -> List[Tuple[str, str]]:
    """Extract explicitly labeled task answers from one assistant response.

    A response contributes an answer only through a complete block in the form
    ``\\boxed{task_N_out = value}``. This makes task association part of the
    model output instead of an evaluator inference. Answers remain in response
    order so a later box for the same task can replace an earlier one.
    """
    response_text = _mask_tool_call_payloads(response_text)
    blocks = _boxed_blocks(response_text)
    found = []
    for _, _, content in blocks:
        task_answer = _task_answer_from_box(content, expected_outputs)
        if task_answer is not None:
            found.append(task_answer)
    return found


def update_task_answers(
    task_answers: Dict[str, Dict[str, Any]],
    response_text: str,
    expected_outputs: List[str],
    turn: int,
) -> List[str]:
    """Merge answers from a turn into the trajectory-wide answer map.

    Each expected output keeps only the latest observed answer and the turn
    index where it appeared. This intentionally lets repair turns or later
    text in the same response correct earlier task answers.
    """
    updated = []
    for output_name, answer in answers_from_response(response_text, expected_outputs):
        task_answers[output_name] = {"answer": answer, "turn": turn}
        updated.append(output_name)
    return updated


def missing_task_outputs(
    expected_outputs: List[str],
    task_answers: Dict[str, Dict[str, Any]],
) -> List[str]:
    """Return expected outputs that do not have a collected answer yet.

    Repair prompting uses this list to identify gaps. Already collected outputs
    are not considered missing, though the repair prompt still allows the model
    to correct them and rely on the latest-answer policy.
    """
    return [output_name for output_name in expected_outputs if output_name not in task_answers]


def target_output_text(
    expected_outputs: List[str],
    task_answers: Dict[str, Dict[str, Any]],
) -> str:
    """Render the final requested output in grading-compatible boxed form.

    Multi-task prompts may contain distractors or chain dependencies, but the
    benchmark target remains the final requested output. When that answer has
    not been collected, an empty string signals that there is no synthesized
    target answer to grade.
    """
    if not expected_outputs:
        return ""
    target_name = expected_outputs[-1]
    record = task_answers.get(target_name)
    if record is None:
        return ""
    return f"\\boxed{{{target_name} = {record['answer']}}}"


def last_turn_output(turn_outputs: Any) -> Optional[str]:
    """Return the latest stored assistant turn text.

    ``turn_outputs`` is the canonical persisted text field for agent runs. In
    normal samples its keys are stringified turn indices, so the highest
    numeric key is the final assistant response. Invalid or empty maps return
    ``None``.
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


def build_final_answer_repair_prompt(
    expected_outputs: List[str],
    task_answers: Dict[str, Dict[str, Any]],
) -> str:
    """Build the final-answer repair message for multi-task prompts.

    The message names missing outputs, requires each answer to be labeled by
    ``task_N_out`` inside its box, and explicitly allows corrections. Because
    :func:`update_task_answers` stores the latest value per output, a repair
    turn can both fill absent answers and replace earlier mistakes.
    """
    missing = missing_task_outputs(expected_outputs, task_answers)
    missing_text = ", ".join(missing) if missing else "none"
    return FINAL_ANSWER_REPAIR_PROMPT + (
        f"The outputs still missing are: {missing_text}. "
        "Provide every missing output. If any previously provided output should be corrected, "
        "provide its corrected version too; the latest version of each output will be used. "
        "Put the exact output name and its value together inside each box, for example "
        "\\boxed{task_1_out = ...}. Labels outside boxes do not count. "
        "Use more calculator calls if needed. Do not restart the solution."
    )


# ---------------------------------------------------------------------------
# Calculator tool
# ---------------------------------------------------------------------------
def safe_calc(expression: str) -> str:
    """Evaluate a simple arithmetic expression safely."""
    try:
        node = ast.parse(expression, mode='eval').body
        def _eval(n):
            if isinstance(n, ast.Constant):
                return n.value
            elif isinstance(n, ast.UnaryOp):    # unary operators
                operand = _eval(n.operand)
                if isinstance(n.op, ast.USub):
                    return -operand
                elif isinstance(n.op, ast.UAdd):
                    return +operand
                raise TypeError(f"Unsupported unary op: {type(n.op)}")
            elif isinstance(n, ast.BinOp):      # binary operators
                left, right = _eval(n.left), _eval(n.right)
                if isinstance(n.op, ast.Add): return left + right
                elif isinstance(n.op, ast.Sub): return left - right
                elif isinstance(n.op, ast.Mult): return left * right
                elif isinstance(n.op, ast.Div): return left / right
                elif isinstance(n.op, ast.FloorDiv): return left // right
                elif isinstance(n.op, ast.Mod): return left % right
                elif isinstance(n.op, ast.Pow): return left ** right
                raise TypeError(f"Unsupported binop: {type(n.op)}")
            raise TypeError(f"Unsupported node: {type(n)}")
        return str(_eval(node))
    except Exception as e:
        return f"Error: {e}"


TOOL_SCHEMA = [{
    "type": "function",
    "function": {
        "name": "calculator",
        "description": (
            "Evaluates one or more mathematical expressions. For list or multi-step arithmetic, "
            "submit as many independent expressions as possible in one calculator call. "
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expressions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Mathematical expressions to evaluate in order, e.g. "
                        "['17 * 43 + 5', '100 // 7']."
                    )
                },
                "expression": {
                    "type": "string",
                    "description": "Legacy single-expression field. Prefer expressions instead."
                }
            },
            "required": ["expressions"]
        }
    }
}]


def _json_loads_maybe(value):
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    try:
        return json.loads(stripped)
    except Exception:
        return value


def _normalize_tool_call_object(obj):
    """Normalize common Hermes/Qwen/OpenAI-style tool call objects."""
    calls = []

    obj = _json_loads_maybe(obj)
    if isinstance(obj, list):
        for item in obj:
            calls.extend(_normalize_tool_call_object(item))
        return calls

    if not isinstance(obj, dict):
        return calls

    if isinstance(obj.get("function"), dict):
        fn = obj["function"]
        name = fn.get("name") or obj.get("name")
        arguments = fn.get("arguments", obj.get("arguments", {}))
        if name:
            calls.append({"name": name, "arguments": arguments})
        return calls

    if isinstance(obj.get("function_call"), dict):
        fn = obj["function_call"]
        name = fn.get("name")
        arguments = fn.get("arguments", {})
        if name:
            calls.append({"name": name, "arguments": arguments})
        return calls

    name = obj.get("name")
    if name:
        calls.append({"name": name, "arguments": obj.get("arguments", {})})
    return calls


def _parse_tool_call_payload(payload):
    payload = payload.strip()
    if not payload:
        return []
    parsed = _json_loads_maybe(payload)
    if parsed is payload:
        return []
    return _normalize_tool_call_object(parsed)


def _parse_legacy_xml_parameters(body):
    params = {}
    for m in re.finditer(r"<parameter=([^>]+)>(.*?)</parameter>", body, re.S):
        key = m.group(1).strip()
        value = m.group(2).strip()
        params[key] = _json_loads_maybe(value)
    return params


def parse_tool_calls(text):
    calls = []
    closed_function_spans = []

    # Hermes/Qwen-style tool call tags. Parse the complete tag payload rather
    # than trying to regex-match a JSON object, since arguments may contain
    # nested braces or a list of tool calls.
    for m in re.finditer(r"<tool_call>(.*?)</tool_call>", text, re.S):
        calls.extend(_parse_tool_call_payload(m.group(1)))

    # Legacy XML style:
    # <function=calculator><parameter=expression>1 + 2</parameter></function>
    for m in re.finditer(r"<function=([^>]+)>(.*?)</function>", text, re.S):
        closed_function_spans.append(m.span())
        params = _parse_legacy_xml_parameters(m.group(2))
        calls.append({"name": m.group(1).strip(), "arguments": params})

    # Compatibility fallback for truncated generations that close the parameter
    # but stop before </function> or </tool_call>.
    for m in re.finditer(r"<function=([^>]+)>.*?<parameter=([^>]+)>(.*?)</parameter>", text, re.S):
        if any(start <= m.start() < end for start, end in closed_function_spans):
            continue
        calls.append({
            "name": m.group(1).strip(),
            "arguments": {m.group(2).strip(): _json_loads_maybe(m.group(3).strip())}
        })

    return calls


_TOOL_SYNTAX_RE = re.compile(
    r"<\s*/?\s*tool_call\b|"
    r"<\s*function\s*=|"
    r"<\s*/\s*function\s*>|"
    r"<\s*parameter\s*=|"
    r"<\s*/\s*parameter\s*>|"
    r'"function_call"\s*:|'
    r'"arguments"\s*:|'
    r'"name"\s*:\s*"calculator"',
    re.I
)


def _looks_like_tool_call(text):
    return bool(_TOOL_SYNTAX_RE.search(text or ""))


def get_initial_calculator_repair_prompt(text, tool_calls):
    if any(tc.get("name") == "calculator" for tc in tool_calls):
        return None
    if tool_calls or _looks_like_tool_call(text):
        return INITIAL_CALCULATOR_MALFORMED_TOOL_REPAIR_PROMPT
    return INITIAL_CALCULATOR_NO_TOOL_REPAIR_PROMPT


def normalize_tool_arguments(arguments):
    if isinstance(arguments, dict):
        if set(arguments) == {"arguments"}:
            return normalize_tool_arguments(arguments["arguments"])

        normalized = dict(arguments)
        if "arguments" in normalized and isinstance(normalized["arguments"], (dict, str, list)):
            nested = normalize_tool_arguments(normalized["arguments"])
            normalized.pop("arguments")
            normalized.update(nested)

        if "expressions" in normalized:
            expressions = _json_loads_maybe(normalized["expressions"])
            if isinstance(expressions, tuple):
                expressions = list(expressions)
            if isinstance(expressions, list):
                normalized["expressions"] = expressions
            elif isinstance(expressions, str):
                normalized["expressions"] = [expressions]

        if "expression" in normalized and not isinstance(normalized["expression"], str):
            normalized["expression"] = str(normalized["expression"])

        return normalized

    if isinstance(arguments, str):
        parsed = _json_loads_maybe(arguments)
        if isinstance(parsed, dict):
            return normalize_tool_arguments(parsed)
        if isinstance(parsed, list):
            return {"expressions": parsed}
        return {"expression": arguments.strip()}

    if isinstance(arguments, list):
        return {"expressions": arguments}

    raise TypeError(
        f"Tool arguments must be a dict, list, string, or JSON object string, got {type(arguments).__name__}"
    )


def run_calculator_tool(arguments):
    args = normalize_tool_arguments(arguments)

    if "expressions" in args:
        expressions = args["expressions"]
        if isinstance(expressions, str):
            expressions = [expressions]
        elif not isinstance(expressions, list):
            expressions = [expressions]

        results = []
        for expression in expressions:
            if not isinstance(expression, str):
                results.append(f"Error: expression must be a string, got {type(expression).__name__}")
            else:
                results.append(safe_calc(expression))
        return json.dumps({"results": results})

    if "expression" in args:
        expression = args["expression"]
        if not isinstance(expression, str):
            return f"Error: expression must be a string, got {type(expression).__name__}"
        return safe_calc(expression)

    raise TypeError("Calculator arguments must include 'expressions' or 'expression'")


async def _generate_text(engine, prompt, sampling_params, request_id):
    results_generator = engine.generate(prompt, sampling_params, request_id)

    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    if final_output is None or not final_output.outputs:
        return ""
    return final_output.outputs[0].text

# ---------------------------------------------------------------------------
# Agent loop (real vLLM)
# ---------------------------------------------------------------------------
async def run_agent_loop(
    engine,
    tokenizer,
    sampling_params,
    system_prompt: str,
    task_prompt: str,
    max_agent_turns: int,
    index: int,
    *,
    repair_sampling_params=None,
    max_model_len: int | None = None,
    require_initial_calculator_call: bool = False,
    initial_tool_repair_attempts: int = 1,
    final_answer_repair_attempts: int = 3,
    cutoff_repair_attempts: int = 5,
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

    requested_outputs = resume_state.get("expected_task_outputs") or expected_task_outputs(task_prompt)
    if not requested_outputs:
        requested_outputs = ["final_answer"]

    messages = resume_state.get("messages") or [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_prompt}
    ]

    total_tool_calls = resume_state.get("total_tool_calls", 0)
    turn_outputs = dict(resume_state.get("turn_outputs") or {})
    task_answers = dict(resume_state.get("task_answers") or {})
    initial_calculator_call_seen = resume_state.get(
        "initial_calculator_call_seen",
        not require_initial_calculator_call,
    )
    initial_calculator_requirement_failed = resume_state.get("initial_calculator_requirement_failed", False)
    tool_repair_attempts_used = resume_state.get("tool_repair_attempts_used", 0)
    final_answer_repair_attempts_used = resume_state.get("final_answer_repair_attempts_used", 0)
    cutoff_repair_attempts_used = resume_state.get("cutoff_repair_attempts_used", 0)
    next_sampling_mode = resume_state.get("next_sampling_mode", "normal")
    start_turn = resume_state.get("next_turn", len(turn_outputs))

    def final_status(missing_requested_answers=False):
        if missing_requested_answers:
            if initial_calculator_requirement_failed:
                return "missing_initial_calculator_call_and_missing_requested_task_answers"
            return "missing_requested_task_answers"
        if initial_calculator_requirement_failed:
            return "missing_initial_calculator_call"
        return "success"

    def build_resume_state(next_turn):
        return {
            "messages": messages,
            "next_turn": next_turn,
            "total_tool_calls": total_tool_calls,
            "initial_calculator_call_seen": initial_calculator_call_seen,
            "initial_calculator_requirement_failed": initial_calculator_requirement_failed,
            "tool_repair_attempts_used": tool_repair_attempts_used,
            "final_answer_repair_attempts_used": final_answer_repair_attempts_used,
            "cutoff_repair_attempts_used": cutoff_repair_attempts_used,
            "next_sampling_mode": next_sampling_mode,
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
        full_prompt = tokenizer.apply_chat_template(messages, tools=TOOL_SCHEMA, tokenize=False, add_generation_prompt=True)
        
        request_id = f"req_{index}_turn_{turn}"
        current_sampling_params = (
            repair_sampling_params
            if next_sampling_mode == "repair" and repair_sampling_params is not None
            else sampling_params
        )
        next_sampling_mode = "normal"
        if max_model_len is not None:
            input_tokens = len(tokenizer.encode(full_prompt, add_special_tokens=False))
            requested_output_tokens = current_sampling_params.max_tokens
            if input_tokens + requested_output_tokens > max_model_len:
                turn_outputs[str(turn)] = ""
                return agent_result("context_length_exceeded", turn)
        out_text = await _generate_text(engine, full_prompt, current_sampling_params, request_id)
        turn_outputs[str(turn)] = out_text
        update_task_answers(task_answers, out_text, requested_outputs, turn)

        tool_calls = parse_tool_calls(out_text)
        has_calculator_call = any(tc.get("name") == "calculator" for tc in tool_calls)

        output_token_count = len(tokenizer.encode(out_text, add_special_tokens=False))
        max_output_tokens = current_sampling_params.max_tokens
        cutoff_detected = output_token_count == max_output_tokens
        missing_outputs = missing_task_outputs(requested_outputs, task_answers)
        if (
            cutoff_detected
            and not tool_calls
            and cutoff_repair_attempts_used < cutoff_repair_attempts
            and (missing_outputs or _looks_like_tool_call(out_text))
        ):
            if _looks_like_tool_call(out_text):
                repair_prompt = get_initial_calculator_repair_prompt(out_text, tool_calls)
            elif "\\boxed" in out_text:
                repair_prompt = build_final_answer_repair_prompt(requested_outputs, task_answers)
            else:
                repair_prompt = CUTOFF_REPAIR_PROMPT
            messages.append({"role": "assistant", "content": out_text})
            messages.append({"role": "user", "content": repair_prompt})
            cutoff_repair_attempts_used += 1
            next_sampling_mode = "repair"
            await checkpoint(turn + 1)
            continue

        if not initial_calculator_call_seen and not has_calculator_call:
            if tool_repair_attempts_used < initial_tool_repair_attempts:
                repair_prompt = get_initial_calculator_repair_prompt(out_text, tool_calls)
                messages.append({"role": "assistant", "content": out_text})
                messages.append({"role": "user", "content": repair_prompt})
                tool_repair_attempts_used += 1
                next_sampling_mode = "repair"
                await checkpoint(turn + 1)
                continue

            initial_calculator_requirement_failed = True
            initial_calculator_call_seen = True
            tool_calls = []

        if not tool_calls:
            if not missing_task_outputs(requested_outputs, task_answers):
                return agent_result(final_status(), turn + 1)

            if final_answer_repair_attempts_used < final_answer_repair_attempts:
                messages.append({"role": "assistant", "content": out_text})
                messages.append({"role": "user", "content": build_final_answer_repair_prompt(requested_outputs, task_answers)})
                final_answer_repair_attempts_used += 1
                next_sampling_mode = "repair"
                await checkpoint(turn + 1)
                continue

            return agent_result(final_status(missing_requested_answers=True), turn + 1)

        # Append the assistant message with the raw tool call XML
        assistant_msg = {"role": "assistant", "content": out_text}
        messages.append(assistant_msg)

        # Execute each tool call and append the result
        for tc in tool_calls:
            total_tool_calls += 1
            name = tc.get("name") or "unknown"
            if name == "calculator":
                try:
                    result = run_calculator_tool(tc.get("arguments", {}))
                except Exception as e:
                    result = f"Failed to parse arguments: {e}"
            else:
                result = f"Unknown tool: {name}"

            messages.append({
                "role": "tool",
                "name": name,
                "content": str(result)
            })

        if has_calculator_call:
            initial_calculator_call_seen = True

        await checkpoint(turn + 1)

    return agent_result("timeout", max_agent_turns)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser( description="Run agent evaluation with calculator tool on context-management benchmark datasets.")
    # Same parameters as run_prompts.py
    parser.add_argument("--input", type=str, required=True, help="Input dataset or results JSON file")
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--output", type=str, default="", help="Exact output results JSON file path.")
    output_group.add_argument("--output-dir", type=str, default="", help="Directory for a timestamped output file.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0.0 for deterministic)")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--max_tokens", type=int, default=16384, help="Max tokens per normal generation turn")
    parser.add_argument("--repair_turn_max_tokens", type=int, default=32768, help="Max tokens per repair turn; defaults to --max_tokens")
    parser.add_argument("--max_model_len", type=int, default=262144, help="Max model sequence length for vLLM")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-27B", help="Model name/path")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs (tensor_parallel_size)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="vLLM gpu_memory_utilization")
    parser.add_argument("--max_cudagraph_capture_size", type=int, default=None, help="Limit vLLM CUDA graph capture batch size")
    parser.add_argument("--hf_overrides_json", type=str, default="", help="JSON object passed to vLLM hf_overrides")
    parser.add_argument("--disable_custom_all_reduce", action="store_true", help="Disable vLLM custom all-reduce kernels/fusions for multi-GPU startup stability")
    parser.add_argument("--disable_language_model_only", action="store_false", dest="language_model_only", default=True, help="Do not pass language_model_only=True to vLLM")
    # Agent-specific parameters
    parser.add_argument("--max_agent_turns", type=int, default=200, help="Maximum number of agent turns per task")
    parser.add_argument("--require_initial_calculator_call", action="store_true", help="Require at least one calculator call")
    parser.add_argument("--initial_tool_repair_attempts", type=int, default=5, help="Number of repair turns to try when the initial calculator call is missing")
    parser.add_argument("--final_answer_repair_attempts", type=int, default=5, help="Number of repair turns to try when requested task outputs are missing boxed answers")
    parser.add_argument("--cutoff_repair_attempts", type=int, default=5, help="Number of repair turns to try after a generation cutoff")

    args = parser.parse_args()
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
           gen_params.get("max_tokens") != args.max_tokens or \
           gen_params.get("repair_turn_max_tokens", gen_params.get("max_tokens")) != args.repair_turn_max_tokens or \
           gen_params.get("max_model_len") != args.max_model_len or \
           gen_params.get("model") != expected_model or \
           gen_params.get("max_agent_turns") != args.max_agent_turns or \
           gen_params.get("require_initial_calculator_call", False) != args.require_initial_calculator_call or \
           gen_params.get("initial_tool_repair_attempts", 1) != args.initial_tool_repair_attempts or \
           gen_params.get("final_answer_repair_attempts", 3) != args.final_answer_repair_attempts or \
           gen_params.get("cutoff_repair_attempts", 5) != args.cutoff_repair_attempts or \
           gen_params.get("hf_overrides") != hf_overrides:
            print(f"Error: Generation parameters do not match the continue file.")
            print(f"  File params: {gen_params}")
            print(f"  CLI params: temp={args.temperature}, top_p={args.top_p}, "
                  f"max_tokens={args.max_tokens}, repair_turn_max_tokens={args.repair_turn_max_tokens}, "
                  f"max_model_len={args.max_model_len}, "
                  f"model={expected_model}, max_agent_turns={args.max_agent_turns}, "
                  f"require_initial_calculator_call={args.require_initial_calculator_call}, "
                  f"initial_tool_repair_attempts={args.initial_tool_repair_attempts}, "
                  f"final_answer_repair_attempts={args.final_answer_repair_attempts}, "
                  f"cutoff_repair_attempts={args.cutoff_repair_attempts}, "
                  f"hf_overrides={hf_overrides}")
            sys.exit(1)

        json_path = args.output if args.output else args.input
        gen_params["repair_turn_max_tokens"] = args.repair_turn_max_tokens
        gen_params["final_answer_repair_attempts"] = args.final_answer_repair_attempts
        gen_params["cutoff_repair_attempts"] = args.cutoff_repair_attempts
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

        new_data = {
            "summary": {
                "start_times": [datetime.datetime.now().isoformat()],
                "generation_params": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": args.max_tokens,
                    "repair_turn_max_tokens": args.repair_turn_max_tokens,
                    "max_model_len": args.max_model_len,
                    "model": expected_model,
                    "max_agent_turns": args.max_agent_turns,
                    "require_initial_calculator_call": args.require_initial_calculator_call,
                    "initial_tool_repair_attempts": args.initial_tool_repair_attempts,
                    "final_answer_repair_attempts": args.final_answer_repair_attempts,
                    "cutoff_repair_attempts": args.cutoff_repair_attempts,
                    "language_model_only": args.language_model_only,
                    "max_cudagraph_capture_size": args.max_cudagraph_capture_size,
                    "hf_overrides": hf_overrides,
                    "disable_custom_all_reduce": args.disable_custom_all_reduce
                },
                "system_prompt": system_prompt,
                "mode": "agent",
                "num_tasks": data["summary"].get("num_tasks"),
                "num_samples_per_task": data["summary"].get("num_samples_per_task"),
                "overall_accuracy": None,
                "average_token_length": None,
                "average_tool_calls": None,
                "average_turns": None,
                "tool_use_rate": None
            },
            "samples": []
        }
        for s in data["samples"]:
            sample = s.copy()
            if "correct" not in sample:
                sample["correct"] = None
            if "token_length" not in sample:
                sample["token_length"] = None
            if "agent_turns" not in sample:
                sample["agent_turns"] = None
            if "agent_tool_calls" not in sample:
                sample["agent_tool_calls"] = None
            if "turn_outputs" not in sample:
                sample["turn_outputs"] = None
            if "expected_task_outputs" not in sample:
                sample["expected_task_outputs"] = None
            if "task_answers" not in sample:
                sample["task_answers"] = None
            if "agent_status" not in sample:
                sample["agent_status"] = None
            if "agent_resume_state" not in sample:
                sample["agent_resume_state"] = None
            new_data["samples"].append(sample)
        data = new_data

    save_json(data, json_path)


    # Filter pending samples
    pending_indices = [
        i
        for i, sample in enumerate(data["samples"])
        if sample.get("agent_status") == "in_progress"
        or last_turn_output(sample.get("turn_outputs")) is None
    ]
    if not pending_indices:
        print("All samples already evaluated.")
        return
    print(f"Found {len(pending_indices)} pending samples.", flush=True)

    async def process_all_samples():
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm import SamplingParams
        from transformers import AutoTokenizer

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

        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens
        )
        repair_sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.repair_turn_max_tokens
        )
        save_lock = asyncio.Lock()

        async def save_sample_checkpoint(index, result):
            async with save_lock:
                sample = data["samples"][index]
                sample["agent_status"] = "in_progress"
                sample["agent_turns"] = result["turns_taken"]
                sample["agent_tool_calls"] = result["total_tool_calls"]
                sample["turn_outputs"] = result.get("turn_outputs", {})
                sample["expected_task_outputs"] = result.get("expected_task_outputs", [])
                sample["task_answers"] = result.get("task_answers", {})
                sample["agent_resume_state"] = result.get("resume_state")
                save_json(data, json_path)

        async def process_sample(index):
            sample = data["samples"][index]
            resume_state = None
            if sample.get("agent_status") == "in_progress":
                resume_state = dict(sample.get("agent_resume_state") or {})
                resume_state["turn_outputs"] = sample.get("turn_outputs") or {}
                resume_state["expected_task_outputs"] = sample.get("expected_task_outputs") or []
                resume_state["task_answers"] = sample.get("task_answers") or {}
            result = await run_agent_loop(
                engine=engine,
                tokenizer=tokenizer,
                sampling_params=sampling_params,
                system_prompt=system_prompt,
                task_prompt=sample["prompt"],
                max_agent_turns=args.max_agent_turns,
                index=index,
                repair_sampling_params=repair_sampling_params,
                max_model_len=args.max_model_len,
                require_initial_calculator_call=args.require_initial_calculator_call,
                initial_tool_repair_attempts=args.initial_tool_repair_attempts,
                final_answer_repair_attempts=args.final_answer_repair_attempts,
                cutoff_repair_attempts=args.cutoff_repair_attempts,
                resume_state=resume_state,
                checkpoint_callback=lambda partial_result: save_sample_checkpoint(index, partial_result),
            )
            
            terminal_output = last_turn_output(result.get("turn_outputs")) or ""
            grading_text = target_output_text(
                result.get("expected_task_outputs", []),
                result.get("task_answers", {}),
            ) or terminal_output
            token_count = 0
            for text in result.get("turn_outputs", {}).values():
                if text:
                    token_count += len(tokenizer.encode(text))
            sample["token_length"] = token_count
            sample["correct"] = grade(grading_text, sample["ground_truth"])
            sample["agent_turns"] = result["turns_taken"]
            sample["agent_tool_calls"] = result["total_tool_calls"]
            sample["turn_outputs"] = result.get("turn_outputs", {})
            sample["expected_task_outputs"] = result.get("expected_task_outputs", [])
            sample["task_answers"] = result.get("task_answers", {})
            sample["agent_status"] = result["status"]
            sample["agent_resume_state"] = None
            return index

        tasks = [asyncio.create_task(process_sample(idx)) for idx in pending_indices]
        for future in async_tqdm.as_completed(tasks, total=len(tasks), desc="Agent evaluation"):
            await future
            async with save_lock:
                save_json(data, json_path)

    asyncio.run(process_all_samples())

    # ---------------------------------------------------------------------------
    # Update summary statistics
    # ---------------------------------------------------------------------------

    total = len(data["samples"])
    correct = sum(1 for s in data["samples"] if s["correct"])
    evaluated = [s for s in data["samples"] if s["token_length"] is not None]
    total_tokens = sum(s["token_length"] for s in evaluated)
    total_tool_calls = sum(s["agent_tool_calls"] for s in evaluated if s["agent_tool_calls"] is not None)
    total_turns = sum(s["agent_turns"] for s in evaluated if s["agent_turns"] is not None)
    agents_using_tools = sum(1 for s in evaluated if s.get("agent_tool_calls") and s["agent_tool_calls"] > 0)
    eval_count = len(evaluated)

    data["summary"]["overall_accuracy"] = correct / total if total > 0 else 0
    data["summary"]["average_token_length"] = total_tokens / eval_count if eval_count > 0 else 0
    data["summary"]["average_tool_calls"] = total_tool_calls / eval_count if eval_count > 0 else 0
    data["summary"]["average_turns"] = total_turns / eval_count if eval_count > 0 else 0
    data["summary"]["tool_use_rate"] = agents_using_tools / eval_count if eval_count > 0 else 0

    save_json(data, json_path)

    print(f"\nFinished! Results saved to: {json_path}")
    print(f"Overall Accuracy: {data['summary']['overall_accuracy']:.2%}")
    print(f"Average Token Length: {data['summary']['average_token_length']:.1f}")
    print(f"Average Tool Calls: {data['summary']['average_tool_calls']:.1f}")
    print(f"Average Agent Turns: {data['summary']['average_turns']:.1f}")
    print(f"Tool Use Rate: {data['summary']['tool_use_rate']:.2%}")


if __name__ == "__main__":
    main()
