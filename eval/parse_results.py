"""Parse what the agent said.

run_agent_calc.py runs the agent; this reads its output. A turn comes back as
free text carrying two things the loop needs: task output assignments
(`task_3_out = {"result": 7}`), which are the answers, and tool calls, which the
model may write in any of several dialects. Both are recovered here.

Nothing in this module imports vLLM or MCP, so anything that only needs to read
model output — a regrade, an offline analysis, an RLVR scorer — can import it
without standing up an engine. grade_results.py is the same split for grading.
"""

import json
import re
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Task output assignments
# ---------------------------------------------------------------------------
def _mask_tool_call_payloads(text: str) -> str:
    """Hide closed tool calls before answer extraction."""
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


TASK_OUTPUT_RE = re.compile(r"(?<![A-Za-z0-9_])(?P<name>task_[1-9]\d*_out)\s*=\s*")


def _parse_result_value(text: str, start: int) -> Any | None:
    while start < len(text) and text[start].isspace():
        start += 1
    try:
        parsed, end = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or set(parsed.keys()) != {"result"}:
        return None
    end += start
    if end < len(text) and (text[end].isalnum() or text[end] == "_"):
        return None
    return parsed["result"]


def answers_from_response(
    response_text: str,
    expected_outputs: List[str],
) -> List[tuple[str, Any]]:
    """Extract valid task output assignments from one assistant response."""
    response_text = _mask_tool_call_payloads(response_text)
    expected = set(expected_outputs)
    answers = []
    for match in TASK_OUTPUT_RE.finditer(response_text):
        output_name = match.group("name")
        if output_name not in expected:
            continue
        value = _parse_result_value(response_text, match.end())
        if value is not None:
            answers.append((output_name, value))
    return answers


def update_task_answers(
    task_answers: Dict[str, Dict[str, Any]],
    response_text: str,
    expected_outputs: List[str],
    turn: int,
) -> None:
    """Merge answers from a turn into the trajectory-wide answer map.
    """
    for output_name, answer in answers_from_response(response_text, expected_outputs):
        task_answers[output_name] = {"answer": answer, "turn": turn}


def missing_task_outputs(
    expected_outputs: List[str],
    task_answers: Dict[str, Dict[str, Any]],
) -> List[str]:
    """Return expected outputs that do not have a collected answer yet.
    """
    return [output_name for output_name in expected_outputs if output_name not in task_answers]


# ---------------------------------------------------------------------------
# Tool calls
# ---------------------------------------------------------------------------
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


def _parse_xml_parameters(body):
    params = {}
    for m in re.finditer(r"<parameter=([^>]+)>(.*?)</parameter>", body, re.S):
        key = m.group(1).strip()
        value = m.group(2).strip()
        params[key] = _json_loads_maybe(value)
    return params


def parse_tool_calls(text):
    calls = []
    closed_function_spans = []

    for m in re.finditer(r"<tool_call>(.*?)</tool_call>", text, re.S):
        calls.extend(_parse_tool_call_payload(m.group(1)))

    # Parse XML-style function wrapper calls.
    for m in re.finditer(r"<function=([^>]+)>(.*?)</function>", text, re.S):
        closed_function_spans.append(m.span())
        params = _parse_xml_parameters(m.group(2))
        calls.append({"name": m.group(1).strip(), "arguments": params})

    # Parse closed XML parameters from truncated function wrapper calls.
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


def looks_like_tool_call(text):
    return bool(_TOOL_SYNTAX_RE.search(text or ""))


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

        if "expression" in normalized:
            expression = normalized.pop("expression")
            if "expressions" not in normalized:
                normalized["expressions"] = [str(expression)]

        return normalized

    if isinstance(arguments, str):
        parsed = _json_loads_maybe(arguments)
        if isinstance(parsed, dict):
            return normalize_tool_arguments(parsed)
        if isinstance(parsed, list):
            return {"expressions": parsed}
        return {"expressions": [arguments.strip()]}

    if isinstance(arguments, list):
        return {"expressions": arguments}

    raise TypeError(
        f"Tool arguments must be a dict, list, string, or JSON object string, got {type(arguments).__name__}"
    )
