"""Parse what the agent said.

run_agent.py runs the agent; this reads its output. A turn comes back as
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
        r"<\|tool_call>.*?(?:<tool_call\|>|<\|tool_response>|<turn\|>)",
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


_HARMONY_MESSAGE_RE = re.compile(
    r"<\|start\|>assistant(?P<header>.*?)<\|message\|>"
    r"(?P<body>.*?)(?:<\|end\|>|<\|return\|>|<\|call\|>|\Z)",
    re.S,
)


def _harmony_messages(text):
    if not (text.startswith("<|channel|>") or text.startswith("<|start|>")):
        return []
    # The chat template ends with <|start|>assistant, so the first generated
    # message normally starts at its channel, without repeating that prefix.
    if not text.startswith("<|start|>"):
        text = "<|start|>assistant" + text
    return [(match.group("header"), match.group("body"))
            for match in _HARMONY_MESSAGE_RE.finditer(text)]


def parse_harmony_tool_calls(text):
    calls = []
    for header, body in _harmony_messages(text):
        recipient = re.search(r"\bto=functions\.([A-Za-z_][A-Za-z_0-9]*)\b", header)
        if not recipient:
            continue
        try:
            arguments = json.loads(body)
        except json.JSONDecodeError:
            continue
        if isinstance(arguments, dict):
            calls.append({"name": recipient.group(1), "arguments": arguments})
    return calls


# ---------------------------------------------------------------------------
# Gemma 4
# ---------------------------------------------------------------------------
# Gemma 4 writes its structure in special tokens: reasoning between <|channel> and
# <channel|>, and a tool call as <|tool_call>call:name{key:value,...}<tool_call|>, in
# a JSON-like format with bare keys and each string between two <|"|> tokens. The
# text keeps these tokens only when decoded with skip_special_tokens=False.
GEMMA_OPEN_THOUGHT = "<|channel>thought\n"
_GEMMA_QUOTE = '<|"|>'
# A call closes with <tool_call|>, or with a stop token the model emits in its place.
_GEMMA_CALL_RE = re.compile(
    r"<\|tool_call>call:([A-Za-z_][A-Za-z_0-9]*)\{(.*?)\}(?:<tool_call\|>|<\|tool_response>|<turn\|>)",
    re.S,
)
_GEMMA_THOUGHT_RE = re.compile(r"<\|channel>(?:thought\n)?(.*?)(?:<channel\|>|\Z)", re.S)
_GEMMA_MARKERS = ("<|channel>", "<channel|>", "<|tool_call>", "<tool_call|>", "<turn|>", "<|tool_response>")


def _gemma_skip(text, i):
    while i < len(text) and text[i] in " \t\r\n":
        i += 1
    return i


def _gemma_scalar(token):
    literals = {"true": True, "false": False, "null": None}
    if token in literals:
        return literals[token]
    for convert in (int, float):
        try:
            return convert(token)
        except ValueError:
            pass
    return _json_loads_maybe(token)


def _gemma_value(text, i):
    """Read one Gemma 4 value starting at text[i]; return it and the index after it."""
    i = _gemma_skip(text, i)
    if text.startswith(_GEMMA_QUOTE, i):
        start = i + len(_GEMMA_QUOTE)
        end = text.find(_GEMMA_QUOTE, start)
        if end < 0:
            raise ValueError("unterminated string")
        return text[start:end], end + len(_GEMMA_QUOTE)
    if text.startswith("{", i) or text.startswith("[", i):
        is_object = text[i] == "{"
        close = "}" if is_object else "]"
        container = {} if is_object else []
        i = _gemma_skip(text, i + 1)
        if text.startswith(close, i):
            return container, i + 1
        while True:
            if is_object:
                colon = text.find(":", i)
                if colon < 0:
                    raise ValueError("key without a value")
                key = text[i:colon].strip().replace(_GEMMA_QUOTE, "").strip('"')
                container[key], i = _gemma_value(text, colon + 1)
            else:
                value, i = _gemma_value(text, i)
                container.append(value)
            i = _gemma_skip(text, i)
            if text.startswith(",", i):
                i = _gemma_skip(text, i + 1)
            elif text.startswith(close, i):
                return container, i + 1
            else:
                raise ValueError(f"expected ',' or {close!r}")
    end = i
    while end < len(text) and text[end] not in ",}]":
        end += 1
    token = text[i:end].strip()
    if not token:
        raise ValueError("missing value")
    return _gemma_scalar(token), end


def _gemma_arguments(body):
    """The arguments inside one Gemma 4 call's braces, or None when they do not parse."""
    text = "{" + body + "}"
    try:
        arguments, end = _gemma_value(text, 0)
    except ValueError:
        return None
    return arguments if end == len(text) else None


def _gemma_turn_parts(text):
    """Split a Gemma 4 turn into its reasoning and its visible text, parsed calls removed."""
    closing = text.find("<channel|>")
    if closing >= 0 and "<|channel>" not in text[:closing]:
        # The prompt opened the thought channel, so the turn starts inside it.
        text = GEMMA_OPEN_THOUGHT + text
    reasoning = "\n".join(
        match.group(1).strip() for match in _GEMMA_THOUGHT_RE.finditer(text) if match.group(1).strip()
    )
    visible = _GEMMA_THOUGHT_RE.sub("", text)
    # A call that does not parse stays in view, so a repair prompt can refer to it.
    visible = _GEMMA_CALL_RE.sub(
        lambda match: "" if _gemma_arguments(match.group(2)) is not None else match.group(0), visible,
    )
    for token in ("<|tool_response>", "<turn|>", "<eos>"):
        visible = visible.replace(token, "")
    return reasoning, visible.strip()


def _tool_call_entries(tool_calls, call_ids):
    """Chat-format tool calls, each carrying the id its tool result will name."""
    return [
        {"id": call_id, "type": "function", "function": {
            "name": call["name"],
            "arguments": json.dumps(call["arguments"], ensure_ascii=False),
        }}
        for call_id, call in zip(call_ids, tool_calls, strict=True)
    ]


def assistant_message_from_response(text, tool_calls, call_ids=()):
    """Make a chat message from a model turn, to carry it into the next prompt.

    A GPT-OSS turn is split into the fields vLLM's Harmony renderer reads:
    earlier reasoning goes in `reasoning`, which it renders to the analysis
    channel, and each tool call carries the id its tool result will name.
    A Gemma 4 turn is split into the fields its chat template reads: without a
    `tool_calls` entry the template drops the tool results that follow, and it
    re-renders earlier reasoning only from `reasoning`. What remains in view is
    kept as `content`, as vLLM's Gemma 4 reasoning and tool parsers would keep it.
    Anything else is kept as the raw text.
    """
    harmony_messages = _harmony_messages(text)
    if not harmony_messages:
        if not any(marker in text for marker in _GEMMA_MARKERS):
            return {"role": "assistant", "content": text}
        reasoning, visible = _gemma_turn_parts(text)
        message = {"role": "assistant"}
        if reasoning:
            message["reasoning"] = reasoning
        if tool_calls:
            message["tool_calls"] = _tool_call_entries(tool_calls, call_ids)
        if visible or len(message) == 1:
            message["content"] = visible
        return message
    reasoning = "\n".join(
        body for header, body in harmony_messages
        if "to=functions." not in header and body
    )
    if tool_calls:
        message = {"role": "assistant", "tool_calls": _tool_call_entries(tool_calls, call_ids)}
        if reasoning:
            message["reasoning"] = reasoning
        return message
    final_bodies = [body for header, body in harmony_messages
                    if "<|channel|>final" in header]
    if final_bodies:
        return {"role": "assistant", "content": final_bodies[-1]}
    # No answer and no call, such as a turn cut off mid-thought: keep it as the
    # reasoning it was, not as an answer.
    return {"role": "assistant", "reasoning": reasoning}


def parse_tool_calls(text):
    if _harmony_messages(text):
        return parse_harmony_tool_calls(text)
    calls = []
    closed_function_spans = []

    for m in re.finditer(r"<tool_call>(.*?)(</tool_call>|(?=<tool_call>)|\Z)", text, re.S):
        parsed = _parse_tool_call_payload(m.group(1)) if m.group(2) == "</tool_call>" else []
        calls.extend((m.start(), call) for call in parsed)

    for m in _GEMMA_CALL_RE.finditer(text):
        arguments = _gemma_arguments(m.group(2))
        if arguments is not None:
            calls.append((m.start(), {"name": m.group(1), "arguments": arguments}))

    # Parse XML-style function wrapper calls.
    for m in re.finditer(r"<function=([^>]+)>(.*?)</function>", text, re.S):
        closed_function_spans.append(m.span())
        calls.append((m.start(), {
            "name": m.group(1).strip(),
            "arguments": _parse_xml_parameters(m.group(2)),
        }))

    # Parse closed XML parameters from truncated function wrapper calls.
    for m in re.finditer(r"<function=([^>]+)>(.*?)(?=<function=|\Z)", text, re.S):
        if any(start <= m.start() < end for start, end in closed_function_spans):
            continue
        params = _parse_xml_parameters(m.group(2))
        if not params:
            continue
        calls.append((m.start(), {"name": m.group(1).strip(), "arguments": params}))

    # Input-bank positions depend on submission order, including mixed call formats.
    return [call for _, call in sorted(calls, key=lambda item: item[0])]


_TOOL_SYNTAX_RE = re.compile(
    r"<\|tool_call>|"
    r"<\s*/?\s*tool_call\b|"
    r"<\s*function\s*=|"
    r"<\s*/\s*function\s*>|"
    r"<\s*parameter\s*=|"
    r"<\s*/\s*parameter\s*>|"
    r'"function_call"\s*:|'
    r'"arguments"\s*:|'
    r'"name"\s*:\s*"calculator"|'
    r"\bto=functions\.",
    re.I
)


def looks_like_tool_call(text):
    return bool(_TOOL_SYNTAX_RE.search(text or ""))


def normalize_tool_arguments(arguments):
    """Rename and retype the calculator's own parameter, which models get wrong.

    Only the two forms models actually produce are repaired: the singular key,
    and the list left as a string. Anything else is passed through as written.
    """
    if isinstance(arguments, dict):
        normalized = dict(arguments)

        if isinstance(normalized.get("expressions"), str):
            expressions = _json_loads_maybe(normalized["expressions"])
            normalized["expressions"] = expressions if isinstance(expressions, list) else [expressions]

        if "expression" in normalized:
            expression = normalized.pop("expression")
            if "expressions" not in normalized:
                normalized["expressions"] = [str(expression)]

        return normalized

    raise TypeError(f"Tool arguments must be an object, got {type(arguments).__name__}")
