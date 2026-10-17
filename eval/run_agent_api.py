"""Vector API evaluation runner.

Use the local runner's environment; additionally install httpx and python-dotenv.
Credentials: VEC_API_KEY in the repository-root .env file.

Use the experiment flags from run_agent.py, with --model selecting the tokenizer
and --api_model selecting Vector's model ID (e.g. Qwen3_8-27B). GPU/server-launch
flags are rejected during argument parsing. No local model is loaded, and only
GPT-OSS needs vLLM installed, for its Harmony prompt rendering: sampling settings
are carried in a plain object and sent to the API as JSON.
Every turn checks that the prompt tokens Vector reports match the local render,
so an API run gives the model exactly what a local run would.
Temporary API/network failures retry the same request every 30 seconds until it
succeeds. Unexpected API errors stop the run with its checkpoints intact rather
than turning an infrastructure failure into an incorrect sample.
"""

import argparse
import asyncio
import dataclasses
import datetime
import json
import os
from pathlib import Path
import sys

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

from util import (
    SAMPLING_FIELDS,
    SUBMISSION_RULE,
    SYSTEM_PROMPT,
    finish_results,
    generation_params,
    harmony_start_date,
    is_harmony_model,
    keeps_special_tokens,
    load_run_data,
    make_sampling_params,
    mcp_server_args,
    new_results,
    prepare_samples,
    process_samples,
    prompt_renderer,
    render_harmony_prompt,
)


@dataclasses.dataclass
class SamplingParams:
    """What one generation turn asks for, in the shape the agent loop expects.

    The local runner gets this from vLLM, which also validates it. Here the
    values only travel to the API as JSON, so they are carried as they came off
    the command line.
    """

    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: float
    repetition_penalty: float


class IncompleteStreamError(Exception):
    """A streamed response that cannot be used as a complete model turn."""


class VectorClient:
    def __init__(self, model, api_model):
        import httpx
        from dotenv import dotenv_values
        from transformers import AutoTokenizer

        api_key = dotenv_values(Path(__file__).resolve().parents[1] / ".env")["VEC_API_KEY"]
        if not api_key:
            raise ValueError("VEC_API_KEY is empty in the repository's .env file")
        self.api_model = api_model
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.client = httpx.AsyncClient(
            base_url="https://proxy.vectorinstitute.ai/v1/",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=None,
        )

    @property
    def harmony(self):
        return is_harmony_model(self.api_model)

    def check_prompt(self, returned, expected, *, messages, tools, chat_template_kwargs):
        """Refuse a turn whose prompt differs from the local render.

        A Harmony prompt carries the server's date, which may differ from the
        local one, so it is rendered again on the server's date; no other token
        may differ.
        """
        if returned == expected:
            return
        date = harmony_start_date(self.tokenizer, returned) if self.harmony else None
        if date is not None and returned == render_harmony_prompt(
            messages, tools, chat_template_kwargs["reasoning_effort"], start_date=date,
        ):
            return
        raise ValueError("Vector's rendered prompt differs from the local model prompt")

    async def generate_text(self, prompt, sampling_params, *, request_id, messages, tools,
                            chat_template_kwargs, prompt_token_ids=None):
        import httpx

        payload = {
            "model": self.api_model,
            "messages": messages,
            "tools": tools,
            # Keep tool calls as raw generated text for the existing parser.
            "tool_choice": "none",
            "chat_template_kwargs": chat_template_kwargs,
            "return_token_ids": True,
            # Receive tokens as they are generated instead of making the
            # upstream wait for the whole turn behind its 10-minute deadline.
            "stream": True,
            **{name: getattr(sampling_params, name) for name in ("max_tokens", *SAMPLING_FIELDS)},
        }
        if self.harmony:
            # Vector ignores this setting in chat_template_kwargs for GPT-OSS;
            # its top-level Chat Completions field controls the Harmony prompt.
            payload["reasoning_effort"] = chat_template_kwargs["reasoning_effort"]
        expected_prompt = (prompt_token_ids if prompt_token_ids is not None
                           else self.tokenizer.encode(prompt, add_special_tokens=False))
        while True:
            try:
                async with self.client.stream("POST", "chat/completions", json=payload) as response:
                    if response.status_code in (408, 429, 500, 502, 503, 504):
                        await response.aread()
                        detail = " ".join(response.text.split())
                        error = f"HTTP {response.status_code}"
                        if detail:
                            error += f": {detail[:500]}"
                    elif response.is_error:
                        await response.aread()
                        raise RuntimeError(f"Vector API HTTP {response.status_code}: {response.text}")
                    else:
                        returned_prompt_ids = None
                        output_token_ids = []
                        finish_reason = None
                        event_lines = []
                        stream_done = False
                        async for line in response.aiter_lines():
                            if line.startswith("data:"):
                                event_lines.append(line[5:].lstrip(" "))
                                continue
                            if line or not event_lines:
                                continue
                            event_text = "\n".join(event_lines)
                            event_lines.clear()
                            if event_text.strip() == "[DONE]":
                                stream_done = True
                                break
                            event = json.loads(event_text)
                            if not isinstance(event, dict):
                                raise IncompleteStreamError("invalid SSE event")
                            if event.get("error"):
                                raise IncompleteStreamError(f"stream error: {event['error']}")
                            if event.get("prompt_token_ids") is not None:
                                returned_prompt_ids = event["prompt_token_ids"]
                            if not event.get("choices"):
                                continue
                            choice = event["choices"][0]
                            if choice.get("token_ids"):
                                output_token_ids.extend(choice["token_ids"])
                            if choice.get("finish_reason") is not None:
                                finish_reason = choice["finish_reason"]

                        if len(event_lines) == 1 and event_lines[0].strip() == "[DONE]":
                            stream_done = True
                            event_lines.clear()
                        if (event_lines or not stream_done or finish_reason is None
                                or returned_prompt_ids is None or not output_token_ids):
                            raise IncompleteStreamError("stream ended before a complete response")
                        self.check_prompt(returned_prompt_ids, expected_prompt, messages=messages,
                                          tools=tools, chat_template_kwargs=chat_template_kwargs)
                        # GPT-OSS and Gemma 4 mark their tool calls with special tokens;
                        # keep them for the parser, as the local runner does.
                        text = self.tokenizer.decode(
                            output_token_ids, skip_special_tokens=not keeps_special_tokens(self.api_model),
                        )
                        return text, finish_reason
            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError,
                    json.JSONDecodeError, IncompleteStreamError) as exc:
                error = f"{type(exc).__name__}: {exc}"
            print(f"[{request_id}] Vector API {error}; retrying in 30 seconds.", flush=True)
            await asyncio.sleep(30)


async def process_api_samples(args, data, pending_indices, checkpoint_dir, server_args, system_prompt):
    client = VectorClient(args.model, args.api_model)
    sampling_params, repair_sampling_params = make_sampling_params(args, SamplingParams)
    async with client.client:
        await process_samples(
            args, data, pending_indices, checkpoint_dir, server_args,
            tokenizer=client.tokenizer, generate_text=client.generate_text,
            sampling_params=sampling_params, repair_sampling_params=repair_sampling_params,
            system_prompt=system_prompt,
            strict=True,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run agent evaluation with calculator and get_input tools on context-management benchmark datasets.",
        allow_abbrev=False,
    )
    # CLI arguments for agent evaluation.
    parser.add_argument("--input", type=str, required=True, help="Input dataset or results JSON file")
    parser.add_argument("--output", type=str, required=True, help="Results JSON file; resume it if it already exists.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0.0 for deterministic)")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k sampling parameter")
    parser.add_argument("--min_p", type=float, default=0.0, help="Min-p sampling parameter")
    parser.add_argument("--presence_penalty", type=float, default=0.0, help="Presence penalty")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty")
    parser.add_argument("--reasoning_effort", type=str, default="xhigh", help="Reasoning depth passed to the chat template (xhigh, medium, low)")
    parser.add_argument("--preserve_thinking", action=argparse.BooleanOptionalAction, default=True, help="Keep reasoning context from earlier turns (--no-preserve_thinking to drop it)")
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction, default=None, help="Turn thinking on or off in chat templates that read enable_thinking, such as Gemma 4's; unset leaves each template's default")
    parser.add_argument("--max_tokens", type=int, default=32768, help="Max tokens per normal generation turn")
    parser.add_argument("--repair_turn_max_tokens", type=int, default=32768, help="Max tokens per repair generation turn")
    parser.add_argument("--max_model_len", type=int, default=262144, help="Max model sequence length for vLLM")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-27B", help="Model name/path")
    parser.add_argument("--example", action="store_true", help="Write the first rendered model prompt to sample_to_view.txt and ask before running")
    # Agent-specific parameters
    parser.add_argument("--max_agent_turns", type=int, default=2000, help="Maximum number of agent turns per task")
    parser.add_argument("--tool_repair_attempts", type=int, default=5, help="Repair turns to try when the model emits a malformed tool call. Flat total per sample, NOT per task.")
    parser.add_argument("--get_input_attempts_per_task", type=int, default=5, help="Valid get_input submissions per task, including correct submissions. Must match the dataset's wrong-input bank size.")
    parser.add_argument("--final_answer_repair_attempts", type=int, default=20, help="Repair turns to try when requested task output assignments are missing. Flat total per sample, NOT per task.")
    parser.add_argument("--cutoff_repair_attempts", type=int, default=20, help="Repair turns to try after a generation cutoff. Flat total per sample, NOT per task.")
    parser.add_argument("--max_turns_without_new_answer", type=int, default=200, help="Abort a sample after this many consecutive turns produce no new task output. Flat budget per sample, NOT per task. Bounds tool-call loops, which no repair budget covers.")
    parser.add_argument("--max_concurrent_samples", type=int, default=12, help="Maximum samples in flight at once (0 for unlimited)")
    parser.add_argument("--api_model", required=True, help="Exact Vector model ID from /v1/models")

    args = parser.parse_args()
    if args.get_input_attempts_per_task < 1:
        parser.error("--get_input_attempts_per_task must be a positive integer")
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        sys.exit(1)

    json_path = args.output
    data = load_run_data(args.input, json_path)

    server_args = mcp_server_args(data["samples"])
    expected_model = args.model
    is_dataset = "generation_params" not in data.get("summary", {})
    system_prompt = SYSTEM_PROMPT

    if not is_dataset:
        system_prompt = data.get("summary", {}).get("system_prompt") or system_prompt
        gen_params = data.get("summary", {}).get("generation_params", {})
        if gen_params.get("temperature") != args.temperature or \
           gen_params.get("top_p") != args.top_p or \
           gen_params.get("top_k") != args.top_k or \
           gen_params.get("min_p") != args.min_p or \
           gen_params.get("presence_penalty") != args.presence_penalty or \
           gen_params.get("repetition_penalty") != args.repetition_penalty or \
           gen_params.get("reasoning_effort") != args.reasoning_effort or \
           gen_params.get("preserve_thinking") != args.preserve_thinking or \
           gen_params.get("enable_thinking") != args.enable_thinking or \
           gen_params.get("max_tokens") != args.max_tokens or \
           gen_params.get("repair_turn_max_tokens", gen_params.get("max_tokens")) != args.repair_turn_max_tokens or \
           gen_params.get("max_model_len") != args.max_model_len or \
           gen_params.get("model") != expected_model or \
           gen_params.get("max_agent_turns") != args.max_agent_turns or \
           gen_params.get("tool_repair_attempts", 5) != args.tool_repair_attempts or \
           gen_params.get("get_input_attempts_per_task") != args.get_input_attempts_per_task or \
           gen_params.get("final_answer_repair_attempts") != args.final_answer_repair_attempts or \
           gen_params.get("cutoff_repair_attempts") != args.cutoff_repair_attempts or \
           gen_params.get("max_turns_without_new_answer") != args.max_turns_without_new_answer or \
           gen_params.get("prompt_renderer", "chat_template") != prompt_renderer(args.model) or \
           gen_params.get("submission_rule", "any_time") != SUBMISSION_RULE or \
           ("api_model" in gen_params and gen_params["api_model"] != args.api_model) or \
           gen_params.get("hf_overrides") is not None:
            print("Error: Generation parameters do not match the continue file.")
            print(f"  File params: {gen_params}")
            print(f"  CLI params: temp={args.temperature}, top_p={args.top_p}, "
                  f"top_k={args.top_k}, min_p={args.min_p}, "
                  f"presence_penalty={args.presence_penalty}, repetition_penalty={args.repetition_penalty}, "
                  f"reasoning_effort={args.reasoning_effort}, preserve_thinking={args.preserve_thinking}, "
                  f"enable_thinking={args.enable_thinking}, "
                  f"max_tokens={args.max_tokens}, repair_turn_max_tokens={args.repair_turn_max_tokens}, "
                  f"max_model_len={args.max_model_len}, "
                  f"model={expected_model}, max_agent_turns={args.max_agent_turns}, "
                  f"tool_repair_attempts={args.tool_repair_attempts}, "
                  f"get_input_attempts_per_task={args.get_input_attempts_per_task}, "
                  f"final_answer_repair_attempts={args.final_answer_repair_attempts}, "
                  f"cutoff_repair_attempts={args.cutoff_repair_attempts}, "
                  f"max_turns_without_new_answer={args.max_turns_without_new_answer}, "
                  f"prompt_renderer={prompt_renderer(args.model)}, "
                  f"submission_rule={SUBMISSION_RULE}, "
                  f"hf_overrides=None")
            print(f"  API model: {args.api_model}")
            sys.exit(1)

        gen_params["repair_turn_max_tokens"] = args.repair_turn_max_tokens
        gen_params["api_model"] = args.api_model
        data["summary"]["start_times"].append(datetime.datetime.now().isoformat())
    else:
        params = {**generation_params(args), "api_model": args.api_model}
        data = new_results(data, params, system_prompt, preserve_sample_state=True)

    checkpoint_dir, pending_indices = prepare_samples(data, json_path)
    if pending_indices:
        asyncio.run(process_api_samples(args, data, pending_indices, checkpoint_dir, server_args, system_prompt))
    finish_results(data, json_path, checkpoint_dir)


if __name__ == "__main__":
    main()
