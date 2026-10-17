import argparse
import asyncio
import datetime
from functools import partial
import json
import os
import sys

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

from util import (
    SUBMISSION_RULE,
    SYSTEM_PROMPT,
    finish_results,
    generation_params,
    keeps_special_tokens,
    load_run_data,
    make_sampling_params,
    mcp_server_args,
    new_results,
    prepare_samples,
    process_samples,
    prompt_renderer,
)


async def _generate_text(engine, prompt, sampling_params, request_id, *, messages, tools,
                         chat_template_kwargs, tokenizer=None, keep_special_tokens=False, prompt_token_ids=None):
    """Generate one completion and report why it stopped.
    """
    # A Harmony prompt arrives already tokenized; hand vLLM those exact tokens.
    engine_prompt = prompt if prompt_token_ids is None else {"prompt_token_ids": prompt_token_ids}
    results_generator = engine.generate(engine_prompt, sampling_params, request_id)

    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    if final_output is None or not final_output.outputs:
        return "", None
    completion = final_output.outputs[0]
    if keep_special_tokens:
        # vLLM's completion.text strips special tokens, and GPT-OSS and Gemma 4 mark
        # their tool calls with them. Keep them for the parser.
        if not completion.token_ids:
            raise ValueError("Completion has no token IDs to decode")
        return tokenizer.decode(completion.token_ids, skip_special_tokens=False), getattr(completion, "finish_reason", None)
    return completion.text, getattr(completion, "finish_reason", None)


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
    parser.add_argument("--get_input_attempts_per_task", type=int, default=5, help="Valid get_input submissions per task, including correct submissions. Must match the dataset's wrong-input bank size.")
    parser.add_argument("--final_answer_repair_attempts", type=int, default=20, help="Repair turns to try when requested task output assignments are missing. Flat total per sample, NOT per task.")
    parser.add_argument("--cutoff_repair_attempts", type=int, default=20, help="Repair turns to try after a generation cutoff. Flat total per sample, NOT per task.")
    parser.add_argument("--max_turns_without_new_answer", type=int, default=200, help="Abort a sample after this many consecutive turns produce no new task output. Flat budget per sample, NOT per task. Bounds tool-call loops, which no repair budget covers.")
    parser.add_argument("--max_concurrent_samples", type=int, default=12, help="Maximum samples in flight at once (0 for unlimited)")

    args = parser.parse_args()
    if args.get_input_attempts_per_task < 1:
        parser.error("--get_input_attempts_per_task must be a positive integer")
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

    json_path = args.output
    data = load_run_data(args.input, json_path)

    server_args = mcp_server_args(data["samples"])
    expected_model = args.model
    is_dataset = "generation_params" not in data.get("summary", {})
    system_prompt = SYSTEM_PROMPT

    if not is_dataset:
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
           gen_params.get("language_model_only") != args.language_model_only or \
           gen_params.get("max_cudagraph_capture_size") != args.max_cudagraph_capture_size or \
           gen_params.get("disable_custom_all_reduce") != args.disable_custom_all_reduce or \
           gen_params.get("hf_overrides") != hf_overrides:
            print(f"Error: Generation parameters do not match the continue file.")
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
                  f"language_model_only={args.language_model_only}, "
                  f"max_cudagraph_capture_size={args.max_cudagraph_capture_size}, "
                  f"disable_custom_all_reduce={args.disable_custom_all_reduce}, "
                  f"hf_overrides={hf_overrides}")
            sys.exit(1)

        gen_params["repair_turn_max_tokens"] = args.repair_turn_max_tokens
        gen_params["language_model_only"] = args.language_model_only
        gen_params["max_cudagraph_capture_size"] = args.max_cudagraph_capture_size
        gen_params["hf_overrides"] = hf_overrides
        gen_params["disable_custom_all_reduce"] = args.disable_custom_all_reduce
        data["summary"]["start_times"].append(datetime.datetime.now().isoformat())
    else:
        # Create a new results structure from a dataset
        params = generation_params(args)
        params.update({
            "language_model_only": args.language_model_only,
            "max_cudagraph_capture_size": args.max_cudagraph_capture_size,
            "hf_overrides": hf_overrides,
            "disable_custom_all_reduce": args.disable_custom_all_reduce,
        })
        data = new_results(data, params, system_prompt, preserve_sample_state=True)

    checkpoint_dir, pending_indices = prepare_samples(data, json_path)

    async def process_all_samples():
        # Preserve dependency failure order: check MCP before initializing vLLM.
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
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

        sampling_params, repair_sampling_params = make_sampling_params(args, SamplingParams)
        await process_samples(
            args, data, pending_indices, checkpoint_dir, server_args,
            tokenizer=tokenizer, generate_text=partial(
                _generate_text, engine, tokenizer=tokenizer,
                keep_special_tokens=keeps_special_tokens(args.model),
            ),
            sampling_params=sampling_params, repair_sampling_params=repair_sampling_params,
            system_prompt=system_prompt,
        )

    if pending_indices:
        asyncio.run(process_all_samples())

    finish_results(data, json_path, checkpoint_dir)


if __name__ == "__main__":
    main()
