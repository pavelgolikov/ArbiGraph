import argparse
import json
import os
import sys
import datetime
import re
import asyncio

import getpass
username = getpass.getuser()
os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_cache_{username}"
os.environ["TORCH_EXTENSIONS_DIR"] = f"/tmp/torch_extensions_{username}"
os.environ["VLLM_CACHE_ROOT"] = f"/tmp/vllm_cache_{username}"
os.environ["VLLM_CONFIG_ROOT"] = f"/tmp/vllm_config_{username}"

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

def save_json(data, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    json_str = json.dumps(data, indent=2)
    # Collapse arrays of simple types (like numbers/strings) horizontally
    json_str = re.sub(r'\[\s+([^\[\]\{\}]*?)\s+\]', lambda m: '[' + re.sub(r'\s+', ' ', m.group(1)) + ']', json_str)
    json_str = re.sub(r'\[\s+\]', '[]', json_str)
    with open(path, "w") as f:
        f.write(json_str)

from grader import grade, extract_boxed
from output_paths import timestamped_output_path
from tqdm.asyncio import tqdm as async_tqdm

SYSTEM_PROMPT = (
    "You are a helpful reasoning assistant. "
    "Think step by step. Put each requested answer in its own \\boxed{} block with "
    "the exact output name inside the box, for example \\boxed{task_1_out = ...}. "
    "When performing calculations, round any floating point numbers to 3 decimals and don't calculate more than 5 decimals. "
    "Do not truncate any lists unless otherwise instructed to."
)


async def run_vllm_async(data, pending_indices, args, json_path, system_prompt):
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm import SamplingParams

    print("Initializing Async vLLM Engine...")
    engine_args = AsyncEngineArgs(
        model=args.model,
        tensor_parallel_size=args.num_gpus,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        skip_mm_profiling=True,
        gdn_prefill_backend="triton",
        language_model_only=True
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # Load tokenizer to format chat templates dynamically
    has_chat_template = False
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        has_chat_template = tokenizer.chat_template is not None or hasattr(tokenizer, "default_chat_template")
    except Exception as e:
        print(f"Warning: Could not load tokenizer for chat template: {e}")

    # Warmup request to compile Triton kernels sequentially and prevent concurrent compilation deadlocks
    print("Running sequential Triton JIT compilation/warmup request...")
    warmup_params = SamplingParams(temperature=0.0, max_tokens=1)
    async for _ in engine.generate("Warmup query.", warmup_params, "warmup_req"):
        pass
    print("Triton JIT compilation complete.")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )

    async def process_sample(index):
        sample = data["samples"][index]
        if has_chat_template:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": sample["prompt"]}
            ]
            full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            full_prompt = f"{system_prompt}\n\n{sample['prompt']}"
            
        request_id = f"req_{index}"
        
        results_generator = engine.generate(full_prompt, sampling_params, request_id)
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
            # print(f"[DEBUG req_{index}] Generated {len(final_output.outputs[0].token_ids)} tokens...", flush=True)
            
        out_text = final_output.outputs[0].text
        tok_len = len(final_output.outputs[0].token_ids)
        
        sample["output"] = out_text
        sample["token_length"] = tok_len
        sample["extracted"] = extract_boxed(out_text)
        sample["correct"] = grade(out_text, sample["ground_truth"])
        return index

    tasks = [asyncio.create_task(process_sample(idx)) for idx in pending_indices]
    
    print("Running vLLM inference...")
    for future in async_tqdm.as_completed(tasks, total=len(tasks)):
        idx = await future
        save_json(data, json_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input dataset or results JSON file")
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--output", type=str, default="", help="Exact output results JSON file path.")
    output_group.add_argument("--output-dir", type=str, default="", help="Directory for a timestamped output file.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=64000)
    parser.add_argument("--max_model_len", type=int, default=64000, help="Max model sequence length to allocate for vLLM")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--num_gpus", type=int, default=1, help="Num GPUs.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="vLLM gpu_memory_utilization")

    args = parser.parse_args()

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
        # Check consistency for continue mode
        gen_params = data.get("summary", {}).get("generation_params", {})
        if gen_params.get("temperature") != args.temperature or \
           gen_params.get("top_p") != args.top_p or \
           gen_params.get("max_tokens") != args.max_tokens or \
           gen_params.get("max_model_len") != args.max_model_len or \
           gen_params.get("model") != expected_model:
            print(f"Error: Generation parameters do not match the continue file. Expected: {gen_params}, Got: temp={args.temperature}, top_p={args.top_p}, max_tokens={args.max_tokens}, max_model_len={args.max_model_len}, model={expected_model}")
            sys.exit(1)
            
        json_path = args.output if args.output else args.input
        data["summary"]["start_times"].append(datetime.datetime.now().isoformat())
    else:
        # Create a new results structure based on the dataset
        output_dir = args.output_dir if args.output_dir else REPO_ROOT
        json_path = args.output if args.output else timestamped_output_path(
            output_dir,
            args.input,
            expected_model,
            "eval",
        )
        
        new_data = {
            "summary": {
                "start_times": [datetime.datetime.now().isoformat()],
                "generation_params": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": args.max_tokens,
                    "max_model_len": args.max_model_len,
                    "model": args.model
                },
                "system_prompt": system_prompt,
                "num_tasks": data["summary"].get("num_tasks"),
                "num_samples_per_task": data["summary"].get("num_samples_per_task"),
                "overall_accuracy": None,
                "average_token_length": None
            },
            "samples": []
        }
        for s in data["samples"]:
            sample = s.copy()
            if "output" not in sample:
                sample["output"] = None
            if "correct" not in sample:
                sample["correct"] = None
            if "token_length" not in sample:
                sample["token_length"] = None
            new_data["samples"].append(sample)
        data = new_data
        
    save_json(data, json_path)

    # Filter pending samples
    pending_indices = [i for i, s in enumerate(data["samples"]) if s["output"] is None]
    if not pending_indices:
        print("All samples already evaluated.")
        return
        
    print(f"Found {len(pending_indices)} pending samples to evaluate.")

    asyncio.run(run_vllm_async(data, pending_indices, args, json_path, system_prompt))

    # Update summary
    total = len(data["samples"])
    correct = sum(1 for s in data["samples"] if s["correct"])
    total_tokens = sum(s["token_length"] for s in data["samples"] if s["token_length"] is not None)
    
    data["summary"]["overall_accuracy"] = correct / total if total > 0 else 0
    evaluated_count = sum(1 for s in data["samples"] if s["token_length"] is not None)
    data["summary"]["average_token_length"] = total_tokens / evaluated_count if evaluated_count > 0 else 0
    
    save_json(data, json_path)
        
    print(f"Finished! Overall Accuracy: {data['summary']['overall_accuracy']:.2%}")
    print(f"Average Token Length: {data['summary']['average_token_length']:.1f}")

if __name__ == "__main__":
    main()
