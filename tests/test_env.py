import os
import getpass
username = getpass.getuser()
os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_cache_{username}"
os.environ["TORCH_EXTENSIONS_DIR"] = f"/tmp/torch_extensions_{username}"
os.environ["VLLM_CACHE_ROOT"] = f"/tmp/vllm_cache_{username}"
os.environ["VLLM_CONFIG_ROOT"] = f"/tmp/vllm_config_{username}"

try:
    from vllm import LLM, SamplingParams
except ModuleNotFoundError:
    if __name__ != "__main__":
        import pytest

        pytest.skip("vLLM is required for the environment smoke test.", allow_module_level=True)
    raise

if __name__ == '__main__':
    # 1. Initialize the LLM with a tiny model to test compilation
    print("Loading model and compiling graphs...")
    llm = LLM(
        # model="Qwen/Qwen2.5-0.5B-Instruct",
        model="Qwen/Qwen3.5-4B",
        # model="Qwen/Qwen3-4B-Instruct-2507",
        # enforce_eager=True, 
        # max_model_len=4096,
        skip_mm_profiling=True,
        tensor_parallel_size=1,
        gdn_prefill_backend="triton"
    )

    # 2. Define the prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]

    # 3. Generate the response
    print("Generating response...")
    sampling_params = SamplingParams(temperature=0.0, max_tokens=4096)

    outputs = llm.chat(
        messages=messages,
        sampling_params=sampling_params
    )

    # 4. Print the result
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print("\n--- Output ---")
        print(f"Response: {generated_text.strip()}")
        print("--------------")

    print("\nSuccess! The environment is working.")
