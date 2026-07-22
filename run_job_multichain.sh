#!/bin/bash
#SBATCH --job-name=golikovp_multichain
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=7:00:00
#SBATCH --output=multichain_eval_out.out
#SBATCH --error=multichain_eval_out.out
#SBATCH --account=aip-gpekhime


# Run Multi-Chain Evaluation
echo "Starting Multi-Chain Evaluation..."

# Set Caches to Project Directory to avoid Quota issues in Home
export HF_HOME=/project/aip-gpekhime/golikovp/cache
export XDG_CACHE_HOME=/project/aip-gpekhime/golikovp/cache
export NLTK_DATA=/project/aip-gpekhime/golikovp/nltk_data
export PYTHONDONTWRITEBYTECODE=1
mkdir -p $HF_HOME
export NCCL_IGNORE_DISABLED_P2P=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# NCCL Fixes

module load python/3.11.5
module load cuda/12.9
module load cudnn
module load gcc opencv/4.13.0


# Qwen3.5-27B ======================================================================
python -B run_agent_calc.py \
    --num_gpus 4 \
    --input results/multichain/Qwen3.5-27B/math/repair/agent_math_multichain_Qwen3.5-27B_20260708_215303.json \
    --output results/multichain/Qwen3.5-27B/math/repair/agent_math_multichain_Qwen3.5-27B_20260708_215303.json \
    --model Qwen/Qwen3.5-27B \
    --max_tokens 16384 \
    --repair_turn_max_tokens 32768 \
    --max_model_len 524288 \
    --gpu_memory_utilization 0.95 \
    --hf_overrides_json '{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":2.0,"original_max_position_embeddings":262144}}}' \
    --initial_tool_repair_attempts 5 \
    --final_answer_repair_attempts 24 \
    --cutoff_repair_attempts 24 \
    --require_initial_calculator_call \
    --max_agent_turns 2000

# python -B run_agent_calc.py \
#     --num_gpus 4 \
#     --input results/multichain/Qwen3.5-27B/python/repair/agent_python_multichain_Qwen3.5-27B_20260711_023520.json \
#     --output results/multichain/Qwen3.5-27B/python/repair/agent_python_multichain_Qwen3.5-27B_20260711_023520.json \
#     --model Qwen/Qwen3.5-27B \
#     --max_tokens 16384 \
#     --repair_turn_max_tokens 32768 \
#     --max_model_len 524288 \
#     --gpu_memory_utilization 0.95 \
#     --hf_overrides_json '{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":2.0,"original_max_position_embeddings":262144}}}' \
#     --initial_tool_repair_attempts 5 \
#     --final_answer_repair_attempts 24 \
#     --cutoff_repair_attempts 24 \
#     --require_initial_calculator_call \
#     --max_agent_turns 2000
