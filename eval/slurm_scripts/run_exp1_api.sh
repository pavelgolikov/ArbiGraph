#!/bin/bash
#SBATCH --job-name=golikovp_arbigraph_exp1_api
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=7:00:00
#SBATCH --output=logs/run_exp1_api_%j.out
#SBATCH --error=logs/run_exp1_api_%j.out
#SBATCH --account=aip-gpekhime

set -euo pipefail
echo "Starting ArbiGraph Experiment 1 via Vector API..."

# Use the same environment as run_exp1.sh; inference runs on Vector.
export HF_HOME=/project/aip-gpekhime/golikovp/cache
export XDG_CACHE_HOME=/project/aip-gpekhime/golikovp/cache
export NLTK_DATA=/project/aip-gpekhime/golikovp/nltk_data
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="/tmp/triton_cache_${USER}"
export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions_${USER}"
export VLLM_CACHE_ROOT="/tmp/vllm_cache_${USER}"
export VLLM_CONFIG_ROOT="/tmp/vllm_config_${USER}"
mkdir -p "$HF_HOME"
export NCCL_IGNORE_DISABLED_P2P=1

module load python/3.11.5
module load cuda/12.9
module load cudnn
module load gcc opencv/4.13.0

cd /home/golikovp/projects/aip-gpekhime/golikovp/ArbiGraph-private/eval || exit 1

MODELS=(
  # Qwen3.8-27B
  # Nemotron-3-Nano-Omni-30B-A3B
  GPT-OSS-120B
)

declare -A MODEL_PATHS=(
  [Qwen3.8-27B]="Qwen/Qwen3.8-27B"
  [Nemotron-3-Nano-Omni-30B-A3B]="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
  [GPT-OSS-120B]="openai/gpt-oss-120b"
)

# Exact Vector model ids, from /v1/models. These are not the Hugging Face paths and
# are not derivable from them: the Qwen id replaces the dot with an underscore and
# gpt-oss is lower case.
declare -A MODEL_API_IDS=(
  [Qwen3.8-27B]=Qwen3_8-27B
  [Nemotron-3-Nano-Omni-30B-A3B]=Nemotron-3-Nano-Omni-30B-A3B
  [GPT-OSS-120B]=gpt-oss-120b
)

declare -A MODEL_CONTEXT_LENGTHS=(
  [Qwen3.8-27B]=262144
  [Nemotron-3-Nano-Omni-30B-A3B]=131072
  [GPT-OSS-120B]=131072
)

# Reasoning effort has no shared vocabulary across these three, so it cannot live in
# COMMON. Qwen takes xhigh (its default) and its template raises on "high"; gpt-oss
# takes low/medium/high and interpolates whatever it is given straight into the
# harmony system block, so "xhigh" there is a string the model never saw in training;
# Nemotron ignores the argument entirely and always reasons.
declare -A MODEL_REASONING_EFFORT=(
  [Qwen3.8-27B]=xhigh
  [Nemotron-3-Nano-Omni-30B-A3B]=xhigh
  [GPT-OSS-120B]=high
)

# The current datasets use the target-specific opening instruction. Keep their
# results separate from completed runs generated with the earlier generic prompt.
# GPT-OSS-120B now renders its prompts with Harmony, as vLLM's chat endpoint does,
# so its runs are tagged apart from the earlier target_only runs, which used the
# chat template. The runners refuse to resume a file across that change.
RUN_TAG=target_only_harmony

# Comment out individual entries to exclude them from the sweep. math and python
# come first: the gap experiment's pools are screened from them.
CATEGORIES=(
  math
  python
  rlve
  rag
)

TOPOLOGIES=(
  single_target_baseline
  # three_independent_target
  # three_chain_target
  # six_independent_target
  # six_chain_target
  # four_way_fan_in_target
  # two_branch_recombine_target
)

COMMON=(
  --temperature 1.0
  --top_p 0.95
  --top_k 20
  --min_p 0.0
  --presence_penalty 0.0
  --repetition_penalty 1.0
  --preserve_thinking
  --max_tokens 16384
  --repair_turn_max_tokens 32768
  --tool_repair_attempts 5
  --final_answer_repair_attempts 20
  --cutoff_repair_attempts 20
  --max_agent_turns 2000
  --max_turns_without_new_answer 200
  --max_concurrent_samples 32
)

for MODEL in "${MODELS[@]}"; do
  MODEL_PATH="${MODEL_PATHS[$MODEL]}"
  API_MODEL="${MODEL_API_IDS[$MODEL]}"
  MAX_MODEL_LEN="${MODEL_CONTEXT_LENGTHS[$MODEL]}"
  REASONING_EFFORT="${MODEL_REASONING_EFFORT[$MODEL]}"
  RESULT_ROOT="results/exp1/$MODEL"
  echo "=== model $MODEL ($API_MODEL), max context $MAX_MODEL_LEN, reasoning $REASONING_EFFORT $(date -Is) ==="

  for category in "${CATEGORIES[@]}"; do
    DATA_ROOT="../generate/generated/exp1/$category"
    GRAPH_ROOT="../generate/topologies/$category/exp1"

    for topology in "${TOPOLOGIES[@]}"; do
      echo "=== $MODEL/$category/$topology $(date -Is) ==="
      python run_agent_api.py \
        --input "$DATA_ROOT/$topology/${topology}_dataset.json" \
        --output "$RESULT_ROOT/$category/agent_${topology}${RUN_TAG:+_$RUN_TAG}_${MODEL}.json" \
        --model "$MODEL_PATH" \
        --api_model "$API_MODEL" \
        --max_model_len "$MAX_MODEL_LEN" \
        --reasoning_effort "$REASONING_EFFORT" \
        "${COMMON[@]}"
      python grade_results.py \
        --input "$RESULT_ROOT/$category/agent_${topology}${RUN_TAG:+_$RUN_TAG}_${MODEL}.json" \
        --graph "$GRAPH_ROOT/$topology.json"
    done
  done
done

echo "All datasets finished at $(date -Is)"
