#!/bin/bash
#SBATCH --job-name=golikovp_arbigraph_branch_api
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=14:00:00
#SBATCH --output=logs/run_branch_pruning_api_%j.out
#SBATCH --error=logs/run_branch_pruning_api_%j.out
#SBATCH --account=aip-gpekhime

set -euo pipefail

export HF_HOME=/project/aip-gpekhime/golikovp/cache
export XDG_CACHE_HOME=/project/aip-gpekhime/golikovp/cache
export NLTK_DATA=/project/aip-gpekhime/golikovp/nltk_data
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="/tmp/triton_cache_${USER}"
export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions_${USER}"
export VLLM_CACHE_ROOT="/tmp/vllm_cache_${USER}"
export VLLM_CONFIG_ROOT="/tmp/vllm_config_${USER}"
export NCCL_IGNORE_DISABLED_P2P=1
mkdir -p "$HF_HOME"

module load python/3.11.5
module load cuda/12.9
module load cudnn
module load gcc opencv/4.13.0

cd /home/golikovp/projects/aip-gpekhime/golikovp/ArbiGraph-private/eval

category="${1:?usage: $0 <math|python> <peninsula|island> [GPT-OSS-120B|Qwen3.8-27B]}"
condition="${2:?usage: $0 <math|python> <peninsula|island> [GPT-OSS-120B|Qwen3.8-27B]}"
model="${3:-GPT-OSS-120B}"
case "$model" in
  GPT-OSS-120B)
    model_path=openai/gpt-oss-120b
    api_model=gpt-oss-120b
    context_length=131072
    reasoning_effort=high
    ;;
  Qwen3.8-27B)
    model_path=Qwen/Qwen3.8-27B
    api_model=Qwen3_8-27B
    context_length=262144
    reasoning_effort=xhigh
    ;;
  *)
    echo "Unsupported API model: $model" >&2
    exit 2
    ;;
esac
name="eighteen_braid_${condition}_target"
base="../generate/generated/branch_pruning/$model/$category/$name"
dataset="$base/${name}_dataset.json"
graph="$base/${name}_input_graph.json"
result="results/branch_pruning/$model/$category/agent_${name}_${model}.json"
[[ -f "$dataset" && -f "$graph" ]] || { echo "Missing dataset or graph for $category/$condition" >&2; exit 1; }

echo "=== $model/$category/$condition $(date -Is) ==="
python run_agent_api.py \
  --input "$dataset" \
  --output "$result" \
  --model "$model_path" \
  --api_model "$api_model" \
  --temperature 1.0 \
  --top_p 0.95 \
  --top_k 20 \
  --min_p 0.0 \
  --presence_penalty 0.0 \
  --repetition_penalty 1.0 \
  --reasoning_effort "$reasoning_effort" \
  --preserve_thinking \
  --max_model_len "$context_length" \
  --max_tokens 16384 \
  --repair_turn_max_tokens 32768 \
  --tool_repair_attempts 5 \
  --get_input_attempts_per_task 5 \
  --final_answer_repair_attempts 20 \
  --cutoff_repair_attempts 20 \
  --max_agent_turns 2000 \
  --max_turns_without_new_answer 200 \
  --max_concurrent_samples 32
python -B grade_results.py --input "$result" --graph "$graph"
echo "Finished $model/$category/$condition at $(date -Is)"
