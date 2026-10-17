#!/bin/bash
#SBATCH --job-name=golikovp_arbigraph_branch_gemma
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=logs/run_branch_pruning_gemma_%j.out
#SBATCH --error=logs/run_branch_pruning_gemma_%j.out
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

model=gemma-4-31B-it
categories=(math python)
conditions=(peninsula island)
if (( $# >= 1 )); then categories=("$1"); fi
if (( $# >= 2 )); then conditions=("${@:2}"); fi

common=(
  --model google/gemma-4-31B-it
  --num_gpus 4
  --temperature 1.0
  --top_p 0.95
  --top_k 64
  --min_p 0.0
  --presence_penalty 0.0
  --repetition_penalty 1.0
  --reasoning_effort unused
  --enable_thinking
  --preserve_thinking
  --max_model_len 262144
  --max_tokens 16384
  --repair_turn_max_tokens 32768
  --tool_repair_attempts 5
  --get_input_attempts_per_task 5
  --final_answer_repair_attempts 20
  --cutoff_repair_attempts 20
  --max_agent_turns 2000
  --max_turns_without_new_answer 200
  --max_concurrent_samples 32
)

for category in "${categories[@]}"; do
  for condition in "${conditions[@]}"; do
    name="eighteen_braid_${condition}_target"
    base="../generate/generated/branch_pruning/$model/$category/$name"
    dataset="$base/${name}_dataset.json"
    graph="$base/${name}_input_graph.json"
    result="results/branch_pruning/$model/$category/agent_${name}_${model}.json"
    [[ -f "$dataset" && -f "$graph" ]] || { echo "Missing dataset or graph for $category/$condition" >&2; exit 1; }
    echo "=== $model/$category/$condition $(date -Is) ==="
    python run_agent.py --input "$dataset" --output "$result" "${common[@]}"
    python -B grade_results.py --input "$result" --graph "$graph"
  done
done
echo "All Gemma branch-pruning runs finished at $(date -Is)"
