#!/bin/bash
# Submit Qwen math/Python peninsula/island API runs, each to its own result file.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

for category in math python; do
  for condition in peninsula island; do
    if [[ "${1:-}" == --dry-run ]]; then
      echo "sbatch run_branch_pruning_api.sh $category $condition Qwen3.8-27B"
    else
      id="$(sbatch --parsable run_branch_pruning_api.sh "$category" "$condition" Qwen3.8-27B)"
      echo "Qwen $category/$condition: job ${id%%;*}"
    fi
  done
done
