#!/bin/bash
# Submit four independent four-H100 Qwen jobs and their local-only analysis.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

jobs=()
for category in math python; do
  for condition in peninsula island; do
    if [[ "${1:-}" == --dry-run ]]; then
      echo "sbatch run_branch_pruning_qwen_local.sh $category $condition"
    else
      id="$(sbatch --parsable run_branch_pruning_qwen_local.sh "$category" "$condition")"
      id="${id%%;*}"
      jobs+=("$id")
      echo "Qwen local $category/$condition: job $id"
    fi
  done
done

if [[ "${1:-}" == --dry-run ]]; then
  echo 'sbatch --dependency=afterok:<four-local-job-ids> run_branch_pruning_analysis.sh --models Qwen3.8-27B --results-root eval/results/branch_pruning_local --output eval/results/branch_pruning_local/analysis_qwen_local.json'
else
  dependency="$(IFS=:; echo "${jobs[*]}")"
  sbatch --dependency="afterok:$dependency" run_branch_pruning_analysis.sh \
    --models Qwen3.8-27B \
    --results-root eval/results/branch_pruning_local \
    --output eval/results/branch_pruning_local/analysis_qwen_local.json
fi
