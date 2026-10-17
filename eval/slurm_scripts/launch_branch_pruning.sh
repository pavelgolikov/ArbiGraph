#!/bin/bash
# Submit Gemma locally and four independent GPT-OSS API runs.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

if [[ "${1:-}" == --dry-run ]]; then
  echo 'gemma=$(sbatch --parsable run_branch_pruning_local.sh)'
  for category in math python; do
    for condition in peninsula island; do
      echo "sbatch run_branch_pruning_api.sh $category $condition"
    done
  done
  echo 'sbatch --dependency=afterok:<all-five-job-ids> run_branch_pruning_analysis.sh'
  exit 0
fi

gemma="$(sbatch --parsable run_branch_pruning_local.sh)"
jobs=("${gemma%%;*}")
echo "Gemma: job ${jobs[0]}"
for category in math python; do
  for condition in peninsula island; do
    id="$(sbatch --parsable run_branch_pruning_api.sh "$category" "$condition")"
    id="${id%%;*}"
    jobs+=("$id")
    echo "GPT-OSS $category/$condition: job $id"
  done
done
dependency="$(IFS=:; echo "${jobs[*]}")"
sbatch --dependency="afterok:$dependency" run_branch_pruning_analysis.sh
