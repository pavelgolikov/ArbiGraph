#!/bin/bash
#SBATCH --job-name=golikovp_arbigraph_branch_qwen_recover
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0:10:00
#SBATCH --chdir=/project/6107991/golikovp/ArbiGraph-private/eval/slurm_scripts
#SBATCH --output=logs/recover_branch_pruning_qwen_local_%j.out
#SBATCH --error=logs/recover_branch_pruning_qwen_local_%j.out
#SBATCH --account=aip-gpekhime

set -euo pipefail
module load python/3.11.5
cd /project/6107991/golikovp/ArbiGraph-private/eval/slurm_scripts

# Each GPU attempt retains the two-hour limit in run_branch_pruning_qwen_local.sh.
# This lightweight dependent job submits only conditions whose graded result is
# still incomplete, then checks once more after those attempts finish.
rounds_left="${1:-2}"
[[ "$rounds_left" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid rounds_left: $rounds_left" >&2; exit 2; }

jobs=()
for category in math python; do
  for condition in peninsula island; do
    result="../results/branch_pruning_local/Qwen3.8-27B/$category/agent_eighteen_braid_${condition}_target_Qwen3.8-27B.json"
    if python - "$result" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    sys.exit(1)
data = json.loads(path.read_text())
sys.exit(0 if data["samples"] and all(
    sample.get("agent_status") not in (None, "in_progress")
    and sample.get("grading") is not None
    for sample in data["samples"]
) else 1)
PY
    then
      echo "Complete: $category/$condition"
    else
      id="$(sbatch --parsable run_branch_pruning_qwen_local.sh "$category" "$condition")"
      id="${id%%;*}"
      jobs+=("$id")
      echo "Resuming $category/$condition: job $id"
    fi
  done
done

if (( ${#jobs[@]} == 0 )); then
  echo "All local Qwen conditions are graded."
  exit 0
fi

dependency="$(IFS=:; echo "${jobs[*]}")"
sbatch --dependency="afterok:$dependency" run_branch_pruning_analysis.sh \
  --models Qwen3.8-27B \
  --results-root eval/results/branch_pruning_local \
  --output eval/results/branch_pruning_local/analysis_qwen_local.json

if (( rounds_left > 1 )); then
  sbatch --dependency="afterany:$dependency" recover_branch_pruning_qwen_local.sh "$((rounds_left - 1))"
fi
