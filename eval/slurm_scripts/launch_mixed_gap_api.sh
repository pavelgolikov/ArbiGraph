#!/bin/bash
# Submit one independent Vector API job per model and mixed topology.
#
#   eval/slurm_scripts/launch_mixed_gap_api.sh             submit the jobs
#   eval/slurm_scripts/launch_mixed_gap_api.sh --dry-run   print the submissions
#
# Resubmitting after an interruption resumes from checkpoints. Never run this while
# its jobs are queued or running: two jobs would write the same results file.
# Stop all jobs by name:
#   scancel -u "$USER" -n golikovp_arbigraph_mixed_gap_api
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

DRY_RUN=
if [[ "${1:-}" == --dry-run ]]; then
  DRY_RUN=1
fi

# submit <model> <category> <stage>
submit() {
  local id
  if [[ -n "$DRY_RUN" ]]; then
    echo "sbatch run_mixed_gap_api.sh $1 $2 $3"
  else
    id="$(sbatch --parsable run_mixed_gap_api.sh "$1" "$2" "$3")"
    id="${id%%;*}"
    echo "$1/$2/$3: job $id"
  fi
}

for model in Qwen3.8-27B GPT-OSS-120B; do
  submit "$model" mixed eighteen_chain_target
  submit "$model" mixed eighteen_braid_target
  submit "$model" mixed eighteen_fan_out_fan_in
done
echo "To stop them all: scancel -u \"\$USER\" -n golikovp_arbigraph_mixed_gap_api" >&2
