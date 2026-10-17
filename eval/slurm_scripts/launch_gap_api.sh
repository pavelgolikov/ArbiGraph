#!/bin/bash
# Submits every prediction-gap run for the model in run_gap_api.sh, one Vector API job
# per category and topology. The calibrations, chains and braids start at once. Each
# fan-out/fan-in and shuffled-label braid waits for its category's chain or braid job to
# end, so no more than six jobs share the API at a time.
#
#   eval/slurm_scripts/launch_gap_api.sh             submit the jobs
#   eval/slurm_scripts/launch_gap_api.sh --dry-run   print what would be submitted
#
# Resubmitting after an interruption resumes every run from its checkpoints. Never run it
# while its jobs are still queued or running: two jobs would write the same results file.
#
# Stop the jobs by name, which also removes the waiting ones:
#   scancel -u "$USER" -n golikovp_arbigraph_gap_api
# A waiting job treats a cancelled job as ended and starts, so cancelling only the running
# jobs would start the fan-out/fan-in and shuffled-label runs in their place.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

DRY_RUN=
if [[ "${1:-}" == --dry-run ]]; then
  DRY_RUN=1
fi

# submit <category> <topology> [sbatch options...]; prints the job id
submit() {
  local id
  if [[ -n "$DRY_RUN" ]]; then
    echo "sbatch ${*:3} run_gap_api.sh $1 $2" >&2
    id="<$1/$2>"
  else
    id="$(sbatch --parsable "${@:3}" run_gap_api.sh "$1" "$2")"
    id="${id%%;*}"
    echo "$1/$2: job $id" >&2
  fi
  echo "$id"
}

for category in math python; do
  submit "$category" single_target_baseline > /dev/null
  chain="$(submit "$category" eighteen_chain_target)"
  braid="$(submit "$category" eighteen_braid_target)"
  submit "$category" eighteen_fan_out_fan_in --dependency="afterany:$chain" > /dev/null
  submit "$category" eighteen_braid_target_shuffled_labels --dependency="afterany:$braid" > /dev/null
done
echo "To stop them all, including the waiting ones: scancel -u \"\$USER\" -n golikovp_arbigraph_gap_api" >&2
