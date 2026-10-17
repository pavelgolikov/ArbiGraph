#!/bin/bash
#SBATCH --job-name=golikovp_arbigraph_branch_analysis
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/run_branch_pruning_analysis_%j.out
#SBATCH --error=logs/run_branch_pruning_analysis_%j.out
#SBATCH --account=aip-gpekhime

set -euo pipefail
module load python/3.11.5
cd /home/golikovp/projects/aip-gpekhime/golikovp/ArbiGraph-private
python -B eval/analyze_branch_pruning.py "$@"
