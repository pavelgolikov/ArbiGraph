#!/bin/bash
#SBATCH --job-name=golikovp_rlvr_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=4:00:00
#SBATCH --output=eval/logs/run_rlvr_smoke_%j.out
#SBATCH --error=eval/logs/run_rlvr_smoke_%j.out
#SBATCH --account=aip-gpekhime

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root with sbatch}"

export HF_HOME=/project/aip-gpekhime/golikovp/cache
export XDG_CACHE_HOME=/project/aip-gpekhime/golikovp/cache
export PYTHONDONTWRITEBYTECODE=1
export TRITON_CACHE_DIR="/tmp/triton_cache_${USER}"
export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions_${USER}"
export NCCL_IGNORE_DISABLED_P2P=1
export OMP_NUM_THREADS=4
export HF_HUB_DISABLE_TELEMETRY=1

module load python/3.11.5 cuda/12.9 cudnn
module load gcc opencv/4.13.0

exec python -m torch.distributed.run --standalone --nproc_per_node=4 \
    rlvr/full_train/train.py "$@" --max-steps 2
