#!/bin/bash
#SBATCH --job-name=jev-frozen-routing
#SBATCH --partition=rtx
#SBATCH --gres=gpu:1g.24gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=Armaan_R/slurm-%x-%j.out
#SBATCH --export=ALL,TMPDIR=/tmp

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
if [[ ! -f Armaan_R/evaluate_frozen_routing.py ]]; then
    printf '%s\n' 'Submit from the repository root: sbatch Armaan_R/run_frozen_routing.sh' >&2
    exit 1
fi

module load python
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONFAULTHANDLER=1
CACHE="${JEV_HF_CACHE_DIR:-${HF_HOME:-/scratch/$USER/huggingface-cache}}"
PREVIOUS_CACHE="/scratch/$USER/jev-efficiency-14610/hf"
MODEL_CACHE="hub/models--Qwen--Qwen3-1.7B"
# The generic cache can exist without the model. Reuse the cache from the
# successful smoke runs when no cache override was supplied.
if [[ -z "${JEV_HF_CACHE_DIR:-}${HF_HOME:-}" &&
      ! -f "$CACHE/$MODEL_CACHE/refs/main" &&
      -f "$PREVIOUS_CACHE/$MODEL_CACHE/refs/main" ]]; then
    CACHE="$PREVIOUS_CACHE"
fi
export JEV_HF_CACHE_DIR="$CACHE" HF_HOME="$CACHE"
if [[ -f "$CACHE/$MODEL_CACHE/refs/main" ]]; then
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
fi
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=600 HF_HUB_ETAG_TIMEOUT=60

WORK="/scratch/$USER/jev-frozen-routing-$SLURM_JOB_ID"
OUTPUT="${JEV_OUTPUT_DIR:-$WORK/output}"
export MPLCONFIGDIR="$WORK/mpl"
mkdir -p "$MPLCONFIGDIR" "$OUTPUT" "$HF_HOME"

srun --unbuffered "$HOME/envs/jev/bin/python" Armaan_R/evaluate_frozen_routing.py \
    --output-dir "$OUTPUT" "$@"
