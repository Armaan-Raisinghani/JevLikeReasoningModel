#!/bin/bash
# Full-data version of run_jev_smoke.sh. The course QoS caps jobs at 4 hours.
# All variants run in one job so their final comparison table is produced together.
#SBATCH --job-name=jev-full
#SBATCH --partition=rtx
#SBATCH --gres=gpu:1g.24gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --export=ALL,TMPDIR=/tmp
#SBATCH --output=/home/armaan.r.ug24/jev/logs/%x-%j.out
#SBATCH --error=/home/armaan.r.ug24/jev/logs/%x-%j.err
set -euo pipefail
module load python
source /home/armaan.r.ug24/envs/jev/bin/activate
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

WORK="/scratch/$USER/jev-efficiency-$SLURM_JOB_ID"
mkdir -p "$WORK/mpl" "$WORK/output" /home/armaan.r.ug24/jev/logs

# Reuse the cache populated by the smoke/debug runs across job ids.
HF_CACHE="${JEV_HF_CACHE_DIR:-/scratch/$USER/huggingface-cache}"
if [[ ! -d "$HF_CACHE" && -d "/scratch/$USER/jev-efficiency-14610/hf" ]]; then
  HF_CACHE="/scratch/$USER/jev-efficiency-14610/hf"
fi
mkdir -p "$HF_CACHE"
export JEV_HF_CACHE_DIR="$HF_CACHE" HF_HOME="$HF_CACHE"
export MPLCONFIGDIR="$WORK/mpl" TMPDIR=/tmp
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=600 HF_HUB_ETAG_TIMEOUT=60

HTTP_CACHE_MARKER="$HF_CACHE/.jev_standard_http_initialized"
if [[ ! -f "$HTTP_CACHE_MARKER" ]]; then
  rm -f "$HF_CACHE"/hub/models--Qwen--Qwen3-1.7B/blobs/*.incomplete
  touch "$HTTP_CACHE_MARKER"
fi

srun python /home/armaan.r.ug24/jev/train_jev_efficiency_experiments.py \
  --full --output-dir "$WORK/output"

FINAL="/home/armaan.r.ug24/jev/jev_efficiency_results/$SLURM_JOB_ID"
mkdir -p "$FINAL"
cp -a "$WORK/output/metrics" "$WORK/output/plots" "$WORK/output/reports" "$FINAL/"
cp "$WORK/output/run_config.json" "$FINAL/"
