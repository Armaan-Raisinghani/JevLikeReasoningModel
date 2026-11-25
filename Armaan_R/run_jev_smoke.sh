#!/bin/bash
#SBATCH --job-name=jev-smoke
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
export PYTHONFAULTHANDLER=1
VARIANTS="${JEV_VARIANTS:-frozen_1_7b,frozen_0_6b,lora_all,lora_attention,lora_mlp,top25_finetune,pruned25_lora}"
LEARNING_RATE="${JEV_LEARNING_RATE:-5e-5}"
WORK="/scratch/$USER/jev-efficiency-$SLURM_JOB_ID"
OUTPUT="${JEV_OUTPUT_DIR:-$WORK/output}"
mkdir -p "$WORK/mpl" "$OUTPUT" /home/armaan.r.ug24/jev/logs

# Reuse the partial Qwen3-1.7B download from job 14610 if available. Keep the
# selected path fixed between jobs; scratch remains suitable for this cache.
HF_CACHE="${JEV_HF_CACHE_DIR:-/scratch/$USER/huggingface-cache}"
if [[ ! -d "$HF_CACHE" && -d "/scratch/$USER/jev-efficiency-14610/hf" ]]; then
  HF_CACHE="/scratch/$USER/jev-efficiency-14610/hf"
fi
mkdir -p "$HF_CACHE"
export JEV_HF_CACHE_DIR="$HF_CACHE" HF_HOME="$HF_CACHE"
export MPLCONFIGDIR="$WORK/mpl"
export TMPDIR=/tmp
# Xet range transfers failed partway through the 1.7B shard in prior jobs.
# Use the regular Hub HTTPS downloader and allow longer quiet periods.
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
# Discard only the failed Xet partial once. Subsequent HTTPS partials remain
# available for the Hub downloader to resume on the next job.
HTTP_CACHE_MARKER="$HF_CACHE/.jev_standard_http_initialized"
if [[ ! -f "$HTTP_CACHE_MARKER" ]]; then
  rm -f "$HF_CACHE"/hub/models--Qwen--Qwen3-1.7B/blobs/*.incomplete
  touch "$HTTP_CACHE_MARKER"
fi
srun --unbuffered python /home/armaan.r.ug24/jev/train_jev_efficiency_experiments.py \
  --smoke --variants "$VARIANTS" --learning-rate "$LEARNING_RATE" --output-dir "$OUTPUT"
FINAL="/home/armaan.r.ug24/jev/jev_efficiency_results/$SLURM_JOB_ID"
mkdir -p "$FINAL"
cp -a "$OUTPUT/metrics" "$OUTPUT/plots" "$OUTPUT/reports" "$FINAL/"
cp "$OUTPUT/run_config.json" "$FINAL/"

