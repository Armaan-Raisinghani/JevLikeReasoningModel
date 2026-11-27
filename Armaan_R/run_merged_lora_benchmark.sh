#!/bin/bash
#SBATCH --job-name=jev-merged-lora
#SBATCH --partition=rtx
#SBATCH --gres=gpu:1g.24gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=Armaan_R/slurm-%x-%j.out
#SBATCH --export=ALL,TMPDIR=/tmp

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"

module load python
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" PYTHONFAULTHANDLER=1
export HF_HOME="${JEV_HF_CACHE_DIR:-/scratch/$USER/jev-efficiency-14610/hf}"
export JEV_HF_CACHE_DIR="$HF_HOME" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

WORK="/scratch/$USER/jev-merged-lora-$SLURM_JOB_ID"
OUTPUT="${JEV_OUTPUT_DIR:-$WORK/output}"
REFERENCE="${JEV_REFERENCE_DIR:-/scratch/$USER/jev-frozen-routing-17248/output}"
LORA="${JEV_LORA_DIR:-/scratch/$USER/jev-lora-routing-17248}"
export MPLCONFIGDIR="$WORK/mpl"
mkdir -p "$MPLCONFIGDIR" "$OUTPUT"

exec 9>"$OUTPUT/.benchmark.lock"
flock -n 9
srun --unbuffered "$HOME/envs/jev/bin/python" Armaan_R/benchmark_merged_lora.py \
    --reference-dir "$REFERENCE" --lora-dir "$LORA" --output-dir "$OUTPUT" "$@"

FINAL="Armaan_R/jev_efficiency_results/$SLURM_JOB_ID"
mkdir -p "$FINAL"
cp "$OUTPUT/metrics/merged_lora_results.json" "$OUTPUT/run_config.json" "$FINAL/"
