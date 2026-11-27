#!/bin/bash
#SBATCH --job-name=jev-lora-routing
#SBATCH --partition=rtx
#SBATCH --gres=gpu:1g.24gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=Armaan_R/slurm-%x-%j.out
#SBATCH --export=ALL,TMPDIR=/tmp

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
STAGE="${1:-}"
case "$STAGE" in
    traces|train|evaluate) shift ;;
    *) printf '%s\n' 'Usage: sbatch Armaan_R/run_lora_routing.sh {traces|train|evaluate} [--smoke]' >&2
       exit 2 ;;
esac

module load python
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" PYTHONFAULTHANDLER=1
export HF_HOME="${JEV_HF_CACHE_DIR:-/scratch/$USER/jev-efficiency-14610/hf}"
export JEV_HF_CACHE_DIR="$HF_HOME" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
REFERENCE="${JEV_REFERENCE_DIR:-/scratch/$USER/jev-frozen-routing-17248/output}"
OUTPUT="${JEV_OUTPUT_DIR:-/scratch/$USER/jev-lora-routing-17248}"
export MPLCONFIGDIR="/scratch/$USER/jev-lora-routing-$SLURM_JOB_ID/mpl"
mkdir -p "$OUTPUT" "$MPLCONFIGDIR"

# Each stage uses the same output directory. Avoid simultaneous writers.
exec 9>"$OUTPUT/.stage.lock"
if ! flock -n 9; then
    printf '%s\n' "Another stage is already using $OUTPUT" >&2
    exit 1
fi

srun --unbuffered "$HOME/envs/jev/bin/python" Armaan_R/run_lora_routing_experiment.py \
    "$STAGE" --reference-dir "$REFERENCE" --output-dir "$OUTPUT" "$@"

if [[ "$STAGE" == evaluate ]]; then
    FINAL="Armaan_R/jev_efficiency_results/$SLURM_JOB_ID"
    mkdir -p "$FINAL"
    cp "$OUTPUT/metrics/routing_results.json" "$OUTPUT/run_config.json" "$FINAL/"
fi
