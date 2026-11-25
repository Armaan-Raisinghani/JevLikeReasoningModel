# Jev efficiency experiments

Focused working repository for the Qwen fast/reasoning cascade experiments.

- `train_jev_efficiency_experiments.py`: experiment pipeline (v1.6).
- `run_jev_smoke.sh`, `run_jev_full.sh`: Slurm launchers.
- `efficiency_variant_comparison.md`: consolidated smoke-test comparison.
- `conversation_handoff.md`: project explanation, history, run validity, and next steps.
- `jev_efficiency_results/14662/`, `14979/`, `17131/`: run audits and reproducibility summaries. Job 17131 includes metrics, plots, and the generated report.

Model weights, datasets, caches, checkpoints, and raw prediction archives are not included. The original cluster working tree in `~/jev` is retained; this folder contains the Git-tracked handoff and selected artifacts.
