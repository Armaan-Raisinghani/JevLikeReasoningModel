# Project conversation handoff

This note consolidates the important decisions, implementation history, results, and next steps from the project conversation. It is a research handoff, not a verbatim chat transcript.

## Project in plain language

The experiment uses Qwen3 as a two-stage answer system. A small/fast path answers every question. A learned gate decides whether that answer is good enough or whether to escalate the question to a longer reasoning generation. The project compares task accuracy against generated tokens, latency, model size, and training cost. It also tests whether adapting a subset of the model or removing transformer blocks can reduce cost while preserving quality.

## Architecture and experiment variants

- **Fast path:** classification-style answer from the model without a long reasoning trace.
- **Reasoning path:** generate a longer answer/trace (capped at 256 new tokens in these runs), then map it to the task answer.
- **Learned gate:** uses fast-path features/confidence to predict whether to escalate. Its operating point is chosen using validation data; reported test results use the fixed threshold.
- **Confidence gate:** a separate confidence-based baseline.
- **Frozen baselines:** no backbone training.
- **LoRA:** small low-rank adapters are trained on all projections, attention only, or MLP only; the original model backbone remains at inference.
- **Top25 fine-tune:** unfreezes and updates the top seven of 28 transformer blocks; it still retains all 28 blocks at inference.
- **Prune25+LoRA:** physically removes seven of 28 blocks, leaving 21, then adapts the remaining model with LoRA. This is actual depth reduction.

So LoRA and top-layer fine-tuning reduce the number of trainable weights compared with full fine-tuning, but do not themselves shrink inference depth. The pruning experiment changes architecture and total parameter count.

## Timeline, bugs, and run validity

- **Job 14662:** early smoke run of top25. It trained, but a checkpoint-loading bug evaluated the frozen base instead of the fine-tuned checkpoint. Its published top25 metric is invalid for fine-tuning. The audit is in `jev_efficiency_results/14662/analysis.md`.
- **Jobs 14615/14617 and nearby retries:** early attempts exposed incomplete Hugging Face checkpoint downloads and an environment mismatch. The user verified that compute-node internet worked and manually downloaded the missing shard into a persistent scratch HF cache. The functioning environment was `~/envs/jev` (Python 3.13.5, torch 2.13.0+cu130), not the broken `~/jev-venv` Python 3.9 environment.
- **Job 14979:** five variants completed with usable results. Top25 was again evaluated incorrectly because the full-checkpoint loader was missing; pruning did not run because `pruned25_lora` was not routed to the LoRA training path. The job failed. Its audit documents valid versus invalid results.
- **Script v1.6 / job 17131:** fixed full-checkpoint loading for top25 and routed `pruned25_lora` through LoRA training. Job completed (Slurm exit 0, 54:43). Both variants were trained and evaluated. This is the valid source for top25 and pruning results.

The earlier download problem was not simply “no internet”: DNS/HTTPS to Hugging Face worked, but large-file download/cache behavior and transient scratch write issues caused incomplete shards. A consistent `HF_HOME` on scratch, `HF_HUB_DISABLE_XET=1`, and manual `hf_hub_download` from an allocated compute node resolved model loading.

## Job 17131 headline results

This was still a smoke test: 20 test examples per task, 100 total, one seed, one epoch, BF16, learning rate 5e-5, max reasoning 256 tokens. The smoke teacher actually used was Qwen3-1.7B, recorded in the run config.

| Variant | Fast accuracy | Reasoning accuracy | Learned gate | Escalation | Gate latency | Total parameters |
|---|---:|---:|---:|---:|---:|---:|
| Top25 fine-tune | 54% | 69% | 61% | 25% | 0.916 s | 1.721B |
| Prune25 + LoRA | 26% | 27% | 30% | 17% | 0.881 s | 1.381B |

Top25 trained 352.45M parameters (about 20.5% of the original model) and kept all 28 blocks for inference. Prune25 removed seven blocks, reducing total parameters by about 19.7%, but lost substantial accuracy. Its timing did not demonstrate a robust speedup. Reasoning generations hit the 256-token cap, which limits interpretation of the reasoning path. The 100-example test set is too small for strong claims.

Across 14979 and 17131, the frozen 1.7B learned-gate baseline was 63% accuracy at 24% escalation. LoRA-all reached 64% at 42% escalation (14979). These are exploratory, cross-variant smoke comparisons, not controlled final conclusions.

## Metrics and interpretation cautions

Runs log accuracy, macro-F1, Brier score, NLL, ECE, escalation rate, average generated tokens, mean/p50/p95 latency, trainable/total parameters, and checkpoint/artifact size. Gate thresholds are chosen from validation data. Latency is measured within a single run; GPU load and generation behavior vary. Always-reason generations hit the token limit, so compare cautiously. Per-task smoke test has n=20, meaning one answer changes task accuracy by five percentage points.

Do not conflate:
- fewer trainable parameters (LoRA/top-layer tuning) with smaller inference compute;
- fewer total parameters / fewer blocks (pruning) with demonstrated wall-clock speedup;
- the invalid 14662 or 14979 top25 metric with the corrected 17131 top25 metric;
- smoke evidence with statistically reliable full-run results.

## Files and artifacts

- `train_jev_efficiency_experiments.py`: v1.6 experiment pipeline.
- `run_jev_smoke.sh`, `run_jev_full.sh`: Slurm launchers.
- `efficiency_variant_comparison.md`: consolidated comparison.
- `jev_efficiency_results/14662/`, `14979/`, `17131/`: run audits/configuration; job 17131 includes machine-readable metrics, report, and plots.
- Large checkpoints and raw prediction archives remain in scratch and are intentionally not copied into this Git repository.

The working copy originally used on the cluster is `~/jev`. This handoff and selected reproducible outputs are copied into the Git-tracked `~/jev-project-repo/Armaan_R` folder. The source working copy is retained so existing Slurm paths remain valid.

## Next steps

1. Run the corrected full experiment (not smoke) with enough wall time and storage.
2. Use multiple seeds and report uncertainty intervals.
3. Test smaller pruning amounts and/or better layer-selection strategies; compare accuracy at matched compute/cost.
4. Consider a shorter reasoning cap or learned stop criterion, since every reasoning generation hit 256 tokens.
5. Repeat latency benchmarks under controlled GPU conditions and report throughput plus peak GPU memory.
6. Keep each run’s config, logs, predictions, metrics, checkpoint identity, and code version together; copy final useful artifacts from scratch back to backed-up home storage.
