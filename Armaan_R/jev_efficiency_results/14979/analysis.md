# Analysis: job 14979 (partial seven-variant smoke run)

## Run status and data integrity

Slurm recorded job 14979 as **FAILED, exit 1:0, elapsed 1:27:25**. It did not time out or hit the system-memory limit. The script raised `ValueError: Unsupported training recipe: pruned25_lora` after processing the other variants.

Five results are valid: `frozen_1_7b`, `frozen_0_6b`, `lora_all`, `lora_attention`, and `lora_mlp`. The `top25_finetune` training step ran, but its evaluation is invalid: the code saved a full fine-tuned model, then the evaluation loader only knew how to attach PEFT adapters. It therefore loaded the original frozen 1.7B model. Its reported predictions and gate validation values duplicate `frozen_1_7b`. The `pruned25_lora` recipe never trained or produced results.

The script exited before writing its aggregate report. I reconstructed the partial metrics from each completed variant's saved validation/test predictions. These are in `metrics/results_partial.json` and `.csv`. The original `metrics/results.json` still in scratch is from job 14662 and must not be read as job 14979's aggregate.

## Overall comparison (100 test examples)

| Variant | Fast accuracy | Always-reason accuracy | Learned-gate accuracy | Escalation | Gate mean latency | Gate tokens/example | Trainable parameters |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen Qwen3-1.7B | 55% | 69% | 63% | 24% | 0.892 s | 61.4 | 0 |
| Frozen Qwen3-0.6B | 39% | 51% | 40% | 3% | 0.118 s | 7.7 | 0 |
| LoRA all projections | 53% | 69% | 64% | 42% | 2.942 s | 107.5 | 17.43M |
| LoRA attention only | 54% | 67% | 57% | 13% | 0.834 s | 33.3 | 6.42M |
| LoRA MLP only | 53% | 69% | 57% | 17% | 0.982 s | 43.5 | 11.01M |
| Top 25% fine-tune | **Invalid evaluation** | **Invalid evaluation** | **Invalid evaluation** | — | — | — | 352.45M trained |
| Prune 25% + LoRA | No result | No result | No result | — | — | — | — |

Every row uses the same 100-example smoke test, 20 examples per task. These are exploratory point estimates; one example changes overall accuracy by one percentage point. Reasoning generated the full 256-token limit for every example, so reasoning results may reflect truncated traces.

## Interpretation

- On frozen 1.7B, the learned gate gained 8 points over fast accuracy and escalated 24% of requests. It is a clear quality/cost trade-off on this sample, though reasoning hit the token cap.
- The 0.6B model is much faster, but its fast accuracy was 39%, versus 55% for frozen 1.7B. Its learned gate escalated only 3% and gained one point, so it did not recover the larger model's quality here.
- Among valid LoRA runs, `lora_all` had the highest learned-gate accuracy (64%) but escalated 42% and averaged 2.94 seconds. Attention-only and MLP-only gates were faster, but both reached 57% accuracy. This suggests a compute/quality trade-off, not a decisive winning adapter target.
- LoRA trained 6.42M–17.43M parameters, roughly 0.37%–1.01% of the 1.72B base. The full base model still ran at inference; LoRA reduced trainable parameters and adapter size, not backbone compute.
- The confidence gate escalated 57%–89% for the 1.7B/LoRA variants, so it generally saved less compute than the learned gate.
- Training logs show finite losses and no `NaN` gradient norms in the completed LoRA variants under BF16 and `5e-5`. Large logged norms are pre-clipping values; training used `max_grad_norm=1.0`.

## Per-task accuracy counts

Each cell is correct answers out of 20, shown as `fast / always reason / learned gate`.

| Task | Frozen 1.7B | Frozen 0.6B | LoRA all | LoRA attention | LoRA MLP |
|---|---:|---:|---:|---:|---:|
| Sentiment | 18 / 18 / 18 | 14 / 16 / 14 | 18 / 18 / 16 | 16 / 18 / 16 | 15 / 18 / 15 |
| Inference | 11 / 13 / 13 | 3 / 8 / 3 | 6 / 11 / 12 | 10 / 11 / 13 | 8 / 11 / 11 |
| Science questions | 15 / 18 / 16 | 11 / 15 / 12 | 17 / 18 / 18 | 16 / 17 / 16 | 17 / 18 / 17 |
| General knowledge | 6 / 12 / 6 | 8 / 7 / 8 | 6 / 11 / 9 | 6 / 11 / 6 | 7 / 11 / 8 |
| Word math (held out from training) | 5 / 8 / 10 | 3 / 5 / 3 | 6 / 11 / 9 | 6 / 10 / 6 | 6 / 11 / 6 |

Per-task values are noisy at n=20. For example, the frozen-1.7B gate's word-math result is 10/20, but it comes from only 20 held-out questions.

## Core conclusion for the project

This run gives useful smoke evidence about the fast/reasoning cascade and LoRA target choices. It does **not** answer the main structural-compression question: the true layer-pruned model never ran. The top-layer fine-tune also lacks valid evaluation due to the loading bug. Consequently, do not present job 14979 as evidence that layer pruning or top-layer fine-tuning preserves accuracy.

The code has now been fixed for a future run: full fine-tuned checkpoints are loaded for `top25_finetune`, and `pruned25_lora` is accepted by the LoRA training path. The fix does not change this completed job's results. A new run is required for valid results from those two variants.

## Reproducibility files

- `run_config.json`: settings and selected variants.
- `logs/jev-smoke-14979.out` and `.err`: original Slurm logs and failure traceback.
- `metrics/results_partial.json` and `.csv`: recovered six-variant output; top25 is explicitly invalid and pruning is absent.
- `raw_artifacts.tar.gz`: splits, teacher traces, per-example validation/test predictions, calibration/gate thresholds, and manifests for recovery/reanalysis.
