# Efficiency experiment comparison

These are smoke-test results with 100 test examples (20 each for sentiment, inference, science, general knowledge, and held-out word math). They are exploratory and have substantial sampling uncertainty. Latency is single-run average seconds/question; generated tokens are average/question. A hybrid is a learned gate that answers fast first and escalates selected questions to reasoning.

| Variant | Fast accuracy | Always-reason accuracy | Learned-gate accuracy | Escalation | Gate latency | Gate tokens | Trainable parameters | Structural change |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Frozen Qwen3-1.7B (14979) | 55% | 69% | 63% | 24% | 0.892 s | 61.4 | 0 | None; reference model |
| Frozen Qwen3-0.6B (14979) | 39% | 51% | 40% | 3% | 0.118 s | 7.7 | 0 | Smaller pretrained model |
| LoRA all projections (14979) | 53% | 69% | 64% | 42% | 2.942 s | 107.5 | 17.43M | Adapters; backbone retained |
| LoRA attention (14979) | 54% | 67% | 57% | 13% | 0.834 s | 33.3 | 6.42M | Adapters; backbone retained |
| LoRA MLP (14979) | 53% | 69% | 57% | 17% | 0.982 s | 43.5 | 11.01M | Adapters; backbone retained |
| Top 25% layers fine-tuned (17131) | 54% | 69% | 61% | 25% | 0.916 s | 64.0 | 352.45M | Updates top 7/28 blocks; full backbone at inference |
| Prune 25% + LoRA (17131) | 26% | 27% | 30% | 17% | 0.881 s | 43.5 | 13.07M | Removes 7/28 blocks; 21 remain |

## Reading the results

The frozen 1.7B learned gate remains a useful reference: 63% accuracy at 24% escalation. LoRA all projections had the highest learned-gate accuracy in the earlier smoke run (64%), but escalated 42% and took 2.942 seconds/question. The 17131 top-layer fine-tune was correctly evaluated from its saved fine-tuned checkpoint; it reaches 61% at 25% escalation, close to the frozen reference, while updating about 20.5% of model parameters. It does not reduce model depth or total inference parameters.

Pruning did reduce depth from 28 to 21 blocks and total parameter count from 1.721B to 1.381B (about 19.7%). It also sharply reduced accuracy: 30% learned-gate accuracy, versus 61% for top25. Its measured latency was not lower in a reliable way (0.881 vs 0.916 seconds for the learned gate; always-reason latency was 5.109 vs 3.614 seconds). These timings are from one smoke run and are noisy, so they do not establish a speedup.

The 14662 top25 result must not be used as a valid fine-tuning result: a loader bug evaluated the frozen base model. Job 14979 also produced invalid top25 evaluation and failed before pruning due to a recipe-dispatch bug. Those implementation issues were fixed before 17131. The five unrelated variants from 14979 remain useful exploratory results; the top25/pruning rows should come from 17131.

## Project status and next work

The project tests whether a cascade can save reasoning cost and whether adapting or structurally shrinking Qwen preserves task quality. Current evidence supports the cascade as a quality/cost trade-off, while this particular 25% block-pruning recipe loses too much accuracy. All results are smoke tests, not final course claims. Next, run the corrected variants on the full prescribed data, use multiple seeds, compare less aggressive pruning (for example 10–15%), and repeat latency measurements under controlled load. Record the selected checkpoint and run config with every final claim.

Per-run detail and plots: [job 17131](jev_efficiency_results/17131/analysis.md). Earlier run audits: [14662](jev_efficiency_results/14662/analysis.md), [14979](jev_efficiency_results/14979/analysis.md).
