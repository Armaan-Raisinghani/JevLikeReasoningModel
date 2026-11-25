# Analysis: job 17131 (corrected top-layer fine-tune and pruning)

## Run status

Slurm reports **COMPLETED, exit 0**, runtime 54:43. This is the first run after fixing the two v1.5 evaluation/training dispatch bugs. Both selected variants were trained, evaluated, and included in the generated report. The run is still a smoke test: 20 test questions per task, 100 total.

Configuration: script 1.6, Qwen3-1.7B, BF16, learning rate `5e-5`, one epoch, seed 2026. It used Qwen3-1.7B as the smoke trace teacher. All reasoning generations hit the 256-token limit.

## Results

| Variant | Fast accuracy | Always-reason accuracy | Learned-gate accuracy | Escalation | Gate tokens/example | Gate mean latency | Trainable params | Total params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Top 25% fine-tune | 54% | 69% | 61% | 25% | 64.0 | 0.916 s | 352.45M | 1.721B |
| Prune 25% + LoRA | 26% | 27% | 30% | 17% | 43.5 | 0.881 s | 13.07M | 1.381B |

The top-layer fine-tune now loads its saved full checkpoint. It updates 352.45M parameters (20.5% of the original model), keeping the full 28-layer backbone for inference. Its test accuracy is close to the frozen 1.7B baseline from job 14979 (55% fast, 69% reasoning, 63% learned gate); this smoke sample shows no clear improvement from fine-tuning.

The pruning recipe removed 7 of the 28 transformer blocks, leaving 21, then trained LoRA adapters on the remaining model. Total parameters fell by about 19.7%, from 1.721B to 1.381B. The saved artifact was about 2.69 GiB, versus 3.30 GiB for the top-layer full checkpoint. Accuracy fell sharply: fast 26%, always reasoning 27%, and learned gate 30%. This is evidence that the current pruning recipe damages task performance at this setting; it is not a successful compression result yet.

## Efficiency interpretation

The pruned variant’s learned-gate average latency (0.881 s) is close to top25’s (0.916 s), but its accuracy is much lower. Its fast latency was 0.023 s versus 0.018 s for top25, and its always-reason latency was 5.109 s versus 3.614 s. These single-run timings do not show a speedup from pruning. Generation ran to the token cap for every reasoning example, and the benchmark has no repeated timing runs, so latency should be treated as noisy.

The learned gate escalated 17% of pruned-model examples and achieved 30% accuracy, compared with 25% escalation and 61% accuracy for top25. The confidence gate reached 29% on the pruned model at 22% escalation. Neither policy compensates for the large accuracy loss from pruning.

## Task pattern

| Task (20 examples) | Top25 fast / reason / learned gate | Pruned fast / reason / learned gate |
|---|---:|---:|
| Sentiment | 16 / 17 / 16 | 10 / 9 / 10 |
| Inference | 7 / 15 / 11 | 4 / 5 / 7 |
| Science questions | 16 / 17 / 16 | 4 / 3 / 5 |
| General knowledge | 8 / 11 / 8 | 4 / 4 / 4 |
| Word math (held out from training) | 7 / 9 / 10 | 4 / 6 / 4 |

Each count is correct answers out of 20. The pruning drop is broad, especially on science questions and inference. With only 20 examples per task, each item changes task accuracy by five points; these are preliminary results.

## What to conclude

- The corrected top25 run is a valid fine-tune evaluation. It updates roughly one-fifth of parameters but does not reduce inference depth or total parameter count. Performance is essentially similar to the frozen 1.7B smoke baseline.
- The corrected pruning run confirms a real depth reduction from 28 to 21 blocks and about a 20% reduction in total parameters. In this smoke run it loses too much accuracy and shows no measured latency improvement.
- Do not claim that pruning is effective yet. A follow-up should test less aggressive pruning (for example, 10–15%), compare several seeds, and use longer reasoning generation or a stop condition. Repeat latency measurements under controlled load before claiming a speed benefit.

## Artifacts

The generated report and machine-readable results are in this folder’s `reports/`, `metrics/`, and `plots/`. `run_config.json` records the settings. Per-example predictions and model checkpoints remain in scratch unless separately archived.
