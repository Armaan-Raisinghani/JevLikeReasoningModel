# Efficiency experiment comparison

Results below are from the 100-question smoke test (20 questions per task). Latency is average seconds per question; tokens are average generated tokens per question. The hybrid is the learned gate that starts with the fast answer and escalates selected questions to reasoning.

| Model variant | Fast: accuracy / latency / tokens | Always reason: accuracy / latency / tokens | Hybrid: accuracy / escalation / latency / tokens | Impact |
|---|---:|---:|---:|---|
| **Frozen Qwen3-1.7B** | 55% / 0.018 s / 0 | 69% / 3.651 s / 256 | 63% / 24% / 0.892 s / 61.4 | **+8 percentage points** accuracy vs. fast; **−2.759 s (76%)** latency vs. always reasoning; **−76%** tokens vs. always reasoning. |
| **Frozen Qwen3-0.6B** | 39% / 0.015 s / 0 | 51% / 3.489 s / 256 | 40% / 3% / 0.118 s / 7.7 | **+1 point** accuracy vs. fast; **−3.371 s (97%)** latency vs. always reasoning; **−97%** tokens vs. always reasoning. |
| **LoRA on all projections** | 53% / 0.031 s / 0 | 69% / 6.792 s / 256 | 64% / 42% / 2.942 s / 107.5 | **+11 points** accuracy vs. fast; **−3.850 s (57%)** latency vs. always reasoning; **−58%** tokens vs. always reasoning. |
| **LoRA on attention** | 54% / 0.027 s / 0 | 67% / 6.229 s / 256 | 57% / 13% / 0.834 s / 33.3 | **+3 points** accuracy vs. fast; **−5.395 s (87%)** latency vs. always reasoning; **−87%** tokens vs. always reasoning. |
| **LoRA on MLP** | 53% / 0.025 s / 0 | 69% / 5.651 s / 256 | 57% / 17% / 0.982 s / 43.5 | **+4 points** accuracy vs. fast; **−4.669 s (83%)** latency vs. always reasoning; **−83%** tokens vs. always reasoning. |
| **Fine-tune top 25% of layers** | Pending | Pending | Pending | Corrected evaluation is running; the earlier result evaluated the frozen base model. |
| **Prune 25% of layers + LoRA** | Pending | Pending | Pending | Awaiting completed run. |

## Best results so far

- **Best overall balance:** Frozen Qwen3-1.7B with the learned hybrid gate: 63% accuracy, 24% escalation, and 0.892 s average latency. It improves accuracy by 8 points over fast answers while reducing latency by 76% compared with always reasoning.
- **Highest completed hybrid accuracy:** LoRA on all projections at 64%, but it escalates 42% of questions and averages 2.942 s per question.
- **Fastest hybrid:** Frozen Qwen3-0.6B at 0.118 s, but its accuracy is only 40%, just 1 point above its fast path.

The impact values compare hybrid accuracy against fast accuracy, and hybrid latency/token use against always reasoning. These are preliminary smoke-test measurements on only 100 questions; the layer fine-tuning and pruning comparisons are not yet available.
