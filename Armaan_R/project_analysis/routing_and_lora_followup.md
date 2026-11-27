# Qwen3 routing and LoRA follow-up

Merged all-projection LoRA with raw-confidence routing reached **70.0% accuracy at 1.015 s per question** on 500 balanced test questions. Frozen Qwen3-1.7B with the same type of gate reached **65.4% at 0.974 s** on the same questions and GPU. That is **23 additional correct answers**, about **41 ms** more mean latency, and nearly identical p95 latency. [Final same-GPU benchmark](../jev_efficiency_results/17764/merged_lora_results.json)

## Architecture

A single Qwen model serves both answer paths. It first scores the valid answer letters from the prompt. A gate can keep that fast answer or ask the same model to generate a reasoning trace and score the answer letters again. Fast-only always keeps the first answer. Always-reason runs the full reasoning path. Model adaptation changes the answer scores and traces. The gate changes which questions receive reasoning.

- **Raw-confidence gate.** Escalate when the highest fast-answer probability falls below a validation-selected threshold.
- **Calibrated-confidence gate.** Apply temperature scaling to the answer probabilities, then use the confidence threshold rule.
- **Learned rescue gate.** Fit logistic regression on fast probabilities and summaries of the final hidden state. Its positive target is a wrong fast answer that the reasoning path corrects.

LoRA keeps the original weights frozen in training and learns low-rank corrections to selected matrices. Direct-answer rows train the correct answer letter. Accepted teacher traces add language-model supervision for the reasoning text and train the final answer readout. [Inference and training code](../train_jev_efficiency_experiments.py) · [Gate evaluator](../evaluate_frozen_routing.py)

## Seven smoke variants

| Variant | Model change | Fast / reason / gated accuracy | Escalation | Mean gate latency | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| Frozen 1.7B | Original 28-block Qwen | 55% / 69% / 63% | 24% | 0.892 s | Strong baseline for accuracy and latency. |
| Frozen 0.6B | Smaller pretrained model | 39% / 51% / 40% | 3% | 0.118 s | Lowest measured latency, with a 23 percentage point gated drop from frozen 1.7B. |
| All-projection LoRA | Adapted attention and MLP matrices | 53% / 69% / 64% | 42% | 2.942 s | One additional correct answer over frozen at 3.3× the mean latency. |
| Attention-only LoRA | Adapted attention matrices | 54% / 67% / 57% | 13% | 0.834 s | Six points below frozen for a 58 ms lower mean. |
| MLP-only LoRA | Adapted feed-forward matrices | 53% / 69% / 57% | 17% | 0.982 s | Tied attention-only accuracy while escalating more. |
| Top-25% fine-tuning | Updated the top seven blocks and retained all 28 for inference | 54% / 69% / 61% | 25% | 0.916 s | No observed gain over the frozen reference despite training more weights. |
| Prune 25% + LoRA | Removed seven blocks and adapted the remaining 21 | 26% / 27% / 30% | 17% | 0.881 s | About 20% fewer parameters and a 31-point gated accuracy drop versus top-layer tuning in the same job. |

Each smoke test used 100 questions, so one answer moved the overall score by one point. The valid top-layer and pruning rows come from the corrected job 17131. Different test escalation rates also affect the latency comparison. The smoke tests made frozen 1.7B the baseline and left open whether larger-scale LoRA training would pay off. [Smoke comparison](../efficiency_variant_comparison.md)

## Follow-up experiments

Shreya's larger LoRA and validation-selected raw-confidence gate demonstrated a practical routing policy. Her reported test accuracy went from 67.84% fast-only to 71.57% gated and 74.77% always-reason. Her test had a different task mix, so the next experiments compared models on a shared split. [Shreya's results](../../Shreya/efficiency.json)

1. **Frozen Qwen with confidence routing.** Job 17248 used 400 validation and 500 test questions. Raw confidence improved accuracy from 57.6% fast-only to **65.4% gated** at **1.032 s per question** and **27.8% escalation**. Always-reason reached 70.8% at 3.679 s. The learned gate's budget point reached 65.8% at 1.519 s, two more correct answers for 0.487 s more mean latency. [Frozen metrics](../jev_efficiency_results/17248/routing_results.json)
2. **Larger LoRA with both gates.** Job 17562 evaluated an adapter trained for two epochs on **1,080 direct-answer examples and 1,500 accepted traces**. Both gates were refit on the adapter's validation outputs and evaluated on the same test questions. Fast-only improved to **62.6% (+5.0 points)** and always-reason to **71.8% (+1.0 point)**. The raw-confidence budget point reached **69.2% at 1.901 s**. The learned budget point reached 65.8% at 1.165 s with 18.8% escalation, but its p95 was 6.144 s. Unmerged LoRA took 6.501 s to always reason with the same 256 generated tokens as frozen. [LoRA metrics](../jev_efficiency_results/17562/routing_results.json) · [Training configuration](../jev_efficiency_results/17562/run_config.json)

## Same-GPU merge benchmark

Unmerged LoRA computes the original output and the correction separately in each adapted layer, producing `Wx + ΔWx`. Merging adds the trained correction to the weight once before inference, so the layer computes `(W + ΔW)x` with a matrix of the original shape. This is a post-training deployment step. The formulas agree in exact arithmetic, but BF16 rounding and autoregressive generation can change predictions. The two LoRA versions agreed on 98.8% of fast answers, 91.0% of always-reason answers, and 94.4% of gated answers.

| Model on the same 500 questions | Fast accuracy | Fast mean | Reason accuracy | Reason mean | Gated accuracy | Escalation | Gate tokens/question | Gate mean | Gate p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen | 57.6% | 16.9 ms | 70.8% | 3.486 s | 65.4% | 27.8% | 71.2 | 0.974 s | 3.474 s |
| Unmerged LoRA | 62.6% | 26.6 ms | 71.8% | 6.199 s | 69.2% | 28.8% | 73.7 | 1.786 s | 6.159 s |
| **Merged LoRA** | **62.0%** | **16.9 ms** | **73.8%** | **3.484 s** | **70.0%** | **29.0%** | **74.2** | **1.015 s** | **3.477 s** |

Merging reduced mean gated latency by **43%** and always-reason latency by **44%** versus unmerged. Both adapter versions used the same validation-selected raw-confidence threshold and routed almost the same fraction of questions. All three always-reason paths generated 256 tokens per question. That makes the speed recovery the clearest deployment finding.

Compared with frozen, merged LoRA improved fast-only by **4.4 percentage points**, always-reason by **3.0 points**, and gated accuracy by **4.6 points**. Its gate fixed 65 errors made by frozen and introduced 42, for 23 net gains. Word math had the largest gated gain, from 43% to 54%, aided by more escalation on that task. [Same-GPU metrics](../jev_efficiency_results/17764/merged_lora_results.json) · [Benchmark code](../benchmark_merged_lora.py)

## Comparison with equal task weights

The Armaan runs give each task 20% weight. Shreya's original test gives word math 11.3%. The equal-weight row averages the five task-level metrics and keeps her saved confidence threshold fixed.

| Raw-confidence system | Fast accuracy | Gated accuracy | Reason accuracy | Escalation | Gate tokens/question | Mean gate latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Shreya, original weighting | 67.84% | 71.57% | 74.77% | 17.01% | 43.6 | 1.214 s |
| **Shreya, equal task weights** | **63.52%** | **69.55%** | **73.43%** | **22.76%** | **58.3** | **1.612 s** |
| Armaan frozen Qwen | 57.60% | 65.40% | 70.80% | 27.80% | 71.2 | 0.974 s |
| Armaan unmerged LoRA | 62.60% | 69.20% | 71.80% | 28.80% | 73.7 | 1.786 s |
| **Armaan merged LoRA** | **62.00%** | **70.00%** | **73.80%** | **29.00%** | **74.2** | **1.015 s** |

With equal task weights, merged LoRA and Shreya's system are within half a point on gated and always-reason accuracy. Shreya retains a 1.52-point fast-only lead and escalates fewer questions. The merged run recorded a lower mean latency. Different test questions, independently fitted thresholds, and separate timing runs limit what can be inferred across these two evaluations. The same-GPU frozen, unmerged, and merged runs provide the direct comparison. [Shreya's per-task results](../../Shreya/final_table.csv) · [Same-GPU results](../jev_efficiency_results/17764/merged_lora_results.json)

## Presentation takeaway

Frozen Qwen was the latency reference. Larger-scale LoRA improved fast answers, and merging recovered the inference speed lost with a separate adapter. On the same 500 questions and GPU, merged LoRA with confidence routing improved accuracy from **65.4% to 70.0%** while staying at about **one second per question**.
