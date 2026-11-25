# Analysis: job 14662 (`top25_finetune`)

## Executive read

This was a **smoke run**, not the final comparison. It trained one top-quarter fine-tune on 100 examples (20 each from five test tasks), but a model-loading bug meant the evaluation reloaded the original frozen Qwen3-1.7B instead of the saved fine-tuned checkpoint. The reported fast/reasoning/gate scores therefore describe the frozen base model, not the trained top-quarter model. The training itself updated 352.4M parameters and saved a full checkpoint, but this job does not provide valid performance metrics for that fine-tune.

## Results

| Mode | Accuracy | 95% Wilson interval | Mean latency / example | Mean generated tokens | Escalation |
|---|---:|---:|---:|---:|---:|
| Fast | 55/100 = 55% | 45.2–64.4% | 0.018 s | 0 | 0% |
| Always reason | 69/100 = 69% | 59.4–77.2% | 3.756 s | 256 | 100% |
| Learned gate | 63/100 = 63% | 53.2–71.8% | 0.920 s | 61.4 | 24% |
| Confidence gate | 69/100 = 69% | 59.4–77.2% | 3.349 s | 227.8 | 89% |

For the **frozen base model actually evaluated**, paired predictions show 20 examples that reasoning got right and fast got wrong, versus 6 examples where reasoning got wrong and fast got right. The net gain is 14 correct answers; an exploratory exact McNemar test gives p=0.009. These intervals and the test are descriptive for this small smoke sample, not a substitute for a larger evaluation.

For that frozen base model, the learned gate trades 6 percentage points of accuracy against always reasoning for a 76% reduction in generated tokens and about a 4.1× reduction in mean latency. Its p50 latency is close to the fast path (0.017 s), while its p95 is 3.81 s because escalated examples still incur reasoning latency. The confidence gate gets the same accuracy as always reasoning, but escalates 89% of examples and therefore saves little compute.

## Task-level pattern

| Task (20 examples) | Fast | Always reason | Learned gate | What it suggests |
|---|---:|---:|---:|---|
| Sentiment | 18/20 (90%) | 18/20 (90%) | 18/20 (90%) | Fast is sufficient on this sample; the learned gate escalated none. |
| Inference | 11/20 (55%) | 13/20 (65%) | 13/20 (65%) | Reasoning recovered two answers; the gate escalated 50%. |
| Science questions | 15/20 (75%) | 18/20 (90%) | 16/20 (80%) | Reasoning helped; the gate captured one extra correct answer at 15% escalation. |
| General knowledge | 6/20 (30%) | 12/20 (60%) | 6/20 (30%) | Reasoning helped substantially, but this gate did not route useful examples well. |
| Word math | 5/20 (25%) | 8/20 (40%) | 10/20 (50%) | This task was held out from training the model/gate; its apparent gate gain is promising but based on only 20 examples. |

For the frozen base model, the learned gate is not consistently effective across tasks. In particular, its overall gain hides a failure to benefit general-knowledge examples. A task-aware or better-trained gate could be worth investigating, but this result alone does not establish that it would generalize.

## What “top 25% fine-tune” means here

The training recipe froze the lower transformer blocks and trained the top blocks plus normalization and language-model-head parameters. The base model has 1,720,574,976 parameters; 352,445,440 were trainable (20.5%). It did **not** remove layers. The saved full checkpoint is about 3.22 GiB (3.46 GB decimal). However, evaluation loaded the unchanged base checkpoint: `load_variant_model` only attached PEFT adapters, and the full fine-tune has no `adapter_config.json`. A code fix is needed before the fine-tune can be evaluated correctly.

The smoke config used Qwen3-1.7B, BF16, learning rate 5e-5, one epoch, 20 training examples per seen task, and 31 accepted teacher traces from 40 candidates. The reported 15.5 seconds of training and memory numbers are specific to this small smoke setup and should not be extrapolated to the full course experiment.

## Limitations to disclose

- Only 100 test examples were used; each task has 20, so per-task accuracy moves in 5-point steps and uncertainty is large.
- Every reasoning example used the full 256-token generation cap. The reasoning accuracy may reflect truncated answers; rerun with a larger cap or a stop condition before drawing strong conclusions.
- There is no same-run baseline or LoRA comparison in job 14662. Do not claim this variant beats the other approaches from these data.
- The report's generic sentence about bootstrap comparisons to `lora_all` does not apply here: only `top25_finetune` was selected, so no such cross-variant comparison was performed.
- The point estimates are after temperature calibration and gating thresholds selected on validation data. The test set was not used to set those thresholds, but calibration and gate estimates are still noisy with this sample size.

## Presentation-safe takeaway

> In a 100-example smoke test of frozen Qwen3-1.7B, reasoning improved accuracy from 55% to 69%. A learned gate reached 63% while escalating 24% of examples. These are preliminary base-model cascade results; the top-quarter fine-tune was trained but accidentally not loaded for evaluation, and every reasoning generation hit the 256-token cap.

## Source artifacts

- `reports/experiment_report.md`: generated summary (its bootstrap-comparison boilerplate does not apply to this one-variant run).
- `metrics/results.json` and `metrics/results.csv`: metrics and efficiency figures.
- `run_config.json`: seed, model, split sizes, precision, learning rate, and selected variant.
- Per-example predictions and validation artifacts were in scratch at run time; the report and aggregate metrics above are the durable job archive.
