# Selective Reasoning with LoRA and Brier Penalties

This repository contains two related experiments that measure whether a language model should answer immediately or spend additional computation generating a reasoning trace. The baseline uses answer confidence and escalation gates. The follow-up trains a fresh LoRA adapter with an added Brier-score penalty and compares its answer and calibration metrics with the baseline.

## Experiments

### Original baseline: Jev-style escalation

The baseline notebook loads five multiple-choice task datasets, constructs deterministic train/validation/test splits, obtains teacher reasoning traces for eligible training examples, and fine-tunes a shared LoRA adapter. It compares fast answer-token readout with generated reasoning followed by a final-answer readout. It calibrates probabilities with temperatures and evaluates confidence-based and learned escalation policies.

The baseline uses two training epochs. It uses four seen tasks for training and reserves Word Math as a held-out task. Test results include all five tasks.

### Brier-penalty follow-up

The Brier notebook reuses `data/splits.json` and `traces/train_traces.json`; it does not regenerate those inputs. It trains a fresh LoRA adapter with a Brier penalty added to language-model and restricted-answer losses. The prepared copy uses one epoch, consistent with the saved experiment configuration, trainer state, and available Brier outputs. The baseline configuration used two epochs.

The Brier notebook writes generated predictions, adapter files, trainer checkpoints, and metrics under `brier_outputs/`. Only compact metrics are included here; predictions and weights are omitted.

## Datasets and splits

The baseline notebook names these Hugging Face datasets and configurations:

| Task | Dataset |
|---|---|
| Sentiment | `stanfordnlp/sst2` |
| Inference | `facebook/anli` |
| Science questions | `allenai/ai2_arc`, configuration `ARC-Challenge` |
| General knowledge | `cais/mmlu`, configuration `all` |
| Word Math | `deepmind/aqua_rat`, configuration `raw` |

The saved split file contains 500 training, 150 validation, and 250 test examples per task. Training uses 500 examples from each of the four seen tasks (2,000 total); Word Math training examples are excluded. Validation contains 750 examples and the test set contains 1,250 examples (250 per task). The accepted teacher-trace file contains 345 traces from seen-task training examples.

The split and trace files are included to support exact reuse by the Brier experiment. Check the upstream dataset terms and redistribution requirements before publishing the derived examples and traces.

## Methodology

Both experiments use the requested base model `Qwen/Qwen3-1.7B` with `Qwen/Qwen3-0.6B` as a fallback if the primary model cannot be loaded. The original notebook configures `Qwen/Qwen3-4B` as its teacher model. Models must be available in the local Hugging Face cache or downloadable from the model host, subject to the model's access and license terms.

The notebooks produce a probability distribution over each question's valid answer options. The fast path reads the answer distribution immediately. The reasoning path generates a scratchpad and then reads the distribution after a final-answer prompt. LoRA settings recorded in the experiment configuration are rank 16, alpha 32, dropout 0.05, learning rate `2e-4`, batch size 1, gradient accumulation 8, maximum sequence length 1,024, and seed 2026. The Brier experiment uses a Brier coefficient of 1.0. The maximum generated reasoning length is 256 tokens.

The notebooks fit temperature calibration using validation data from the four seen tasks. Word Math is excluded from fitting those temperatures. The baseline learned gate uses maximum fast-path probability, the margin between the top two probabilities, entropy, option count, and the fast-path final hidden state. A standardized, class-balanced logistic-regression pipeline is fit on seen-task validation data with an internal 25% holdout when class counts permit; Word Math is excluded from gate fitting. Confidence-threshold and learned-gate curves are also calculated on test data as descriptive outputs, not as policy-selection data. Evaluation reports task-level and overall results. The Brier objective adds the mean per-example sum of squared differences between valid-option probabilities and one-hot labels, weighted by 1.0, to the language-model and answer cross-entropy losses.

## Metrics

- **Accuracy:** fraction of examples with the correct top-probability answer; higher is better.
- **ECE:** 10-bin expected calibration error comparing confidence with accuracy; lower is better.
- **NLL:** negative log likelihood of the correct option; lower is better.
- **Brier score:** mean squared error between the predicted option probabilities and one-hot labels; lower is better.
- **Escalation rate:** fraction routed to reasoning by a policy. The baseline also records average generated tokens and time for policy sweeps.

The baseline sweep and headline escalation files contain test-set descriptive results. Do not use test-set curves or rates to tune a policy; policy selection should use validation data.

## Reported results

Overall test metrics from the included result files:

| Experiment | Mode | Accuracy | ECE | NLL | Brier |
|---|---|---:|---:|---:|---:|
| Original baseline | Fast | 0.6280 | 0.0830 | 0.8938 | 0.4715 |
| Original baseline | Reasoning | 0.7400 | 0.0558 | 0.7619 | 0.3889 |
| Brier penalty | Fast | 0.6336 | 0.0449 | 0.8613 | 0.4539 |
| Brier penalty | Reasoning | 0.7328 | 0.0586 | 0.7391 | 0.3791 |

These are the saved experiment results, not a new run. In these outputs, the Brier variant has slightly higher fast-path accuracy and lower fast-path ECE/NLL/Brier than the baseline, while its reasoning accuracy is lower than the baseline's. The Brier trainer state records a completed one-epoch run.

The compact per-task tables and JSON summaries are under `metrics/` and `brier_outputs/metrics/`. Baseline gate sweeps and escalation rates are under `metrics/`.

## Dependencies and execution

The environment snapshot records Python 3.11.18 and PyTorch 2.11.0+cu128. The notebooks use Python packages including NumPy, pandas, PyTorch, Transformers, PEFT, SciPy, scikit-learn, Matplotlib, and Hugging Face Datasets (the baseline data-loading workflow). The full recorded environment is in `environment/pip_freeze.txt`; it is a host-specific snapshot and includes CUDA-related packages.

The Brier notebook expects its working directory to be this repository root, where `data/`, `traces/`, and `metrics/` are located. It creates its outputs in `brier_outputs/`. Training requires suitable GPU memory and access to the configured model weights; a CPU fallback is coded but is not practical for the full training/inference workload.

The baseline notebook contains an existing machine-specific absolute output path. It is retained unchanged in this prepared copy per the project owner's instruction and may need an environment-specific adjustment to run elsewhere. The SLURM runner and job logs are not included.

## Repository structure

```text
README.md
.gitignore
brier_experiment.ipynb
source/
  current_jev.ipynb
data/
  splits.json
traces/
  train_traces.json
environment/
  experiment_config.txt
  python_torch_info.txt
  pip_freeze.txt
metrics/
  final_table.csv
  results.json
  gate_sweeps.json
  headline_escalation_rates.json
  temperatures.json
brier_outputs/
  metrics/
    brier_final_table.csv
    brier_results.json
    comparison_original_vs_brier.csv
    temperatures.json
```

## Files intentionally excluded

Per-example prediction JSON files, LoRA weights and tokenizer artifacts, optimizer/trainer checkpoints, the duplicate compressed package archive, the duplicate Brier notebook copy, SLURM logs, and the site-specific SLURM runner are excluded. The available plot directory contained no plot files. The Brier adapter may be retained separately if needed; it is not required to inspect the included metrics.