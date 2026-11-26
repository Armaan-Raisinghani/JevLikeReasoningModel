# Phase 1 — Repository Audit

This document audits the existing `JevLikeReasoningModel` repository (`https://github.com/Armaan-Raisinghani/JevLikeReasoningModel`) and local cached experiment artifacts before executing the Leave-One-Task-Out (LOTO) gate generalisation experiment. All values below are extracted directly from tracked source files, configuration JSONs, executed notebooks, and saved logs.

---

## 1. Overview of Existing Implementations

The repository contains three contributor folders:

| Folder | Primary Entrypoint | Model / Scope | Learned Gate Type | Cached Raw Predictions Available Locally |
| :--- | :--- | :--- | :--- | :--- |
| `Gitanjali/JevLikeReasoningModel/` | `notebooks/current_jev_executed.ipynb` | LoRA-fine-tuned `Qwen/Qwen3-1.7B` (`SEED=2026`, 500/150/250 splits) | 2,052-feature `StandardScaler` + `LogisticRegression` (`learned_gate.json`) | **Yes** (`/Users/gracytanna/Downloads/jev result/nanbugfix_medium/predictions/`) |
| `Shreya/` | `run_jev_hpc_pipeline.py` | LoRA-fine-tuned `Qwen/Qwen3-1.7B` (`SEED=2026`, 2,000/300/500 target splits, 2,251 test queries) | Confidence gate only (`max(fast_probs) < 0.82`); no logistic regression gate | No (only `final_table.csv` and `efficiency.json` tracked) |
| `Armaan_R/` | `train_jev_efficiency_experiments.py` (v1.6) | 7 efficiency variants (`frozen_1_7b`, `frozen_0_6b`, `lora_all`, `lora_attention`, `lora_mlp`, `top25_finetune`, `pruned25_lora`) on 20/20/20 smoke splits (100 test total) | 39-feature (`4` prob/option + `35` pooled hidden stats) `StandardScaler` + `LogisticRegression` | No (only `run_config.json` and `analysis.md` tracked for jobs 14662, 14979, 17131) |

### Relevant Implementation Selected for This Experiment

**`Gitanjali/JevLikeReasoningModel/`** (`notebooks/current_jev_executed.ipynb`, executed run `nanbugfix_medium` / Slurm job `current-jev-14894`) is the exact target implementation for this study because:
1. It implements the **exact 2,052-feature learned escalation gate** specified in the research design (`[max_prob, top2_margin, entropy, n_options, 2048-d fast_hidden]`).
2. Complete frozen-model raw prediction caches (`stage2_val.json` and `stage5_test.json`, plus untuned Stage 0 base-model caches `stage0_val.json` and `stage0_test.json`) and canonical splits (`data/splits.json`) exist locally and match the repository's `results.json`, `final_table.csv`, `temperatures.json`, `headline_escalation_rates.json`, and `learned_gate.json` byte-for-byte.

---

## 2. Detailed 14-Point Audit (Primary Reference: `Gitanjali/JevLikeReasoningModel/`)

### 1. Exact Model Name and Version
- **Primary Base Model**: `Qwen/Qwen3-1.7B` (`1,720,574,976` base parameters; `1,738,007,552` total with LoRA adapter).
- **Fallback Model Configured**: `Qwen/Qwen3-0.6B` (not triggered in the executed run).
- **Trace Teacher Model**: `Qwen/Qwen3-4B`.
- **Execution Environment Versions** (recorded in `current_jev_executed.ipynb`, `requirements.txt`, `learned_gate.json`, and `logs/current-jev-14894.out`):
  - Python `3.11` (`/home/gitanjali.a.ug24/envs/jev_gpu/lib/python3.11`)
  - PyTorch `2.11.0+cu128`
  - CUDA Driver `610.43.02`, CUDA UMD `13.3` on `NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb` (node `cn03`)
  - `scikit-learn == 1.9.1` when `learned_gate.pkl` was saved (inspected under `1.8.0` when exporting `learned_gate.json`).

*(Comparator notes: `Shreya/run_jev_hpc_pipeline.py` also used `Qwen/Qwen3-1.7B` with `Qwen/Qwen3-1.7B` as default teacher; `Armaan_R` jobs 14662/14979/17131 used `Qwen/Qwen3-1.7B` as both primary and smoke teacher and `Qwen/Qwen3-0.6B` as small baseline.)*

### 2. Dataset Names and Sources
Loaded via Hugging Face `datasets.load_dataset`:
1. **Sentiment**: `stanfordnlp/sst2` (`train`: 67,349; `validation`: 872; `test`: 1,821). Prompted with `"Review: " + sentence`, options `["negative", "positive"]`.
2. **Inference**: `facebook/anli` (`train_r1`: 16,946; `dev_r1`: 1,000; `test_r1`: 1,000). Prompted with `"Premise: " + premise + "\nHypothesis: " + hypothesis`, options `["entailment", "neutral", "contradiction"]`.
3. **Science questions**: `allenai/ai2_arc`, subset `ARC-Challenge` (`train`: 1,119; `validation`: 299; `test`: 1,172).
4. **General knowledge**: `cais/mmlu`, subset `all` (`auxiliary_train`: 99,842; `dev`: 285; `test`: 14,042).
5. **Word math (Held-out)**: `deepmind/aqua_rat`, subset `raw` (`train`: 97,467; `validation`: 254; `test`: 254).

*(Same 5 dataset IDs and source splits are used across `Gitanjali`, `Shreya`, and `Armaan_R`.)*

### 3. Training, Validation, and Testing Splits
- **Per-task allocation in `Gitanjali/JevLikeReasoningModel/`**:
  - `TRAIN_PER_TASK = 500`, `VAL_PER_TASK = 150`, `TEST_PER_TASK = 250`.
  - For **sentiment** (`sst2`), because Hugging Face `sst2/test` lacks public labels, `sst2/validation` (872 examples) was shuffled with `SEED = 2026` and partitioned into disjoint pools: first `150` → `val`, next `250` → `test`, and `500` from `sst2/train` → `train`.
  - For all other tasks: `train` sampled with `SEED`, `val` sampled with `SEED + 1`, `test` sampled with `SEED + 2`.
  - **Option shuffling**: Options are deterministically shuffled per example using `int(sha256(f"{SEED}{task}{question}").hexdigest()[:8], 16)`.
  - **Leakage assertions**: SHA-256 content hashes (`id`) enforce zero overlap between `train_ids` and `(val_ids | test_ids)`, and verify `word math` never enters `train`. Verified on local `stage2_val.json` and `stage5_test.json`: `val_ids & test_ids == empty` (0 overlapping IDs).
- *(Comparator split sizes: `Shreya` used `2,000 / 300 / 500` target caps yielding 2,251 test queries (`497` sentiment, `500` inference, `500` science, `500` general knowledge, `254` word math) and 1,438 val queries; `Armaan_R` smoke runs used `20 / 20 / 20` = 100 test examples.)*

### 4. Number of Samples Used for Each Task
In `Gitanjali/JevLikeReasoningModel/` (`splits.json`, `train_traces.json`, `stage2_val.json`, `stage5_test.json`):

| Task | LoRA Direct Train | Accepted Teacher Traces | Total Val | Total Test | Options (`n_options`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `sentiment` | 500 | 98 | 150 | 250 | 2 |
| `inference` | 500 | 78 | 150 | 250 | 3 |
| `science questions` | 500 | 80 | 150 | 250 | 3–5 (val), 3–4 (test) |
| `general knowledge` | 500 | 89 | 150 | 250 | 4 |
| `word math` (held out) | 0 | 0 | 150 | 250 | 5 |
| **Total** | **2,000** (+345 traces = 2,345 rows) | **345** (from 345 attempted) | **750** (600 seen) | **1,250** (1,000 seen) | 2–5 |

### 5. Random Seeds
- `SEED = 2026` everywhere (`transformers.set_seed(2026)`, `random.seed(2026)`, `np.random.seed(2026)`, `torch.manual_seed(2026)`, `torch.cuda.manual_seed_all(2026)`, `train_test_split(..., random_state=2026)`, `LogisticRegression(..., random_state=2026)`).
- Split sampling seeds: `2026` (train & sentiment val/test pool), `2027` (`SEED+1`, val), `2028` (`SEED+2`, test).

### 6. Training Configurations, Epochs, and Learning Rates
- **Gitanjali (`current_jev_executed.ipynb`)**:
  - `EPOCHS = 2`, `LR = 2e-4`, `BATCH_SIZE = 1`, `GRAD_ACCUM = 8` (effective batch size 8), `MAX_SEQ_LEN = 1024`, `fp16 = True` (CUDA), `max_grad_norm = 1.0`, default `AdamW` + linear scheduler via Hugging Face `Trainer`.
  - Custom `ReadoutTrainer`: joint loss `lm_loss + answer_loss`, where `lm_loss` masks prompt tokens up to `prompt + "\nReasoning:"` (`-100`) and `answer_loss` applies cross-entropy over restricted option letter logits at `readout_pos = len(ids) - 1` (unused options masked with `-1e4`).
  - Teacher traces: `TRACE_TOTAL = 345` attempted from seen-task training pools (`max_new_tokens = 220`, `do_sample = False`), rejection-sampled on regex `Final answer:\s*([ABCDE])`.
- *(Comparator: `Shreya` used `epochs = 2`, `lr = 2e-4`, `bf16` on RTX PRO 6000, `1,500` accepted traces from `1,912` attempts; `Armaan_R` smoke runs used `epochs = 1`, `lr = 5e-5`, `bf16`, `max_seq_len = 1024`, `40` trace attempts.)*

### 7. LoRA Configurations
- **Gitanjali (`current_jev_executed.ipynb`)**:
  - `LORA_R = 16`, `LORA_ALPHA = 32`, `LORA_DROPOUT = 0.05`, `task_type = 'CAUSAL_LM'`
  - Discovered target modules: `['down_proj', 'gate_proj', 'k_proj', 'o_proj', 'q_proj', 'up_proj', 'v_proj']`
  - Trainable parameters: `17,432,576` / `1,738,007,552` (`1.0030%`).
- *(Comparator: `Shreya` used identical LoRA `r=16, alpha=32, dropout=0.05` on all 7 projections; `Armaan_R` tested `lora_all` [17.43M], `lora_attention` [6.42M], `lora_mlp` [11.01M], `top25_finetune` [352.45M], and `pruned25_lora` [13.07M on 21/28 layers].)*

### 8. Fast and Reasoning Inference Configurations
- **Token IDs** (verified single tokens on Qwen3 tokenizer): `{'A': 362, 'B': 425, 'C': 356, 'D': 422, 'E': 468}`.
- **Fast mode**: Single forward pass on `prompt_for(ex, reasoning=False)` ending with `"Answer:"`. Extracts restricted softmax over `LETTERS[:n_options]` and final-layer hidden state `out.hidden_states[-1][0, -1]` (dimension `2048`). `generated_tokens = 0`. Mean test latency in `stage5_test.json`: `0.0288 s` (`28.8 ms`).
- **Reasoning mode**: Greedy generation (`do_sample = False`, `use_cache = True`) from `prompt_for(ex, reasoning=True)` ending with `"Answer:\nReasoning:"` up to `MAX_REASON_TOKENS = 256`, followed by appending `"\nFinal answer:"` and one forward pass for restricted option softmax. Every example in `stage5_test.json` generated all `256` tokens (min=256, mean=256.0, max=256). Mean test latency in `stage5_test.json`: `6.1013 s`.

### 9. Maximum Reasoning Token Limit
- `MAX_REASON_TOKENS = 256` across all three implementations (`Gitanjali`, `Shreya`, `Armaan_R`).

### 10. Gate Architecture and Features
- **Gitanjali (`current_jev_executed.ipynb` / `learned_gate.json`)**:
  - Classifier: `sklearn.pipeline.make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced', random_state=2026))` (L2 penalty, `C=1.0`, `lbfgs`, converged in 108 iterations).
  - Input feature vector (length `2,052`), computed strictly from **raw** fast-mode outputs (`row['fast']['probs']` and `row['fast']['hidden']`):
    - Index `0`: `max(p)` (maximum raw fast probability)
    - Index `1`: `sorted(p)[-1] - sorted(p)[-2]` (top-two raw probability margin)
    - Index `2`: `-sum(p * log(max(p, 1e-12)))` (Shannon entropy of raw fast probabilities)
    - Index `3`: `n_options` (number of valid answer choices, `2..5`)
    - Indices `4..2051`: `fast.hidden` (2,048-dim final hidden vector at fast prompt readout position, `float32`)
- *(Note: `Armaan_R/train_jev_efficiency_experiments.py` differs: it uses calibrated probabilities and compresses the 2,048-dim hidden vector into 35 summary statistics [16 chunk means/stds + global mean, std, L2 norm] for a 39-dim feature vector.)*

### 11. Rescue-Label Construction
- **Gitanjali (`current_jev_executed.ipynb`, Cell 16)**:
  ```python
  y = int(
      np.argmax(r['fast']['calibrated_probs']) != r['label']
      and np.argmax(r['reasoning']['calibrated_probs']) == r['label']
  )
  ```
  Positive class (`1`) indicates that calibrated fast mode is wrong AND calibrated reasoning mode is right.
  In `stage3_val_calibrated.json` (`seen_val`, `n=600`), rescue base rates are:
  - `sentiment`: `4 / 150` (`2.67%`)
  - `inference`: `31 / 150` (`20.67%`)
  - `science questions`: `14 / 150` (`9.33%`)
  - `general knowledge`: `16 / 150` (`10.67%`)
  - Overall seen validation: `65 / 600` (`10.83%`).

### 12. Existing Threshold-Selection Methodology (and Identified Leakage Gaps)
Auditing all three implementations reveals critical methodological differences:
1. **`Gitanjali/JevLikeReasoningModel/` (`current_jev_executed.ipynb`)**:
   - Fits temperature scaling (`T_fast = 3.5781`, `T_reasoning = 5.6998`) on **all 600** `seen_val` examples.
   - Splits the **same** `seen_val` 75%/25% (`450` train / `150` holdout) to fit `learned_gate.pkl` (reporting holdout rescue classification accuracy `0.8867`).
   - **Does not select a fixed validation operating threshold**: Cell 19 sweeps `th in np.linspace(0, 1, 101)` descriptively on `test_rows`, and `headline_escalation_rates.json` computes test-set post-hoc confidence-ranked escalation fractions.
2. **`Armaan_R/train_jev_efficiency_experiments.py`**:
   - Fits temperatures and gate on **all** `seen_val`, then selects `threshold` on **the exact same `seen_val` rows used to fit the gate** (`utility = accuracy - 0.02 * escalation_rate`), introducing in-sample overfitting in threshold choice.
3. **`Shreya/run_jev_hpc_pipeline.py`**:
   - Confidence gate only (`max(probs) < th`), selecting `τ* = 0.82` on validation data (`min escalation_rate` s.t. `accuracy >= val_reasoning_acc - 0.03`), but `val_examples` in `Shreya` included `word math` during threshold calibration.

**Implication for Phase 3**: Our new LOTO experiment fixes all three issues by enforcing a **strict 4-way disjoint partition** (`val_temp`, `val_gate_train`, `val_thresh`, `test`) using **only eligible training tasks** for temperature calibration, gate fitting, and threshold selection.

### 13. Previously Completed Experiments and Results
- **Gitanjali (`nanbugfix_medium`, `n = 1,250` test, `250/task`)**:
  - Overall: Fast acc `0.6280` (ECE `0.0830`, NLL `0.8938`, Brier `0.4715`); Reasoning acc `0.7400` (ECE `0.0558`, NLL `0.7619`, Brier `0.3889`).
  - Per-task (Fast → Reasoning): Sentiment `0.9320 → 0.9440`; Inference `0.5560 → 0.6800`; Science `0.8320 → 0.8720`; General Knowledge `0.5640 → 0.5960`; Word Math `0.2560 → 0.6080`.
  - Stage 0 untuned `Qwen/Qwen3-1.7B` (verified from `stage0_test.json`, `n = 1,250`): Fast raw acc `0.5552`, Reasoning raw acc `0.6752`.
- **Shreya (`n = 2,251` test, confidence gate `τ = 0.82`)**:
  - Overall: Fast `67.84%`, Always Reason `74.77%`, Confidence-Gated `71.57%` at `17.01%` escalation (`43.56` avg tokens, `1.214 s` avg latency).
- **Armaan_R (smoke tests, `n = 100` test, `20/task`)**:
  - Job `14662`: `top25_finetune` evaluation invalid due to checkpoint-loader bug (loaded frozen 1.7B: 55% fast / 69% reason / 63% learned gate at 24% escalation).
  - Job `14979`: Failed on `pruned25_lora`; 5 valid smoke rows (`frozen_1_7b`: 63% @ 24% esc; `frozen_0_6b`: 40% @ 3%; `lora_all`: 64% @ 42%; `lora_attention`: 57% @ 13%; `lora_mlp`: 57% @ 17%).
  - Job `17131` (v1.6 bugfix): `top25_finetune` 54% fast / 69% reason / 61% learned gate @ 25% esc; `pruned25_lora` (21/28 layers, 1.381B params) 26% fast / 27% reason / 30% learned gate @ 17% esc.

### 14. Recorded Experiment Dates
- **`Gitanjali` (`nanbugfix_medium` / Slurm job `14894`)**: Started `Thu Nov 19 13:02:37 2026` (`logs/current-jev-14894.out`); files timestamped `Nov 18 21:09` (`splits.json`) through `Nov 19 16:10` (`stage5_test_calibrated.json`, `current_jev_executed.ipynb`).
- **GitHub Repository Commits**:
  - `Armaan_R`: `2026-11-25 23:01:54 +0530` (initial pipeline, 14662/14979 audits) and `2026-11-25 23:28:04 +0530` (handoff + job 17131).
  - `Shreya`: `2026-11-25 23:24:30 +0530` (latency experiment).
  - `Gitanjali`: `2026-11-26 00:37:13 +0530` (`JevLikeReasoningModel` notebook, metrics, plots).
- **Armaan_R Jobs `14662`, `14979`, `17131`**: Exact calendar dates not recorded in tracked files (elapsed wall times recorded: `1:27:25` for 14979, `54:43` for 17131).

---

## 3. Reusable Artifacts & Decision for Phase 2–7

| Artifact | Location | Reusable? | Usage in New LOTO Experiment |
| :--- | :--- | :---: | :--- |
| `stage2_val.json` (750 val, raw fast/reason probs + 2,048-d hidden) | `/Users/gracytanna/Downloads/jev result/nanbugfix_medium/predictions/` | **Yes (Primary)** | Partitioned per eligible task into disjoint `val_temp`, `val_gate_train`, `val_thresh`. |
| `stage5_test.json` (1,250 test, raw fast/reason probs + 2,048-d hidden) | `/Users/gracytanna/Downloads/jev result/nanbugfix_medium/predictions/` | **Yes (Primary)** | Final frozen test set across all 5 tasks (`250/task`). |
| `stage0_val.json`, `stage0_test.json` (untuned `Qwen/Qwen3-1.7B`) | `/Users/gracytanna/Downloads/jev result/nanbugfix_medium/predictions/` | **Yes (Zero-shot Backbone Control)** | Secondary sensitivity run verifying LOTO generalisation without any backbone LoRA exposure to seen tasks. |
| `stage3_val_calibrated.json`, `stage5_test_calibrated.json`, `temperatures.json` | `/Users/gracytanna/Downloads/jev result/nanbugfix_medium/` | **Do NOT reuse directly for LOTO** | Calibrated with temperatures fitted on *all 4* seen tasks (would leak excluded tasks and overlap with gate/threshold partitions). Temperatures are re-fitted strictly on each configuration's `val_temp`. |
| `learned_gate.json` | `Gitanjali/JevLikeReasoningModel/results/metrics/learned_gate.json` | **Reference / Verification** | Used in sanity verification to confirm exact 2,052-feature extraction parity. |
