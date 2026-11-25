# Experimental Findings & Analysis

## 1. Research Questions Revisited

1. **Can a learned escalation gate trained on a subset of reasoning tasks reliably generalise to tasks excluded from gate training?**
   **Partially, and asymmetrically across tasks.**
   - **Science questions (Config D)** transfers strongly: rescue-prediction ROC-AUC is `0.8425` (vs `0.8296` for Config A trained on all four tasks; 5-seed mean `0.7864 ± 0.0539`), and matched-escalation accuracy at `10%–40%` (`0.856, 0.860, 0.868, 0.872`) matches Config A (`0.848, 0.860, 0.864, 0.872`).
   - **General knowledge (Config E)** transfers well in ranking (`ROC-AUC = 0.6643` vs `0.6244` in Config A; 5-seed `0.7053 ± 0.0315` vs `0.6696 ± 0.0340`) and matched-escalation accuracy (`0.580` vs `0.576` at 20%), though its validation-selected threshold (`τ* = 0.40`) over-escalates (`40.0%` on General Knowledge, `36.08%` overall).
   - **Sentiment (Config B)** suffers a drop in rescue ranking (`ROC-AUC = 0.6759` vs `0.8082` in Config A; 5-seed `0.6898 ± 0.0570` vs `0.9066 ± 0.0572`), likely because binary 2-option classification differs structurally from 3–4-option QA/NLI and has only 5 true test rescues (`2.0%`). However, gated accuracy remains high (`0.9360` at `1.2%` escalation) because fast-mode accuracy is already `0.9320`.
   - **Inference (Config C)** is the **hardest excluded seen task** to generalise to. ANLI inference accounts for **47.7%** (`31 / 65`) of all seen-task validation rescues. Excluding it halves gate-training positives (`17` vs `33`) and leaves only `10` threshold-selection rescues, causing `τ*` to collapse to `0.87` (`0.0%` escalation on test inference, `0.5560` gated accuracy = fast baseline, `0.1240` regret vs reasoning). Even at matched `20%–30%` escalation, Config C trails Config A by `2.0–2.8` percentage points (`0.596` vs `0.616` at 20%; `0.608` vs `0.636` at 30%), and ROC-AUC drops from `0.7429` to `0.6699`.

2. **Does the learned gate generalise to held-out word mathematics (AQuA-RAT)?**
   **No.** Across Configs A, B, C, and D, rescue-prediction ROC-AUC on AQuA-RAT is **below chance (`0.4238–0.4767`)**, and only `0.5358` in Config E (5-seed means: `0.4195–0.5129`). At matched escalation rates (`10%–40%`), all learned gates perform near the random-escalation line (`0.291–0.397`). At validation-selected thresholds, Configs A–D escalate only `8.8%–32.0%` of AQuA-RAT queries (`0.2720–0.3360` accuracy vs `0.6080` always-reasoning, regret `0.2720–0.3360`). Config E reaches `0.5920` on AQuA-RAT only because its decision boundary triggers `95.6%` escalation (`τ* = 0.40`), matching the confidence gate (`0.6000` at `95.6%` escalation) via high escalation volume rather than better per-query discrimination.

3. **How does the learned gate compare against the confidence-based gate?**
   - **On the 4 seen tasks (`n = 1,000`)**: The learned gate (Config A) is **more compute-efficient** than the confidence gate. At validation-selected `τ*`, Config A achieves `0.7430` accuracy at `7.30%` escalation vs `0.7400` at `17.40%` escalation for the confidence gate (more than **2.3× fewer reasoning calls** for slightly higher accuracy). At matched escalation rates on seen tasks, Config A beats the confidence gate at `10%` (`0.7440` vs `0.7330`), `20%` (`0.7530` vs `0.7470`), and `30%` (`0.7600` vs `0.7510`), especially on `inference` (`+5.2 pp` at `10%`, `+4.8 pp` at `20%` and `30%`).
   - **When including out-of-distribution Word Math (`n = 1,250`)**: The confidence gate outperforms Configs A–D overall (`0.7120` at `33.04%` escalation vs Config A's `0.6608` at `9.84%`, and `0.6920` vs `0.6752` at matched `20%` global escalation). Because fast-mode AQuA-RAT accuracy is near random (`0.2560`), low fast-mode confidence (`1 - max(p)`) reliably flags AQuA-RAT as uncertain (`95.6%` escalated at `τ* = 0.42`), whereas the 2,052-dim logistic gate—having never seen 5-option arithmetic problems—fails to assign calibrated rescue probabilities under domain shift.

---

## 2. Why Threshold Selection Without Overlap Matters

In the original `Armaan_R` script (`train_jev_efficiency_experiments.py`), `choose_threshold` optimized `accuracy - 0.02 * escalation_rate` on the **same `seen_val` examples** used to train the logistic gate. Under our strict 4-way disjoint split (`val_temp` / `val_gate_train` / `val_thresh` / `test`):
- `utility = acc - 0.02 * esc` collapses to `τ* = 0.00` (~100% escalation) on 4 of 5 configs because out-of-sample `val_thresh` accuracy gains (`~7–9 pp`) outweigh a `0.02` penalty.
- Using `Shreya`'s constrained accuracy-gap criterion (`argmin escalation_rate` s.t. `val_acc >= val_reasoning_acc - 0.03`) yields non-degenerate thresholds (`0.31–0.87`), though threshold stability across small validation pools (`n = 135–180`, `10–19` rescues) remains noisy across seeds (`esc_rate_std = 0.0287–0.1619` overall; `0.1814–0.3719` on Word Math).
- This confirms why **matched escalation rate comparisons (`10%, 20%, 30%, 40%`)** are essential alongside fixed-threshold reporting.

---

## 3. Zero-Shot Untuned Backbone Control (`stage0_val` / `stage0_test`)

Running the identical LOTO protocol on `stage0` predictions (base `Qwen/Qwen3-1.7B` prior to LoRA fine-tuning) tests whether LOTO transfer depends on the backbone having been LoRA-tuned on all four seen tasks:
- Untuned fast/reasoning test accuracies are lower (`0.5552 / 0.6752` overall vs `0.6280 / 0.7400` after LoRA).
- Rescue-prediction ROC-AUC on excluded tasks drops sharply on `stage0`: Sentiment `0.3510` (vs `0.8724` in `stage0` Config A), Inference `0.4512` (vs `0.5727`), Science `0.6041` (vs `0.7206`), General Knowledge `0.6482` (vs `0.5101`).
- Thus, LoRA fine-tuning with `ReadoutTrainer` substantially improves hidden-state separability for rescue prediction across seen tasks, **and** even with LoRA tuning, the 2,052-dim linear gate still fails to generalise zero-shot to AQuA-RAT.
