#!/usr/bin/env python3
"""Leave-One-Task-Out (LOTO) Generalisation Experiment for the JEV Escalation Gate.

Reuses frozen Qwen/Qwen3-1.7B (LoRA-tuned) cached predictions from
Gitanjali/JevLikeReasoningModel (stage2_val.json, stage5_test.json) and also
evaluates the untuned Stage 0 base-model predictions (stage0_val.json,
stage0_test.json) as a zero-shot backbone control.

Enforces strict 4-way disjoint partitioning per configuration:
  1. val_temp       (20% = 30/task on eligible tasks): temperature calibration
  2. val_gate_train (50% = 75/task on eligible tasks): logistic regression gate fit
  3. val_thresh     (30% = 45/task on eligible tasks): threshold selection
  4. test           (250/task across all 5 tasks): final frozen evaluation
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from verify_inputs import (
    calibrate,
    check_prediction_file,
    extract_features,
    fit_temperature,
    partition_eligible_val,
    resolve_cache_dir,
)


def compute_ece_nll_brier(probs: list[list[float]], labels: list[int], bins: int = 10) -> dict[str, float]:
    if not probs:
        return {"accuracy": 0.0, "ece": 0.0, "nll": 0.0, "brier": 0.0}
    conf = np.asarray([max(p) for p in probs], dtype=float)
    pred = np.asarray([int(np.argmax(p)) for p in probs], dtype=int)
    y = np.asarray(labels, dtype=int)
    ok = pred == y

    ece = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        mask = (conf >= lo) & (conf < hi if i < bins - 1 else conf <= hi + 1e-8)
        if mask.any():
            ece += float(mask.mean() * abs(float(ok[mask].mean()) - float(conf[mask].mean())))

    nll_val = float(-np.mean([math.log(max(float(p[lbl]), 1e-12)) for p, lbl in zip(probs, labels)]))
    width = max(len(p) for p in probs)
    one_hot = np.zeros((len(labels), width), dtype=float)
    padded = np.zeros_like(one_hot)
    for idx, (p, lbl) in enumerate(zip(probs, labels)):
        one_hot[idx, lbl] = 1.0
        padded[idx, : len(p)] = p
    brier = float(np.mean(np.sum((padded - one_hot) ** 2, axis=1)))
    return {"accuracy": float(ok.mean()), "ece": float(ece), "nll": nll_val, "brier": brier}


def rescue_labels(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            int(
                np.argmax(r["fast"]["calibrated_probs"]) != r["label"]
                and np.argmax(r["reasoning"]["calibrated_probs"]) == r["label"]
            )
            for r in rows
        ],
        dtype=int,
    )


def select_threshold_gap(
    rows: list[dict[str, Any]], scores: np.ndarray, target_gap: float = 0.03, grid_points: int = 101
) -> dict[str, Any]:
    """Select tau* minimizing validation escalation rate subject to acc >= reason_acc - gap."""
    reason_acc = float(np.mean([int(np.argmax(r["reasoning"]["calibrated_probs"]) == r["label"]) for r in rows]))
    candidates = []
    for tau in np.linspace(0.0, 1.0, grid_points):
        esc = scores >= tau
        acc = float(
            np.mean(
                [
                    int(np.argmax((r["reasoning"] if e else r["fast"])["calibrated_probs"]) == r["label"])
                    for r, e in zip(rows, esc)
                ]
            )
        )
        rate = float(np.mean(esc))
        candidates.append((rate, -acc, float(tau), acc))
    eligible = [c for c in candidates if c[3] >= reason_acc - target_gap]
    chosen = min(eligible or candidates, key=lambda x: (x[0], x[1]))
    return {
        "threshold": float(chosen[2]),
        "val_reasoning_acc": reason_acc,
        "val_gated_acc": float(chosen[3]),
        "val_escalation_rate": float(chosen[0]),
        "feasible": bool(eligible),
    }


def evaluate_slice(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    escalate_mask: np.ndarray,
    confident_thresh: float = 0.95,
) -> dict[str, Any]:
    n = len(rows)
    assert n > 0
    fast_probs = [r["fast"]["calibrated_probs"] for r in rows]
    reas_probs = [r["reasoning"]["calibrated_probs"] for r in rows]
    gated_probs = [
        r["reasoning"]["calibrated_probs"] if esc else r["fast"]["calibrated_probs"]
        for r, esc in zip(rows, escalate_mask)
    ]
    labels = [int(r["label"]) for r in rows]

    m_fast = compute_ece_nll_brier(fast_probs, labels)
    m_reas = compute_ece_nll_brier(reas_probs, labels)
    m_gate = compute_ece_nll_brier(gated_probs, labels)

    fast_pred = np.asarray([int(np.argmax(p)) for p in fast_probs])
    reas_pred = np.asarray([int(np.argmax(p)) for p in reas_probs])
    y_arr = np.asarray(labels)
    y_rescue = ((fast_pred != y_arr) & (reas_pred == y_arr)).astype(int)

    esc_bool = np.asarray(escalate_mask, dtype=bool)
    esc_rate = float(np.mean(esc_bool))
    avg_tokens = float(np.mean([r["reasoning"]["generated_tokens"] if e else 0 for r, e in zip(rows, esc_bool)]))
    avg_seconds = float(
        np.mean([r["reasoning"]["seconds"] if e else r["fast"]["seconds"] for r, e in zip(rows, esc_bool)])
    )

    # Rescue classification metrics
    rescue_acc = float(np.mean(esc_bool.astype(int) == y_rescue))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_rescue, esc_bool.astype(int), average="binary", zero_division=0
    )
    roc_auc = float(roc_auc_score(y_rescue, scores)) if len(np.unique(y_rescue)) > 1 else None

    fast_errors = fast_pred != y_arr
    rescued_mask = esc_bool & (y_rescue == 1)
    harmed_mask = esc_bool & (fast_pred == y_arr) & (reas_pred != y_arr)

    # Confidently wrong fast predictions
    fast_conf = np.asarray([max(p) for p in fast_probs])
    conf_mask = fast_conf >= confident_thresh
    conf_wrong_mask = conf_mask & fast_errors
    conf_count = int(conf_mask.sum())
    conf_wrong_share = float(conf_wrong_mask.sum() / conf_count) if conf_count > 0 else None
    conf_wrong_caught = float((conf_wrong_mask & esc_bool).sum() / conf_wrong_mask.sum()) if conf_wrong_mask.any() else None

    denom = m_reas["accuracy"] - m_fast["accuracy"]
    recovery = ((m_gate["accuracy"] - m_fast["accuracy"]) / denom) if abs(denom) > 1e-12 else None

    return {
        "n": n,
        "fast_accuracy": m_fast["accuracy"],
        "reasoning_accuracy": m_reas["accuracy"],
        "gated_accuracy": m_gate["accuracy"],
        "regret_vs_reasoning": m_reas["accuracy"] - m_gate["accuracy"],
        "accuracy_gain_over_fast": m_gate["accuracy"] - m_fast["accuracy"],
        "accuracy_recovery_fraction": recovery,
        "escalation_rate": esc_rate,
        "avg_generated_tokens": avg_tokens,
        "avg_latency_seconds": avg_seconds,
        "utility_cost_002": m_gate["accuracy"] - 0.02 * esc_rate,
        "utility_cost_010": m_gate["accuracy"] - 0.10 * esc_rate,
        "true_rescue_count": int(y_rescue.sum()),
        "true_rescue_rate": float(y_rescue.mean()),
        "realized_rescue_count": int(rescued_mask.sum()),
        "realized_rescue_rate_over_all": float(rescued_mask.mean()),
        "fast_error_rescue_rate": float(rescued_mask.sum() / fast_errors.sum()) if fast_errors.any() else None,
        "harmed_correct_fast_count": int(harmed_mask.sum()),
        "gate_rescue_accuracy": rescue_acc,
        "gate_rescue_precision": float(prec),
        "gate_rescue_recall": float(rec),
        "gate_rescue_f1": float(f1),
        "gate_rescue_roc_auc": roc_auc,
        "confident_fast_count": conf_count,
        "confidently_wrong_fast_count": int(conf_wrong_mask.sum()),
        "confidently_wrong_fast_share": conf_wrong_share,
        "confidently_wrong_fast_overall_rate": float(conf_wrong_mask.mean()),
        "confidently_wrong_caught_rate": conf_wrong_caught,
        "fast_ece": m_fast["ece"],
        "reasoning_ece": m_reas["ece"],
        "gated_ece": m_gate["ece"],
        "fast_nll": m_fast["nll"],
        "reasoning_nll": m_reas["nll"],
        "gated_nll": m_gate["nll"],
        "fast_brier": m_fast["brier"],
        "reasoning_brier": m_reas["brier"],
        "gated_brier": m_gate["brier"],
    }


def top_k_mask(scores: np.ndarray, rate: float) -> np.ndarray:
    """Deterministic exact top-k escalation mask matching round(rate * len(scores))."""
    k = int(round(rate * len(scores)))
    mask = np.zeros(len(scores), dtype=bool)
    if k <= 0:
        return mask
    if k >= len(scores):
        mask[:] = True
        return mask
    # Stable descending sort so ties break deterministically by index
    order = np.argsort(-scores, kind="mergesort")
    mask[order[:k]] = True
    return mask


def run_single_seed(
    val_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    config: dict[str, Any],
    seed: int,
    save_gates_dir: Path | None = None,
) -> dict[str, Any]:
    all_tasks = config["tasks"]["all"]
    seen_tasks = config["tasks"]["seen"]
    rates = config["matched_escalation_rates"]
    gap = config["threshold_selection"]["target_accuracy_gap"]
    grid = config["threshold_selection"]["candidate_grid_points"]

    partition_records: list[dict[str, Any]] = []
    threshold_records: list[dict[str, Any]] = []
    matched_global_records: list[dict[str, Any]] = []
    matched_per_task_records: list[dict[str, Any]] = []
    sweep_curves: dict[str, Any] = {}

    X_test_all = np.stack([extract_features(r) for r in test_rows])

    for cfg_id, cfg in config["configurations"].items():
        train_tasks = cfg["train_tasks"]
        excluded_seen = cfg["excluded_seen_task"]

        # 1. Disjoint 4-way partition (eligible tasks only for val_temp, val_gate_train, val_thresh)
        vt, vg, vth = partition_eligible_val(val_rows, train_tasks, seed, config)

        # Verify excluded task & word math never enter validation partitions
        disallowed = {"word math"} | ({excluded_seen} if excluded_seen else set())
        assert not any(r["task"] in disallowed for r in vt + vg + vth), f"Leakage in {cfg_id}"

        # 2. Temperature calibration on val_temp ONLY
        temps = {m: fit_temperature(vt, m) for m in ("fast", "reasoning")}
        vg_cal = calibrate(vg, temps)
        vth_cal = calibrate(vth, temps)
        test_cal = calibrate(test_rows, temps)

        # 3. Gate training on val_gate_train ONLY
        X_tr = np.stack([extract_features(r) for r in vg_cal])
        y_tr = rescue_labels(vg_cal)
        gate = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=config["gate"]["max_iter"],
                class_weight=config["gate"]["class_weight"],
                random_state=seed,
            ),
        )
        gate.fit(X_tr, y_tr)

        if save_gates_dir is not None:
            save_gates_dir.mkdir(parents=True, exist_ok=True)
            export = {
                "config_id": cfg_id,
                "seed": seed,
                "train_tasks": train_tasks,
                "excluded_seen_task": excluded_seen,
                "temperatures": temps,
                "scaler_mean": gate.named_steps["standardscaler"].mean_.tolist(),
                "scaler_scale": gate.named_steps["standardscaler"].scale_.tolist(),
                "coef": gate.named_steps["logisticregression"].coef_.tolist(),
                "intercept": gate.named_steps["logisticregression"].intercept_.tolist(),
            }
            (save_gates_dir / f"gate_{cfg_id}.json").write_text(json.dumps(export), encoding="utf-8")

        # 4. Threshold selection on val_thresh ONLY
        X_th = np.stack([extract_features(r) for r in vth_cal])
        y_th = rescue_labels(vth_cal)
        learned_scores_th = gate.predict_proba(X_th)[:, 1]
        conf_scores_th = np.asarray([1.0 - max(r["fast"]["calibrated_probs"]) for r in vth_cal], dtype=float)

        learned_sel = select_threshold_gap(vth_cal, learned_scores_th, target_gap=gap, grid_points=grid)
        conf_sel = select_threshold_gap(vth_cal, conf_scores_th, target_gap=gap, grid_points=grid)

        partition_records.append(
            {
                "config_id": cfg_id,
                "label": cfg["label"],
                "seed": seed,
                "train_tasks": ",".join(train_tasks),
                "excluded_seen_task": excluded_seen or "none",
                "n_val_temp": len(vt),
                "n_val_gate_train": len(vg),
                "n_gate_train_rescues": int(y_tr.sum()),
                "n_val_thresh": len(vth),
                "n_val_thresh_rescues": int(y_th.sum()),
                "n_test_total": len(test_cal),
                "T_fast": temps["fast"],
                "T_reasoning": temps["reasoning"],
                "learned_tau_star": learned_sel["threshold"],
                "learned_val_thresh_acc": learned_sel["val_gated_acc"],
                "learned_val_thresh_esc": learned_sel["val_escalation_rate"],
                "conf_tau_star": conf_sel["threshold"],
                "conf_val_thresh_acc": conf_sel["val_gated_acc"],
                "conf_val_thresh_esc": conf_sel["val_escalation_rate"],
                "val_thresh_reasoning_acc": learned_sel["val_reasoning_acc"],
                "leakage_checks_passed": True,
            }
        )

        # 5. Final testing on frozen test set
        learned_scores_test = gate.predict_proba(X_test_all)[:, 1]
        conf_scores_test = np.asarray([1.0 - max(r["fast"]["calibrated_probs"]) for r in test_cal], dtype=float)

        # Define evaluation groups
        groups: dict[str, list[int]] = {t: [i for i, r in enumerate(test_cal) if r["task"] == t] for t in all_tasks}
        groups["seen_tasks_all_4"] = [i for i, r in enumerate(test_cal) if r["task"] in seen_tasks]
        groups["eligible_train_tasks"] = [i for i, r in enumerate(test_cal) if r["task"] in train_tasks]
        if excluded_seen:
            groups["excluded_seen_task"] = [i for i, r in enumerate(test_cal) if r["task"] == excluded_seen]
        groups["overall_all_5"] = list(range(len(test_cal)))

        # Threshold-based evaluation (tau* from val_thresh, plus fixed 0.50 for learned gate)
        policies = [
            ("learned_gate_val_tau", learned_scores_test, learned_sel["threshold"]),
            ("learned_gate_tau_050", learned_scores_test, 0.50),
            ("confidence_gate_val_tau", conf_scores_test, conf_sel["threshold"]),
        ]
        for policy_name, sc_all, tau in policies:
            for gname, idxs in groups.items():
                sub_rows = [test_cal[i] for i in idxs]
                sub_sc = sc_all[idxs]
                mask = sub_sc >= tau
                metrics = evaluate_slice(sub_rows, sub_sc, mask, config["confident_wrong_threshold"])
                threshold_records.append(
                    {
                        "seed": seed,
                        "config_id": cfg_id,
                        "excluded_seen_task": excluded_seen or "none",
                        "policy": policy_name,
                        "threshold_used": tau,
                        "eval_group": gname,
                        **metrics,
                    }
                )

        # Matched escalation rate evaluation (10%, 20%, 30%, 40%)
        # (A) Global-budget allocation across overall_all_5 and seen_tasks_all_4
        for pool_name in ("overall_all_5", "seen_tasks_all_4"):
            pool_idx = groups[pool_name]
            pool_rows = [test_cal[i] for i in pool_idx]
            for gate_kind, sc_all in [("learned", learned_scores_test), ("confidence", conf_scores_test)]:
                pool_sc = sc_all[pool_idx]
                for rate in rates:
                    pool_mask = top_k_mask(pool_sc, rate)
                    m_pool = evaluate_slice(pool_rows, pool_sc, pool_mask, config["confident_wrong_threshold"])
                    row_entry = {
                        "seed": seed,
                        "config_id": cfg_id,
                        "excluded_seen_task": excluded_seen or "none",
                        "pool": pool_name,
                        "gate_kind": gate_kind,
                        "target_escalation_rate": rate,
                        **m_pool,
                    }
                    # Add per-task allocation & accuracy inside this global budget
                    for t in all_tasks:
                        t_local = [j for j, r in enumerate(pool_rows) if r["task"] == t]
                        if t_local:
                            t_sub = [pool_rows[j] for j in t_local]
                            t_sc = pool_sc[t_local]
                            t_mask = pool_mask[t_local]
                            tm = evaluate_slice(t_sub, t_sc, t_mask, config["confident_wrong_threshold"])
                            slug = t.replace(" ", "_")
                            row_entry[f"task_{slug}_acc"] = tm["gated_accuracy"]
                            row_entry[f"task_{slug}_esc"] = tm["escalation_rate"]
                    matched_global_records.append(row_entry)

            # Analytical random-escalation baseline at exact rate
            fast_acc_p = float(np.mean([int(np.argmax(r["fast"]["calibrated_probs"]) == r["label"]) for r in pool_rows]))
            reas_acc_p = float(np.mean([int(np.argmax(r["reasoning"]["calibrated_probs"]) == r["label"]) for r in pool_rows]))
            for rate in rates:
                matched_global_records.append(
                    {
                        "seed": seed,
                        "config_id": cfg_id,
                        "excluded_seen_task": excluded_seen or "none",
                        "pool": pool_name,
                        "gate_kind": "random_expected",
                        "target_escalation_rate": rate,
                        "n": len(pool_rows),
                        "fast_accuracy": fast_acc_p,
                        "reasoning_accuracy": reas_acc_p,
                        "gated_accuracy": (1.0 - rate) * fast_acc_p + rate * reas_acc_p,
                        "regret_vs_reasoning": (1.0 - rate) * (reas_acc_p - fast_acc_p),
                        "accuracy_gain_over_fast": rate * (reas_acc_p - fast_acc_p),
                        "escalation_rate": rate,
                        "avg_generated_tokens": 256.0 * rate,
                    }
                )

        # (B) Per-slice matched escalation (escalate top r% within each slice independently)
        for gname, idxs in groups.items():
            sub_rows = [test_cal[i] for i in idxs]
            for gate_kind, sc_all in [("learned", learned_scores_test), ("confidence", conf_scores_test)]:
                sub_sc = sc_all[idxs]
                for rate in rates:
                    mask = top_k_mask(sub_sc, rate)
                    m = evaluate_slice(sub_rows, sub_sc, mask, config["confident_wrong_threshold"])
                    matched_per_task_records.append(
                        {
                            "seed": seed,
                            "config_id": cfg_id,
                            "excluded_seen_task": excluded_seen or "none",
                            "eval_group": gname,
                            "gate_kind": gate_kind,
                            "target_escalation_rate": rate,
                            **m,
                        }
                    )

        # (C) Fine sweep curve (0% to 100% by 5%) for Pareto plotting (only primary seed)
        if save_gates_dir is not None:
            sweep_curves[cfg_id] = {}
            for gname in ("overall_all_5", "word math", "seen_tasks_all_4") + (("excluded_seen_task",) if excluded_seen else ()):
                idxs = groups[gname]
                sub_rows = [test_cal[i] for i in idxs]
                sweep_curves[cfg_id][gname] = {}
                for gate_kind, sc_all in [("learned", learned_scores_test), ("confidence", conf_scores_test)]:
                    sub_sc = sc_all[idxs]
                    pts = []
                    for rate in np.linspace(0.0, 1.0, 21):
                        mask = top_k_mask(sub_sc, float(rate))
                        m = evaluate_slice(sub_rows, sub_sc, mask, config["confident_wrong_threshold"])
                        pts.append(
                            {
                                "target_rate": float(rate),
                                "escalation_rate": m["escalation_rate"],
                                "gated_accuracy": m["gated_accuracy"],
                                "avg_generated_tokens": m["avg_generated_tokens"],
                                "avg_latency_seconds": m["avg_latency_seconds"],
                            }
                        )
                    sweep_curves[cfg_id][gname][gate_kind] = pts

    return {
        "partitions": partition_records,
        "threshold_eval": threshold_records,
        "matched_global": matched_global_records,
        "matched_per_task": matched_per_task_records,
        "sweep_curves": sweep_curves,
    }


def build_loto_summary(thresh_df: pd.DataFrame, matched_pt_df: pd.DataFrame) -> pd.DataFrame:
    """Create focused LOTO generalisation comparison table across Configs A-E and Confidence Gate."""
    rows = []
    loto_configs = [
        ("B_excl_sentiment", "sentiment"),
        ("C_excl_inference", "inference"),
        ("D_excl_science", "science questions"),
        ("E_excl_general_knowledge", "general knowledge"),
    ]
    for cfg_id, target_task in loto_configs:
        for eval_target in (target_task, "word math"):
            sub_t = thresh_df[(thresh_df["eval_group"] == eval_target) & (thresh_df["policy"] == "learned_gate_val_tau")]
            a_row = sub_t[sub_t["config_id"] == "A_all_seen"].iloc[0]
            l_row = sub_t[sub_t["config_id"] == cfg_id].iloc[0]
            c_row = thresh_df[
                (thresh_df["eval_group"] == eval_target)
                & (thresh_df["policy"] == "confidence_gate_val_tau")
                & (thresh_df["config_id"] == cfg_id)
            ].iloc[0]
            ca_row = thresh_df[
                (thresh_df["eval_group"] == eval_target)
                & (thresh_df["policy"] == "confidence_gate_val_tau")
                & (thresh_df["config_id"] == "A_all_seen")
            ].iloc[0]

            entry = {
                "loto_config": cfg_id,
                "excluded_task": target_task,
                "evaluated_on": eval_target,
                "n": int(a_row["n"]),
                "fast_acc": a_row["fast_accuracy"],
                "reason_acc": a_row["reasoning_accuracy"],
                "loto_learned_tau_acc": l_row["gated_accuracy"],
                "loto_learned_tau_esc": l_row["escalation_rate"],
                "loto_learned_tau_regret": l_row["regret_vs_reasoning"],
                "all4_learned_tau_acc": a_row["gated_accuracy"],
                "all4_learned_tau_esc": a_row["escalation_rate"],
                "conf_loto_tau_acc": c_row["gated_accuracy"],
                "conf_loto_tau_esc": c_row["escalation_rate"],
                "conf_all4_tau_acc": ca_row["gated_accuracy"],
                "conf_all4_tau_esc": ca_row["escalation_rate"],
                "loto_rescue_roc_auc": l_row["gate_rescue_roc_auc"],
                "all4_rescue_roc_auc": a_row["gate_rescue_roc_auc"],
            }
            for rate in (0.10, 0.20, 0.30, 0.40):
                pct = int(round(rate * 100))
                mp_loto = matched_pt_df[
                    (matched_pt_df["config_id"] == cfg_id)
                    & (matched_pt_df["eval_group"] == eval_target)
                    & (matched_pt_df["gate_kind"] == "learned")
                    & (np.isclose(matched_pt_df["target_escalation_rate"], rate))
                ].iloc[0]
                mp_all4 = matched_pt_df[
                    (matched_pt_df["config_id"] == "A_all_seen")
                    & (matched_pt_df["eval_group"] == eval_target)
                    & (matched_pt_df["gate_kind"] == "learned")
                    & (np.isclose(matched_pt_df["target_escalation_rate"], rate))
                ].iloc[0]
                mp_conf = matched_pt_df[
                    (matched_pt_df["config_id"] == "A_all_seen")
                    & (matched_pt_df["eval_group"] == eval_target)
                    & (matched_pt_df["gate_kind"] == "confidence")
                    & (np.isclose(matched_pt_df["target_escalation_rate"], rate))
                ].iloc[0]
                entry[f"loto_acc_at_{pct}pct"] = mp_loto["gated_accuracy"]
                entry[f"all4_acc_at_{pct}pct"] = mp_all4["gated_accuracy"]
                entry[f"conf_acc_at_{pct}pct"] = mp_conf["gated_accuracy"]
                entry[f"loto_minus_all4_at_{pct}pct"] = mp_loto["gated_accuracy"] - mp_all4["gated_accuracy"]
                entry[f"loto_minus_conf_at_{pct}pct"] = mp_loto["gated_accuracy"] - mp_conf["gated_accuracy"]
            rows.append(entry)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = repo_root / "gracy" / "experiment_config" / "loto_config.json"
    config = json.loads(cfg_path.read_text(encoding="utf-8"))
    cache_dir = resolve_cache_dir(args.cache_dir, config, repo_root)

    out_dir = repo_root / "gracy" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_tasks = config["tasks"]["all"]

    # Primary run: LoRA-tuned frozen Qwen3-1.7B (stage2_val + stage5_test)
    s2_val = check_prediction_file(cache_dir / "predictions" / "stage2_val.json", all_tasks, 150)
    s5_test = check_prediction_file(cache_dir / "predictions" / "stage5_test.json", all_tasks, 250)
    assert {r["id"] for r in s2_val}.isdisjoint({r["id"] for r in s5_test})

    primary = run_single_seed(s2_val, s5_test, config, config["seed"], save_gates_dir=out_dir / "exported_gates")
    part_df = pd.DataFrame(primary["partitions"])
    thresh_df = pd.DataFrame(primary["threshold_eval"])
    mg_df = pd.DataFrame(primary["matched_global"])
    mpt_df = pd.DataFrame(primary["matched_per_task"])
    loto_sum_df = build_loto_summary(thresh_df, mpt_df)

    part_df.to_csv(out_dir / "partition_summary.csv", index=False)
    thresh_df.to_csv(out_dir / "threshold_eval_detailed.csv", index=False)
    mg_df.to_csv(out_dir / "matched_escalation_global.csv", index=False)
    mpt_df.to_csv(out_dir / "matched_escalation_per_task.csv", index=False)
    loto_sum_df.to_csv(out_dir / "loto_generalisation_summary.csv", index=False)
    (out_dir / "sweep_curves.json").write_text(json.dumps(primary["sweep_curves"], indent=2), encoding="utf-8")

    # Multi-seed validation-split robustness check
    robust_thresh, robust_mpt = [], []
    for s in config["robustness_seeds"]:
        res = run_single_seed(s2_val, s5_test, config, s, save_gates_dir=None)
        robust_thresh.extend(res["threshold_eval"])
        robust_mpt.extend(res["matched_per_task"])
    rt_df = pd.DataFrame(robust_thresh)
    rm_df = pd.DataFrame(robust_mpt)
    rt_sub = rt_df[rt_df["policy"] == "learned_gate_val_tau"]
    rob_summary = (
        rt_sub.groupby(["config_id", "eval_group"], as_index=False)
        .agg(
            n=("n", "first"),
            gated_acc_mean=("gated_accuracy", "mean"),
            gated_acc_std=("gated_accuracy", "std"),
            esc_rate_mean=("escalation_rate", "mean"),
            esc_rate_std=("escalation_rate", "std"),
            regret_mean=("regret_vs_reasoning", "mean"),
            roc_auc_mean=("gate_rescue_roc_auc", "mean"),
            roc_auc_std=("gate_rescue_roc_auc", "std"),
        )
    )
    # Also add matched 20% accuracy mean/std across seeds
    rm20 = rm_df[(rm_df["gate_kind"] == "learned") & (np.isclose(rm_df["target_escalation_rate"], 0.20))]
    rm20_agg = (
        rm20.groupby(["config_id", "eval_group"], as_index=False)
        .agg(acc_at_20pct_mean=("gated_accuracy", "mean"), acc_at_20pct_std=("gated_accuracy", "std"))
    )
    rob_summary = rob_summary.merge(rm20_agg, on=["config_id", "eval_group"], how="left")
    rob_summary.to_csv(out_dir / "robustness_multiseed.csv", index=False)

    # Zero-shot untuned backbone control (stage0_val + stage0_test)
    s0_val = check_prediction_file(cache_dir / "predictions" / "stage0_val.json", all_tasks, 150)
    s0_test = check_prediction_file(cache_dir / "predictions" / "stage0_test.json", all_tasks, 250)
    s0_res = run_single_seed(s0_val, s0_test, config, config["seed"], save_gates_dir=None)
    s0_thresh = pd.DataFrame(s0_res["threshold_eval"])
    s0_mpt = pd.DataFrame(s0_res["matched_per_task"])
    s0_summary = build_loto_summary(s0_thresh, s0_mpt)
    s0_thresh.to_csv(out_dir / "stage0_untuned_threshold_eval.csv", index=False)
    s0_summary.to_csv(out_dir / "stage0_untuned_loto_summary.csv", index=False)

    print("Completed full LOTO experiment successfully.")
    print(f"Results written to: {out_dir}")
    print("\n--- Partition & Threshold Summary (Primary Seed 2026) ---")
    print(part_df[["config_id", "n_val_temp", "n_val_gate_train", "n_gate_train_rescues", "n_val_thresh", "learned_tau_star", "conf_tau_star"]].to_string(index=False))
    print("\n--- LOTO Generalisation Summary (Primary Seed 2026) ---")
    cols = [
        "loto_config", "evaluated_on", "fast_acc", "reason_acc",
        "loto_learned_tau_acc", "loto_learned_tau_esc",
        "all4_learned_tau_acc", "all4_learned_tau_esc",
        "conf_all4_tau_acc", "conf_all4_tau_esc",
        "loto_acc_at_20pct", "all4_acc_at_20pct", "conf_acc_at_20pct",
        "loto_rescue_roc_auc", "all4_rescue_roc_auc",
    ]
    print(loto_sum_df[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
