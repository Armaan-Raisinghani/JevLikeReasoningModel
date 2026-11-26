#!/usr/bin/env python3
"""Pre-execution verification and sanity checks for the JEV LOTO gate experiment.

Checks:
1. Required cached prediction files exist and are compatible.
2. Zero val/test ID overlap and exact per-task counts.
3. Feature extraction matches Gitanjali/JevLikeReasoningModel/results/metrics/learned_gate.json.
4. Strict task exclusions and disjoint 4-way partitioning across all 5 configurations.
5. Small sanity run on Config A and Config B inspecting validation threshold selection.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def resolve_cache_dir(cli_dir: str | None, config: dict[str, Any], repo_root: Path) -> Path:
    candidates = []
    if cli_dir:
        candidates.append(Path(cli_dir).expanduser())
    for raw in config.get("default_cache_dir_candidates", []):
        p = Path(raw)
        if not p.is_absolute():
            p = (repo_root / "gracy" / "experiment_config" / p).resolve()
        candidates.append(p)
    for c in candidates:
        if (c / "predictions" / "stage2_val.json").exists() and (c / "predictions" / "stage5_test.json").exists():
            return c
    raise FileNotFoundError(
        "Missing cached prediction files (stage2_val.json / stage5_test.json). "
        f"Searched: {[str(c) for c in candidates]}"
    )


def extract_features(row: dict[str, Any]) -> np.ndarray:
    """Exact 2,052-dim feature vector matching Gitanjali/JevLikeReasoningModel."""
    p = np.asarray(row["fast"]["probs"], dtype=np.float64)
    s = np.sort(p)
    ent = float(-(p * np.log(np.maximum(p, 1e-12))).sum())
    h = np.asarray(row["fast"]["hidden"], dtype=np.float32)
    return np.r_[s[-1], s[-1] - s[-2], ent, float(row["n_options"]), h]


def check_prediction_file(path: Path, expected_tasks: list[str], expected_per_task: int) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(rows, list), f"{path} must be a JSON list"
    counts: dict[str, int] = {}
    for i, r in enumerate(rows):
        for key in ("id", "task", "label", "n_options", "fast", "reasoning"):
            assert key in r, f"{path}[{i}] missing {key}"
        assert len(r["fast"]["probs"]) == r["n_options"], f"{path}[{i}] fast probs length mismatch"
        assert len(r["reasoning"]["probs"]) == r["n_options"], f"{path}[{i}] reasoning probs length mismatch"
        assert len(r["fast"]["hidden"]) == 2048, f"{path}[{i}] hidden dim != 2048"
        assert math.isclose(sum(r["fast"]["probs"]), 1.0, abs_tol=1e-3), f"{path}[{i}] fast probs sum != 1"
        assert math.isclose(sum(r["reasoning"]["probs"]), 1.0, abs_tol=1e-3), f"{path}[{i}] reason probs sum != 1"
        counts[r["task"]] = counts.get(r["task"], 0) + 1
    for t in expected_tasks:
        assert counts.get(t, 0) == expected_per_task, f"{path}: expected {expected_per_task} for {t}, got {counts.get(t, 0)}"
    return rows


def verify_learned_gate_parity(repo_root: Path, val_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Verify our feature extraction reproduces Gitanjali's exported learned_gate.json."""
    gate_json_path = repo_root / "Gitanjali" / "JevLikeReasoningModel" / "results" / "metrics" / "learned_gate.json"
    temps_path = repo_root / "Gitanjali" / "JevLikeReasoningModel" / "results" / "metrics" / "temperatures.json"
    gate_spec = json.loads(gate_json_path.read_text(encoding="utf-8"))
    temps = json.loads(temps_path.read_text(encoding="utf-8"))

    seen_val = [r for r in val_rows if r["task"] != "word math"]
    # Calibrate with the original temps to replicate Cell 16
    cal_fast = []
    cal_reas = []
    for r in seen_val:
        pf = np.asarray(r["fast"]["probs"], dtype=float) ** (1.0 / temps["fast"])
        pr = np.asarray(r["reasoning"]["probs"], dtype=float) ** (1.0 / temps["reasoning"])
        cal_fast.append(pf / pf.sum())
        cal_reas.append(pr / pr.sum())
    X = np.stack([extract_features(r) for r in seen_val])
    y = np.array([int(np.argmax(pf) != r["label"] and np.argmax(pr) == r["label"])
                  for r, pf, pr in zip(seen_val, cal_fast, cal_reas)])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=2026, stratify=y)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=2026))
    pipe.fit(Xtr, ytr)

    mean_diff = float(np.max(np.abs(pipe.named_steps["standardscaler"].mean_ - np.asarray(gate_spec["model"]["standard_scaler"]["mean"]))))
    coef_diff = float(np.max(np.abs(pipe.named_steps["logisticregression"].coef_ - np.asarray(gate_spec["model"]["logistic_regression"]["coef"]))))
    holdout_acc = float(pipe.score(Xte, yte))
    return {
        "n_train": int(len(Xtr)),
        "n_holdout": int(len(Xte)),
        "max_scaler_mean_abs_diff": mean_diff,
        "max_logreg_coef_abs_diff": coef_diff,
        "reproduced_holdout_acc": holdout_acc,
    }


def deduplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove exact duplicate content IDs while preserving original order."""
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for r in rows:
        if r["id"] not in seen:
            seen.add(r["id"])
            unique.append(r)
    return unique


def partition_eligible_val(
    val_rows: list[dict[str, Any]],
    train_tasks: list[str],
    seed: int,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically partition each eligible task's unique val rows into (val_temp, val_gate_train, val_thresh)."""
    counts = config["validation_partitions"]["per_task_counts"]
    n_temp = counts["val_temp"]
    n_thresh = counts["val_thresh"]
    val_temp, val_gate, val_thresh = [], [], []

    for task in train_tasks:
        rows = deduplicate_rows([r for r in val_rows if r["task"] == task])
        assert len(rows) >= n_temp + n_thresh + 10, f"Too few unique rows for {task}: {len(rows)}"
        raw_rescue = np.array([
            int(np.argmax(r["fast"]["probs"]) != r["label"] and np.argmax(r["reasoning"]["probs"]) == r["label"])
            for r in rows
        ])
        idx = np.arange(len(rows))
        strat1 = raw_rescue if min(np.bincount(raw_rescue, minlength=2)) >= 2 else None
        idx_temp, idx_rest = train_test_split(idx, train_size=n_temp, random_state=seed, stratify=strat1)
        rest_rescue = raw_rescue[idx_rest]
        strat2 = rest_rescue if min(np.bincount(rest_rescue, minlength=2)) >= 2 else None
        idx_gate, idx_thresh = train_test_split(idx_rest, test_size=n_thresh, random_state=seed + 100, stratify=strat2)

        val_temp.extend(rows[i] for i in idx_temp)
        val_gate.extend(rows[i] for i in idx_gate)
        val_thresh.extend(rows[i] for i in idx_thresh)

    # Strict disjointness & exclusion assertions
    id_temp = {r["id"] for r in val_temp}
    id_gate = {r["id"] for r in val_gate}
    id_thresh = {r["id"] for r in val_thresh}
    assert id_temp.isdisjoint(id_gate), "Overlap between val_temp and val_gate_train"
    assert id_temp.isdisjoint(id_thresh), "Overlap between val_temp and val_thresh"
    assert id_gate.isdisjoint(id_thresh), "Overlap between val_gate_train and val_thresh"
    allowed = set(train_tasks)
    assert all(r["task"] in allowed for r in val_temp + val_gate + val_thresh), "Disallowed task entered validation"
    return val_temp, val_gate, val_thresh


def fit_temperature(rows: list[dict[str, Any]], mode: str) -> float:
    ps = [np.asarray(r[mode]["probs"], dtype=np.float64) for r in rows]
    ys = [int(r["label"]) for r in rows]

    def loss(logt: float) -> float:
        t = math.exp(logt)
        total = 0.0
        for p, y in zip(ps, ys):
            q = p ** (1.0 / t)
            q /= q.sum()
            total -= math.log(max(float(q[y]), 1e-12))
        return total / len(ys)

    return float(math.exp(minimize_scalar(loss, bounds=(-3.0, 3.0), method="bounded").x))


def calibrate(rows: list[dict[str, Any]], temps: dict[str, float]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        nr = {
            "id": r["id"],
            "task": r["task"],
            "label": r["label"],
            "n_options": r["n_options"],
            "fast": {**r["fast"]},
            "reasoning": {**r["reasoning"]},
        }
        for mode in ("fast", "reasoning"):
            p = np.asarray(r[mode]["probs"], dtype=np.float64) ** (1.0 / temps[mode])
            nr[mode]["calibrated_probs"] = (p / p.sum()).tolist()
        out.append(nr)
    return out


def select_threshold_gap_sanity(
    rows: list[dict[str, Any]], scores: np.ndarray, target_gap: float = 0.03, grid_points: int = 101
) -> float:
    """Constrained accuracy-gap threshold selection matching run_loto_experiment.py."""
    reason_acc = float(np.mean([int(np.argmax(r["reasoning"]["calibrated_probs"]) == r["label"]) for r in rows]))
    candidates = []
    for tau in np.linspace(0.0, 1.0, grid_points):
        esc = scores >= tau
        acc = float(
            np.mean([
                int(np.argmax((r["reasoning"] if e else r["fast"])["calibrated_probs"]) == r["label"])
                for r, e in zip(rows, esc)
            ])
        )
        candidates.append((float(np.mean(esc)), -acc, float(tau), acc))
    eligible = [c for c in candidates if c[3] >= reason_acc - target_gap]
    chosen = min(eligible or candidates, key=lambda x: (x[0], x[1]))
    return float(chosen[2])


def select_threshold_utility_sanity(
    rows: list[dict[str, Any]], scores: np.ndarray, cost: float = 0.02, grid_points: int = 101
) -> float:
    """Legacy utility-based threshold diagnostic (accuracy - cost * escalation_rate)."""
    best_u, best_tau = -1e9, 1.0
    for tau in np.linspace(0.0, 1.0, grid_points):
        esc = scores >= tau
        acc = float(
            np.mean([
                int(np.argmax((r["reasoning"] if e else r["fast"])["calibrated_probs"]) == r["label"])
                for r, e in zip(rows, esc)
            ])
        )
        u = acc - cost * float(np.mean(esc))
        if u > best_u:
            best_u, best_tau = u, float(tau)
    return best_tau


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = repo_root / "gracy" / "experiment_config" / "loto_config.json"
    config = json.loads(cfg_path.read_text(encoding="utf-8"))
    cache_dir = resolve_cache_dir(args.cache_dir, config, repo_root)
    print(f"[Verify] Using cache directory: {cache_dir}")

    all_tasks = config["tasks"]["all"]
    s2_val = check_prediction_file(cache_dir / "predictions" / "stage2_val.json", all_tasks, 150)
    s5_test = check_prediction_file(cache_dir / "predictions" / "stage5_test.json", all_tasks, 250)
    s0_val = check_prediction_file(cache_dir / "predictions" / "stage0_val.json", all_tasks, 150)
    s0_test = check_prediction_file(cache_dir / "predictions" / "stage0_test.json", all_tasks, 250)

    val_ids = {r["id"] for r in s2_val}
    test_ids = {r["id"] for r in s5_test}
    assert val_ids.isdisjoint(test_ids), "Val/Test ID overlap detected!"
    assert {r["id"] for r in s0_val} == val_ids, "stage0_val IDs differ from stage2_val!"
    assert {r["id"] for r in s0_test} == test_ids, "stage0_test IDs differ from stage5_test!"
    print(f"[Verify] Prediction integrity passed: val={len(s2_val)}, test={len(s5_test)}, overlap=0")

    parity = verify_learned_gate_parity(repo_root, s2_val)
    print(f"[Verify] Original learned_gate.json parity: {json.dumps(parity, indent=2)}")

    target_gap = config["threshold_selection"]["target_accuracy_gap"]
    grid_points = config["threshold_selection"]["candidate_grid_points"]
    esc_cost = config["threshold_selection"]["escalation_cost"]

    # Verify all 5 configs' partitions + threshold selection on val_thresh ONLY (no test peeking)
    for cfg_id, cfg in config["configurations"].items():
        vt, vg, vth = partition_eligible_val(s2_val, cfg["train_tasks"], config["seed"], config)
        temps = {m: fit_temperature(vt, m) for m in ("fast", "reasoning")}
        vg_cal = calibrate(vg, temps)
        vth_cal = calibrate(vth, temps)
        X_tr = np.stack([extract_features(r) for r in vg_cal])
        y_tr = np.array([
            int(np.argmax(r["fast"]["calibrated_probs"]) != r["label"] and np.argmax(r["reasoning"]["calibrated_probs"]) == r["label"])
            for r in vg_cal
        ])
        X_th = np.stack([extract_features(r) for r in vth_cal])
        y_th = np.array([
            int(np.argmax(r["fast"]["calibrated_probs"]) != r["label"] and np.argmax(r["reasoning"]["calibrated_probs"]) == r["label"])
            for r in vth_cal
        ])
        gate = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=2026))
        gate.fit(X_tr, y_tr)
        scores_th = gate.predict_proba(X_th)[:, 1]
        conf_scores_th = np.array([1.0 - max(r["fast"]["calibrated_probs"]) for r in vth_cal])

        tau_gap = select_threshold_gap_sanity(vth_cal, scores_th, target_gap=target_gap, grid_points=grid_points)
        ctau_gap = select_threshold_gap_sanity(vth_cal, conf_scores_th, target_gap=target_gap, grid_points=grid_points)
        tau_util = select_threshold_utility_sanity(vth_cal, scores_th, cost=esc_cost, grid_points=grid_points)

        print(
            f"[Sanity] {cfg_id}: n_temp={len(vt)}, n_gate={len(vg)} (rescues={y_tr.sum()}), "
            f"n_thresh={len(vth)} (rescues={y_th.sum()}), T_fast={temps['fast']:.3f}, T_reas={temps['reasoning']:.3f}, "
            f"learned_tau*(gap={target_gap})={tau_gap:.2f}, conf_tau*(gap={target_gap})={ctau_gap:.2f}, "
            f"legacy_utility_tau*(cost={esc_cost})={tau_util:.2f}"
        )


if __name__ == "__main__":
    main()
