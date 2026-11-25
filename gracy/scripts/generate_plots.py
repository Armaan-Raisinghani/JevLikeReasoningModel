#!/usr/bin/env python3
"""Generate publication-quality plots for the JEV LOTO gate generalisation study."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    res_dir = repo_root / "gracy" / "results"
    plot_dir = repo_root / "gracy" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    thresh_df = pd.read_csv(res_dir / "threshold_eval_detailed.csv")
    mg_df = pd.read_csv(res_dir / "matched_escalation_global.csv")
    mpt_df = pd.read_csv(res_dir / "matched_escalation_per_task.csv")
    loto_df = pd.read_csv(res_dir / "loto_generalisation_summary.csv")
    sweeps = json.loads((res_dir / "sweep_curves.json").read_text(encoding="utf-8"))

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    # 1. Excluded-task generalisation comparison at matched 20% escalation & at val-selected tau*
    excl_rows = loto_df[loto_df["evaluated_on"] != "word math"].copy()
    tasks = excl_rows["excluded_task"].tolist()
    x = np.arange(len(tasks))
    w = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    # Left: Validation-selected threshold tau*
    ax = axes[0]
    ax.bar(x - 1.5 * w, excl_rows["fast_acc"], w, label="Fast-only", color="#8c8c8c")
    ax.bar(x - 0.5 * w, excl_rows["conf_all4_tau_acc"], w, label="Confidence gate (val τ*)", color="#e07a5f")
    ax.bar(x + 0.5 * w, excl_rows["all4_learned_tau_acc"], w, label="Learned Config A (All 4)", color="#3d5a80")
    ax.bar(x + 1.5 * w, excl_rows["loto_learned_tau_acc"], w, label="Learned LOTO (Excluded)", color="#2a9d8f")
    ax.plot(x, excl_rows["reason_acc"], "k--o", lw=1.3, ms=5, label="Always-reason")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace(" ", "\n") for t in tasks])
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("Accuracy")
    ax.set_title("Excluded Task Accuracy (Validation-Selected τ*)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")

    # Right: Matched 20% escalation rate
    ax = axes[1]
    ax.bar(x - 1.5 * w, excl_rows["fast_acc"], w, label="Fast-only", color="#8c8c8c")
    ax.bar(x - 0.5 * w, excl_rows["conf_acc_at_20pct"], w, label="Confidence @ 20%", color="#e07a5f")
    ax.bar(x + 0.5 * w, excl_rows["all4_acc_at_20pct"], w, label="Learned Config A @ 20%", color="#3d5a80")
    ax.bar(x + 1.5 * w, excl_rows["loto_acc_at_20pct"], w, label="Learned LOTO @ 20%", color="#2a9d8f")
    ax.plot(x, excl_rows["reason_acc"], "k--o", lw=1.3, ms=5, label="Always-reason")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace(" ", "\n") for t in tasks])
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("Accuracy")
    ax.set_title("Excluded Task Accuracy (Matched 20% Escalation)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")

    fig.tight_layout()
    fig.savefig(plot_dir / "loto_excluded_task_comparison.png", dpi=180)
    plt.close(fig)

    # 2. Matched escalation curves (10%, 20%, 30%, 40%) across Overall, Seen-4, Word Math
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    cfgs = [
        ("A_all_seen", "Learned A (All 4)", "#1d3557", "-o"),
        ("B_excl_sentiment", "LOTO B (-Sentiment)", "#2a9d8f", "--s"),
        ("C_excl_inference", "LOTO C (-Inference)", "#e9c46a", "--^"),
        ("D_excl_science", "LOTO D (-Science)", "#f4a261", "--d"),
        ("E_excl_general_knowledge", "LOTO E (-GenKnow)", "#9b5de5", "--v"),
    ]
    panels = [
        ("overall_all_5", "Overall (5 Tasks, n=1,250)", True),
        ("seen_tasks_all_4", "Seen Tasks (4 Tasks, n=1,000)", True),
        ("word math", "Held-Out Word Math (AQuA-RAT, n=250)", False),
    ]
    for ax, (target, title, is_global) in zip(axes, panels):
        if is_global:
            df_src = mg_df[mg_df["pool"] == target]
            fast_acc = df_src["fast_accuracy"].iloc[0]
            reas_acc = df_src["reasoning_accuracy"].iloc[0]
            conf_sub = df_src[(df_src["config_id"] == "A_all_seen") & (df_src["gate_kind"] == "confidence")].sort_values("target_escalation_rate")
            rand_sub = df_src[(df_src["config_id"] == "A_all_seen") & (df_src["gate_kind"] == "random_expected")].sort_values("target_escalation_rate")
            ax.plot(rand_sub["target_escalation_rate"] * 100, rand_sub["gated_accuracy"], "k:", lw=1.2, label="Random baseline")
        else:
            df_src = mpt_df[mpt_df["eval_group"] == target]
            fast_acc = df_src["fast_accuracy"].iloc[0]
            reas_acc = df_src["reasoning_accuracy"].iloc[0]
            conf_sub = df_src[(df_src["config_id"] == "A_all_seen") & (df_src["gate_kind"] == "confidence")].sort_values("target_escalation_rate")
            rates_r = np.array([10, 20, 30, 40])
            ax.plot(rates_r, fast_acc + (rates_r / 100.0) * (reas_acc - fast_acc), "k:", lw=1.2, label="Random baseline")

        ax.axhline(fast_acc, color="#8c8c8c", ls="--", lw=1.0, label=f"Fast ({fast_acc:.3f})")
        ax.axhline(reas_acc, color="#222222", ls="-.", lw=1.0, label=f"Reason ({reas_acc:.3f})")
        ax.plot(conf_sub["target_escalation_rate"] * 100, conf_sub["gated_accuracy"], color="#e07a5f", marker="X", lw=2.0, label="Confidence gate (A)")

        for cid, cname, col, fmt in cfgs:
            sub = df_src[(df_src["config_id"] == cid) & (df_src["gate_kind"] == "learned")].sort_values("target_escalation_rate")
            ax.plot(sub["target_escalation_rate"] * 100, sub["gated_accuracy"], fmt, color=col, lw=1.7, ms=5, label=cname)

        ax.set_xlabel("Matched Escalation Rate (%)")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.set_xticks([10, 20, 30, 40])
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=7.5, loc="lower right", ncol=1)
    fig.tight_layout()
    fig.savefig(plot_dir / "matched_escalation_overall_and_tasks.png", dpi=180)
    plt.close(fig)

    # 3. Word math generalisation summary (val-selected tau* + matched escalation rates)
    wm_t = thresh_df[(thresh_df["eval_group"] == "word math") & (thresh_df["policy"] == "learned_gate_val_tau")].copy()
    wm_c = thresh_df[(thresh_df["eval_group"] == "word math") & (thresh_df["policy"] == "confidence_gate_val_tau") & (thresh_df["config_id"] == "A_all_seen")].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    ax = axes[0]
    labels = ["Conf (A)", "A (All 4)", "B (-Sent)", "C (-Inf)", "D (-Sci)", "E (-GK)"]
    accs = [wm_c["gated_accuracy"]] + wm_t["gated_accuracy"].tolist()
    escs = [wm_c["escalation_rate"]] + wm_t["escalation_rate"].tolist()
    colors = ["#e07a5f", "#1d3557", "#2a9d8f", "#e9c46a", "#f4a261", "#9b5de5"]
    bars = ax.bar(labels, accs, color=colors)
    ax.axhline(wm_c["fast_accuracy"], color="#8c8c8c", ls="--", label=f"Fast ({wm_c['fast_accuracy']:.3f})")
    ax.axhline(wm_c["reasoning_accuracy"], color="black", ls=":", label=f"Always-Reason ({wm_c['reasoning_accuracy']:.3f})")
    for bar, esc in zip(bars, escs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012, f"esc\n{esc:.0%}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 0.78)
    ax.set_ylabel("Accuracy on Word Math (n=250)")
    ax.set_title("Held-Out AQuA-RAT at Validation τ*")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(axis="y", alpha=0.25)

    # Right: Full accuracy vs escalation sweep on Word Math
    ax = axes[1]
    pts_c = sweeps["A_all_seen"]["word math"]["confidence"]
    ax.plot([p["escalation_rate"] * 100 for p in pts_c], [p["gated_accuracy"] for p in pts_c], color="#e07a5f", lw=2.0, label="Confidence (A)")
    for cid, cname, col, fmt in cfgs:
        pts = sweeps[cid]["word math"]["learned"]
        ax.plot([p["escalation_rate"] * 100 for p in pts], [p["gated_accuracy"] for p in pts], color=col, lw=1.6, ls="-" if cid == "A_all_seen" else "--", label=cname)
    ax.axhline(wm_c["fast_accuracy"], color="#8c8c8c", ls="--", lw=1.0)
    ax.axhline(wm_c["reasoning_accuracy"], color="black", ls=":", lw=1.0)
    ax.set_xlabel("Escalation Rate (%)")
    ax.set_ylabel("Accuracy on Word Math")
    ax.set_title("Held-Out AQuA-RAT Full Accuracy–Escalation Frontier")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "word_math_generalisation.png", dpi=180)
    plt.close(fig)

    # 4. Pareto: Accuracy vs Average Generated Tokens (Overall & Word Math)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    for ax, target, title in zip(axes, ["overall_all_5", "word math"], ["Overall (n=1,250)", "Held-Out Word Math (n=250)"]):
        pts_c = sweeps["A_all_seen"][target]["confidence"]
        ax.plot([p["avg_generated_tokens"] for p in pts_c], [p["gated_accuracy"] for p in pts_c], color="#e07a5f", lw=2.0, label="Confidence (A)")
        for cid, cname, col, _ in cfgs:
            pts = sweeps[cid][target]["learned"]
            ax.plot([p["avg_generated_tokens"] for p in pts], [p["gated_accuracy"] for p in pts], color=col, lw=1.6, ls="-" if cid == "A_all_seen" else "--", label=cname)
        ax.set_xlabel("Average Generated Tokens / Query")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Accuracy vs Token Cost — {title}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(plot_dir / "pareto_accuracy_vs_tokens.png", dpi=180)
    plt.close(fig)

    # 5. Heatmap of Gate Rescue ROC-AUC across Configs x Tasks
    task_order = ["sentiment", "inference", "science questions", "general knowledge", "word math"]
    cfg_order = ["A_all_seen", "B_excl_sentiment", "C_excl_inference", "D_excl_science", "E_excl_general_knowledge"]
    cfg_labels = ["Config A\n(All 4)", "Config B\n(-Sent)", "Config C\n(-Inf)", "Config D\n(-Sci)", "Config E\n(-GenKnow)"]
    sub = thresh_df[(thresh_df["policy"] == "learned_gate_val_tau") & (thresh_df["eval_group"].isin(task_order))]
    mat = np.zeros((len(cfg_order), len(task_order)))
    for i, cid in enumerate(cfg_order):
        for j, t in enumerate(task_order):
            val = sub[(sub["config_id"] == cid) & (sub["eval_group"] == t)]["gate_rescue_roc_auc"].iloc[0]
            mat[i, j] = float(val)

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0.45, vmax=0.90)
    ax.set_xticks(np.arange(len(task_order)))
    ax.set_xticklabels([t.replace(" ", "\n") for t in task_order])
    ax.set_yticks(np.arange(len(cfg_order)))
    ax.set_yticklabels(cfg_labels)
    for i in range(len(cfg_order)):
        for j in range(len(task_order)):
            is_loto_excl = (i >= 1 and j == i - 1)
            is_word_math = (j == 4)
            tag = " *" if is_loto_excl else (" †" if is_word_math else "")
            txt_color = "white" if mat[i, j] > 0.72 else "black"
            ax.text(j, i, f"{mat[i, j]:.3f}{tag}", ha="center", va="center", color=txt_color, fontweight="bold" if is_loto_excl else "normal")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Rescue-Prediction ROC-AUC")
    ax.set_title("Gate Rescue ROC-AUC (* = LOTO-Excluded Task, † = Held-Out Word Math)")
    fig.tight_layout()
    fig.savefig(plot_dir / "rescue_roc_auc_heatmap.png", dpi=180)
    plt.close(fig)
    print(f"Generated 5 plots in {plot_dir}")


if __name__ == "__main__":
    main()
