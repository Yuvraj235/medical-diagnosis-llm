# -*- coding: utf-8 -*-
"""
evaluation/generate_final_figures.py
Generate final dissertation figures from ALL real evaluation results.

Includes:
  1. Method comparison bar chart (accuracy + F1-macro)
  2. Per-class F1 heatmap (yes / no / maybe)
  3. LoRA v2 calibration effect (raw vs calibrated)
  4. Prediction distribution stacked bar
  5. Training progression (LoRA v2)
  6. Radar chart — best model vs baselines

Run:
    python evaluation/generate_final_figures.py
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
FIGURES_DIR = os.path.join(Path(__file__).parent.parent, "results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─── All real evaluation results ─────────────────────────────────────────────
METHODS = {
    "Zero-shot\n(orig format)": {
        "acc": 0.4867, "f1m": 0.3566,
        "yes_f1": 0.5286, "no_f1": 0.5414, "maybe_f1": 0.000,
        "yes_r": 0.4933, "no_r": 0.7059, "maybe_r": 0.000,
        "pred": {"yes": 73, "no": 75, "maybe": 2},
    },
    "Zero-shot\n(correct fmt)": {
        "acc": 0.6533, "f1m": 0.4797,
        "yes_f1": 0.7160, "no_f1": 0.7027, "maybe_f1": 0.000,
        "yes_r": 0.8800, "no_r": 0.6667, "maybe_r": 0.000,
        "pred": {"yes": 100, "no": 50, "maybe": 0},
    },
    "3-Shot ICL": {
        "acc": 0.5333, "f1m": 0.3336,
        "yes_f1": 0.6870, "no_f1": 0.3140, "maybe_f1": 0.000,
        "yes_r": 0.9200, "no_r": 0.2160, "maybe_r": 0.000,
        "pred": {"yes": 126, "no": 19, "maybe": 5},
    },
    "LoRA v2\n(raw)": {
        "acc": 0.6533, "f1m": 0.4957,
        "yes_f1": 0.7290, "no_f1": 0.6860, "maybe_f1": 0.071,
        "yes_r": 0.8270, "no_r": 0.6860, "maybe_r": 0.0420,
        "pred": {"yes": 95, "no": 51, "maybe": 4},
    },
    "LoRA v2\n+calibration": {
        "acc": 0.6533, "f1m": 0.5471,
        "yes_f1": 0.7260, "no_f1": 0.6730, "maybe_f1": 0.242,
        "yes_r": 0.7600, "no_r": 0.7250, "maybe_r": 0.1670,
        "pred": {"yes": 82, "no": 59, "maybe": 9},
    },
    "PubMedBERT\nclassifier": {
        "acc": 0.5400, "f1m": 0.4534,
        "yes_f1": 0.641, "no_f1": 0.519, "maybe_f1": 0.200,
        "yes_r": 0.667, "no_r": 0.529, "maybe_r": 0.167,
        "pred": {"yes": 81, "no": 53, "maybe": 16},
    },
}

COLORS = {
    "acc": "#2196F3",
    "f1m": "#FF9800",
    "yes": "#4CAF50",
    "no":  "#F44336",
    "maybe": "#9C27B0",
}

BEST = "LoRA v2\n+calibration"   # best overall method

N_TEST = 150
TRUE_DIST = {"yes": 75, "no": 51, "maybe": 24}


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Method comparison — accuracy + F1-macro
# ─────────────────────────────────────────────────────────────────────────────
def fig_method_comparison():
    names = list(METHODS.keys())
    accs  = [METHODS[n]["acc"]  for n in names]
    f1ms  = [METHODS[n]["f1m"]  for n in names]
    x     = np.arange(len(names))
    w     = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars_a = ax.bar(x - w/2, accs, w, label="Accuracy", color=COLORS["acc"],
                    alpha=0.85, edgecolor="white")
    bars_f = ax.bar(x + w/2, f1ms, w, label="F1 Macro", color=COLORS["f1m"],
                    alpha=0.85, edgecolor="white")

    for bar, val in zip(bars_a, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for bar, val in zip(bars_f, f1ms):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Highlight best
    best_idx = names.index(BEST)
    for bar in [bars_a[best_idx], bars_f[best_idx]]:
        bar.set_edgecolor("#212121"); bar.set_linewidth(2.5)
    ax.annotate("Best\nmodel", xy=(best_idx, max(accs[best_idx], f1ms[best_idx]) + 0.05),
                ha="center", fontsize=9, color="#212121",
                arrowprops=dict(arrowstyle="->", color="#212121"),
                xytext=(best_idx, max(accs[best_idx], f1ms[best_idx]) + 0.12))

    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 0.88)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("PubMedQA Evaluation — Accuracy & F1-Macro by Method\n"
                 f"(n=150 test samples, 3-class: yes/no/maybe)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.axhline(0.70, ls="--", lw=1.2, color="gray", alpha=0.6, label="70% target")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "method_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(); print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Per-class F1 heatmap
# ─────────────────────────────────────────────────────────────────────────────
def fig_perclass_f1():
    names    = list(METHODS.keys())
    classes  = ["YES", "NO", "MAYBE"]
    keys     = ["yes_f1", "no_f1", "maybe_f1"]
    data     = np.array([[METHODS[n][k] for k in keys] for n in names])

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticks(range(len(names)));   ax.set_yticklabels(names, fontsize=9)

    for i in range(len(names)):
        for j in range(len(classes)):
            v   = data[i, j]
            col = "white" if v > 0.55 else "#212121"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=9, color=col, fontweight="bold")

    plt.colorbar(im, ax=ax, label="F1 Score")
    ax.set_title("Per-class F1 Score by Method\n(higher = better)", fontsize=12,
                 fontweight="bold")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "perclass_f1_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(); print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Calibration effect (raw vs calibrated vs baseline)
# ─────────────────────────────────────────────────────────────────────────────
def fig_calibration_effect():
    labels = ["YES F1", "NO F1", "MAYBE F1", "F1 Macro", "Accuracy"]
    zero   = [0.716, 0.703, 0.000, 0.480, 0.653]
    raw    = [0.729, 0.686, 0.071, 0.496, 0.653]
    cal    = [0.726, 0.673, 0.242, 0.547, 0.653]

    x = np.arange(len(labels))
    w = 0.26

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w,   zero, w, label="Zero-shot (correct fmt)", color="#78909C", alpha=0.85)
    ax.bar(x,       raw,  w, label="LoRA v2 raw",             color="#42A5F5", alpha=0.85)
    ax.bar(x + w,   cal,  w, label="LoRA v2 + calibration",  color="#66BB6A", alpha=0.85)

    for bars, vals in [([x-w], zero), ([x], raw), ([x+w], cal)]:
        for xi, v in zip(bars[0], vals):
            ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 0.82)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Effect of LoRA Fine-tuning & Per-class Calibration on PubMedQA Metrics",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "calibration_effect.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(); print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Prediction distribution stacked bar
# ─────────────────────────────────────────────────────────────────────────────
def fig_pred_distribution():
    names = list(METHODS.keys())
    yes_c = [METHODS[n]["pred"]["yes"]   for n in names]
    no_c  = [METHODS[n]["pred"]["no"]    for n in names]
    may_c = [METHODS[n]["pred"]["maybe"] for n in names]
    x     = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.bar(x, yes_c, label="Pred YES",   color=COLORS["yes"],   alpha=0.85)
    b2 = ax.bar(x, no_c,  bottom=yes_c,       label="Pred NO",       color=COLORS["no"],    alpha=0.85)
    b3 = ax.bar(x, may_c, bottom=[y+n for y,n in zip(yes_c,no_c)],
                label="Pred MAYBE", color=COLORS["maybe"], alpha=0.85)

    ax.axhline(TRUE_DIST["yes"],                         ls="--", color=COLORS["yes"],   lw=1.5,
               alpha=0.6, label=f"True YES={TRUE_DIST['yes']}")
    ax.axhline(TRUE_DIST["yes"]+TRUE_DIST["no"],         ls="--", color=COLORS["no"],    lw=1.5,
               alpha=0.6, label=f"True NO={TRUE_DIST['no']}")
    ax.axhline(N_TEST,                                   ls=":",  color="gray",           lw=1)

    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Predicted count (out of 150)", fontsize=11)
    ax.set_title("Prediction Distribution by Method\n"
                 f"(True dist — YES:{TRUE_DIST['yes']}  NO:{TRUE_DIST['no']}  MAYBE:{TRUE_DIST['maybe']})",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.5, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "pred_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(); print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Radar chart — best vs others
# ─────────────────────────────────────────────────────────────────────────────
def fig_radar():
    categories  = ["Accuracy", "F1 Macro", "YES F1", "NO F1", "MAYBE F1"]
    n_cat       = len(categories)
    angles      = np.linspace(0, 2*np.pi, n_cat, endpoint=False).tolist()
    angles     += angles[:1]

    def values(name):
        m = METHODS[name]
        v = [m["acc"], m["f1m"], m["yes_f1"], m["no_f1"], m["maybe_f1"]]
        return v + v[:1]

    sel = ["Zero-shot\n(correct fmt)", "LoRA v2\n(raw)",
           "LoRA v2\n+calibration", "PubMedBERT\nclassifier"]
    color_map = ["#78909C", "#42A5F5", "#66BB6A", "#FF7043"]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    for name, col in zip(sel, color_map):
        v   = values(name)
        ax.plot(angles, v, "o-", linewidth=2, color=col, label=name.replace("\n"," "))
        ax.fill(angles, v, alpha=0.1, color=col)

    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    ax.set_title("Multi-dimensional Performance Comparison\n(PubMedQA, n=150 test)",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8.5)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "radar_chart_final.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(); print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: MAYBE recall improvement across methods
# ─────────────────────────────────────────────────────────────────────────────
def fig_maybe_recall():
    names    = list(METHODS.keys())
    maybe_r  = [METHODS[n]["maybe_r"] for n in names]
    colors   = [COLORS["maybe"] if n == BEST else "#B0BEC5" for n in names]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(range(len(names)), maybe_r, color=colors, alpha=0.87, edgecolor="white")
    for bar, val in zip(bars, maybe_r):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                f"{val:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("MAYBE Recall", fontsize=11)
    ax.set_ylim(0, 0.32)
    ax.set_title('MAYBE-Class Recall — Critical for Clinical "Uncertain" Detection\n'
                 '(Higher = model better handles ambiguous medical evidence)',
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "maybe_recall.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(); print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: Updated baseline_vs_finetuned (replacing old one)
# ─────────────────────────────────────────────────────────────────────────────
def fig_baseline_vs_finetuned():
    """Grouped bar: zero-shot vs LoRA v2 vs LoRA v2 + calibration."""
    metrics = ["Accuracy", "F1 Macro", "YES F1", "NO F1", "MAYBE F1"]
    z_vals  = [0.6533, 0.4797, 0.716, 0.703, 0.000]
    l_vals  = [0.6533, 0.4957, 0.729, 0.686, 0.071]
    c_vals  = [0.6533, 0.5471, 0.726, 0.673, 0.242]

    x = np.arange(len(metrics))
    w = 0.26

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - w,   z_vals, w, label="Zero-shot (correct format)", color="#546E7A", alpha=0.85)
    ax.bar(x,       l_vals, w, label="LoRA v2 fine-tuned",         color="#1565C0", alpha=0.85)
    ax.bar(x + w,   c_vals, w, label="LoRA v2 + calibration",     color="#2E7D32", alpha=0.85)

    for offset, vals in [(-w, z_vals), (0, l_vals), (w, c_vals)]:
        for xi, v in zip(x + offset, vals):
            ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 0.88)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Baseline vs. Fine-tuned vs. Calibrated — PubMedQA (n=150)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "baseline_vs_finetuned.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(); print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8: Summary metrics table image
# ─────────────────────────────────────────────────────────────────────────────
def fig_metrics_table():
    names   = [n.replace("\n", " ") for n in METHODS.keys()]
    headers = ["Method", "Accuracy", "F1 Macro", "YES F1", "NO F1", "MAYBE F1"]
    rows    = [[n,
                f"{METHODS[k]['acc']:.3f}",
                f"{METHODS[k]['f1m']:.3f}",
                f"{METHODS[k]['yes_f1']:.3f}",
                f"{METHODS[k]['no_f1']:.3f}",
                f"{METHODS[k]['maybe_f1']:.3f}"]
               for n, k in zip(names, METHODS)]

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=headers,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.9)

    # Style header
    for j in range(len(headers)):
        tbl[(0, j)].set_facecolor("#1565C0")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    # Highlight best row
    best_row = list(METHODS.keys()).index(BEST) + 1
    for j in range(len(headers)):
        tbl[(best_row, j)].set_facecolor("#C8E6C9")
        tbl[(best_row, j)].set_text_props(fontweight="bold")

    ax.set_title("Comprehensive PubMedQA Evaluation Results  (★ = best model)",
                 fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "metrics_table.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(); print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating final dissertation figures ...")
    fig_method_comparison()
    fig_perclass_f1()
    fig_calibration_effect()
    fig_pred_distribution()
    fig_radar()
    fig_maybe_recall()
    fig_baseline_vs_finetuned()
    fig_metrics_table()
    print(f"\nAll figures saved to: {FIGURES_DIR}")
    print("Best method: LoRA v2 + calibration")
    print("  Accuracy : 65.3%")
    print("  F1 Macro : 0.547  (+14% vs zero-shot)")
    print("  MAYBE F1 : 0.242  (was 0.000 zero-shot)")
