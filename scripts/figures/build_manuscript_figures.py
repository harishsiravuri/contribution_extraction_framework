"""Build the four manuscript figures from on-disk experimental artifacts.

No LLM calls, no Together/OpenRouter spend — pure matplotlib + numpy plotting.
Each figure is saved at 300 DPI in both .png and .pdf, transparent background,
to `figures_out/` at the project root.

INPUT SOURCES + EXACT NUMBERS USED
==================================

Figure 1 — fig_architecture
  - Non-data schematic. Uses matplotlib patches to draw the pipeline
    boxes (Extractor / Voting / Critic / Consolidator) and arrows.
  - Family-diversity colour cue: Extractor box in the "DeepSeek" colour;
    Critic + Consolidator boxes in the "Llama" colour. Both colours are
    drawn from the seaborn 'colorblind' palette.

Figure 2 — fig_benchmark_f1
  Source A: outputs/paper_data_v3/benchmarks/scirex/evaluation.json
    (SciREX dev, n=62 paired, default multi-agent vs single-LLM baseline)
      methods   multi 0.4839 [0.3666, 0.5968]  base 0.4957 [0.3667, 0.6097]  p=0.6429
      tasks     multi 0.5591 [0.4516, 0.6613]  base 0.3495 [0.2473, 0.4571]  p=0.0004 *
      datasets  multi 0.4589 [0.3776, 0.5513]  base 0.4360 [0.3483, 0.5248]  p=0.4549
      metrics   multi 0.4227 [0.3354, 0.5150]  base 0.3068 [0.2162, 0.4039]  p=0.0002 *
  Source B: outputs/paper_data_v6/benchmarks_ft_70b_test/scirex/evaluation.json
    (SciREX test, n=65 paired, specialized 70B FT framework)
      methods   multi 0.5821 [0.4640, 0.6924]
      tasks     multi 0.7677 [0.6779, 0.8590]
      datasets  multi 0.5271 [0.4331, 0.6216]
      metrics   multi 0.6394 [0.5414, 0.7323]
  Source C: Jain et al. 2020 ACL — author-reported on SciREX test split
      methods 0.567   tasks 0.610   datasets 0.553   metrics 0.553

Figure 3 — fig_reliability_diagram
  Source: outputs/paper_data_v2/calibration/calibration.json
    per_field['method.name']:
      n=61, T=20.0
      ece_uncalibrated = 0.4292
      ece_calibrated   = 0.1110
      reliability_uncalibrated = list of 10 bins {bin_lo, bin_hi, n,
                                  mean_confidence, accuracy}
      reliability_calibrated   = same shape after temperature scaling

Figure 4 — fig_cost_vs_quality
  Task-F1 source: Figure 2 sources (dev for first two, test for third)
    Single-LLM baseline (DeepSeek)       Task F1 = 0.3495 [0.2473, 0.4571]
    Multi-agent open-weights default     Task F1 = 0.5591 [0.4516, 0.6613]
    Specialized (Llama 3.1 70B FT)       Task F1 = 0.7677 [0.6779, 0.8590]
  Cost-per-paper source: outputs/paper_data_v6/RESULTS.md headline numbers
    (DeepSeek baseline ≈ $0.001/paper; default multi-agent ≈ $0.005/paper;
     specialized 70B FT endpoint ≈ $0.18/paper from $48 endpoint / ~260
     papers across runs ⇒ ~$0.18/paper amortised on the test run).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures_out"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
PALETTE = sns.color_palette("colorblind")
# Stable named colours so labels are consistent across figures
COL_DEEPSEEK = PALETTE[0]     # blue
COL_LLAMA = PALETTE[1]        # orange
COL_MULTI_DEV = PALETTE[0]    # blue
COL_BASE_DEV = PALETTE[7]     # grey
COL_SPECIALIZED = PALETTE[2]  # green
COL_JAIN = PALETTE[3]         # red

mpl.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.transparent": True,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


def _save(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf"):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, dpi=300, transparent=True, bbox_inches="tight")
        print(f"  wrote {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 — Architecture diagram
# ---------------------------------------------------------------------------
def fig_architecture() -> None:
    print("[1/4] fig_architecture")
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def _box(x, y, w, h, label, fc, ec="black", lw=1.2, fontsize=10, fontweight="normal"):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.15",
            facecolor=fc, edgecolor=ec, linewidth=lw,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, fontweight=fontweight, wrap=True)

    def _arrow(x0, y0, x1, y1, lw=1.4, color="black"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", lw=lw, color=color))

    # Paper input
    _box(0.2, 2.0, 1.4, 1.0, "Paper\ntext", fc="white", lw=1.2, fontsize=10)

    # Extractor — 3 samples at three temperatures
    _box(2.2, 3.2, 2.6, 1.2, "Extractor\n(DeepSeek Chat)\nt=0.0", fc=(*COL_DEEPSEEK, 0.45), fontsize=9, fontweight="bold")
    _box(2.2, 1.8, 2.6, 1.2, "Extractor\n(DeepSeek Chat)\nt=0.3", fc=(*COL_DEEPSEEK, 0.45), fontsize=9, fontweight="bold")
    _box(2.2, 0.4, 2.6, 1.2, "Extractor\n(DeepSeek Chat)\nt=0.7", fc=(*COL_DEEPSEEK, 0.45), fontsize=9, fontweight="bold")

    # Self-consistency voting (small box)
    _box(5.3, 1.9, 1.4, 1.0, "Self-\nconsistency\nvoting", fc="white", lw=1.0, fontsize=8)

    # Critic
    _box(7.2, 1.9, 2.0, 1.0, "Critic\n(Llama 3.3 70B)",
         fc=(*COL_LLAMA, 0.45), fontsize=9, fontweight="bold")

    # Consolidator
    _box(9.7, 1.9, 2.0, 1.0, "Consolidator\n(Llama 3.3 70B)",
         fc=(*COL_LLAMA, 0.45), fontsize=9, fontweight="bold")

    # Output below
    _box(9.7, 0.1, 2.0, 0.8, "Contribution\nRecord", fc="white", lw=1.2, fontsize=9, fontweight="bold")

    # Arrows: Paper -> 3 extractors
    for y in (3.8, 2.4, 1.0):
        _arrow(1.6, 2.5, 2.2, y)

    # Arrows: 3 extractors -> Voting
    for y in (3.8, 2.4, 1.0):
        _arrow(4.8, y, 5.3, 2.4)

    # Voting -> Critic -> Consolidator
    _arrow(6.7, 2.4, 7.2, 2.4)
    _arrow(9.2, 2.4, 9.7, 2.4)

    # Consolidator -> ContributionRecord
    _arrow(10.7, 1.9, 10.7, 0.9)

    # Family-diversity legend
    legend_handles = [
        mpatches.Patch(facecolor=(*COL_DEEPSEEK, 0.45), edgecolor="black", label="DeepSeek family"),
        mpatches.Patch(facecolor=(*COL_LLAMA, 0.45), edgecolor="black", label="Llama family"),
    ]
    ax.legend(handles=legend_handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)

    ax.set_title("Multi-agent extraction pipeline", fontweight="bold")
    _save(fig, "fig_architecture")


# ---------------------------------------------------------------------------
# Figure 2 — Per-field F1 across deployments
# ---------------------------------------------------------------------------
def fig_benchmark_f1() -> None:
    print("[2/4] fig_benchmark_f1")
    # Load source A: dev split
    dev = json.loads((ROOT / "outputs/paper_data_v3/benchmarks/scirex/evaluation.json").read_text())["fields"]
    # Load source B: test split (specialized 70B FT)
    test = json.loads((ROOT / "outputs/paper_data_v6/benchmarks_ft_70b_test/scirex/evaluation.json").read_text())["fields"]
    # Source C: Jain 2020 reported on test
    jain = {"methods": 0.567, "tasks": 0.610, "datasets": 0.553, "metrics": 0.553}

    fields = ("methods", "tasks", "datasets", "metrics")
    field_labels = ("Method", "Task", "Dataset", "Metric")

    rows = []
    for f, label in zip(fields, field_labels):
        m = dev[f]["multi_agent"]
        b = dev[f]["baseline"]
        s = test[f]["multi_agent"]
        rows.append({
            "field": label,
            "multi_dev": (m["f1"], m["ci_lo"], m["ci_hi"]),
            "base_dev": (b["f1"], b["ci_lo"], b["ci_hi"]),
            "specialized": (s["f1"], s["ci_lo"], s["ci_hi"]),
            "jain": jain[f],
            "p_multi_vs_base": dev[f]["p_value"],
        })

    # Print extracted numbers for sanity-check
    print("  extracted F1 values:")
    for r in rows:
        print(f"    {r['field']:8s}  multi-dev={r['multi_dev'][0]:.3f}  base-dev={r['base_dev'][0]:.3f}  "
              f"specialized-test={r['specialized'][0]:.3f}  jain-test={r['jain']:.3f}  p={r['p_multi_vs_base']:.4f}")

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5.0))
    n_groups = len(rows)
    n_bars = 4  # multi-dev, base-dev, specialized, jain
    bar_w = 0.20
    x = np.arange(n_groups)

    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_w

    multi_f1 = [r["multi_dev"][0] for r in rows]
    multi_err = np.array([[r["multi_dev"][0] - r["multi_dev"][1] for r in rows],
                          [r["multi_dev"][2] - r["multi_dev"][0] for r in rows]])
    base_f1 = [r["base_dev"][0] for r in rows]
    base_err = np.array([[r["base_dev"][0] - r["base_dev"][1] for r in rows],
                         [r["base_dev"][2] - r["base_dev"][0] for r in rows]])
    spec_f1 = [r["specialized"][0] for r in rows]
    spec_err = np.array([[r["specialized"][0] - r["specialized"][1] for r in rows],
                         [r["specialized"][2] - r["specialized"][0] for r in rows]])
    jain_f1 = [r["jain"] for r in rows]

    ax.bar(x + offsets[0], multi_f1, bar_w, yerr=multi_err, capsize=3,
           color=COL_MULTI_DEV, edgecolor="black", linewidth=0.6,
           label="Multi-agent (dev)")
    ax.bar(x + offsets[1], base_f1, bar_w, yerr=base_err, capsize=3,
           color=COL_BASE_DEV, edgecolor="black", linewidth=0.6,
           label="Single-LLM (dev)")
    ax.bar(x + offsets[2], spec_f1, bar_w, yerr=spec_err, capsize=3,
           color=COL_SPECIALIZED, edgecolor="black", linewidth=0.6,
           label="Specialized 70B FT (test)")
    ax.bar(x + offsets[3], jain_f1, bar_w,
           color=COL_JAIN, edgecolor="black", linewidth=0.6,
           label="Jain et al. 2020 (test, reported)")

    # Significance asterisks above the multi-agent (dev) bar where p<0.05
    for i, r in enumerate(rows):
        if r["p_multi_vs_base"] < 0.05:
            y = r["multi_dev"][2] + 0.03
            ax.text(x[i] + offsets[0], y, "*", ha="center", va="bottom",
                    fontsize=12, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(field_labels)
    ax.set_ylabel(r"$F_1$ (set-level lenient)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Per-field $F_1$ across deployments and baselines on SciREX",
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Caption-internal honest-split note
    ax.text(0.5, -0.18,
            "Dev split: multi-agent + single-LLM baseline (n=62 paired).  "
            "Test split: specialized framework (n=65) and Jain et al. 2020 (reported).\n"
            "* = paired permutation p < 0.05 vs single-LLM baseline.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
            style="italic", color="dimgray")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28),
              ncol=4, frameon=False)
    _save(fig, "fig_benchmark_f1")


# ---------------------------------------------------------------------------
# Figure 3 — Reliability diagram (Method field, before vs after T-scaling)
# ---------------------------------------------------------------------------
def fig_reliability_diagram() -> None:
    print("[3/4] fig_reliability_diagram")
    calib = json.loads((ROOT / "outputs/paper_data_v2/calibration/calibration.json").read_text())
    m = calib["per_field"]["method.name"]
    print(f"  source: paper_data_v2/calibration/calibration.json[per_field][method.name]")
    print(f"    n={m['n']}  T={m['T']:.3f}  ECE pre={m['ece_uncalibrated']:.4f}  post={m['ece_calibrated']:.4f}")

    uncal = m["reliability_uncalibrated"]
    cal = m["reliability_calibrated"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)

    def _draw(ax, bins, ece, title):
        # Diagonal
        ax.plot([0, 1], [0, 1], "--", color="dimgray", lw=1.0, alpha=0.7, label="Perfect calibration")
        # Bars at bin midpoints with observed accuracy
        # Skip empty bins (n==0)
        for b in bins:
            mid = (b["bin_lo"] + b["bin_hi"]) / 2.0
            if b["n"] == 0:
                continue
            ax.bar(mid, b["accuracy"], width=0.09,
                   color=COL_MULTI_DEV, edgecolor="black", linewidth=0.7,
                   alpha=0.85)
            # Annotate bin population just above the bar (or at 0.02 if accuracy is 0)
            y_anno = max(b["accuracy"], 0.0) + 0.025
            ax.text(mid, y_anno, f"n={b['n']}", ha="center", va="bottom",
                    fontsize=7.5, color="black")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Predicted confidence")
        ax.set_title(title, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.text(0.02, 0.95, f"ECE = {ece:.3f}",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=11, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="black",
                          boxstyle="round,pad=0.3"))

    _draw(axes[0], uncal, m["ece_uncalibrated"],
          "Uncalibrated")
    _draw(axes[1], cal, m["ece_calibrated"],
          f"Temperature-scaled (T = {m['T']:.1f})")

    axes[0].set_ylabel("Observed accuracy")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Reliability diagram — Method field, before vs after temperature scaling",
                 fontweight="bold", y=1.02)
    _save(fig, "fig_reliability_diagram")


# ---------------------------------------------------------------------------
# Figure 4 — Cost vs Task F1 scatter
# ---------------------------------------------------------------------------
def fig_cost_vs_quality() -> None:
    print("[4/4] fig_cost_vs_quality")
    dev = json.loads((ROOT / "outputs/paper_data_v3/benchmarks/scirex/evaluation.json").read_text())["fields"]
    test = json.loads((ROOT / "outputs/paper_data_v6/benchmarks_ft_70b_test/scirex/evaluation.json").read_text())["fields"]

    base_t = dev["tasks"]["baseline"]
    multi_t = dev["tasks"]["multi_agent"]
    spec_t = test["tasks"]["multi_agent"]

    # Cost per paper (USD), from RESULTS.md headline numbers
    points = [
        {
            "label": "Single-LLM baseline\n(DeepSeek Chat, dev)",
            "cost": 0.001,
            "f1": base_t["f1"],
            "ci_lo": base_t["ci_lo"],
            "ci_hi": base_t["ci_hi"],
            "color": COL_BASE_DEV,
            "marker": "o",
        },
        {
            "label": "Multi-agent default\n(open-weights, dev)",
            "cost": 0.005,
            "f1": multi_t["f1"],
            "ci_lo": multi_t["ci_lo"],
            "ci_hi": multi_t["ci_hi"],
            "color": COL_MULTI_DEV,
            "marker": "s",
        },
        {
            "label": "Specialized\n(Llama 3.1 70B FT, test)",
            "cost": 0.18,
            "f1": spec_t["f1"],
            "ci_lo": spec_t["ci_lo"],
            "ci_hi": spec_t["ci_hi"],
            "color": COL_SPECIALIZED,
            "marker": "^",
        },
    ]

    print("  extracted points:")
    for p in points:
        print(f"    {p['label'].replace(chr(10),' '):60s}  cost=${p['cost']:.3f}  F1={p['f1']:.3f} [{p['ci_lo']:.3f}, {p['ci_hi']:.3f}]")

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    # Dashed connector line in cost order
    sorted_pts = sorted(points, key=lambda p: p["cost"])
    ax.plot([p["cost"] for p in sorted_pts],
            [p["f1"] for p in sorted_pts],
            "--", color="dimgray", lw=1.0, alpha=0.7, zorder=1)

    for p in points:
        yerr = np.array([[p["f1"] - p["ci_lo"]], [p["ci_hi"] - p["f1"]]])
        ax.errorbar(p["cost"], p["f1"], yerr=yerr,
                    fmt=p["marker"], markersize=12, mec="black", mew=0.8,
                    color=p["color"], capsize=4, lw=1.2, zorder=3)

    # Labels offset from points
    label_offsets = [(1.5, -0.06), (1.5, -0.08), (0.32, -0.10)]
    for p, (xmult, ydy) in zip(points, label_offsets):
        ax.annotate(p["label"],
                    xy=(p["cost"], p["f1"]),
                    xytext=(p["cost"] * xmult, p["f1"] + ydy),
                    fontsize=9, ha="left", va="top",
                    arrowprops=None)

    ax.set_xscale("log")
    ax.set_xlim(0.0005, 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Cost per paper (USD, log scale)")
    ax.set_ylabel(r"Task $F_1$")
    ax.set_title("Cost vs Task $F_1$ across deployments",
                 fontweight="bold")
    ax.grid(alpha=0.3, which="both")

    _save(fig, "fig_cost_vs_quality")


def main() -> None:
    print(f"figures_out → {OUT}")
    fig_architecture()
    fig_benchmark_f1()
    fig_reliability_diagram()
    fig_cost_vs_quality()
    print("\nAll 4 figures written (8 files total: 4 .png + 4 .pdf).")


if __name__ == "__main__":
    main()
