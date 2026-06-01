"""Regenerate fig_cost_vs_quality with the GPT-4o (E1) points added.

Adds:
  - GPT-4o multi-agent (E1, 25-paper SciREX dev subset)
  - GPT-4o single-LLM baseline (E1, 25-paper subset)

Inputs:
  - outputs/paper_data_v3/benchmarks/scirex/evaluation.json (DeepSeek dev numbers)
  - outputs/paper_data_v6/benchmarks_ft_70b_test/scirex/evaluation.json (specialized test)
  - outputs/paper_data_v7/closed_source_comparison/results.json (GPT-4o standalone F1 + cost)

Output: figures_out/fig_cost_vs_quality_v2.{png,pdf}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures_out"
OUT.mkdir(parents=True, exist_ok=True)

PALETTE = sns.color_palette("colorblind")
COL_BASE_DEV = PALETTE[7]      # grey
COL_MULTI_DEV = PALETTE[0]     # blue
COL_SPECIALIZED = PALETTE[2]   # green
COL_GPT_MULTI = PALETTE[4]     # purple
COL_GPT_BASE = PALETTE[5]      # brown

mpl.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.transparent": True,
    "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
})


def _save(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf"):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, dpi=300, transparent=True, bbox_inches="tight")
        print(f"  wrote {p}")
    plt.close(fig)


def main() -> None:
    dev = json.loads((ROOT / "outputs/paper_data_v3/benchmarks/scirex/evaluation.json").read_text())["fields"]
    test = json.loads((ROOT / "outputs/paper_data_v6/benchmarks_ft_70b_test/scirex/evaluation.json").read_text())["fields"]
    e1 = json.loads((ROOT / "outputs/paper_data_v7/closed_source_comparison/results.json").read_text())

    base_t = dev["tasks"]["baseline"]
    multi_t = dev["tasks"]["multi_agent"]
    spec_t = test["tasks"]["multi_agent"]
    gpt_multi_t = e1["f1_standalone"]["gpt4o_multi"]["tasks"]
    gpt_base_t = e1["f1_standalone"]["gpt4o_baseline"]["tasks"]

    # Per-paper cost (USD)
    e1_costs = e1["cost_usd_per_system"]
    n_sampled = e1["papers_sampled"]
    gpt_multi_cost_per = e1_costs["gpt4o_multi"] / n_sampled if n_sampled else 0.0
    gpt_base_cost_per = e1_costs["gpt4o_baseline"] / n_sampled if n_sampled else 0.0

    points = [
        {"label": "DeepSeek single-LLM\n(open, dev)", "cost": 0.001,
         "f1": base_t["f1"], "ci_lo": base_t["ci_lo"], "ci_hi": base_t["ci_hi"],
         "color": COL_BASE_DEV, "marker": "o"},
        {"label": "DeepSeek multi-agent\n(open, dev)", "cost": 0.005,
         "f1": multi_t["f1"], "ci_lo": multi_t["ci_lo"], "ci_hi": multi_t["ci_hi"],
         "color": COL_MULTI_DEV, "marker": "s"},
        {"label": "GPT-4o single-LLM\n(closed, dev)", "cost": gpt_base_cost_per,
         "f1": gpt_base_t["a_f1"], "ci_lo": gpt_base_t["a_ci_lo"], "ci_hi": gpt_base_t["a_ci_hi"],
         "color": COL_GPT_BASE, "marker": "v"},
        {"label": "GPT-4o multi-agent\n(closed, dev)", "cost": gpt_multi_cost_per,
         "f1": gpt_multi_t["a_f1"], "ci_lo": gpt_multi_t["a_ci_lo"], "ci_hi": gpt_multi_t["a_ci_hi"],
         "color": COL_GPT_MULTI, "marker": "D"},
        {"label": "Specialized 70B FT\n(test)", "cost": 0.18,
         "f1": spec_t["f1"], "ci_lo": spec_t["ci_lo"], "ci_hi": spec_t["ci_hi"],
         "color": COL_SPECIALIZED, "marker": "^"},
    ]

    print("  points:")
    for p in points:
        print(f"    {p['label'].replace(chr(10),' '):42s}  ${p['cost']:.4f}/paper  "
              f"F1={p['f1']:.3f} [{p['ci_lo']:.3f}, {p['ci_hi']:.3f}]")

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    sorted_pts = sorted(points, key=lambda p: p["cost"])
    ax.plot([p["cost"] for p in sorted_pts],
            [p["f1"] for p in sorted_pts],
            "--", color="dimgray", lw=1.0, alpha=0.5, zorder=1)

    for p in points:
        yerr = np.array([[p["f1"] - p["ci_lo"]], [p["ci_hi"] - p["f1"]]])
        ax.errorbar(p["cost"], p["f1"], yerr=yerr,
                    fmt=p["marker"], markersize=12, mec="black", mew=0.7,
                    color=p["color"], capsize=4, lw=1.2, zorder=3,
                    label=p["label"].replace("\n", " "))

    ax.set_xscale("log")
    ax.set_xlim(0.0005, 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Cost per paper (USD, log scale)")
    ax.set_ylabel(r"Task $F_1$")
    ax.set_title("Cost vs Task $F_1$ — open-weights, closed-source, and specialized",
                 fontweight="bold")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92)

    _save(fig, "fig_cost_vs_quality_v2")


if __name__ == "__main__":
    main()
