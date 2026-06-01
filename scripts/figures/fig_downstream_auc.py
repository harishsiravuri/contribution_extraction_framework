"""GNN AUC bar chart with 95% CIs."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.figures._common import PALETTE, save


def main() -> None:
    path = Path("outputs/paper_data/phase7_downstream/results.json")
    if not path.exists():
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "Phase 7 not run", ha="center", va="center")
        ax.axis("off")
        save(fig, "fig_downstream_auc")
        return
    r = json.loads(path.read_text())
    means = [r["multi_agent"]["mean"], r["baseline"]["mean"]]
    los = [r["multi_agent"]["mean"] - r["multi_agent"]["ci_lo"], r["baseline"]["mean"] - r["baseline"]["ci_lo"]]
    his = [r["multi_agent"]["ci_hi"] - r["multi_agent"]["mean"], r["baseline"]["ci_hi"] - r["baseline"]["mean"]]
    labels = ["multi-agent", "baseline"]
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=[los, his], color=[PALETTE[0], PALETTE[1]], capsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Test ROC-AUC (shared-dataset link prediction)")
    ax.set_title(f"GNN downstream AUC (p={r['p_value_paired_permutation']:.3f})")
    save(fig, "fig_downstream_auc")


if __name__ == "__main__":
    main()
