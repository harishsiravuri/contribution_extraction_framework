"""Per-field stability with 95% CI error bars."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.figures._common import PALETTE, load_json, save


def main() -> None:
    agg = load_json(Path("outputs/paper_data/phase2_pilot_10k/aggregate.json"))
    fields = ["method.name", "task.name", "datasets", "metrics", "claim_strength"]
    means = []
    lows = []
    highs = []
    for f in fields:
        d = agg["stability"]["per_field"][f]
        means.append(d["mean"])
        lows.append(d["mean"] - d["ci_lo"])
        highs.append(d["ci_hi"] - d["mean"])
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(fields))
    ax.bar(x, means, yerr=[lows, highs], color=PALETTE[0], capsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(fields, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Jaccard stability")
    ax.set_title("Per-field stability (multi-agent run A vs re-run C)")
    save(fig, "fig_stability_per_field")


if __name__ == "__main__":
    main()
