"""Clustered bar: % non-null per field, multi-agent vs baseline."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.figures._common import PALETTE, load_json, save


def main() -> None:
    agg = load_json(Path("outputs/paper_data/phase2_pilot_10k/aggregate.json"))
    fields = ["method.name", "task.name", "datasets", "metrics", "claim_strength"]
    multi, base = [], []
    for f in fields:
        c = agg["coverage"]["per_field"].get(f) or {"multi_agent_coverage": 0.0, "baseline_coverage": 0.0}
        multi.append(100 * c["multi_agent_coverage"])
        base.append(100 * c["baseline_coverage"])
    x = np.arange(len(fields))
    width = 0.4
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, multi, width, label="multi-agent", color=PALETTE[0])
    ax.bar(x + width / 2, base, width, label="baseline", color=PALETTE[1])
    ax.set_xticks(x)
    ax.set_xticklabels(fields, rotation=20, ha="right")
    ax.set_ylabel("Coverage (% papers with non-null field)")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.set_title("Coverage comparison")
    save(fig, "fig_coverage_comparison")


if __name__ == "__main__":
    main()
