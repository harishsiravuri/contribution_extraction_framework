"""Projected cost vs corpus size."""

from pathlib import Path

import matplotlib.pyplot as plt

from scripts.figures._common import PALETTE, load_json, save


def main() -> None:
    agg = load_json(Path("outputs/paper_data/phase2_pilot_10k/aggregate.json"))
    sizes = []
    multi = []
    base = []
    for k, v in agg["cost_projections"].items():
        sizes.append(int(k.split("=")[1]))
        multi.append(v["multi_agent_usd"])
        base.append(v["baseline_usd"])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sizes, multi, "o-", color=PALETTE[0], label="multi-agent")
    ax.plot(sizes, base, "s-", color=PALETTE[1], label="baseline")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Corpus size (papers)")
    ax.set_ylabel("Projected OpenRouter spend (USD)")
    ax.set_title("Cost scaling — extrapolated from pilot run")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    save(fig, "fig_cost_scaling")


if __name__ == "__main__":
    main()
