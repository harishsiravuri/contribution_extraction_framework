"""Bar chart: overall stability with bootstrap 95% CI error bars.

Reads outputs/paper_data/phase2_pilot_10k/aggregate.json.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from scripts.figures._common import PALETTE, load_json, save


def main() -> None:
    agg = load_json(Path("outputs/paper_data/phase2_pilot_10k/aggregate.json"))
    ov = agg["stability"]["overall"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    yerr = [[ov["mean"] - ov["ci_lo"]], [ov["ci_hi"] - ov["mean"]]]
    ax.bar(["multi-agent (run A vs run C)"], [ov["mean"]], yerr=yerr, color=PALETTE[0], capsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Jaccard stability")
    ax.set_title(f"Overall stability (n={agg['stability']['n']})")
    save(fig, "fig_stability_overall")


if __name__ == "__main__":
    main()
