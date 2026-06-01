"""Reliability diagram for method.name."""

from pathlib import Path

import matplotlib.pyplot as plt

from scripts.figures._common import PALETTE, load_json, save


def main() -> None:
    cal_path = Path("outputs/paper_data/phase5_calibration/calibration.json")
    if not cal_path.exists():
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.text(0.5, 0.5, "Phase 5 calibration data not present", ha="center", va="center")
        ax.axis("off")
        save(fig, "fig_reliability_diagram")
        return
    cal = load_json(cal_path)
    field_data = cal.get("per_field", {}).get("method.name")
    if not field_data:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.text(0.5, 0.5, "No method.name calibration data", ha="center", va="center")
        ax.axis("off")
        save(fig, "fig_reliability_diagram")
        return

    bins = field_data["reliability_bins"]
    centres = [(b["bin_lo"] + b["bin_hi"]) / 2 for b in bins]
    accs = [b["accuracy"] for b in bins]
    counts = [b["n"] for b in bins]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="ideal calibration")
    ax.bar(centres, accs, width=1.0 / len(bins) * 0.9, color=PALETTE[0], alpha=0.85, label="observed accuracy")
    for c, a, n in zip(centres, accs, counts):
        if n > 0:
            ax.text(c, a + 0.02, f"n={n}", fontsize=7, ha="center")
    ax.set_xlabel("predicted self_consistency (confidence bin)")
    ax.set_ylabel("empirical accuracy")
    ax.set_title(f"Reliability diagram — method.name (ECE={field_data['ece']:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper left")
    save(fig, "fig_reliability_diagram")


if __name__ == "__main__":
    main()
