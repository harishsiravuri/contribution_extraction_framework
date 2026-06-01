"""Histogram of per-paper wall-time."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from scripts.figures._common import PALETTE, save


def main() -> None:
    summary_path = Path("outputs/paper_data/phase2_pilot_10k/multi_agent/summary.json")
    if not summary_path.exists():
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "Phase 2 summary not present", ha="center", va="center")
        ax.axis("off")
        save(fig, "fig_walltime_distribution")
        return
    summary = json.loads(summary_path.read_text())
    times = [r["wall_time_seconds"] for r in summary["per_paper"] if r["status"] == "ok"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(times, bins=40, color=PALETTE[0], edgecolor="white")
    ax.set_xlabel("wall-time per paper (seconds)")
    ax.set_ylabel("number of papers")
    ax.set_title(f"Multi-agent per-paper wall-time (n={len(times)})")
    save(fig, "fig_walltime_distribution")


if __name__ == "__main__":
    main()
