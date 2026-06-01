"""Pie chart of error categories."""

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from scripts.figures._common import PALETTE, save


def _category(err: str) -> str:
    if "ConsolidatorParseError" in err:
        return "Consolidator parse"
    if "CriticParseError" in err:
        return "Critic parse"
    if "ExtractorParseError" in err:
        return "Extractor parse"
    if "SingleLLMParseError" in err:
        return "Baseline parse"
    if "Timeout" in err or "timeout" in err:
        return "Timeout"
    if "OpenRouter" in err:
        return "OpenRouter API"
    return "Other"


def main() -> None:
    err_counts: Counter[str] = Counter()
    for phase in ("multi_agent", "baseline", "stability"):
        path = Path(f"outputs/paper_data/phase2_pilot_10k/{phase}/summary.json")
        if not path.exists():
            continue
        s = json.loads(path.read_text())
        for r in s["per_paper"]:
            if r["status"] == "error" and r["error"]:
                err_counts[_category(r["error"])] += 1

    if not err_counts:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.text(0.5, 0.5, "No errors recorded — perfect run!", ha="center", va="center")
        ax.axis("off")
        save(fig, "fig_error_breakdown")
        return

    labels = list(err_counts.keys())
    sizes = [err_counts[l] for l in labels]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sizes, labels=labels, autopct=lambda p: f"{p:.1f}%\n({int(round(p * sum(sizes) / 100))})", colors=PALETTE[: len(labels)])
    ax.set_title(f"Error category breakdown ({sum(sizes)} errors)")
    save(fig, "fig_error_breakdown")


if __name__ == "__main__":
    main()
