"""Frontier vs open-weights extractor F1, grouped bar."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.figures._common import PALETTE, load_json, save


def _f1(eval_path: Path, field: str) -> float | None:
    if not eval_path.exists():
        return None
    ev = load_json(eval_path)
    if field == "triples":
        t = ev.get("triples")
        return t["multi_agent"]["f1"] if t else None
    row = ev.get("fields", {}).get(field)
    return row["multi_agent"]["f1"] if row else None


def main() -> None:
    open_dir = Path("outputs/paper_data/phase4_benchmarks")
    fr_dir = Path("outputs/paper_data/phase6_frontier")

    targets = [
        ("SciREX/methods", open_dir / "scirex" / "evaluation.json", fr_dir / "scirex" / "evaluation.json", "methods"),
        ("SciREX/datasets", open_dir / "scirex" / "evaluation.json", fr_dir / "scirex" / "evaluation.json", "datasets"),
        ("TDMSci/triple", open_dir / "tdmsci" / "evaluation.json", fr_dir / "tdmsci" / "evaluation.json", "triples"),
    ]

    rows = []
    for label, op, fp, field in targets:
        o = _f1(op, field)
        f = _f1(fp, field)
        rows.append((label, o, f))

    if not rows or not any(r[2] is not None for r in rows):
        # No frontier data — render placeholder note
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "Frontier ablation skipped — see RESULTS.md", ha="center", va="center")
        ax.axis("off")
        save(fig, "fig_frontier_vs_openweights")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(rows))
    width = 0.4
    open_vals = [(r[1] or 0.0) for r in rows]
    fr_vals = [(r[2] or 0.0) for r in rows]
    ax.bar(x - width / 2, open_vals, width, label="open-weights (DeepSeek)", color=PALETTE[0])
    ax.bar(x + width / 2, fr_vals, width, label="frontier (Opus)", color=PALETTE[3])
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], rotation=20, ha="right")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Open-weights vs frontier extractor")
    save(fig, "fig_frontier_vs_openweights")


if __name__ == "__main__":
    main()
