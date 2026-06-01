"""Bar chart: our F1 vs published baselines on SciREX & TDMSci."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.figures._common import PALETTE, load_json, save


def main() -> None:
    bench_dir = Path("outputs/paper_data/phase4_benchmarks")
    rows = []  # (label, our_f1, our_lo, our_hi, published)
    if (bench_dir / "scirex" / "evaluation.json").exists():
        ev = load_json(bench_dir / "scirex" / "evaluation.json")
        for f, pub in (("methods", 0.567), ("tasks", 0.610), ("datasets", 0.553), ("metrics", 0.553)):
            row = ev["fields"].get(f)
            if row:
                rows.append((f"SciREX/{f}", row["multi_agent"]["f1"], row["multi_agent"]["ci_lo"], row["multi_agent"]["ci_hi"], pub))
    if (bench_dir / "tdmsci" / "evaluation.json").exists():
        ev = load_json(bench_dir / "tdmsci" / "evaluation.json")
        if ev.get("triples"):
            t = ev["triples"]
            rows.append(("TDMSci/triple", t["multi_agent"]["f1"], t["multi_agent"]["ci_lo"], t["multi_agent"]["ci_hi"], 0.452))
    if (bench_dir / "nlp_tdms" / "evaluation.json").exists():
        ev = load_json(bench_dir / "nlp_tdms" / "evaluation.json")
        if ev.get("triples"):
            t = ev["triples"]
            rows.append(("NLP-TDMS/triple", t["multi_agent"]["f1"], t["multi_agent"]["ci_lo"], t["multi_agent"]["ci_hi"], 0.317))

    if not rows:
        print("No benchmark evaluations found")
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(rows))
    width = 0.4
    ours = [r[1] for r in rows]
    lows = [r[1] - r[2] for r in rows]
    highs = [r[3] - r[1] for r in rows]
    pubs = [r[4] for r in rows]
    ax.bar(x - width / 2, ours, width, yerr=[lows, highs], label="ours (multi-agent)", color=PALETTE[0], capsize=4)
    ax.bar(x + width / 2, pubs, width, label="published prior", color=PALETTE[2])
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], rotation=20, ha="right")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Multi-agent vs published prior on public benchmarks")
    save(fig, "fig_benchmark_f1")


if __name__ == "__main__":
    main()
