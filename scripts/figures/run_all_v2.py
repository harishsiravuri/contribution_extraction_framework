"""Generate all v2 figures, reading from outputs/paper_data_v2/."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.figures._common_v2 import FIG_DIR, PALETTE, load_json, save

V2 = Path("outputs/paper_data_v2")
V1 = Path("outputs/paper_data")


def _safe_load(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def fig_architecture():
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, "Multi-agent architecture\n(Extractor × 3 → Critic → Consolidator)\n\nplaceholder — final draw in Inkscape",
            ha="center", va="center", fontsize=14)
    ax.axis("off")
    save(fig, "fig_architecture")


def fig_stability_overall():
    a2 = _safe_load(V2 / "pilot_5k" / "aggregate.json")
    if not a2:
        return
    a1 = _safe_load(V1 / "phase2_pilot_10k" / "aggregate.json")
    means, los, his, labels = [], [], [], []
    for label, agg in (("v1 (n=" + str(a1["stability"]["n"]) + ")", a1), ("v2 (n=" + str(a2["stability"]["n"]) + ")", a2)) if a1 else (("v2", a2),):
        ov = agg["stability"]["overall"]
        means.append(ov["mean"])
        los.append(ov["mean"] - ov["ci_lo"])
        his.append(ov["ci_hi"] - ov["mean"])
        labels.append(label)
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=[los, his], capsize=8, color=[PALETTE[0]] * len(labels))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Jaccard stability")
    ax.set_title("Multi-agent stability (overall)")
    save(fig, "fig_stability_overall")


def fig_stability_per_field():
    agg = _safe_load(V2 / "pilot_5k" / "aggregate.json")
    if not agg:
        return
    fields = ["method.name", "task.name", "datasets", "metrics", "claim_strength"]
    means, los, his = [], [], []
    for f in fields:
        d = agg["stability"]["per_field"][f]
        means.append(d["mean"])
        los.append(d["mean"] - d["ci_lo"])
        his.append(d["ci_hi"] - d["mean"])
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(fields))
    ax.bar(x, means, yerr=[los, his], color=PALETTE[0], capsize=6)
    ax.set_xticks(x); ax.set_xticklabels(fields, rotation=20, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("Jaccard stability")
    ax.set_title(f"Per-field stability (n={agg['stability']['n']})")
    save(fig, "fig_stability_per_field")


def fig_coverage_comparison():
    agg = _safe_load(V2 / "pilot_5k" / "aggregate.json")
    if not agg:
        return
    fields = ["method.name", "task.name", "datasets", "metrics", "claim_strength"]
    multi, base = [], []
    for f in fields:
        c = agg["coverage"]["per_field"].get(f) or {"multi_agent_coverage": 0, "baseline_coverage": 0}
        multi.append(100 * c["multi_agent_coverage"])
        base.append(100 * c["baseline_coverage"])
    x = np.arange(len(fields)); w = 0.4
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, multi, w, label="multi-agent", color=PALETTE[0])
    ax.bar(x + w/2, base, w, label="baseline", color=PALETTE[1])
    ax.set_xticks(x); ax.set_xticklabels(fields, rotation=20, ha="right")
    ax.set_ylim(0, 105); ax.set_ylabel("Coverage (% non-null)")
    ax.legend(); ax.set_title("Coverage comparison (v2)")
    save(fig, "fig_coverage_comparison")


def fig_benchmark_f1():
    bench_dir = V2 / "benchmarks"
    rows = []
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
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(rows)); w = 0.4
    ours = [r[1] for r in rows]
    los = [r[1] - r[2] for r in rows]
    his = [r[3] - r[1] for r in rows]
    pubs = [r[4] for r in rows]
    ax.bar(x - w/2, ours, w, yerr=[los, his], capsize=4, label="ours (multi-agent v2)", color=PALETTE[0])
    ax.bar(x + w/2, pubs, w, label="published prior", color=PALETTE[2])
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], rotation=20, ha="right")
    ax.set_ylabel("F1"); ax.set_ylim(0, 1); ax.legend()
    ax.set_title("Benchmark F1 (v2) vs published prior")
    save(fig, "fig_benchmark_f1")


def fig_frontier_vs_openweights():
    fr_dir = V2 / "frontier"
    op_dir = V2 / "benchmarks"
    rows = []
    for bench in ("tdmsci", "scirex"):
        op_eval = op_dir / bench / "evaluation.json"
        fr_eval = fr_dir / bench / "evaluation.json"
        if not (op_eval.exists() and fr_eval.exists()):
            continue
        op = load_json(op_eval)
        fr = load_json(fr_eval)
        for field in ("methods", "tasks", "datasets", "metrics"):
            o = op["fields"].get(field)
            f = fr["fields"].get(field)
            if o and f:
                rows.append((f"{bench}/{field}", o["multi_agent"]["f1"], f["multi_agent"]["f1"]))
    if not rows:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "Frontier ablation incomplete — see RESULTS.md", ha="center", va="center")
        ax.axis("off"); save(fig, "fig_frontier_vs_openweights"); return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(rows)); w = 0.4
    op_vals = [r[1] for r in rows]
    fr_vals = [r[2] for r in rows]
    ax.bar(x - w/2, op_vals, w, label="open-weights (DeepSeek)", color=PALETTE[0])
    ax.bar(x + w/2, fr_vals, w, label="frontier (Opus 4.6)", color=PALETTE[3])
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], rotation=20, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("F1"); ax.legend()
    ax.set_title("Open-weights vs frontier extractor (v2)")
    save(fig, "fig_frontier_vs_openweights")


def fig_reliability_diagram():
    cal = _safe_load(V2 / "calibration" / "calibration.json")
    if not cal:
        return
    field_data = cal["per_field"].get("method.name")
    if not field_data:
        return
    bins_un = field_data["reliability_uncalibrated"]
    bins_cal = field_data["reliability_calibrated"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, bins, title in ((axs[0], bins_un, "Uncalibrated"), (axs[1], bins_cal, "Temperature scaled")):
        centres = [(b["bin_lo"] + b["bin_hi"]) / 2 for b in bins]
        accs = [b["accuracy"] for b in bins]
        counts = [b["n"] for b in bins]
        ax.plot([0, 1], [0, 1], "--", color="gray", label="ideal")
        ax.bar(centres, accs, width=1.0/len(bins)*0.9, color=PALETTE[0], alpha=0.85)
        for c, a, n in zip(centres, accs, counts):
            if n > 0:
                ax.text(c, a + 0.02, f"n={n}", fontsize=7, ha="center")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.1)
        ax.set_xlabel("predicted confidence")
        ax.set_title(title)
    axs[0].set_ylabel("empirical accuracy")
    fig.suptitle(f"Reliability — method.name (T={field_data['T']:.2f}, ECE: {field_data['ece_uncalibrated']:.3f} → {field_data['ece_calibrated']:.3f})")
    save(fig, "fig_reliability_diagram")


def fig_cost_scaling():
    agg = _safe_load(V2 / "pilot_5k" / "aggregate.json")
    if not agg:
        return
    sizes, multi, base = [], [], []
    for k, v in agg["cost_projections"].items():
        sizes.append(int(k.split("=")[1]))
        multi.append(v["multi_agent_usd"]); base.append(v["baseline_usd"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sizes, multi, "o-", color=PALETTE[0], label="multi-agent")
    ax.plot(sizes, base, "s-", color=PALETTE[1], label="baseline")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Corpus size (papers)"); ax.set_ylabel("Projected spend (USD)")
    ax.set_title("Cost scaling (v2)"); ax.legend(); ax.grid(True, which="both", alpha=0.3)
    save(fig, "fig_cost_scaling")


def fig_walltime_distribution():
    sm = _safe_load(V2 / "pilot_5k" / "multi_agent" / "summary.json")
    if not sm:
        return
    times = [r["wall_time_seconds"] for r in sm["per_paper"] if r["status"] == "ok"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(times, bins=40, color=PALETTE[0], edgecolor="white")
    ax.set_xlabel("wall-time per paper (s)"); ax.set_ylabel("count")
    ax.set_title(f"Multi-agent wall-time distribution (n={len(times)})")
    save(fig, "fig_walltime_distribution")


def fig_error_breakdown():
    from collections import Counter
    counts: Counter[str] = Counter()
    for phase in ("multi_agent", "baseline", "stability"):
        s = _safe_load(V2 / "pilot_5k" / phase / "summary.json")
        if not s:
            continue
        for r in s["per_paper"]:
            if r["status"] == "error" and r["error"]:
                counts[r["error"].split(":", 1)[0]] += 1
    if not counts:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.text(0.5, 0.5, "No errors in v2 summaries", ha="center", va="center"); ax.axis("off")
        save(fig, "fig_error_breakdown"); return
    labels = list(counts.keys()); sizes = [counts[l] for l in labels]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sizes, labels=labels, autopct=lambda p: f"{p:.1f}%\n({int(round(p*sum(sizes)/100))})", colors=PALETTE[:len(labels)])
    ax.set_title(f"Error breakdown ({sum(sizes)} errors, v2)")
    save(fig, "fig_error_breakdown")


def fig_downstream_auc():
    r = _safe_load(V2 / "downstream" / "results.json")
    if not r:
        return
    means = [r["multi_agent"]["mean"], r["baseline"]["mean"]]
    los = [r["multi_agent"]["mean"] - r["multi_agent"]["ci_lo"], r["baseline"]["mean"] - r["baseline"]["ci_lo"]]
    his = [r["multi_agent"]["ci_hi"] - r["multi_agent"]["mean"], r["baseline"]["ci_hi"] - r["baseline"]["mean"]]
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(2)
    ax.bar(x, means, yerr=[los, his], color=[PALETTE[0], PALETTE[1]], capsize=8)
    ax.set_xticks(x); ax.set_xticklabels(["multi-agent", "baseline"])
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("Test ROC-AUC")
    ax.set_title(f"Downstream GNN AUC (v2; p={r['p_value_paired_permutation']:.3f})")
    save(fig, "fig_downstream_auc")


def fig_calibration_ece():
    cal = _safe_load(V2 / "calibration" / "calibration.json")
    if not cal:
        return
    fields = ["method.name", "task.name", "datasets", "metrics"]
    before, after = [], []
    for f in fields:
        d = cal["per_field"].get(f)
        if d:
            before.append(d["ece_uncalibrated"])
            after.append(d["ece_calibrated"])
        else:
            before.append(0); after.append(0)
    x = np.arange(len(fields)); w = 0.4
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, before, w, label="before scaling", color=PALETTE[1])
    ax.bar(x + w/2, after, w, label="after scaling", color=PALETTE[0])
    ax.set_xticks(x); ax.set_xticklabels(fields, rotation=15, ha="right")
    ax.set_ylabel("ECE"); ax.set_title("Calibration: ECE before vs after temperature scaling")
    ax.legend()
    save(fig, "fig_calibration_ece")


FIGS = [
    ("fig_architecture", fig_architecture),
    ("fig_stability_overall", fig_stability_overall),
    ("fig_stability_per_field", fig_stability_per_field),
    ("fig_coverage_comparison", fig_coverage_comparison),
    ("fig_benchmark_f1", fig_benchmark_f1),
    ("fig_frontier_vs_openweights", fig_frontier_vs_openweights),
    ("fig_reliability_diagram", fig_reliability_diagram),
    ("fig_cost_scaling", fig_cost_scaling),
    ("fig_walltime_distribution", fig_walltime_distribution),
    ("fig_error_breakdown", fig_error_breakdown),
    ("fig_downstream_auc", fig_downstream_auc),
    ("fig_calibration_ece", fig_calibration_ece),
]


def main():
    for name, fn in FIGS:
        try:
            fn()
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
    desc = {
        "fig_architecture": ("Architecture diagram (placeholder)", "n/a", "Section 3"),
        "fig_stability_overall": ("Overall Jaccard stability v1 vs v2 with 95% CI", "v2/pilot_5k/aggregate.json", "Section 6"),
        "fig_stability_per_field": ("Per-field stability with CI", "v2/pilot_5k/aggregate.json", "Section 6"),
        "fig_coverage_comparison": ("Coverage multi vs base", "v2/pilot_5k/aggregate.json", "Section 6"),
        "fig_benchmark_f1": ("Multi-agent F1 vs published prior", "v2/benchmarks/*/evaluation.json", "Section 6"),
        "fig_frontier_vs_openweights": ("Open-weights vs frontier", "v2/{benchmarks,frontier}/*", "Section 7"),
        "fig_reliability_diagram": ("Reliability before+after temperature scaling", "v2/calibration/calibration.json", "Section 6"),
        "fig_cost_scaling": ("Cost vs corpus size", "v2/pilot_5k/aggregate.json", "Section 5"),
        "fig_walltime_distribution": ("Wall-time histogram", "v2/pilot_5k/multi_agent/summary.json", "Section 5"),
        "fig_error_breakdown": ("Error categories (v2)", "v2/pilot_5k/*/summary.json", "Section 8"),
        "fig_downstream_auc": ("GNN AUC", "v2/downstream/results.json", "Section 6"),
        "fig_calibration_ece": ("ECE before vs after temperature scaling", "v2/calibration/calibration.json", "Section 6"),
    }
    lines = ["# Figures (v2)\n", "All saved at 300 DPI as both .png and .pdf, palette seaborn 'colorblind'.\n",
             "| File | What it shows | Source | Section |", "|---|---|---|---|"]
    for name, (what, src, sect) in desc.items():
        lines.append(f"| `{name}.pdf` / `.png` | {what} | `{src}` | {sect} |")
    (FIG_DIR / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
