"""Generate v3 figures (3 new + regenerated benchmark/calibration variants)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

V3 = Path("outputs/paper_data_v3")
V2 = Path("outputs/paper_data_v2")
FIG_DIR = V3 / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PALETTE = sns.color_palette("colorblind")
sns.set_theme(context="paper", style="whitegrid", font_scale=1.1, palette="colorblind")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _load(p):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def fig_span_grounding():
    """Headline figure: span-grounding F1, full vs no_critic vs baseline,
    raw and resolved variants side by side."""
    eval_path = V3 / "span_grounding" / "evaluation.json"
    data = _load(eval_path)
    if not data:
        return
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    for ax, variant in zip(axs, ("raw", "resolved")):
        means, los, his, labels = [], [], [], []
        for cond in ("baseline", "no_critic", "full"):
            d = data["by_condition"].get(cond)
            if d is None:
                continue
            v = d[variant]["f1"]
            means.append(v["mean"])
            los.append(v["mean"] - v["ci_lo"])
            his.append(v["ci_hi"] - v["mean"])
            labels.append(cond)
        x = np.arange(len(labels))
        colors = [PALETTE[1], PALETTE[2], PALETTE[0]][: len(labels)]
        ax.bar(x, means, yerr=[los, his], capsize=8, color=colors)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("Span-grounding F1" if variant == "raw" else "")
        ax.set_title(f"{variant.capitalize()} (LLM-emitted spans)" if variant == "raw" else "Resolved (string-match in source text)")
    fig.suptitle("Phase Y2 — Span-grounding accuracy on SciREX dev (n≈60)")
    save(fig, "fig_span_grounding")


def fig_critic_validation():
    """Critic precision/recall on UNSUPPORTED verdicts."""
    data = _load(V3 / "critic_analysis" / "critic_validation.json")
    if not data:
        return
    fields = ["method.name", "task.name", "datasets", "metrics"]
    n_supp, n_correct, n_false, n_missed = [], [], [], []
    for f in fields:
        d = data["aggregate"]["by_field"][f]
        n_supp.append(d["n_critic_suppressions"])
        n_correct.append(d["correct_supp"])
        n_false.append(d["false_supp"])
        n_missed.append(d["missed_supp"])
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(fields))
    w = 0.22
    ax.bar(x - 1.5*w, n_supp, w, label="explicit suppressions", color=PALETTE[0])
    ax.bar(x - 0.5*w, n_correct, w, label="correct (truly wrong)", color=PALETTE[2])
    ax.bar(x + 0.5*w, n_false, w, label="false (gold had match)", color=PALETTE[3])
    ax.bar(x + 1.5*w, n_missed, w, label="missed (full retained, no gold)", color=PALETTE[1])
    ax.set_xticks(x); ax.set_xticklabels(fields, rotation=15, ha="right")
    ax.set_ylabel("count")
    ax.set_title(f"Critic suppression behaviour on SciREX dev (n={data['n_papers']})")
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "fig_critic_validation")


def fig_benchmark_f1_v3():
    """v3 benchmark F1 vs published prior."""
    bench = V3 / "benchmarks"
    rows = []
    sx = _load(bench / "scirex" / "evaluation.json")
    if sx:
        for f, pub in (("methods", 0.567), ("tasks", 0.610), ("datasets", 0.553), ("metrics", 0.553)):
            r = sx.get("fields", {}).get(f)
            if r:
                rows.append((f"SciREX/{f}", r["multi_agent"]["f1"], r["baseline"]["f1"], pub))
    td = _load(bench / "tdmsci" / "evaluation.json")
    if td and td.get("triples"):
        t = td["triples"]
        rows.append(("TDMSci/triple", t["multi_agent"]["f1"], t["baseline"]["f1"], 0.452))
    nl = _load(bench / "nlp_tdms" / "evaluation.json")
    if nl and nl.get("triples"):
        t = nl["triples"]
        rows.append(("NLP-TDMS/triple", t["multi_agent"]["f1"], t["baseline"]["f1"], 0.317))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(rows))
    w = 0.27
    ax.bar(x - w, [r[1] for r in rows], w, label="multi-agent (ours)", color=PALETTE[0])
    ax.bar(x, [r[2] for r in rows], w, label="single-LLM baseline", color=PALETTE[1])
    ax.bar(x + w, [r[3] for r in rows], w, label="published prior", color=PALETTE[2])
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1"); ax.legend()
    ax.set_title("Phase Y4 — benchmark F1 with binding-rule extractor")
    save(fig, "fig_benchmark_f1_v3")


def fig_calibration_carry():
    """Re-render calibration ECE from v2 (no v3 calibration re-run)."""
    cal = _load(V2 / "calibration" / "calibration.json")
    if not cal:
        return
    fields = ["method.name", "task.name", "datasets", "metrics"]
    before, after = [], []
    for f in fields:
        d = cal["per_field"].get(f)
        before.append(d["ece_uncalibrated"] if d else 0)
        after.append(d["ece_calibrated"] if d else 0)
    x = np.arange(len(fields)); w = 0.4
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, before, w, label="uncalibrated", color=PALETTE[1])
    ax.bar(x + w/2, after, w, label="temperature-scaled", color=PALETTE[0])
    ax.set_xticks(x); ax.set_xticklabels(fields, rotation=15, ha="right")
    ax.set_ylabel("ECE"); ax.set_title("Calibration: ECE before vs after temperature scaling (carried from v2)")
    ax.legend()
    save(fig, "fig_calibration_ece")


def fig_architecture():
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, "Multi-agent architecture\n(Extractor × 3 → Critic → Consolidator)\n\nplaceholder — final draw in Inkscape",
            ha="center", va="center", fontsize=14)
    ax.axis("off"); save(fig, "fig_architecture")


def fig_error_breakdown_v3():
    counts: Counter[str] = Counter()
    for sub in (V3 / "span_grounding").glob("*/summary.json"):
        s = _load(sub)
        if not s:
            continue
        for r in s.get("per_paper", []):
            if r["status"] == "error" and r.get("error"):
                counts[r["error"].split(":", 1)[0]] += 1
    if not counts:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.text(0.5, 0.5, "v3 span-grounding: no errors", ha="center", va="center")
        ax.axis("off"); save(fig, "fig_error_breakdown_v3"); return
    labels = list(counts.keys()); sizes = [counts[l] for l in labels]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sizes, labels=labels, autopct=lambda p: f"{p:.1f}%\n({int(round(p*sum(sizes)/100))})", colors=PALETTE[:len(labels)])
    ax.set_title(f"v3 error breakdown ({sum(sizes)} errors)")
    save(fig, "fig_error_breakdown_v3")


def fig_frontier_vs_openweights_v3():
    """Open-weights (DeepSeek) vs frontier (GPT-5) extractor on SciREX dev."""
    op_eval = V3 / "benchmarks" / "scirex" / "evaluation.json"
    fr_eval = V3 / "frontier_gpt5" / "scirex" / "evaluation.json"
    if not (op_eval.exists() and fr_eval.exists()):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "GPT-5 frontier evaluation missing", ha="center", va="center")
        ax.axis("off"); save(fig, "fig_frontier_vs_openweights"); return
    op = _load(op_eval); fr = _load(fr_eval)
    fields = ("methods", "tasks", "datasets", "metrics")
    rows = []
    for f in fields:
        ro = op.get("fields", {}).get(f)
        rf = fr.get("fields", {}).get(f)
        if ro and rf:
            rows.append((f, ro["multi_agent"]["f1"], rf["multi_agent"]["f1"]))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(rows)); w = 0.4
    op_vals = [r[1] for r in rows]
    fr_vals = [r[2] for r in rows]
    ax.bar(x - w/2, op_vals, w, label="open-weights (DeepSeek)", color=PALETTE[0])
    ax.bar(x + w/2, fr_vals, w, label="frontier (GPT-5)", color=PALETTE[3])
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylim(0, 1); ax.set_ylabel("F1"); ax.legend()
    ax.set_title("SciREX dev — DeepSeek vs GPT-5 extractor")
    save(fig, "fig_frontier_vs_openweights")


FIGS = [
    ("fig_span_grounding", fig_span_grounding),
    ("fig_critic_validation", fig_critic_validation),
    ("fig_benchmark_f1_v3", fig_benchmark_f1_v3),
    ("fig_frontier_vs_openweights", fig_frontier_vs_openweights_v3),
    ("fig_calibration_ece", fig_calibration_carry),
    ("fig_architecture", fig_architecture),
    ("fig_error_breakdown_v3", fig_error_breakdown_v3),
]


def main():
    for name, fn in FIGS:
        try:
            fn()
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
    desc = {
        "fig_span_grounding": ("Span-grounding F1 (raw + resolved) for full / no_critic / baseline", "v3/span_grounding/evaluation.json"),
        "fig_critic_validation": ("Critic suppression behaviour (correct / false / missed)", "v3/critic_analysis/critic_validation.json"),
        "fig_benchmark_f1_v3": ("Multi-agent vs baseline vs published prior on benchmarks", "v3/benchmarks/*/evaluation.json"),
        "fig_calibration_ece": ("ECE before vs after temperature scaling (v2 data)", "v2/calibration/calibration.json"),
        "fig_architecture": ("Architecture diagram placeholder", "n/a"),
        "fig_error_breakdown_v3": ("Error categories across v3 runs", "v3/span_grounding/*/summary.json"),
    }
    lines = ["# Figures (v3)\n", "All saved at 300 DPI as both .png and .pdf.\n",
             "| File | What it shows | Source |", "|---|---|---|"]
    for name, (what, src) in desc.items():
        lines.append(f"| `{name}.pdf` / `.png` | {what} | `{src}` |")
    (FIG_DIR / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
