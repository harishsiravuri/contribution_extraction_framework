"""E5 — Precision/coverage at confidence thresholds (pure analysis).

Loads the existing v2 SciREX multi-agent records + gold, builds per-field
(self_consistency, correctness) pairs, then sweeps confidence thresholds
tau ∈ {0.5, 0.6, 0.7, 0.8, 0.9} both for the raw confidence and for the
temperature-scaled confidence using the per-field T from
outputs/paper_data_v2/calibration/calibration.json.

For each (field, tau, scaling) cell, reports precision, recall, coverage,
and bootstrap 95% CIs on precision. Writes results.json + a
precision-vs-coverage plot.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import typer
from rich.console import Console

from paper1.loaders import load_scirex
from paper1.metrics import bootstrap_ci
from paper1.metrics.temperature_scaling import apply_temperature
from paper1.schema import ContributionRecord
from paper1.voting import _norm

SEED = 42
RNG = np.random.default_rng(SEED)
N_BOOT = 1000

app = typer.Typer(no_args_is_help=False, add_completion=False)
console = Console()


def _correct(pred_set: set[str], gold_set: set[str]) -> int | None:
    """Return 1 if any pred matches any gold (substring lenient), 0 if not, None if no gold."""
    if not gold_set:
        return None
    if not pred_set:
        return 0
    for p in pred_set:
        for g in gold_set:
            if p in g or g in p:
                return 1
    return 0


def _build_pairs(records: dict[str, ContributionRecord],
                 papers) -> dict[str, list[tuple[float, int]]]:
    """For each field, build a list of (confidence, correct_indicator) pairs.
    One per first-contribution per paper (matches the v2 calibration convention).
    A pair is also a 'gold has at least one truth label' instance — needed for
    precision/recall computation.
    """
    by_id = {p.paper_id: p for p in papers}
    pairs: dict[str, list[tuple[float, int]]] = {
        "method.name": [], "task.name": [], "datasets": [], "metrics": []
    }
    for pid, rec in records.items():
        p = by_id.get(pid)
        if p is None or not rec.contributions:
            continue
        c0 = rec.contributions[0]
        conf = c0.self_consistency
        ps = {
            "method.name": {_norm(c0.method.name)} if c0.method and c0.method.name else set(),
            "task.name":   {_norm(c0.task.name)} if c0.task and c0.task.name else set(),
            "datasets":    {_norm(d.name) for d in c0.datasets if d.name},
            "metrics":     {_norm(m.name) for m in c0.metrics if m.name},
        }
        gold = {
            "method.name": p.gold.methods,
            "task.name": p.gold.tasks,
            "datasets": p.gold.datasets,
            "metrics": p.gold.metrics,
        }
        for f in pairs:
            k = _correct(ps[f] - {""}, gold[f])
            if k is not None:
                pairs[f].append((conf, k))
    return pairs


def _threshold_metrics(pairs: list[tuple[float, int]], tau: float,
                       n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Compute precision, recall, coverage for records with conf >= tau.

    precision = TP / (TP + FP) among RETAINED records (conf >= tau)
    recall    = TP / total positive in gold (= count of correct==1 in full set)
    coverage  = #retained / #total
    """
    if not pairs:
        return None
    confs = np.array([c for c, _ in pairs])
    correct = np.array([k for _, k in pairs], dtype=float)
    total = len(pairs)
    total_positive = float(correct.sum())  # gold "positives" we could have caught
    retained_mask = confs >= tau
    n_retained = int(retained_mask.sum())
    if n_retained == 0:
        return {
            "tau": tau, "n_total": total, "n_retained": 0,
            "coverage": 0.0, "precision": 0.0,
            "precision_ci_lo": 0.0, "precision_ci_hi": 0.0,
            "recall": 0.0, "n_total_positive": total_positive,
        }
    retained_correct = correct[retained_mask]
    precision = float(retained_correct.mean())
    # Bootstrap CI on precision (over the retained subset)
    rng = np.random.default_rng(seed)
    n = retained_correct.size
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = retained_correct[idx].mean()
    p_lo = float(np.quantile(means, 0.025))
    p_hi = float(np.quantile(means, 0.975))
    recall = float(retained_correct.sum() / total_positive) if total_positive > 0 else 0.0
    coverage = n_retained / total
    return {
        "tau": tau, "n_total": total, "n_retained": n_retained,
        "coverage": coverage,
        "precision": precision,
        "precision_ci_lo": p_lo,
        "precision_ci_hi": p_hi,
        "recall": recall,
        "n_total_positive": total_positive,
    }


@app.command()
def run(
    bench_dir: Path = typer.Option(Path("outputs/paper_data_v2/benchmarks/scirex/multi_agent"), "--bench-dir"),
    calib_path: Path = typer.Option(Path("outputs/paper_data_v2/calibration/calibration.json"), "--calib-path"),
    output_dir: Path = typer.Option(Path("outputs/paper_data_v8/threshold_analysis"), "--output-dir"),
    fig_dup_to: Path = typer.Option(Path("figures_out/fig_precision_vs_coverage.png"), "--fig-dup-to"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "seed.txt").write_text(f"{SEED}\n")

    # Load records and gold
    records: dict[str, ContributionRecord] = {}
    for f in bench_dir.glob("*.json"):
        try:
            rec = ContributionRecord.model_validate_json(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        pid = f.stem.replace("__", ":")
        records[pid] = rec
    papers = [p for p in load_scirex(splits=("dev",)) if p.full_text and len(p.full_text) >= 50]
    console.print(f"  records={len(records)} dev_papers={len(papers)}")

    pairs = _build_pairs(records, papers)
    for f, ps in pairs.items():
        console.print(f"    {f}: {len(ps)} (conf, correct) pairs, base accuracy={np.mean([k for _,k in ps]):.3f}")

    # Per-field T from existing calibration
    T_per_field: dict[str, float | None] = {}
    if calib_path.exists():
        c = json.loads(calib_path.read_text())["per_field"]
        for f in pairs:
            T_per_field[f] = (c.get(f) or {}).get("T")
    console.print(f"  T per field: {T_per_field}")

    taus = [0.5, 0.6, 0.7, 0.8, 0.9]
    results: dict = {
        "experiment": "E5_precision_at_threshold",
        "seed": SEED,
        "n_resamples_bootstrap": N_BOOT,
        "thresholds": taus,
        "per_field": {},
    }
    for f, ps in pairs.items():
        if not ps:
            results["per_field"][f] = None
            continue
        # Raw (un-temperature-scaled)
        raw_rows = [_threshold_metrics(ps, tau) for tau in taus]
        # T-scaled (using per-field T from calibration.json)
        scaled_rows = None
        T = T_per_field.get(f)
        if T:
            confs = [c for c, _ in ps]
            scaled = apply_temperature(confs, T).tolist()
            scaled_pairs = list(zip(scaled, [k for _, k in ps]))
            scaled_rows = [_threshold_metrics(scaled_pairs, tau) for tau in taus]
        results["per_field"][f] = {
            "n": len(ps),
            "T": T,
            "raw": raw_rows,
            "T_scaled": scaled_rows,
        }

    out_path = output_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    console.print(f"[green]Wrote {out_path}[/green]")

    # Figure: precision vs coverage, per field, raw confidence
    _build_figure(results, output_dir / "fig_precision_vs_coverage.png", fig_dup_to)


def _build_figure(results: dict, primary_path: Path, dup_path: Path) -> None:
    palette = sns.color_palette("colorblind")
    field_colors = {
        "method.name": palette[0],
        "task.name": palette[2],
        "datasets": palette[1],
        "metrics": palette[3],
    }
    mpl.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.transparent": True,
        "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
        "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    })
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, mode in zip(axes, ("raw", "T_scaled")):
        ax.set_title(f"Confidence: {'uncalibrated' if mode=='raw' else 'temperature-scaled'}", fontweight="bold")
        ax.set_xlabel("Coverage (fraction of records retained)")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        for f, d in results["per_field"].items():
            if d is None or d.get(mode) is None:
                continue
            covs = [r["coverage"] for r in d[mode]]
            precs = [r["precision"] for r in d[mode]]
            errs_lo = [p - r["precision_ci_lo"] for p, r in zip(precs, d[mode])]
            errs_hi = [r["precision_ci_hi"] - p for p, r in zip(precs, d[mode])]
            ax.errorbar(covs, precs, yerr=[errs_lo, errs_hi],
                        fmt="o-", color=field_colors[f], mec="black", mew=0.5,
                        capsize=3, label=f)
            # Annotate tau values
            for tau, cov, prec in zip(results["thresholds"], covs, precs):
                ax.annotate(f"τ={tau}", xy=(cov, prec),
                            xytext=(3, 5), textcoords="offset points",
                            fontsize=7, color=field_colors[f], alpha=0.7)
    axes[0].set_ylabel("Precision (lenient match)")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Precision vs coverage as confidence threshold τ is swept (SciREX dev, multi-agent)",
                 fontweight="bold", y=1.02)
    for p in (primary_path, dup_path):
        p.parent.mkdir(parents=True, exist_ok=True)
        # Save both png and pdf for primary; just png for dup
        fig.savefig(p, dpi=300, transparent=True, bbox_inches="tight")
        if p == primary_path:
            pdf = p.with_suffix(".pdf")
            fig.savefig(pdf, dpi=300, transparent=True, bbox_inches="tight")
            print(f"  wrote {pdf}")
        print(f"  wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    app()
