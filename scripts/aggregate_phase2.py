"""Aggregate a completed Phase 2 run with bootstrap CIs, permutation tests, and
distribution stats. Reads the by_paper/ directories and summary.json files from
a phase2 output directory and writes report.md + aggregate.json.

Usage:
    python scripts/aggregate_phase2.py --output-dir outputs/paper_data/phase2_pilot_10k
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import typer
from rich.console import Console

from paper1.metrics import (
    bootstrap_ci,
    paired_permutation_test,
    percentiles,
)
from paper1.metrics.stability import jaccard_stability
from paper1.schema import ContributionRecord
from paper1.voting import _norm

app = typer.Typer(no_args_is_help=False, add_completion=False)
console = Console()


def _load(path: Path) -> ContributionRecord | None:
    try:
        return ContributionRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _has(rec: ContributionRecord, field: str) -> int:
    if not rec.contributions:
        return 0
    c = rec.contributions[0]
    if field == "method.name":
        return int(bool(_norm(c.method.name)))
    if field == "task.name":
        return int(bool(_norm(c.task.name)))
    if field == "datasets":
        return int(any(_norm(d.name) for d in c.datasets))
    if field == "metrics":
        return int(any(_norm(m.name) for m in c.metrics))
    if field == "claim_strength":
        return int(c.claim_strength is not None)
    return 0


@app.command()
def run(
    output_dir: Path = typer.Option(..., "--output-dir"),
) -> None:
    multi_dir = output_dir / "multi_agent" / "by_paper"
    base_dir = output_dir / "baseline" / "by_paper"
    stab_dir = output_dir / "stability" / "by_paper"

    multi_summary = json.loads((output_dir / "multi_agent" / "summary.json").read_text())
    base_summary = json.loads((output_dir / "baseline" / "summary.json").read_text())
    stab_summary = (
        json.loads((output_dir / "stability" / "summary.json").read_text())
        if (output_dir / "stability" / "summary.json").exists()
        else None
    )

    fields = ("method.name", "task.name", "datasets", "metrics", "claim_strength")

    # Stability with bootstrap CI per-field and overall
    stab_overall: list[float] = []
    stab_per_field: dict[str, list[float]] = {f: [] for f in fields}
    if stab_dir.exists():
        for stab_path in sorted(stab_dir.glob("*.json")):
            multi_path = multi_dir / stab_path.name
            if not multi_path.exists():
                continue
            a = _load(multi_path)
            b = _load(stab_path)
            if a is None or b is None:
                continue
            s = jaccard_stability(a, b)
            stab_overall.append(s["overall"])
            for f in fields:
                stab_per_field[f].append(s["per_field"][f])

    stability_block: dict[str, Any] = {"n": len(stab_overall), "per_field": {}}
    if stab_overall:
        m, lo, hi = bootstrap_ci(stab_overall, n_resamples=1000)
        stability_block["overall"] = {"mean": m, "ci_lo": lo, "ci_hi": hi, "std": statistics.pstdev(stab_overall) if len(stab_overall) > 1 else 0.0}
    else:
        stability_block["overall"] = {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "std": 0.0}
    for f in fields:
        vals = stab_per_field[f]
        if vals:
            m, lo, hi = bootstrap_ci(vals, n_resamples=1000)
            stability_block["per_field"][f] = {
                "mean": m,
                "ci_lo": lo,
                "ci_hi": hi,
                "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            }
        else:
            stability_block["per_field"][f] = {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "std": 0.0}

    # Coverage with paired permutation test (over papers present in both)
    coverage_block: dict[str, Any] = {"per_field": {}}
    paired_a: dict[str, list[float]] = {f: [] for f in fields}
    paired_b: dict[str, list[float]] = {f: [] for f in fields}
    matched = 0
    for multi_path in sorted(multi_dir.glob("*.json")):
        base_path = base_dir / multi_path.name
        if not base_path.exists():
            continue
        a = _load(multi_path)
        b = _load(base_path)
        if a is None or b is None:
            continue
        matched += 1
        for f in fields:
            paired_a[f].append(_has(a, f))
            paired_b[f].append(_has(b, f))
    coverage_block["n_paired_papers"] = matched
    bonferroni_factor = len(fields)
    for f in fields:
        a_arr = paired_a[f]
        b_arr = paired_b[f]
        if not a_arr:
            coverage_block["per_field"][f] = None
            continue
        cov_a = float(np.mean(a_arr))
        cov_b = float(np.mean(b_arr))
        p = paired_permutation_test(a_arr, b_arr, n_permutations=5000)
        p_corr = min(1.0, p * bonferroni_factor)
        coverage_block["per_field"][f] = {
            "multi_agent_coverage": cov_a,
            "baseline_coverage": cov_b,
            "delta": cov_a - cov_b,
            "p_value_raw": p,
            "p_value_bonferroni": p_corr,
        }

    # Token & wall-time distributions per phase
    distributions: dict[str, Any] = {}
    for name, sm in (
        ("multi_agent", multi_summary),
        ("baseline", base_summary),
    ) + (((("stability", stab_summary),)) if stab_summary else ()):
        ok = [r for r in sm["per_paper"] if r["status"] == "ok"]
        distributions[name] = {
            "tokens_in": percentiles([r["tokens_in"] for r in ok]),
            "tokens_out": percentiles([r["tokens_out"] for r in ok]),
            "cost_usd": percentiles([r["cost_usd"] for r in ok]),
            "wall_time_s": percentiles([r["wall_time_seconds"] for r in ok]),
            "n_ok": len(ok),
            "n_error": sum(1 for r in sm["per_paper"] if r["status"] == "error"),
            "error_rate": (
                sum(1 for r in sm["per_paper"] if r["status"] == "error")
                / max(1, len(sm["per_paper"]))
            ),
        }

    # Cost projection
    if distributions["multi_agent"]["n_ok"] > 0:
        per_paper_multi = distributions["multi_agent"]["cost_usd"]["mean"]
    else:
        per_paper_multi = 0.0
    if distributions["baseline"]["n_ok"] > 0:
        per_paper_base = distributions["baseline"]["cost_usd"]["mean"]
    else:
        per_paper_base = 0.0

    projections = {}
    for n in (1000, 5000, 10000, 50000, 100000):
        projections[f"n={n}"] = {
            "multi_agent_usd": per_paper_multi * n,
            "baseline_usd": per_paper_base * n,
        }

    # Aggregate JSON
    aggregate = {
        "stability": stability_block,
        "coverage": coverage_block,
        "distributions": distributions,
        "cost_projections": projections,
        "phase_costs": {
            "multi_agent": multi_summary["total_cost_usd"],
            "baseline": base_summary["total_cost_usd"],
            "stability": stab_summary["total_cost_usd"] if stab_summary else 0.0,
            "total": multi_summary["total_cost_usd"]
            + base_summary["total_cost_usd"]
            + (stab_summary["total_cost_usd"] if stab_summary else 0.0),
        },
        "phase_wall_times_s": {
            "multi_agent": multi_summary["wall_time_seconds"],
            "baseline": base_summary["wall_time_seconds"],
            "stability": stab_summary["wall_time_seconds"] if stab_summary else 0.0,
        },
    }
    (output_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )

    # Markdown report
    lines: list[str] = []
    lines.append("# Phase 2 — Pilot statistical report\n")
    lines.append("## Headline\n")
    ov = stability_block["overall"]
    lines.append(
        f"- **Stability (Jaccard, n={stability_block['n']}):** "
        f"{ov['mean']:.3f} (95% CI [{ov['ci_lo']:.3f}, {ov['ci_hi']:.3f}], std {ov['std']:.3f})"
    )
    lines.append(f"- **Total spend:** ${aggregate['phase_costs']['total']:.4f}")
    wt = aggregate["phase_wall_times_s"]
    lines.append(
        f"- **Wall-time:** multi {wt['multi_agent']:.1f}s, "
        f"baseline {wt['baseline']:.1f}s, stability {wt['stability']:.1f}s"
    )
    lines.append(f"- **Multi-agent error rate:** {distributions['multi_agent']['error_rate']*100:.1f}%")
    lines.append(f"- **Baseline error rate:** {distributions['baseline']['error_rate']*100:.1f}%\n")

    lines.append("## Stability per field (with 95% bootstrap CIs)\n")
    lines.append("| Field | Mean | 95% CI | Std |")
    lines.append("|---|---:|---|---:|")
    for f in fields:
        s = stability_block["per_field"][f]
        lines.append(f"| {f} | {s['mean']:.3f} | [{s['ci_lo']:.3f}, {s['ci_hi']:.3f}] | {s['std']:.3f} |")
    lines.append(f"| **overall** | **{ov['mean']:.3f}** | [{ov['ci_lo']:.3f}, {ov['ci_hi']:.3f}] | {ov['std']:.3f} |\n")

    lines.append("## Coverage comparison with significance (Bonferroni-corrected)\n")
    lines.append(
        f"_Paired papers (present in both multi-agent and baseline): {coverage_block['n_paired_papers']}_\n"
    )
    lines.append("| Field | Multi-agent | Baseline | Δ | p (raw) | p (Bonferroni) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for f in fields:
        c = coverage_block["per_field"].get(f)
        if c is None:
            lines.append(f"| {f} | — | — | — | — | — |")
            continue
        sig = "*" if c["p_value_bonferroni"] < 0.05 else ""
        lines.append(
            f"| {f} | {c['multi_agent_coverage']*100:.1f}% | {c['baseline_coverage']*100:.1f}% | "
            f"{c['delta']*100:+.1f}pp | {c['p_value_raw']:.4f} | {c['p_value_bonferroni']:.4f}{sig} |"
        )
    lines.append("")

    lines.append("## Per-paper distribution (papers with status=ok)\n")
    lines.append("| Phase | n | tokens_in (mean/p50/p95/p99) | tokens_out (mean/p50/p95/p99) | wall_time_s (mean/p50/p95/p99) | $/paper |")
    lines.append("|---|---:|---|---|---|---:|")
    for name, d in distributions.items():
        if d["n_ok"] == 0:
            lines.append(f"| {name} | 0 | — | — | — | — |")
            continue
        ti, to, wt2, cu = d["tokens_in"], d["tokens_out"], d["wall_time_s"], d["cost_usd"]
        lines.append(
            f"| {name} | {d['n_ok']} | "
            f"{ti['mean']:.0f} / {ti['p50']:.0f} / {ti['p95']:.0f} / {ti['p99']:.0f} | "
            f"{to['mean']:.0f} / {to['p50']:.0f} / {to['p95']:.0f} / {to['p99']:.0f} | "
            f"{wt2['mean']:.1f} / {wt2['p50']:.1f} / {wt2['p95']:.1f} / {wt2['p99']:.1f} | "
            f"${cu['mean']:.5f} |"
        )
    lines.append("")

    lines.append("## Cost projection (extrapolated from observed mean cost/paper)\n")
    lines.append("| Corpus size | Multi-agent (USD) | Baseline (USD) |")
    lines.append("|---|---:|---:|")
    for k, v in projections.items():
        lines.append(f"| {k} | ${v['multi_agent_usd']:.2f} | ${v['baseline_usd']:.2f} |")
    lines.append("")

    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    console.print(f"[green]Wrote {output_dir / 'report.md'} and aggregate.json[/green]")


if __name__ == "__main__":
    app()
