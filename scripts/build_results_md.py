"""Build outputs/paper_data/RESULTS.md — single-source-of-truth summary
synthesised from all completed phase outputs. Skips sections gracefully if
phase data is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=False, add_completion=False)
console = Console()

ROOT = Path("outputs/paper_data")


def _load(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


@app.command()
def run() -> None:
    out = []
    out.append("# Paper 1 — RESULTS.md\n")
    out.append("_Single-source summary aggregated from all completed phases. Honest reading only — phases not run are flagged._\n")

    # 1. Headline
    out.append("## 1. Headline numbers\n")
    agg = _load(ROOT / "phase2_pilot_10k" / "aggregate.json")
    if agg:
        ov = agg["stability"]["overall"]
        out.append(f"- **Stability (Jaccard, n={agg['stability']['n']})**: {ov['mean']:.3f} (95% CI [{ov['ci_lo']:.3f}, {ov['ci_hi']:.3f}])")
        cost = agg["phase_costs"]
        out.append(f"- **Phase 2 spend**: ${cost['total']:.4f} (multi ${cost['multi_agent']:.3f}, base ${cost['baseline']:.3f}, stab ${cost['stability']:.3f})")
        d = agg["distributions"]
        if d.get("multi_agent"):
            out.append(f"- **Multi-agent error rate**: {d['multi_agent']['error_rate']*100:.1f}% (was 22% before Phase 1 fixes)")
        if d.get("baseline"):
            out.append(f"- **Baseline error rate**: {d['baseline']['error_rate']*100:.1f}%")
    else:
        out.append("- _Phase 2 aggregate not present_")
    out.append("")

    # 2. Stability
    out.append("## 2. Stability table\n")
    if agg:
        out.append("| Field | n | Mean Jaccard | 95% CI | Std |")
        out.append("|---|---:|---:|---|---:|")
        for f in ("method.name", "task.name", "datasets", "metrics", "claim_strength"):
            s = agg["stability"]["per_field"][f]
            out.append(f"| {f} | {agg['stability']['n']} | {s['mean']:.3f} | [{s['ci_lo']:.3f}, {s['ci_hi']:.3f}] | {s['std']:.3f} |")
        ov = agg["stability"]["overall"]
        out.append(f"| **overall** | {agg['stability']['n']} | **{ov['mean']:.3f}** | [{ov['ci_lo']:.3f}, {ov['ci_hi']:.3f}] | {ov['std']:.3f} |")
    else:
        out.append("_Phase 2 not yet run._")
    out.append("")

    # 3. Coverage
    out.append("## 3. Coverage (multi-agent vs baseline) with Bonferroni-corrected p-values\n")
    if agg:
        out.append(f"_paired papers: {agg['coverage']['n_paired_papers']}_\n")
        out.append("| Field | Multi | Base | Δ | p (raw) | p (Bonf) |")
        out.append("|---|---:|---:|---:|---:|---:|")
        for f in ("method.name", "task.name", "datasets", "metrics", "claim_strength"):
            c = agg["coverage"]["per_field"].get(f)
            if c is None:
                out.append(f"| {f} | — | — | — | — | — |")
                continue
            sig = "*" if c["p_value_bonferroni"] < 0.05 else ""
            out.append(
                f"| {f} | {c['multi_agent_coverage']*100:.1f}% | {c['baseline_coverage']*100:.1f}% | "
                f"{c['delta']*100:+.1f}pp | {c['p_value_raw']:.4f} | {c['p_value_bonferroni']:.4f}{sig} |"
            )
    else:
        out.append("_Phase 2 not run._")
    out.append("")

    # 4. Public benchmarks
    out.append("## 4. Public benchmark comparison (vs published prior)\n")
    bench_dir = ROOT / "phase4_benchmarks"
    table_path = bench_dir / "results_table.md"
    if table_path.exists():
        out.append(table_path.read_text())
    else:
        out.append("_Phase 4 not run / not finished._")
    out.append("")

    # 5. Frontier vs open weights
    out.append("## 5. Frontier vs open-weights extractor\n")
    fr_path = ROOT / "phase6_frontier" / "report.md"
    if fr_path.exists():
        out.append(fr_path.read_text())
    else:
        out.append("_Phase 6 frontier ablation not run (see end-of-run report for reason)._")
    out.append("")

    # 6. Calibration
    out.append("## 6. Calibration\n")
    cal_path = ROOT / "phase5_calibration" / "report.md"
    if cal_path.exists():
        out.append(cal_path.read_text())
    else:
        out.append("_Phase 5 calibration not run._")
    out.append("")

    # 7. Downstream
    out.append("## 7. Downstream GNN link prediction\n")
    ds_path = ROOT / "phase7_downstream" / "results.md"
    if ds_path.exists():
        out.append(ds_path.read_text())
    else:
        out.append("_Phase 7 GNN not run._")
    out.append("")

    # 8. Cost
    out.append("## 8. Cost analysis\n")
    if agg:
        d = agg["distributions"]
        if d.get("multi_agent"):
            out.append(f"- $/paper observed (multi-agent, n={d['multi_agent']['n_ok']}): ${d['multi_agent']['cost_usd']['mean']:.5f}")
        if d.get("baseline"):
            out.append(f"- $/paper observed (baseline,    n={d['baseline']['n_ok']}): ${d['baseline']['cost_usd']['mean']:.5f}")
        out.append("\n| Corpus | Multi-agent | Baseline |")
        out.append("|---|---:|---:|")
        for k, v in agg["cost_projections"].items():
            out.append(f"| {k} | ${v['multi_agent_usd']:.2f} | ${v['baseline_usd']:.2f} |")
    out.append("")

    # 9. Errors
    out.append("## 9. Error analysis\n")
    cats = {}
    for phase in ("multi_agent", "baseline", "stability"):
        s = _load(ROOT / "phase2_pilot_10k" / phase / "summary.json")
        if not s:
            continue
        for r in s["per_paper"]:
            if r["status"] == "error" and r["error"]:
                key = r["error"].split(":", 1)[0]
                cats[(phase, key)] = cats.get((phase, key), 0) + 1
    if cats:
        out.append("| Phase | Category | n |")
        out.append("|---|---|---:|")
        for (phase, k), n in sorted(cats.items(), key=lambda kv: -kv[1]):
            out.append(f"| {phase} | {k} | {n} |")
    else:
        out.append("_No errors recorded._")
    out.append("")

    # 10. Limitations
    out.append("## 10. Limitations\n")
    out.append(
        "- **Abstracts only**, not full text. The Phase 2 corpus is arXiv abstracts; "
        "SciREX in Phase 4 uses full-text. Results on metric *values* (numeric F1 scores etc.) are not assessed.\n"
        "- **Topic skew**. The arXiv pull is cs.LG / cs.CL / cs.CV from 2022+, sorted by submission date. "
        "Recent + ML-heavy. Generalisation to bio/chem/physics-of-science is untested.\n"
        "- **Self-consistency as confidence**. Calibration uses the Consolidator's "
        "`self_consistency` field. This is the agreement-fraction across 3 voting runs, "
        "not a learned probability — it can be miscalibrated even if labels are correct.\n"
        "- **Set-level F1, not span F1**. Phase 4 reports lenient set-match F1 (predicted entity-name "
        "set vs gold entity-name set). The original SciREX joint model reports document-level entity F1, "
        "but our setup does not directly produce mention-level spans for SciREX's exact protocol — these "
        "numbers are directional, not head-to-head.\n"
        "- **No 1k → 10k scaling validation**. Cost projections at 50k–100k are linear extrapolations, "
        "not measured.\n"
    )

    # 11. Figures
    out.append("## 11. Figures\n")
    fig_readme = ROOT / "figures" / "README.md"
    if fig_readme.exists():
        out.append(fig_readme.read_text())
    else:
        out.append("_Figures not generated yet._")
    out.append("")

    # 12. Supplementary tables
    out.append("## 12. Supplementary tables\n")
    out.append("- `outputs/paper_data/phase1_bug_fixes.md` — Phase 1 changelog")
    out.append("- `outputs/paper_data/phase2_pilot_10k/aggregate.json` — full Phase 2 stats")
    out.append("- `outputs/paper_data/phase2_pilot_10k/multi_agent/summary.json` / `baseline/...` / `stability/...`")
    out.append("- `outputs/paper_data/phase3_loaders.md` — gold-benchmark loader documentation")
    out.append("- `outputs/paper_data/phase4_benchmarks/<bench>/evaluation.json`")
    out.append("- `outputs/paper_data/phase5_calibration/calibration.json`")
    out.append("- `outputs/paper_data/phase6_frontier/...` (if present)")
    out.append("- `outputs/paper_data/phase7_downstream/results.json` (if present)")

    (ROOT / "RESULTS.md").write_text("\n".join(out), encoding="utf-8")
    console.print(f"[green]Wrote {ROOT / 'RESULTS.md'}[/green]")


if __name__ == "__main__":
    app()
