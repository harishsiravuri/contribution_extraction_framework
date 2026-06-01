"""Build outputs/paper_data_v2/RESULTS.md — v2 summary, with v1→v2 deltas
where comparable."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=False, add_completion=False)
console = Console()

V1 = Path("outputs/paper_data")
V2 = Path("outputs/paper_data_v2")


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
    out.append("# Paper 1 — RESULTS.md (v2)\n")
    out.append("_v2 run targets the four blockers in v1: (1) 11% error rate, (2) near-zero triple F1, (3) GNN AUC=0.5, (4) frontier n=4._\n")

    # Headline + v1→v2 deltas
    out.append("## 1. Headline (v1 → v2 deltas)\n")
    v1_agg = _load(V1 / "phase2_pilot_10k" / "aggregate.json")
    v2_agg = _load(V2 / "pilot_5k" / "aggregate.json")
    rows = []
    if v1_agg and v2_agg:
        rows.append(("Stability overall (mean Jaccard)", v1_agg["stability"]["overall"]["mean"], v2_agg["stability"]["overall"]["mean"]))
        rows.append(("Multi-agent error rate", v1_agg["distributions"]["multi_agent"]["error_rate"], v2_agg["distributions"]["multi_agent"]["error_rate"]))
        rows.append(("Baseline error rate", v1_agg["distributions"]["baseline"]["error_rate"], v2_agg["distributions"]["baseline"]["error_rate"]))
        rows.append(("Mean $/paper (multi)", v1_agg["distributions"]["multi_agent"]["cost_usd"]["mean"], v2_agg["distributions"]["multi_agent"]["cost_usd"]["mean"]))
        rows.append(("Coverage Δ (datasets, pp)", v1_agg["coverage"]["per_field"]["datasets"]["delta"]*100, v2_agg["coverage"]["per_field"]["datasets"]["delta"]*100))
        rows.append(("Coverage Δ (metrics, pp)", v1_agg["coverage"]["per_field"]["metrics"]["delta"]*100, v2_agg["coverage"]["per_field"]["metrics"]["delta"]*100))
        out.append("| Metric | v1 | v2 | Δ |")
        out.append("|---|---:|---:|---:|")
        for name, a, b in rows:
            out.append(f"| {name} | {a:.4f} | {b:.4f} | {b-a:+.4f} |")
    else:
        out.append("_v1 or v2 aggregate missing._")
    out.append("")

    # v2 phase B detail
    out.append("## 2. Phase B — pilot 5K (full)\n")
    rb = (V2 / "pilot_5k" / "report.md")
    if rb.exists():
        out.append(rb.read_text())
    out.append("")

    # Phase C benchmarks
    out.append("## 3. Phase C — benchmarks (with canonicalizer)\n")
    rc = (V2 / "benchmarks" / "results_table.md")
    if rc.exists():
        out.append(rc.read_text())
    else:
        out.append("_Phase C not run._")
    out.append("")

    # Phase D frontier
    out.append("## 4. Phase D — frontier ablation\n")
    rd = (V2 / "frontier" / "report.md")
    if rd.exists():
        out.append(rd.read_text())
    else:
        out.append("_Phase D not run._")
    out.append("")

    # Phase E calibration
    out.append("## 5. Phase E — calibration with temperature scaling\n")
    re_path = (V2 / "calibration" / "report.md")
    if re_path.exists():
        out.append(re_path.read_text())
    else:
        out.append("_Phase E not run._")
    out.append("")

    # Phase F GNN
    out.append("## 6. Phase F — downstream GNN (re-run at scale)\n")
    rf = (V2 / "downstream" / "results.md")
    if rf.exists():
        out.append(rf.read_text())
    else:
        out.append("_Phase F not run._")
    out.append("")

    # Limitations
    out.append("## 7. Limitations\n")
    out.append("- **Pilot scaled to ≤ 5K papers**, not 10K, because OpenRouter throughput in this session capped at ~5–6 papers/min sustained. The architecture and runner support 10K when more wall-time is available.")
    out.append("- **TDMSci is sentence-granular** so triple F1 there is intrinsically hard — the gold (T, D, M) triple often spans multiple sentences.")
    out.append("- **Frontier ablation** uses Opus 4.6 only on the SciREX dev set + 20 TDMSci sentences — n is now meaningful but cost-per-paper limits the scale.")
    out.append("- **Calibration via self_consistency** is a proxy for confidence. Temperature scaling helps but a learned reliability layer would help more.")
    out.append("- **Span-F1 vs SciREX exact protocol**: we report set-level lenient F1, not the joint mention+coref F1 the SciREX paper reports. Numbers are directional.")

    out.append("\n## 8. Figures\n")
    fig_readme = V2 / "figures" / "README.md"
    if fig_readme.exists():
        out.append(fig_readme.read_text())
    else:
        out.append("_Figures not regenerated._")

    (V2 / "RESULTS.md").write_text("\n".join(out), encoding="utf-8")
    console.print(f"[green]Wrote {V2 / 'RESULTS.md'}[/green]")


if __name__ == "__main__":
    app()
