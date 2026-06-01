"""Generate every figure in scripts/figures/, then write figures/README.md."""

import importlib
import sys
from pathlib import Path

FIG_NAMES = [
    "fig_architecture",
    "fig_stability_overall",
    "fig_stability_per_field",
    "fig_coverage_comparison",
    "fig_benchmark_f1",
    "fig_frontier_vs_openweights",
    "fig_reliability_diagram",
    "fig_cost_scaling",
    "fig_walltime_distribution",
    "fig_error_breakdown",
    "fig_downstream_auc",
]


def main() -> None:
    out_dir = Path("outputs/paper_data/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))
    for name in FIG_NAMES:
        mod = importlib.import_module(f"scripts.figures.{name}")
        try:
            mod.main()
            print(f"  ✓ {name}")
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {name}: {type(e).__name__}: {e}")

    # README
    desc = {
        "fig_architecture": ("Multi-agent architecture (placeholder)", "n/a — manual draw target", "Section 3 (system overview)"),
        "fig_stability_overall": ("Overall Jaccard stability with 95% CI", "outputs/paper_data/phase2_pilot_10k/aggregate.json", "Section 6 — stability headline"),
        "fig_stability_per_field": ("Per-field Jaccard stability with 95% CI", "outputs/paper_data/phase2_pilot_10k/aggregate.json", "Section 6 — per-field stability"),
        "fig_coverage_comparison": ("Multi-agent vs baseline non-null coverage per field", "outputs/paper_data/phase2_pilot_10k/aggregate.json", "Section 6 — coverage"),
        "fig_benchmark_f1": ("Our F1 vs published priors on SciREX/TDMSci/NLP-TDMS", "outputs/paper_data/phase4_benchmarks/*/evaluation.json", "Section 6 — benchmark comparison"),
        "fig_frontier_vs_openweights": ("Open-weights vs frontier extractor F1", "outputs/paper_data/phase4_benchmarks + phase6_frontier", "Section 7 — frontier ablation"),
        "fig_reliability_diagram": ("Reliability diagram for method.name", "outputs/paper_data/phase5_calibration/calibration.json", "Section 6 — calibration"),
        "fig_cost_scaling": ("Projected cost vs corpus size", "outputs/paper_data/phase2_pilot_10k/aggregate.json", "Section 5 — cost"),
        "fig_walltime_distribution": ("Per-paper wall-time histogram", "outputs/paper_data/phase2_pilot_10k/multi_agent/summary.json", "Section 5 — runtime"),
        "fig_error_breakdown": ("Error category breakdown across phases", "outputs/paper_data/phase2_pilot_10k/*/summary.json", "Section 8 — limitations"),
        "fig_downstream_auc": ("GNN link-prediction AUC: multi-agent vs baseline graphs", "outputs/paper_data/phase7_downstream/results.json", "Section 6 — downstream task"),
    }

    lines = ["# Figures\n", "All figures saved at 300 DPI as both .png and .pdf.\n", "Palette: seaborn 'colorblind'.\n", "| File | What it shows | Data source | Paper section |", "|---|---|---|---|"]
    for name, (what, src, sect) in desc.items():
        lines.append(f"| `{name}.pdf` / `.png` | {what} | `{src}` | {sect} |")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir / 'README.md'}")


if __name__ == "__main__":
    main()
