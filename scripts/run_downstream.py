"""Phase 7 — train R-GCN link prediction on multi-agent vs baseline graphs.

For each input directory we build a graph, train across 3 seeds, and report
mean ± std test ROC-AUC. Then a paired permutation test on the test-set
predictions (same test pairs across both graphs).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np
import typer
from rich.console import Console

from paper1.downstream.graph_builder import build_graph, stats, to_json
from paper1.downstream.link_prediction import train_and_eval
from paper1.metrics import bootstrap_ci, paired_permutation_test

app = typer.Typer(no_args_is_help=False, add_completion=False)
console = Console()


@app.command()
def run(
    multi_dir: Path = typer.Option(
        Path("outputs/paper_data/phase2_pilot_10k/multi_agent/by_paper"), "--multi-dir"
    ),
    base_dir: Path = typer.Option(
        Path("outputs/paper_data/phase2_pilot_10k/baseline/by_paper"), "--base-dir"
    ),
    output_dir: Path = typer.Option(Path("outputs/paper_data/phase7_downstream"), "--output-dir"),
    seeds: int = typer.Option(3, "--seeds"),
    epochs: int = typer.Option(30, "--epochs"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if not multi_dir.exists() or not list(multi_dir.glob("*.json")):
        console.print(f"[red]No multi-agent records in {multi_dir}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Building graphs from:[/bold]")
    console.print(f"  multi: {multi_dir}")
    console.print(f"  base:  {base_dir}")

    multi_graph = build_graph(multi_dir)
    base_graph = build_graph(base_dir) if base_dir.exists() else None

    multi_stats = stats(multi_graph)
    base_stats = stats(base_graph) if base_graph else {}
    console.print(f"  multi graph stats: {multi_stats}")
    console.print(f"  base  graph stats: {base_stats}")
    (output_dir / "graph_multi_agent.json").write_text(json.dumps({"stats": multi_stats, "graph": to_json(multi_graph)}, indent=2))
    if base_graph:
        (output_dir / "graph_baseline.json").write_text(json.dumps({"stats": base_stats, "graph": to_json(base_graph)}, indent=2))

    multi_aucs: list[float] = []
    base_aucs: list[float] = []
    multi_runs = []
    base_runs = []
    for seed in range(seeds):
        console.print(f"  [seed={seed}] training multi...")
        m = train_and_eval(multi_graph, seed=seed, epochs=epochs)
        multi_aucs.append(m["test_auc"])
        multi_runs.append({"seed": seed, "test_auc": m["test_auc"], "n_train": m["n_train"], "n_test": m["n_test"]})
        if base_graph:
            console.print(f"  [seed={seed}] training baseline...")
            b = train_and_eval(base_graph, seed=seed, epochs=epochs)
            base_aucs.append(b["test_auc"])
            base_runs.append({"seed": seed, "test_auc": b["test_auc"], "n_train": b["n_train"], "n_test": b["n_test"]})

    multi_mean, multi_lo, multi_hi = (
        bootstrap_ci(multi_aucs, n_resamples=2000) if len(multi_aucs) >= 2 else (multi_aucs[0], multi_aucs[0], multi_aucs[0])
    )
    base_mean, base_lo, base_hi = (
        bootstrap_ci(base_aucs, n_resamples=2000) if len(base_aucs) >= 2 else (base_aucs[0], base_aucs[0], base_aucs[0])
    ) if base_aucs else (0.0, 0.0, 0.0)
    p_perm = paired_permutation_test(multi_aucs, base_aucs, n_permutations=2000) if base_aucs else 1.0

    summary = {
        "n_seeds": seeds,
        "multi_agent": {
            "aucs": multi_aucs,
            "mean": multi_mean,
            "ci_lo": multi_lo,
            "ci_hi": multi_hi,
            "std": statistics.pstdev(multi_aucs) if len(multi_aucs) > 1 else 0.0,
        },
        "baseline": {
            "aucs": base_aucs,
            "mean": base_mean,
            "ci_lo": base_lo,
            "ci_hi": base_hi,
            "std": statistics.pstdev(base_aucs) if len(base_aucs) > 1 else 0.0,
        },
        "p_value_paired_permutation": p_perm,
        "graph_stats": {"multi_agent": multi_stats, "baseline": base_stats},
        "per_seed": {"multi_agent": multi_runs, "baseline": base_runs},
    }
    (output_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    sig = " *" if p_perm < 0.05 else ""
    lines = [
        "# Phase 7 — Downstream GNN link prediction\n",
        "Task: shared-dataset link prediction. R-GCN-style 2-layer model, 64 dim hidden,",
        "5 negatives per positive, 80/10/10 split, ROC-AUC on the test split.",
        f"Seeds: {seeds}.\n",
        "## Graph stats\n",
        f"- Multi-agent: {multi_stats}",
        f"- Baseline:    {base_stats}\n",
        "## ROC-AUC\n",
        "| Graph | n_seeds | Mean | 95% CI | Std |",
        "|---|---:|---:|---|---:|",
        f"| multi-agent | {seeds} | {multi_mean:.4f} | [{multi_lo:.4f}, {multi_hi:.4f}] | {summary['multi_agent']['std']:.4f} |",
        f"| baseline    | {seeds} | {base_mean:.4f} | [{base_lo:.4f}, {base_hi:.4f}] | {summary['baseline']['std']:.4f} |",
        "",
        f"Paired permutation p-value (over per-seed AUCs): **{p_perm:.4f}**{sig}\n",
    ]
    (output_dir / "results.md").write_text("\n".join(lines), encoding="utf-8")
    console.print(f"[green]Wrote {output_dir / 'results.md'}[/green]")


if __name__ == "__main__":
    app()
