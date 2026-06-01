"""Compute calibration: confidence-vs-correctness over Phase 4 benchmark runs.

For each multi-agent ContributionRecord and matching GoldPaper:
- Per field, treat the predicted entity as "correct" if any normalized name
  appears in gold (lenient set match).
- Use self_consistency from the record (the contribution-level confidence
  the Consolidator emits) as the model's confidence.
- Aggregate (confidence, correctness) pairs across all papers, compute ECE,
  reliability diagram bins.

Saves outputs/paper_data/phase5_calibration/calibration.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from paper1.loaders import GoldPaper, load_nlp_tdms, load_scirex, load_tdmsci
from paper1.metrics import expected_calibration_error, reliability_diagram_data
from paper1.schema import ContributionRecord
from paper1.voting import _norm

app = typer.Typer(no_args_is_help=False, add_completion=False)
console = Console()


def _safe_id(paper_id: str) -> str:
    return paper_id.replace(":", "__").replace("/", "_")


def _load_record(path: Path) -> ContributionRecord | None:
    try:
        return ContributionRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _is_correct(pred_set: set[str], gold_set: set[str]) -> int:
    if not gold_set:
        return -1  # skip; no gold for this field
    if not pred_set:
        return 0
    # Lenient: any predicted name overlaps any gold name (substring either direction)
    for p in pred_set:
        for g in gold_set:
            if p in g or g in p:
                return 1
    return 0


@app.command()
def run(
    benchmarks_dir: Path = typer.Option(Path("outputs/paper_data/phase4_benchmarks"), "--benchmarks-dir"),
    output_dir: Path = typer.Option(Path("outputs/paper_data/phase5_calibration"), "--output-dir"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    bench_funcs = {"scirex": load_scirex, "tdmsci": load_tdmsci, "nlp_tdms": load_nlp_tdms}
    pairs_per_field: dict[str, list[tuple[float, int]]] = {
        "method.name": [], "task.name": [], "datasets": [], "metrics": []
    }

    for benchmark, loader in bench_funcs.items():
        bench_dir = benchmarks_dir / benchmark / "multi_agent"
        if not bench_dir.exists():
            continue
        gold_papers: dict[str, GoldPaper] = {p.paper_id: p for p in loader()}
        for rec_path in sorted(bench_dir.glob("*.json")):
            paper_id = rec_path.stem.replace("__", ":")
            gold = gold_papers.get(paper_id)
            if gold is None:
                continue
            rec = _load_record(rec_path)
            if rec is None or not rec.contributions:
                continue
            c0 = rec.contributions[0]
            confidence = c0.self_consistency
            # Per field
            method_pred = {_norm(c0.method.name)} - {None, ""}
            task_pred = {_norm(c0.task.name)} - {None, ""}
            ds_pred = {n for n in (_norm(d.name) for d in c0.datasets) if n}
            mt_pred = {n for n in (_norm(m.name) for m in c0.metrics) if n}

            for field, pred, gold_set in (
                ("method.name", method_pred, gold.gold.methods),
                ("task.name", task_pred, gold.gold.tasks),
                ("datasets", ds_pred, gold.gold.datasets),
                ("metrics", mt_pred, gold.gold.metrics),
            ):
                correct = _is_correct(pred, gold_set)
                if correct < 0:
                    continue
                pairs_per_field[field].append((confidence, correct))

    out: dict = {"per_field": {}}
    for field, pairs in pairs_per_field.items():
        if not pairs:
            out["per_field"][field] = None
            continue
        confs = [c for c, _ in pairs]
        corrects = [k for _, k in pairs]
        ece = expected_calibration_error(confs, corrects, n_bins=10)
        bins = reliability_diagram_data(confs, corrects, n_bins=10)
        out["per_field"][field] = {
            "n": len(pairs),
            "ece": ece,
            "mean_confidence": sum(confs) / len(confs),
            "mean_accuracy": sum(corrects) / len(corrects),
            "reliability_bins": bins,
        }

    (output_dir / "calibration.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Markdown
    lines = ["# Phase 5 — Calibration\n"]
    lines.append("ECE measured against the lenient set-match definition of correctness used in Phase 4.")
    lines.append("Confidence per contribution = `self_consistency` from the Consolidator.\n")
    lines.append("| Field | n | ECE | Mean conf. | Mean acc. |")
    lines.append("|---|---:|---:|---:|---:|")
    for f in ("method.name", "task.name", "datasets", "metrics"):
        d = out["per_field"][f]
        if d is None:
            lines.append(f"| {f} | — | — | — | — |")
            continue
        lines.append(f"| {f} | {d['n']} | {d['ece']:.4f} | {d['mean_confidence']:.3f} | {d['mean_accuracy']:.3f} |")
    lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    console.print(f"[green]Wrote {output_dir / 'report.md'}[/green]")


if __name__ == "__main__":
    app()
