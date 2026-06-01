"""Phase E v2 — calibration with temperature scaling on the v2 benchmark runs.

Reads multi-agent records from outputs/paper_data_v2/benchmarks/<bench>/multi_agent/
and the matching gold from the loaders, builds (confidence, correctness) pairs
per field, fits a temperature on a 20% held-out split, evaluates ECE on 80%.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from paper1.canonicalize import (
    canonicalize_dataset_full,
    canonicalize_method_name,
    canonicalize_metric_name,
    canonicalize_task_name,
)
from paper1.loaders import load_nlp_tdms, load_scirex, load_tdmsci
from paper1.metrics import expected_calibration_error, reliability_diagram_data
from paper1.metrics.temperature_scaling import calibrate_and_evaluate, apply_temperature, fit_temperature
from paper1.schema import ContributionRecord
from paper1.voting import _norm

app = typer.Typer(no_args_is_help=False, add_completion=False)
console = Console()


def _correct(pred_set: set[str], gold_set: set[str]) -> int:
    if not gold_set:
        return -1
    if not pred_set:
        return 0
    for p in pred_set:
        for g in gold_set:
            if p in g or g in p:
                return 1
    return 0


def _load(path: Path) -> ContributionRecord | None:
    try:
        return ContributionRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        return None


@app.command()
def run(
    benchmarks_dir: Path = typer.Option(Path("outputs/paper_data_v2/benchmarks"), "--benchmarks-dir"),
    output_dir: Path = typer.Option(Path("outputs/paper_data_v2/calibration"), "--output-dir"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    bench_funcs = {"scirex": load_scirex, "tdmsci": load_tdmsci, "nlp_tdms": load_nlp_tdms}
    pairs: dict[str, list[tuple[float, int]]] = {
        "method.name": [], "task.name": [], "datasets": [], "metrics": []
    }

    for benchmark, loader in bench_funcs.items():
        bench_dir = benchmarks_dir / benchmark / "multi_agent"
        if not bench_dir.exists():
            continue
        gold = {p.paper_id: p for p in loader()}
        for rec_path in sorted(bench_dir.glob("*.json")):
            paper_id = rec_path.stem.replace("__", ":")
            g = gold.get(paper_id)
            if g is None:
                continue
            rec = _load(rec_path)
            if rec is None or not rec.contributions:
                continue
            c0 = rec.contributions[0]
            conf = c0.self_consistency
            method_pred = {n for n in [_norm(canonicalize_method_name(c0.method.name))] if n}
            task_pred = {n for n in [_norm(canonicalize_task_name(c0.task.name))] if n}
            ds_pred = {n for n in (_norm(canonicalize_dataset_full(d.name)) for d in c0.datasets) if n}
            mt_pred = {n for n in (_norm(canonicalize_metric_name(m.name)) for m in c0.metrics) if n}
            for field, pred, gold_set in (
                ("method.name", method_pred, g.gold.methods),
                ("task.name", task_pred, g.gold.tasks),
                ("datasets", ds_pred, g.gold.datasets),
                ("metrics", mt_pred, g.gold.metrics),
            ):
                k = _correct(pred, gold_set)
                if k < 0:
                    continue
                pairs[field].append((conf, k))

    out: dict = {"per_field": {}}
    for field, ps in pairs.items():
        if not ps:
            out["per_field"][field] = None
            continue
        confs = [c for c, _ in ps]
        corrects = [k for _, k in ps]
        cal = calibrate_and_evaluate(confs, corrects, fit_frac=0.2, n_bins=10)
        # Reliability bins on the eval split, scaled
        T = cal["T"]
        scaled_full = apply_temperature(confs, T).tolist()
        bins_uncal = reliability_diagram_data(confs, corrects, n_bins=10)
        bins_cal = reliability_diagram_data(scaled_full, corrects, n_bins=10)
        out["per_field"][field] = {**cal, "reliability_uncalibrated": bins_uncal, "reliability_calibrated": bins_cal}

    (output_dir / "calibration.json").write_text(json.dumps(out, indent=2))

    lines = ["# Phase E — Calibration with temperature scaling\n",
             "Confidence per contribution = `self_consistency` (Consolidator output).",
             "Temperature T is fit on 20% of records, ECE evaluated on the remaining 80%.\n",
             "| Field | n | T | ECE before | ECE after | Mean conf (cal) | Mean acc |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for f in ("method.name", "task.name", "datasets", "metrics"):
        d = out["per_field"][f]
        if d is None:
            lines.append(f"| {f} | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {f} | {d['n']} | {d['T']:.3f} | {d['ece_uncalibrated']:.4f} | "
            f"{d['ece_calibrated']:.4f} | {d['mean_conf_cal']:.3f} | {d['mean_acc']:.3f} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines))
    console.print(f"[green]Wrote {output_dir / 'report.md'}[/green]")


if __name__ == "__main__":
    app()
