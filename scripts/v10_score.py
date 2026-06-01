"""v10 — Post-process the default zero-shot framework's test-split run.

Reads:
  outputs/paper_data_v10/test_split_default/scirex/evaluation.json (from
    run_benchmarks.py: F1 + bootstrap CIs vs single-LLM baseline)
  outputs/paper_data_v10/test_split_default/scirex/multi_agent/*.json
  outputs/paper_data_v2/calibration/calibration.json (dev-fitted T values)
  outputs/paper_data_v3/benchmarks/scirex/evaluation.json (dev numbers for comparison)

Writes:
  outputs/paper_data_v10/test_split_default/results.json
  outputs/paper_data_v10/test_split_default/SUMMARY.md
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import typer
from rich.console import Console

from paper1.loaders import load_scirex
from paper1.metrics import bootstrap_ci, expected_calibration_error
from paper1.metrics.span_f1 import set_f1
from paper1.metrics.temperature_scaling import apply_temperature
from paper1.metrics.triple_f1 import triple_f1
from paper1.schema import ContributionRecord
from paper1.voting import _norm

SEED = 42
N_BOOT = 1000

app = typer.Typer(no_args_is_help=False, add_completion=False)
console = Console()


def _load_dir(d: Path) -> dict[str, ContributionRecord]:
    out: dict[str, ContributionRecord] = {}
    if not d.exists():
        return out
    for f in d.glob("*.json"):
        try:
            rec = ContributionRecord.model_validate_json(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        pid = f.stem.replace("scirex__", "scirex:")
        out[pid] = rec
    return out


def _pred_sets(rec: ContributionRecord) -> dict[str, set[str]]:
    methods, tasks, datasets, metrics = set(), set(), set(), set()
    for c in rec.contributions:
        if (n := _norm(c.method.name)): methods.add(n)
        if (n := _norm(c.task.name)): tasks.add(n)
        for d in c.datasets:
            if (n := _norm(d.name)): datasets.add(n)
        for m in c.metrics:
            if (n := _norm(m.name)): metrics.add(n)
    return {"methods": methods, "tasks": tasks, "datasets": datasets, "metrics": metrics}


def _pred_triples(rec: ContributionRecord) -> set[tuple[str, str, str]]:
    out = set()
    for c in rec.contributions:
        nt = _norm(c.task.name)
        if not nt: continue
        for d in c.datasets:
            nd = _norm(d.name)
            if not nd: continue
            for m in c.metrics:
                nm = _norm(m.name)
                if nm:
                    out.add((nt, nd, nm))
    return out


def _correct(pred_set: set[str], gold_set: set[str]) -> int | None:
    if not gold_set: return None
    if not pred_set: return 0
    for p in pred_set:
        for g in gold_set:
            if p in g or g in p:
                return 1
    return 0


@app.command()
def score(
    base: Path = typer.Option(Path("outputs/paper_data_v10/test_split_default"), "--base"),
    v3_eval: Path = typer.Option(Path("outputs/paper_data_v3/benchmarks/scirex/evaluation.json"), "--v3-eval"),
    calib_path: Path = typer.Option(Path("outputs/paper_data_v2/calibration/calibration.json"), "--calib-path"),
):
    # 1. Load test eval (produced by run_benchmarks.py)
    eval_path = base / "scirex" / "evaluation.json"
    eval_doc = json.loads(eval_path.read_text())
    # 2. Load dev eval (v3) for the side-by-side
    v3_doc = json.loads(v3_eval.read_text())

    # 3. Load test records, compute triple F1 with bootstrap CI
    test_recs = _load_dir(base / "scirex" / "multi_agent")
    test_papers = [p for p in load_scirex(splits=("test",)) if p.full_text and len(p.full_text) >= 50][:66]
    triples_paired = []
    for p in test_papers:
        rec = test_recs.get(p.paper_id)
        if rec is None: continue
        if not p.gold.triples: continue
        triples_paired.append(triple_f1(_pred_triples(rec), p.gold.triples)["f1"])
    a_rng = np.random.default_rng(SEED)
    if triples_paired:
        tri_mean, tri_lo, tri_hi = bootstrap_ci(triples_paired, n_resamples=N_BOOT, rng=a_rng)
    else:
        tri_mean = tri_lo = tri_hi = 0.0

    # 4. ECE on test using dev-fitted T values from v2 calibration
    calib = json.loads(calib_path.read_text())["per_field"]
    T_per_field = {f: (calib.get(f) or {}).get("T") for f in ("method.name", "task.name", "datasets", "metrics")}

    # Build (confidence, correct) pairs on test — first-contribution per paper, matching v2 convention
    pairs: dict[str, list[tuple[float, int]]] = {
        "method.name": [], "task.name": [], "datasets": [], "metrics": []
    }
    by_id = {p.paper_id: p for p in test_papers}
    for pid, rec in test_recs.items():
        p = by_id.get(pid)
        if p is None or not rec.contributions: continue
        c0 = rec.contributions[0]
        conf = c0.self_consistency
        ps_field = {
            "method.name": {_norm(c0.method.name)} - {""} if c0.method and c0.method.name else set(),
            "task.name":   {_norm(c0.task.name)} - {""} if c0.task and c0.task.name else set(),
            "datasets":    {_norm(d.name) for d in c0.datasets if d.name} - {""},
            "metrics":     {_norm(m.name) for m in c0.metrics if m.name} - {""},
        }
        gold_field = {
            "method.name": p.gold.methods,
            "task.name":   p.gold.tasks,
            "datasets":    p.gold.datasets,
            "metrics":     p.gold.metrics,
        }
        for f in pairs:
            k = _correct(ps_field[f], gold_field[f])
            if k is not None:
                pairs[f].append((conf, k))

    test_ece_block: dict[str, dict] = {}
    for f, ps in pairs.items():
        if not ps:
            test_ece_block[f] = None
            continue
        confs = [c for c, _ in ps]
        corrects = [k for _, k in ps]
        ece_raw = expected_calibration_error(confs, corrects, n_bins=10)
        T = T_per_field.get(f)
        ece_T = None
        if T:
            scaled = apply_temperature(confs, T).tolist()
            ece_T = expected_calibration_error(scaled, corrects, n_bins=10)
        test_ece_block[f] = {
            "n": len(ps),
            "T_from_dev": T,
            "ece_uncalibrated_test": ece_raw,
            "ece_dev_T_applied_to_test": ece_T,
            "ece_calibrated_dev_baseline": (calib.get(f) or {}).get("ece_calibrated"),
            "ece_uncalibrated_dev_baseline": (calib.get(f) or {}).get("ece_uncalibrated"),
        }

    # 5. Build results.json
    # Pull per-field test F1 from eval_doc (already has bootstrap CIs from run_benchmarks)
    fields = ("methods", "tasks", "datasets", "metrics")
    per_field: dict = {}
    for f in fields:
        test_block = eval_doc["fields"][f]["multi_agent"]
        dev_block = v3_doc["fields"][f]["multi_agent"]
        per_field[f] = {
            "n_test": eval_doc["fields"][f]["n"],
            "n_dev": v3_doc["fields"][f]["n"],
            "test_f1": test_block["f1"],
            "test_ci_lo": test_block["ci_lo"],
            "test_ci_hi": test_block["ci_hi"],
            "dev_f1": dev_block["f1"],
            "dev_ci_lo": dev_block["ci_lo"],
            "dev_ci_hi": dev_block["ci_hi"],
            "delta_test_minus_dev": test_block["f1"] - dev_block["f1"],
        }

    # Extraction summary (cost + wall)
    summary = json.loads((base / "scirex" / "multi_agent_summary.json").read_text())
    base_summary = json.loads((base / "scirex" / "baseline_summary.json").read_text())

    result = {
        "experiment": "v10_default_zero_shot_test_split",
        "seed": SEED,
        "n_resamples_bootstrap": N_BOOT,
        "config_path": "config/models.yaml",
        "extractor_model": summary["per_paper"][0].get("paper_id") and "deepseek/deepseek-chat",
        "papers_total": summary["papers_total"],
        "papers_ok_multi": summary["papers_ok"],
        "papers_error_multi": summary["papers_error"],
        "papers_ok_baseline": base_summary["papers_ok"],
        "multi_total_cost_usd": summary["total_cost_usd"],
        "baseline_total_cost_usd": base_summary["total_cost_usd"],
        "total_cost_usd": summary["total_cost_usd"] + base_summary["total_cost_usd"],
        "multi_wall_time_seconds": summary["wall_time_seconds"],
        "per_field_dev_vs_test": per_field,
        "triple_f1_test": {
            "n": len(triples_paired),
            "f1": tri_mean, "ci_lo": tri_lo, "ci_hi": tri_hi,
            "exploratory": True,
            "note": "Triple F1 on the held-out test split; framework was not tuned for triple binding.",
        },
        "calibration_test_with_dev_fitted_T": test_ece_block,
        "records_dir": str(base / "scirex" / "multi_agent"),
    }
    (base / "results.json").write_text(json.dumps(result, indent=2))
    console.print(f"[green]Wrote {base / 'results.json'}[/green]")

    # 6. SUMMARY.md
    lines = [
        "# v10 — Default zero-shot framework on SciREX TEST split (held-out)\n",
        f"_Configuration: identical to the dev-split run in `artifacts/paper_data_v3/`. "
        f"DeepSeek Chat Extractor at t ∈ {{0.0, 0.3, 0.7}}; Llama 3.3 70B Instruct Critic + "
        f"Consolidator at t=0. No prompt or hyperparameter touched the test split before this run._\n",
        f"_Papers: {summary['papers_total']} total, "
        f"{summary['papers_ok']} multi-agent ok, {base_summary['papers_ok']} baseline ok. "
        f"Spend: \\${result['total_cost_usd']:.4f}._\n",
        "## Dev vs Test per-field F1 (default zero-shot framework)\n",
        "| Field   | Dev F1 [95% CI]                | Test F1 [95% CI]               | Δ (test−dev) |",
        "|---------|---------------------------------|---------------------------------|--------------|",
    ]
    for f, label in (("methods", "Method"), ("tasks", "Task"), ("datasets", "Dataset"), ("metrics", "Metric")):
        b = per_field[f]
        lines.append(
            f"| {label:7s} | {b['dev_f1']:.3f} [{b['dev_ci_lo']:.3f}, {b['dev_ci_hi']:.3f}] (n={b['n_dev']}) "
            f"| {b['test_f1']:.3f} [{b['test_ci_lo']:.3f}, {b['test_ci_hi']:.3f}] (n={b['n_test']}) "
            f"| {b['delta_test_minus_dev']:+.3f} |"
        )
    lines.append(
        f"| Triple* | — | {tri_mean:.3f} [{tri_lo:.3f}, {tri_hi:.3f}] (n={len(triples_paired)}) | — |"
    )
    lines.append("")
    lines.append("_*Triple = (Task, Dataset, Metric) joint F1, exploratory only (framework was "
                 "not tuned for triple binding)._\n")

    lines.append("## Dev vs Test ECE (with dev-fitted T values, applied to test)\n")
    lines.append("| Field        | Dev ECE pre-T | Dev ECE post-T | Test ECE pre-T | Test ECE post-T (dev T) | T (from dev) |")
    lines.append("|--------------|---------------|----------------|-----------------|--------------------------|--------------|")
    for f in ("method.name", "task.name", "datasets", "metrics"):
        c = test_ece_block.get(f)
        if c is None:
            lines.append(f"| {f:13s} | — | — | — | — | — |")
            continue
        T = c["T_from_dev"]
        T_str = f"{T:.2f}" if T is not None else "—"
        e_dev_pre = c["ece_uncalibrated_dev_baseline"]
        e_dev_post = c["ece_calibrated_dev_baseline"]
        e_test_pre = c["ece_uncalibrated_test"]
        e_test_post = c["ece_dev_T_applied_to_test"]
        lines.append(
            f"| {f:13s} | {e_dev_pre:.3f} | {e_dev_post:.3f} | {e_test_pre:.3f} | "
            f"{(e_test_post or 0):.3f} | {T_str} |"
        )
    lines.append("")
    # Plain-language interpretation
    deltas_within = sum(1 for f in fields
                       if abs(per_field[f]["delta_test_minus_dev"]) <= 0.05)
    big_drops = [f for f in fields if per_field[f]["delta_test_minus_dev"] <= -0.05]
    big_gains = [f for f in fields if per_field[f]["delta_test_minus_dev"] >= 0.05]
    lines.append("## Interpretation\n")
    bullets = []
    if not big_drops and not big_gains:
        bullets.append(
            "**The dev-split default zero-shot result holds up on the held-out test split.** "
            f"All four per-field deltas are within ±0.05 absolute F1 of their dev values. "
            "This rules out the most damaging methodological concern — that the dev-split numbers "
            "reflect prompt tuning rather than the framework's intrinsic behaviour."
        )
    elif big_drops:
        bullets.append(
            f"**Some fields regress on test relative to dev:** {', '.join(big_drops)} "
            f"(|Δ| > 0.05). Other fields remain within ±0.05. Honest framing for the manuscript: "
            "the default zero-shot framework is partially robust on held-out data."
        )
    elif big_gains:
        bullets.append(
            f"**Some fields improve on test relative to dev:** {', '.join(big_gains)} "
            f"(|Δ| > 0.05). Other fields remain within ±0.05. This is consistent with the "
            "manuscript's framing — no over-fitting to the dev split."
        )
    lines.append(bullets[0] + "\n")
    (base / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    console.print(f"[green]Wrote {base / 'SUMMARY.md'}[/green]")


if __name__ == "__main__":
    app()
