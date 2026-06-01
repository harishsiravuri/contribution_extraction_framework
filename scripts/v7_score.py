"""Compute per-experiment results.json files for the v7 experiments.

Reads per-paper ContributionRecord JSONs from each experiment's output dir,
computes per-field F1 with bootstrap 95% CIs (1000 resamples) and paired
permutation tests (10000 permutations) where applicable, plus calibration
metrics for E2 / E3.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich.console import Console

from paper1.loaders import GoldPaper, load_scirex, load_tdmsci
from paper1.metrics import (
    bootstrap_ci,
    expected_calibration_error,
    paired_permutation_test,
    reliability_diagram_data,
)
from paper1.metrics.span_f1 import set_f1
from paper1.metrics.temperature_scaling import (
    apply_temperature,
    calibrate_and_evaluate,
)
from paper1.schema import ContributionRecord
from paper1.voting import _norm

SEED = 42
RNG = np.random.default_rng(SEED)
N_BOOT = 1000
N_PERM = 10000

app = typer.Typer(no_args_is_help=False, add_completion=False)
console = Console()


def _safe_id(paper_id: str) -> str:
    return paper_id.replace(":", "__").replace("/", "_")


def _load_dir(d: Path) -> dict[str, ContributionRecord]:
    out: dict[str, ContributionRecord] = {}
    if not d.exists():
        return out
    for f in d.glob("*.json"):
        try:
            rec = ContributionRecord.model_validate_json(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        # Translate filename back to paper_id form
        stem = f.stem
        # Match v3/v6 convention: scirex__<sha> -> scirex:<sha>; tdmsci__<split>__<n> -> tdmsci:<split>:<n>
        if "__" in stem:
            parts = stem.split("__")
            pid = ":".join(parts)
        else:
            pid = stem
        out[pid] = rec
    return out


def _extract_pred_sets(rec: ContributionRecord) -> dict[str, set[str]]:
    methods, tasks, datasets, metrics = set(), set(), set(), set()
    for c in rec.contributions:
        if (n := _norm(c.method.name)):
            methods.add(n)
        if (n := _norm(c.task.name)):
            tasks.add(n)
        for d in c.datasets:
            if (n := _norm(d.name)):
                datasets.add(n)
        for m in c.metrics:
            if (n := _norm(m.name)):
                metrics.add(n)
    return {"methods": methods, "tasks": tasks, "datasets": datasets, "metrics": metrics}


def _paired_f1(records_a: dict[str, ContributionRecord],
               records_b: Optional[dict[str, ContributionRecord]],
               papers: list[GoldPaper],
               fields: tuple[str, ...] = ("methods", "tasks", "datasets", "metrics")) -> dict:
    """Per-field F1 for system A (and B if provided). If B given, paired test."""
    kind_map = {"methods": "method", "tasks": "task", "datasets": "dataset", "metrics": "metric"}
    out: dict = {}
    for f in fields:
        a_scores, b_scores = [], []
        for p in papers:
            rec_a = records_a.get(p.paper_id)
            if rec_a is None:
                continue
            gold = getattr(p.gold, f)
            if not gold:
                continue
            a_pred = _extract_pred_sets(rec_a)[f]
            a_f1 = set_f1(a_pred, gold, lenient=True, kind=kind_map[f])["f1"]
            if records_b is not None:
                rec_b = records_b.get(p.paper_id)
                if rec_b is None:
                    continue
                b_pred = _extract_pred_sets(rec_b)[f]
                b_f1 = set_f1(b_pred, gold, lenient=True, kind=kind_map[f])["f1"]
                a_scores.append(a_f1)
                b_scores.append(b_f1)
            else:
                a_scores.append(a_f1)
        if not a_scores:
            out[f] = None
            continue
        a_rng = np.random.default_rng(SEED)
        a_mean, a_lo, a_hi = bootstrap_ci(a_scores, n_resamples=N_BOOT, rng=a_rng)
        entry: dict = {"n": len(a_scores), "a_f1": a_mean,
                       "a_ci_lo": a_lo, "a_ci_hi": a_hi}
        if records_b is not None:
            b_rng = np.random.default_rng(SEED)
            b_mean, b_lo, b_hi = bootstrap_ci(b_scores, n_resamples=N_BOOT, rng=b_rng)
            p_rng = np.random.default_rng(SEED)
            p_val = paired_permutation_test(a_scores, b_scores,
                                            n_permutations=N_PERM, rng=p_rng)
            entry.update({
                "b_f1": b_mean, "b_ci_lo": b_lo, "b_ci_hi": b_hi,
                "delta": a_mean - b_mean, "p_value": p_val,
            })
        out[f] = entry
    return out


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


def _calibration_pairs(records: dict[str, ContributionRecord],
                       papers: list[GoldPaper]) -> dict[str, list[tuple[float, int]]]:
    pairs: dict[str, list[tuple[float, int]]] = {
        "method.name": [], "task.name": [], "datasets": [], "metrics": [],
    }
    by_id = {p.paper_id: p for p in papers}
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
            if k >= 0:
                pairs[f].append((conf, k))
    return pairs


def _calibration_block(pairs: dict[str, list[tuple[float, int]]]) -> dict:
    out: dict = {}
    for f, ps in pairs.items():
        if not ps:
            out[f] = None
            continue
        confs = [c for c, _ in ps]
        corrects = [k for _, k in ps]
        cal = calibrate_and_evaluate(confs, corrects, fit_frac=0.2, n_bins=10)
        T = cal["T"]
        scaled = apply_temperature(confs, T).tolist()
        bins_uncal = reliability_diagram_data(confs, corrects, n_bins=10)
        bins_cal = reliability_diagram_data(scaled, corrects, n_bins=10)
        out[f] = {**cal,
                  "reliability_uncalibrated": bins_uncal,
                  "reliability_calibrated": bins_cal}
    return out


# ---------- E2 critic ablation ----------
@app.command()
def critic_ablation():
    base = Path("outputs/paper_data_v7/critic_ablation")
    papers = [p for p in load_scirex(splits=("dev",)) if p.full_text and len(p.full_text) >= 50][:66]
    crit_off = _load_dir(base / "multi_agent")
    crit_on = _load_dir(Path("outputs/paper_data_v3/benchmarks/scirex/multi_agent"))

    # Paired vs critic-on on intersection
    paired_papers = [p for p in papers if p.paper_id in crit_off and p.paper_id in crit_on]
    console.print(f"  critic_off={len(crit_off)}  critic_on(v3)={len(crit_on)}  paired={len(paired_papers)}")

    # F1: critic-off vs critic-on
    f1_paired = _paired_f1(crit_off, crit_on, paired_papers)
    # Standalone F1 for critic-off (all 66 papers)
    f1_off_all = _paired_f1(crit_off, None, papers)

    # Calibration: critic-off (this run) and critic-on (v2 baseline)
    cal_off = _calibration_block(_calibration_pairs(crit_off, papers))
    # Load v2 critic-on calibration as the comparison baseline
    v2_calib_path = Path("outputs/paper_data_v2/calibration/calibration.json")
    cal_on_v2 = json.loads(v2_calib_path.read_text())["per_field"] if v2_calib_path.exists() else {}

    extr_summary_path = base / "extraction_summary.json"
    cost = json.loads(extr_summary_path.read_text())["total_cost_usd"] if extr_summary_path.exists() else 0.0

    result = {
        "experiment": "E2_critic_ablation",
        "seed": SEED,
        "n_resamples_bootstrap": N_BOOT,
        "n_permutations_paired": N_PERM,
        "papers_total": len(papers),
        "papers_critic_off": len(crit_off),
        "papers_paired_vs_v3": len(paired_papers),
        "f1_critic_off_standalone": f1_off_all,
        "f1_critic_off_vs_critic_on_v3_paired": f1_paired,
        "calibration_critic_off": cal_off,
        "calibration_critic_on_v2_baseline_per_field": {
            f: {k: v for k, v in (cal_on_v2.get(f) or {}).items() if k != "reliability_uncalibrated" and k != "reliability_calibrated"}
            for f in ("method.name", "task.name", "datasets", "metrics")
        },
        "extractor_cost_usd": cost,
    }
    out_path = base / "results.json"
    out_path.write_text(json.dumps(result, indent=2))
    console.print(f"[green]Wrote {out_path}[/green]")


# ---------- E1 closed-source comparison ----------
@app.command()
def closed_source():
    base = Path("outputs/paper_data_v7/closed_source_comparison")
    sampled = json.loads((base / "sampled_papers.json").read_text())
    sampled_set = set(sampled)
    papers = [p for p in load_scirex(splits=("dev",)) if p.paper_id in sampled_set]

    deepseek_multi = _load_dir(base / "deepseek_multi_subset")
    deepseek_base = _load_dir(base / "deepseek_baseline_subset")
    gpt4o_multi = _load_dir(base / "gpt4o_multi")
    gpt4o_base = _load_dir(base / "gpt4o_baseline")

    console.print(f"  n_sampled={len(papers)} ds_multi={len(deepseek_multi)} ds_base={len(deepseek_base)} "
                  f"gpt_multi={len(gpt4o_multi)} gpt_base={len(gpt4o_base)}")

    # Paired: multi-agent DeepSeek vs GPT-4o
    f1_multi_paired = _paired_f1(gpt4o_multi, deepseek_multi, papers)
    # Paired: single-LLM DeepSeek vs GPT-4o
    f1_base_paired = _paired_f1(gpt4o_base, deepseek_base, papers)

    # Standalone F1 per system
    f1_ds_multi = _paired_f1(deepseek_multi, None, papers)
    f1_gpt_multi = _paired_f1(gpt4o_multi, None, papers)
    f1_ds_base = _paired_f1(deepseek_base, None, papers)
    f1_gpt_base = _paired_f1(gpt4o_base, None, papers)

    # Per-paper cost
    cost_by_system = {}
    try:
        ext = json.loads((base / "extraction_summary.json").read_text())
        for sys_name in ("gpt4o_multi", "gpt4o_baseline"):
            if sys_name in ext:
                cost_by_system[sys_name] = ext[sys_name]["total_cost_usd"]
    except Exception:
        pass

    # DeepSeek cost (from copied records' _meta)
    def _sum_cost(recs):
        return float(sum(r.meta.cost_usd for r in recs.values() if r.meta))
    cost_by_system["deepseek_multi (from v3 records)"] = _sum_cost(deepseek_multi)
    cost_by_system["deepseek_baseline (from v3 records)"] = _sum_cost(deepseek_base)

    result = {
        "experiment": "E1_closed_source_comparison",
        "seed": SEED,
        "n_resamples_bootstrap": N_BOOT,
        "n_permutations_paired": N_PERM,
        "papers_sampled": len(papers),
        "sampled_paper_ids": sampled,
        "f1_standalone": {
            "deepseek_multi": f1_ds_multi,
            "gpt4o_multi": f1_gpt_multi,
            "deepseek_baseline": f1_ds_base,
            "gpt4o_baseline": f1_gpt_base,
        },
        "paired": {
            "multi_gpt4o_vs_deepseek (a=gpt4o, b=deepseek)": f1_multi_paired,
            "baseline_gpt4o_vs_deepseek (a=gpt4o, b=deepseek)": f1_base_paired,
        },
        "cost_usd_per_system": cost_by_system,
    }
    out_path = base / "results.json"
    out_path.write_text(json.dumps(result, indent=2))
    console.print(f"[green]Wrote {out_path}[/green]")


# ---------- E3 TDMSci ----------
@app.command()
def tdmsci():
    base = Path("outputs/paper_data_v7/tdmsci")
    papers = [p for p in load_tdmsci() if "test" in p.paper_id
              and p.full_text and len(p.full_text) >= 50][:376]
    recs = _load_dir(base / "multi_agent")
    console.print(f"  papers={len(papers)} records={len(recs)}")

    # TDMSci has no Method field; report tasks/datasets/metrics
    fields = ("tasks", "datasets", "metrics")
    f1 = _paired_f1(recs, None, papers, fields=fields)

    # OOD calibration using SciREX-fitted T-scaling
    pairs = _calibration_pairs(recs, papers)
    v2_calib_path = Path("outputs/paper_data_v2/calibration/calibration.json")
    scirex_T = {}
    if v2_calib_path.exists():
        v2 = json.loads(v2_calib_path.read_text())["per_field"]
        for f, d in v2.items():
            if d:
                scirex_T[f] = d.get("T")

    ece_block = {}
    for f, ps in pairs.items():
        if not ps:
            ece_block[f] = None
            continue
        confs = [c for c, _ in ps]
        corrects = [k for _, k in ps]
        ece_raw = expected_calibration_error(confs, corrects, n_bins=10)
        scaled_ece = None
        scaled_bins = None
        if f in scirex_T and scirex_T[f] is not None:
            scaled = apply_temperature(confs, scirex_T[f]).tolist()
            scaled_ece = expected_calibration_error(scaled, corrects, n_bins=10)
            scaled_bins = reliability_diagram_data(scaled, corrects, n_bins=10)
        ece_block[f] = {
            "n": len(ps),
            "ece_uncalibrated": ece_raw,
            "ece_scirex_T_applied": scaled_ece,
            "scirex_T": scirex_T.get(f),
            "reliability_uncalibrated": reliability_diagram_data(confs, corrects, n_bins=10),
            "reliability_scirex_T_applied": scaled_bins,
        }

    # Confusion summary — for each field, distribution of pred/gold sizes
    confusion = {}
    by_id = {p.paper_id: p for p in papers}
    for f in fields:
        sizes = {"pred_empty_gold_nonempty": 0, "both_empty": 0,
                 "both_nonempty_no_match": 0, "any_match": 0, "gold_empty_pred_nonempty": 0}
        for pid, rec in recs.items():
            p = by_id.get(pid)
            if p is None:
                continue
            g = getattr(p.gold, f)
            pred = _extract_pred_sets(rec)[f]
            if not g and not pred:
                sizes["both_empty"] += 1
            elif not g and pred:
                sizes["gold_empty_pred_nonempty"] += 1
            elif g and not pred:
                sizes["pred_empty_gold_nonempty"] += 1
            else:
                matched = False
                for pp in pred:
                    for gg in g:
                        if pp in gg or gg in pp:
                            matched = True
                            break
                    if matched:
                        break
                if matched:
                    sizes["any_match"] += 1
                else:
                    sizes["both_nonempty_no_match"] += 1
        confusion[f] = sizes

    extr_summary_path = base / "extraction_summary.json"
    cost = json.loads(extr_summary_path.read_text())["total_cost_usd"] if extr_summary_path.exists() else 0.0
    result = {
        "experiment": "E3_tdmsci_cross_corpus",
        "seed": SEED,
        "n_resamples_bootstrap": N_BOOT,
        "papers_total": len(papers),
        "papers_with_record": len(recs),
        "f1_per_field": f1,
        "calibration_ood": ece_block,
        "confusion": confusion,
        "extractor_cost_usd": cost,
    }
    out_path = base / "results.json"
    out_path.write_text(json.dumps(result, indent=2))
    console.print(f"[green]Wrote {out_path}[/green]")


# ---------- E4 specialized ablation ----------
@app.command()
def specialized_ablation():
    base = Path("outputs/paper_data_v7/specialized_ablation")
    papers = [p for p in load_scirex(splits=("test",)) if p.full_text and len(p.full_text) >= 50][:66]
    voting_off = _load_dir(base / "voting_off")
    critic_off = _load_dir(base / "critic_off")
    full = _load_dir(Path("outputs/paper_data_v6/benchmarks_ft_70b_test/scirex/multi_agent"))
    console.print(f"  full={len(full)} voting_off={len(voting_off)} critic_off={len(critic_off)}")

    # Paired comparisons against full
    f1_full = _paired_f1(full, None, papers)
    f1_vo_vs_full = _paired_f1(voting_off, full, papers)
    f1_co_vs_full = _paired_f1(critic_off, full, papers)
    f1_vo_standalone = _paired_f1(voting_off, None, papers)
    f1_co_standalone = _paired_f1(critic_off, None, papers)

    extr_summary_path = base / "extraction_summary.json"
    extr_summary = json.loads(extr_summary_path.read_text()) if extr_summary_path.exists() else {}
    cost_breakdown = {k: v.get("total_cost_usd", 0.0) for k, v in extr_summary.items()}

    result = {
        "experiment": "E4_specialized_ablation",
        "seed": SEED,
        "n_resamples_bootstrap": N_BOOT,
        "n_permutations_paired": N_PERM,
        "papers_total": len(papers),
        "papers_full": len(full),
        "papers_voting_off": len(voting_off),
        "papers_critic_off": len(critic_off),
        "f1_standalone": {
            "full_specialized_v6": f1_full,
            "voting_off": f1_vo_standalone,
            "critic_off": f1_co_standalone,
        },
        "paired_vs_full": {
            "voting_off (a=vo, b=full)": f1_vo_vs_full,
            "critic_off (a=co, b=full)": f1_co_vs_full,
        },
        "cost_usd_per_condition": cost_breakdown,
    }
    out_path = base / "results.json"
    out_path.write_text(json.dumps(result, indent=2))
    console.print(f"[green]Wrote {out_path}[/green]")


@app.command()
def all():
    critic_ablation()
    closed_source()
    tdmsci()
    specialized_ablation()


if __name__ == "__main__":
    app()
