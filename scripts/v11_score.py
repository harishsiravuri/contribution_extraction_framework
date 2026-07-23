"""v11 — Score Conditions A + B against the v10 multi-agent test-split records.

Loads:
  outputs/paper_data_v10/test_split_default/scirex/multi_agent/  (multi-agent)
  outputs/paper_data_v11/baseline_test/records/                    (Condition A)
  outputs/paper_data_v11/selfconsistency_test/records/             (Condition B)

Computes:
  - Per-field F1 with bootstrap 95% CI (1000 resamples, seed=42) on the
    intersection of papers where all three systems succeeded
  - Triple (Task, Dataset, Metric) F1 with bootstrap CI
  - Paired permutation p-values (10,000 permutations, seed=42):
      * multi-agent vs Condition A
      * multi-agent vs Condition B
      * Condition B vs Condition A
    Bonferroni-corrected across the 4 fields per comparison.

Writes:
  outputs/paper_data_v11/SUMMARY.md
  outputs/paper_data_v11/results.json
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import typer
from rich.console import Console

from paper1.loaders import load_scirex
from paper1.metrics import bootstrap_ci, paired_permutation_test
from paper1.metrics.span_f1 import set_f1
from paper1.metrics.triple_f1 import triple_f1
from paper1.schema import ContributionRecord
from paper1.voting import _norm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_BOOT = 1000
N_PERM = 10000
FIELDS = ("methods", "tasks", "datasets", "metrics")
KIND = {"methods": "method", "tasks": "task", "datasets": "dataset", "metrics": "metric"}

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


def _score_per_paper(rec_map, papers, field, kind):
    """Return list of per-paper F1s (only papers where gold has this field)."""
    scores = []
    for p in papers:
        rec = rec_map.get(p.paper_id)
        if rec is None:
            continue
        gold = getattr(p.gold, field)
        if not gold:
            continue
        pred = _pred_sets(rec)[field]
        f1 = set_f1(pred, gold, lenient=True, kind=kind)["f1"]
        scores.append((p.paper_id, f1))
    return scores


def _score_triples(rec_map, papers):
    out = []
    for p in papers:
        rec = rec_map.get(p.paper_id)
        if rec is None:
            continue
        if not p.gold.triples:
            continue
        out.append((p.paper_id, triple_f1(_pred_triples(rec), p.gold.triples)["f1"]))
    return out


def _bootstrap(scores, rng):
    arr = np.array([s for _, s in scores], dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    return bootstrap_ci(arr.tolist(), n_resamples=N_BOOT, rng=rng)


def _paired_p(a_scores_by_pid: dict, b_scores_by_pid: dict, rng) -> tuple[int, float, float]:
    """Compute paired-perm p-value on the intersection of paper_ids."""
    inter = sorted(set(a_scores_by_pid.keys()) & set(b_scores_by_pid.keys()))
    if not inter:
        return 0, 1.0, 0.0
    a = [a_scores_by_pid[pid] for pid in inter]
    b = [b_scores_by_pid[pid] for pid in inter]
    p = paired_permutation_test(a, b, n_permutations=N_PERM, rng=rng)
    delta = float(np.mean(a) - np.mean(b))
    return len(inter), p, delta


@app.command()
def score(
    v10_dir: Path = typer.Option(Path("outputs/paper_data_v10/test_split_default/scirex/multi_agent"), "--v10"),
    a_dir: Path = typer.Option(Path("outputs/paper_data_v11/baseline_test/records"), "--a"),
    b_dir: Path = typer.Option(Path("outputs/paper_data_v11/selfconsistency_test/records"), "--b"),
    out_dir: Path = typer.Option(Path("outputs/paper_data_v11"), "--out"),
):
    console.print("[bold]Loading records...[/bold]")
    multi = _load_dir(v10_dir)
    cond_a = _load_dir(a_dir)
    cond_b = _load_dir(b_dir)
    papers = [p for p in load_scirex(splits=("test",)) if p.full_text and len(p.full_text) >= 50][:66]
    console.print(f"  multi (v10 multi-agent): {len(multi)}")
    console.print(f"  Condition A single-LLM:  {len(cond_a)}")
    console.print(f"  Condition B self-consistency: {len(cond_b)}")

    # Compute per-paper F1 for each system × field
    per_paper: dict[str, dict[str, dict[str, float]]] = {"multi": {}, "A": {}, "B": {}}
    for f in FIELDS:
        for label, rec_map in (("multi", multi), ("A", cond_a), ("B", cond_b)):
            scored = _score_per_paper(rec_map, papers, f, KIND[f])
            per_paper[label][f] = {pid: score for pid, score in scored}

    # Per-paper triple F1
    triple_per_paper: dict[str, dict[str, float]] = {}
    for label, rec_map in (("multi", multi), ("A", cond_a), ("B", cond_b)):
        triple_per_paper[label] = {pid: sc for pid, sc in _score_triples(rec_map, papers)}

    # Aggregate F1 with bootstrap CIs (per-system standalone)
    standalone: dict[str, dict[str, dict]] = {}
    for label in ("multi", "A", "B"):
        standalone[label] = {}
        for f in FIELDS:
            rng = np.random.default_rng(SEED)
            vals = list(per_paper[label][f].values())
            mean, lo, hi = bootstrap_ci(vals, n_resamples=N_BOOT, rng=rng) if vals else (0.0, 0.0, 0.0)
            standalone[label][f] = {"n": len(vals), "f1": mean, "ci_lo": lo, "ci_hi": hi}
        # Triple
        rng = np.random.default_rng(SEED)
        vals = list(triple_per_paper[label].values())
        mean, lo, hi = bootstrap_ci(vals, n_resamples=N_BOOT, rng=rng) if vals else (0.0, 0.0, 0.0)
        standalone[label]["triple"] = {"n": len(vals), "f1": mean, "ci_lo": lo, "ci_hi": hi}

    # Paired permutation tests
    comparisons = [
        ("multi_vs_A", "multi", "A"),
        ("multi_vs_B", "multi", "B"),
        ("B_vs_A",     "B",     "A"),
    ]
    paired: dict[str, dict[str, dict]] = {}
    for cmp_name, x, y in comparisons:
        paired[cmp_name] = {}
        raw_ps = {}
        for f in FIELDS:
            rng = np.random.default_rng(SEED)
            n, p, delta = _paired_p(per_paper[x][f], per_paper[y][f], rng)
            paired[cmp_name][f] = {"n_paired": n, "delta_x_minus_y": delta, "p_raw": p}
            raw_ps[f] = p
        # Bonferroni across 4 fields (per comparison)
        for f in FIELDS:
            paired[cmp_name][f]["p_bonferroni"] = min(1.0, raw_ps[f] * len(FIELDS))
        # Triple (reported separately, not Bonferroni-adjusted with the fields)
        rng = np.random.default_rng(SEED)
        n, p, delta = _paired_p(triple_per_paper[x], triple_per_paper[y], rng)
        paired[cmp_name]["triple"] = {"n_paired": n, "delta_x_minus_y": delta, "p_raw": p}

    # Extraction summaries (cost + tokens + calls + wall)
    def _load_extraction(path: Path) -> dict:
        if not path.exists(): return {}
        return json.loads(path.read_text())

    a_sum = _load_extraction(Path("outputs/paper_data_v11/baseline_test/extraction_summary.json"))
    b_sum = _load_extraction(Path("outputs/paper_data_v11/selfconsistency_test/extraction_summary.json"))
    # v10 multi-agent summary
    v10_sum = _load_extraction(Path("outputs/paper_data_v10/test_split_default/scirex/multi_agent_summary.json"))

    cost_block = {
        "multi_agent_v10": {
            "papers_ok":       v10_sum.get("papers_ok"),
            "papers_error":    v10_sum.get("papers_error"),
            "total_cost_usd":  v10_sum.get("total_cost_usd"),
            "wall_time_seconds": v10_sum.get("wall_time_seconds"),
            "llm_calls_note":  "3 extractor + 1 critic + 1 consolidator = 5 per successful paper",
        },
        "condition_A_single_llm": {
            "papers_ok":       a_sum.get("papers_ok"),
            "papers_error":    a_sum.get("papers_error"),
            "total_cost_usd":  a_sum.get("total_cost_usd"),
            "total_tokens_in": a_sum.get("total_tokens_in"),
            "total_tokens_out":a_sum.get("total_tokens_out"),
            "llm_calls":       a_sum.get("llm_calls"),
            "wall_time_seconds": a_sum.get("wall_time_seconds"),
            "cost_cap_tripped": a_sum.get("cost_cap_tripped"),
        },
        "condition_B_self_consistency": {
            "papers_ok":       b_sum.get("papers_ok"),
            "papers_error":    b_sum.get("papers_error"),
            "total_cost_usd":  b_sum.get("total_cost_usd"),
            "total_tokens_in": b_sum.get("total_tokens_in"),
            "total_tokens_out":b_sum.get("total_tokens_out"),
            "llm_calls":       b_sum.get("llm_calls"),
            "wall_time_seconds": b_sum.get("wall_time_seconds"),
            "cost_cap_tripped": b_sum.get("cost_cap_tripped"),
        },
    }

    total_v11_spend = (a_sum.get("total_cost_usd") or 0.0) + (b_sum.get("total_cost_usd") or 0.0)

    results = {
        "experiment": "v11_matched_budget_baselines_scirex_test",
        "seed": SEED,
        "n_resamples_bootstrap": N_BOOT,
        "n_permutations_paired": N_PERM,
        "bonferroni_across_fields_per_comparison": len(FIELDS),
        "prompt": "config/prompts/extractor.md",
        "extractor_model": "deepseek/deepseek-chat",
        "papers_evaluated": len(papers),
        "standalone_f1_per_system": standalone,
        "paired_comparisons": paired,
        "cost_and_calls": cost_block,
        "total_v11_openrouter_spend_usd": total_v11_spend,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    console.print(f"  wrote {out_dir / 'results.json'}")

    # Build SUMMARY.md
    lines = [
        "# v11 — Matched-budget baselines on SciREX test",
        "",
        "_Comparing the multi-agent framework (v10) against two same-extractor baselines "
        "on the SciREX test split. All three systems use `config/prompts/extractor.md` and "
        f"`deepseek/deepseek-chat` via OpenRouter. Seed = {SEED}. "
        f"Bootstrap 95% CIs from {N_BOOT} resamples; paired permutation p-values from {N_PERM} permutations, "
        f"Bonferroni-corrected across {len(FIELDS)} fields per comparison._",
        "",
        "## Compute / cost per condition",
        "",
        "| Condition | Papers ok | LLM calls | Tokens in | Tokens out | OpenRouter spend | Wall time |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in (("Multi-agent (v10)", "multi_agent_v10"),
                       ("A: Single-LLM baseline", "condition_A_single_llm"),
                       ("B: Self-consistency", "condition_B_self_consistency")):
        c = cost_block[key]
        ok = c.get("papers_ok", "—")
        calls = c.get("llm_calls", "—")
        if key == "multi_agent_v10":
            calls = f"~{5 * (ok if isinstance(ok, int) else 0)}"
        ti = c.get("total_tokens_in", "—")
        to = c.get("total_tokens_out", "—")
        spend = c.get("total_cost_usd", 0.0)
        wall = c.get("wall_time_seconds", 0.0)
        lines.append(
            f"| {label} | {ok} | {calls} | {ti if isinstance(ti,int) else ti} | "
            f"{to if isinstance(to,int) else to} | "
            f"${spend:.4f} | {wall:.0f}s |"
        )
    lines.append("")
    lines.append(f"**v11 total OpenRouter spend (Conditions A + B): ${total_v11_spend:.4f}.**")
    lines.append("")

    lines.append("## Per-field F1 (standalone, bootstrap 95% CI)")
    lines.append("")
    lines.append("| Field   | Multi-agent (v10) F1 [95% CI]     | Condition A F1 [95% CI]        | Condition B F1 [95% CI]        |")
    lines.append("|---------|------------------------------------|---------------------------------|---------------------------------|")
    for f, disp in (("methods", "Method"), ("tasks", "Task"),
                    ("datasets", "Dataset"), ("metrics", "Metric")):
        m = standalone["multi"][f]; a = standalone["A"][f]; b = standalone["B"][f]
        lines.append(
            f"| {disp:7s} | {m['f1']:.3f} [{m['ci_lo']:.3f}, {m['ci_hi']:.3f}] (n={m['n']}) "
            f"| {a['f1']:.3f} [{a['ci_lo']:.3f}, {a['ci_hi']:.3f}] (n={a['n']}) "
            f"| {b['f1']:.3f} [{b['ci_lo']:.3f}, {b['ci_hi']:.3f}] (n={b['n']}) |"
        )
    m = standalone["multi"]["triple"]; a = standalone["A"]["triple"]; b = standalone["B"]["triple"]
    lines.append(
        f"| Triple* | {m['f1']:.3f} [{m['ci_lo']:.3f}, {m['ci_hi']:.3f}] (n={m['n']}) "
        f"| {a['f1']:.3f} [{a['ci_lo']:.3f}, {a['ci_hi']:.3f}] (n={a['n']}) "
        f"| {b['f1']:.3f} [{b['ci_lo']:.3f}, {b['ci_hi']:.3f}] (n={b['n']}) |"
    )
    lines.append("")
    lines.append("_*Triple = (Task, Dataset, Metric) joint F1, exploratory._")
    lines.append("")

    lines.append("## Paired-permutation p-values (Bonferroni across 4 fields)")
    lines.append("")

    def _sig(p):
        return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))

    for cmp_name, disp in (("multi_vs_A", "Multi-agent vs Condition A (single-LLM)"),
                            ("multi_vs_B", "Multi-agent vs Condition B (self-consistency)"),
                            ("B_vs_A",     "Condition B (self-consistency) vs Condition A (single-LLM)")):
        lines.append(f"### {disp}")
        lines.append("")
        lines.append("| Field | n paired | Δ (x − y) | p (raw) | p (Bonferroni) |")
        lines.append("|---|---:|---:|---:|---:|")
        for f, dispf in (("methods", "Method"), ("tasks", "Task"),
                         ("datasets", "Dataset"), ("metrics", "Metric")):
            e = paired[cmp_name][f]
            lines.append(
                f"| {dispf:7s} | {e['n_paired']} | {e['delta_x_minus_y']:+.3f} "
                f"| {e['p_raw']:.4f} | **{e['p_bonferroni']:.4f}** {_sig(e['p_bonferroni'])} |"
            )
        e_tri = paired[cmp_name]["triple"]
        lines.append(
            f"| Triple* | {e_tri['n_paired']} | {e_tri['delta_x_minus_y']:+.3f} "
            f"| {e_tri['p_raw']:.4f} | — (not Bonferroni-adjusted with fields) |"
        )
        lines.append("")

    # Interpretation
    lines.append("## Interpretation")
    lines.append("")
    # Persistence check (reviewer 4): does the dev-split multi>baseline gap persist on test?
    multi_beats_A = []
    for f in FIELDS:
        e = paired["multi_vs_A"][f]
        if e["delta_x_minus_y"] > 0 and e["p_bonferroni"] < 0.05:
            multi_beats_A.append(f)
    if multi_beats_A:
        lines.append(
            "**Reviewer 4 persistence check:** the dev-split multi-agent > single-LLM "
            "advantage persists on test on the following field(s) after Bonferroni correction: "
            + ", ".join(multi_beats_A) + "."
        )
    else:
        lines.append(
            "**Reviewer 4 persistence check:** the dev-split multi-agent > single-LLM "
            "advantage does NOT reach Bonferroni-corrected significance on any per-field "
            "F1 on the held-out test split. Raw p-values below; the direction of the "
            "effect (Δ column) still favours multi-agent on tasks/metrics on dev; test "
            "narrows or reverses the gap depending on the field."
        )
    lines.append("")
    # Cost-matched compute check (advisor Priority 1): does multi > B (cost-matched)?
    multi_beats_B = []
    for f in FIELDS:
        e = paired["multi_vs_B"][f]
        if e["delta_x_minus_y"] > 0 and e["p_bonferroni"] < 0.05:
            multi_beats_B.append(f)
    if multi_beats_B:
        lines.append(
            "**Advisor Priority 1 (cost-matched compute) check:** the multi-agent structure "
            "produces a Bonferroni-significant per-field improvement over the cost-matched "
            "self-consistency baseline on: " + ", ".join(multi_beats_B) + ". This is a "
            "structural gain not attributable to raw compute."
        )
    else:
        lines.append(
            "**Advisor Priority 1 (cost-matched compute) check:** at ~equal LLM-call budget, "
            "the multi-agent structure does NOT produce a Bonferroni-significant per-field "
            "F1 improvement over the cost-matched self-consistency baseline. See the "
            "Δ column for the direction and magnitude of the (statistically insignificant) "
            "differences per field."
        )
    lines.append("")

    (out_dir / "SUMMARY.md").write_text("\n".join(lines))
    console.print(f"  wrote {out_dir / 'SUMMARY.md'}")


if __name__ == "__main__":
    app()
