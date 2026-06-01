"""v9 — Quantitative error analysis on 50 SciREX dev papers.

For each sampled paper:
  - Load framework records from outputs/paper_data_v3/benchmarks/scirex/multi_agent/
  - Load gold sets
  - Enumerate per-field errors:
      FP (framework emitted but not lenient-matched to any gold)
      FN (gold entry not lenient-matched to any framework prediction)
      WV (only for singleton fields method.name / task.name — pred and gold
          both non-empty but no lenient match)

For each error, query meta-llama/llama-3.3-70b-instruct with a structured
prompt containing the paper excerpt + framework value + gold values + error
type; ask for one of 5 categories. Save raw responses.

Aggregate counts per category × field with bootstrap 95% CIs (1000 resamples).

Outputs:
  outputs/paper_data_v9/error_analysis/
    results.json
    raw_categorizations.jsonl
    SUMMARY.md
    seed.txt
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import typer
from rich.console import Console
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn,
)

from paper1.config import load_config
from paper1.loaders import load_scirex
from paper1.openrouter import OpenRouterClient, parse_json_response
from paper1.schema import ContributionRecord
from paper1.voting import _norm

SEED = 42
N_BOOT = 1000
SAMPLE_SIZE = 50
CATEGORIZER_MODEL = "meta-llama/llama-3.3-70b-instruct"
PAPER_EXCERPT_CHARS = 4000  # truncate paper text to keep prompt small

CATEGORIES = {
    1: "Passing-mention method",
    2: "Shared-baseline binding error",
    3: "Ambiguous task description",
    4: "Schema-coverage gap",
    5: "Other / annotation disagreement",
}

app = typer.Typer(no_args_is_help=False, add_completion=False)
console = Console()


def _safe_id(paper_id: str) -> str:
    return paper_id.replace(":", "__").replace("/", "_")


def _lenient_match(a: str, b: str) -> bool:
    if not a or not b:
        return False
    return a in b or b in a


def _set_with_originals(items) -> list[tuple[str, str]]:
    """Return [(normalized, original), ...] for each non-empty item."""
    out = []
    for x in items:
        n = _norm(x)
        if n:
            out.append((n, x))
    return out


def enumerate_errors(rec: ContributionRecord, gold) -> list[dict[str, Any]]:
    """Enumerate per-field errors as a list of dicts.

    Each dict: {field, error_type, framework_value, gold_values_for_context}
    """
    errors: list[dict[str, Any]] = []

    # Framework predictions, aggregated across all contributions
    pred = {
        "method": [],   # list of (norm, original)
        "task": [],
        "datasets": [],
        "metrics": [],
    }
    for c in rec.contributions:
        if c.method and c.method.name:
            n = _norm(c.method.name)
            if n:
                pred["method"].append((n, c.method.name))
        if c.task and c.task.name:
            n = _norm(c.task.name)
            if n:
                pred["task"].append((n, c.task.name))
        for d in c.datasets:
            if d.name:
                n = _norm(d.name)
                if n:
                    pred["datasets"].append((n, d.name))
        for m in c.metrics:
            if m.name:
                n = _norm(m.name)
                if n:
                    pred["metrics"].append((n, m.name))

    gold_norm = {
        "method": [(_norm(g), g) for g in gold.methods],
        "task": [(_norm(g), g) for g in gold.tasks],
        "datasets": [(_norm(g), g) for g in gold.datasets],
        "metrics": [(_norm(g), g) for g in gold.metrics],
    }

    # Helper for set-level FP/FN enumeration
    def _set_errs(field_label: str):
        p = pred[field_label]
        g = gold_norm[field_label]
        # Track which pred and gold entries got matched (lenient)
        p_matched = [False] * len(p)
        g_matched = [False] * len(g)
        for i, (pn, _) in enumerate(p):
            for j, (gn, _) in enumerate(g):
                if _lenient_match(pn, gn):
                    p_matched[i] = True
                    g_matched[j] = True
        # FPs
        for i, (_, po) in enumerate(p):
            if not p_matched[i]:
                errors.append({
                    "field": field_label,
                    "error_type": "FP",
                    "framework_value": po,
                    "gold_values": [g_orig for _, g_orig in g],
                })
        # FNs
        for j, (_, go) in enumerate(g):
            if not g_matched[j]:
                errors.append({
                    "field": field_label,
                    "error_type": "FN",
                    "framework_value": None,
                    "gold_values": [g_orig for _, g_orig in g],
                    "missed_gold_value": go,
                })

    # Singleton-field special case (method, task): also distinguish "wrong value"
    # WV = both non-empty but no lenient match → emit a single WV instead of FP+FN pair
    def _singleton_errs(field_label: str):
        p = pred[field_label]
        g = gold_norm[field_label]
        # For singleton at the framework level — use the first contribution's value
        # (matches the v3 set-F1 convention where we aggregate across contribs, but
        # the spec asks for distinct error events; we use full set semantics for
        # method/task too since framework may emit several methods across contribs)
        if not p and not g:
            return
        if not p and g:
            for _, go in g:
                errors.append({
                    "field": field_label,
                    "error_type": "FN",
                    "framework_value": None,
                    "gold_values": [g_orig for _, g_orig in g],
                    "missed_gold_value": go,
                })
            return
        if p and not g:
            for _, po in p:
                errors.append({
                    "field": field_label,
                    "error_type": "FP",
                    "framework_value": po,
                    "gold_values": [],
                })
            return
        # Both non-empty: try lenient matching
        p_matched = [False] * len(p)
        g_matched = [False] * len(g)
        for i, (pn, _) in enumerate(p):
            for j, (gn, _) in enumerate(g):
                if _lenient_match(pn, gn):
                    p_matched[i] = True
                    g_matched[j] = True
        # Unmatched pred + unmatched gold → "wrong value" pairing
        unmatched_p = [po for i, (_, po) in enumerate(p) if not p_matched[i]]
        unmatched_g = [go for j, (_, go) in enumerate(g) if not g_matched[j]]
        # Greedy pair unmatched p with unmatched g as WV
        while unmatched_p and unmatched_g:
            po = unmatched_p.pop(0)
            go = unmatched_g.pop(0)
            errors.append({
                "field": field_label,
                "error_type": "WV",
                "framework_value": po,
                "gold_values": [g_orig for _, g_orig in g],
                "missed_gold_value": go,
            })
        # Leftover FPs / FNs
        for po in unmatched_p:
            errors.append({
                "field": field_label, "error_type": "FP",
                "framework_value": po,
                "gold_values": [g_orig for _, g_orig in g],
            })
        for go in unmatched_g:
            errors.append({
                "field": field_label, "error_type": "FN",
                "framework_value": None,
                "gold_values": [g_orig for _, g_orig in g],
                "missed_gold_value": go,
            })

    _singleton_errs("method")
    _singleton_errs("task")
    _set_errs("datasets")
    _set_errs("metrics")
    return errors


CATEGORIZER_PROMPT = """You are a careful research analyst. Your job is to assign each extraction error to exactly ONE of five failure-mode categories.

Categories:
1. Passing-mention method — the paper mentions a method in passing (e.g., in related work or as a baseline reference) rather than as its primary contribution; the framework treated it as a contribution.
2. Shared-baseline binding error — the paper reports several methods sharing the same baseline; the framework bound the wrong (method, dataset, metric) tuple.
3. Ambiguous task description — the paper describes its task in synonymous or non-canonical terms; the framework's normalized form differs from the gold form.
4. Schema-coverage gap — the paper's contribution does not fit the 4-field (method × task × dataset × metric) schema cleanly; the framework emits something close but not equal to gold.
5. Other / annotation disagreement — the framework's output is defensible against the source text but does not match gold (could be a gold-annotation oversight).

You will receive:
- A paper text excerpt (first ~4k chars).
- The field being analyzed (method / task / datasets / metrics).
- The error type:
  * FP = framework emitted a value not in gold
  * FN = gold has a value the framework did not emit
  * WV = framework's value and gold's value are both non-empty and disagree
- The framework's emitted value (if any).
- The gold values for that field.

Output ONLY a JSON object with this exact shape, nothing else:
{{"category": <integer 1-5>, "reason": "<one short sentence>"}}

Paper excerpt:
---
{paper_excerpt}
---

Field: {field}
Error type: {error_type}
Framework value: {framework_value}
Gold values: {gold_values}

Assign exactly one category."""


@dataclass
class CategorizationResult:
    paper_id: str
    field: str
    error_type: str
    framework_value: str | None
    gold_values: list[str]
    missed_gold_value: str | None
    category: int | None
    reason: str
    raw_response: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    error: str | None = None


async def _categorize_one(client: OpenRouterClient, paper_text: str,
                          error: dict[str, Any], paper_id: str,
                          sem: asyncio.Semaphore, progress: Progress,
                          task_id) -> CategorizationResult:
    excerpt = paper_text[:PAPER_EXCERPT_CHARS]
    prompt = CATEGORIZER_PROMPT.format(
        paper_excerpt=excerpt,
        field=error["field"],
        error_type=error["error_type"],
        framework_value=str(error.get("framework_value")),
        gold_values=", ".join(error.get("gold_values", [])) or "(none)",
    )
    async with sem:
        try:
            result = await client.complete(
                model_id=CATEGORIZER_MODEL,
                system="You are a careful research analyst. Output ONLY valid JSON.",
                user=prompt,
                temperature=0.0,
                max_tokens=200,
                top_p=0.95,
            )
            cost = (result.tokens_in / 1e6) * 0.59 + (result.tokens_out / 1e6) * 0.79
            cat = None
            reason = ""
            try:
                data = parse_json_response(result.text)
                cat = int(data.get("category"))
                reason = str(data.get("reason", ""))
                if cat not in (1, 2, 3, 4, 5):
                    cat = 5  # default to "Other"
                    reason = f"out-of-range cat={cat}; defaulted to 5; original: {reason}"
            except Exception as e:
                cat = 5
                reason = f"parse-failed; defaulted to 5; raw: {result.text[:120]}"
            return CategorizationResult(
                paper_id=paper_id,
                field=error["field"],
                error_type=error["error_type"],
                framework_value=error.get("framework_value"),
                gold_values=error.get("gold_values", []),
                missed_gold_value=error.get("missed_gold_value"),
                category=cat, reason=reason, raw_response=result.text,
                tokens_in=result.tokens_in, tokens_out=result.tokens_out, cost_usd=cost,
            )
        except Exception as e:  # noqa: BLE001
            return CategorizationResult(
                paper_id=paper_id, field=error["field"], error_type=error["error_type"],
                framework_value=error.get("framework_value"),
                gold_values=error.get("gold_values", []),
                missed_gold_value=error.get("missed_gold_value"),
                category=None, reason="", raw_response="",
                tokens_in=0, tokens_out=0, cost_usd=0.0,
                error=f"{type(e).__name__}: {str(e)[:200]}",
            )
        finally:
            progress.advance(task_id)


def _bootstrap_pct_ci(counts: list[int], total: int, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float, float]:
    """Bootstrap CI on a percentage = sum(counts==target)/total. Pass list of 0/1 indicators per error."""
    if total == 0:
        return 0.0, 0.0, 0.0
    arr = np.asarray(counts, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    n = arr.size
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = arr[idx].mean()
    return float(arr.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


@app.command()
def run(
    output_dir: Path = typer.Option(Path("outputs/paper_data_v9/error_analysis"), "--output-dir"),
    records_dir: Path = typer.Option(Path("outputs/paper_data_v3/benchmarks/scirex/multi_agent"), "--records-dir"),
    config_path: Path = typer.Option(Path("config/models.yaml"), "--config"),
    concurrency: int = typer.Option(10, "--concurrency"),
):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "seed.txt").write_text(f"{SEED}\n")

    cfg = load_config(config_path=config_path)
    or_client = OpenRouterClient(
        api_key=cfg.api_key, base_url=cfg.base_url,
        timeout_s=cfg.defaults.request_timeout_s,
        max_retries=cfg.defaults.max_retries,
        retry_backoff_s=cfg.defaults.retry_backoff_s,
        referer=cfg.referer, title=cfg.title,
    )

    # Sample 50 dev papers
    all_papers = [p for p in load_scirex(splits=("dev",)) if p.full_text and len(p.full_text) >= 50][:66]
    by_id = {p.paper_id: p for p in all_papers}
    rng = random.Random(SEED)
    sampled = rng.sample(all_papers, k=min(SAMPLE_SIZE, len(all_papers)))
    sampled_ids = sorted(p.paper_id for p in sampled)
    (output_dir / "sampled_papers.json").write_text(json.dumps(sampled_ids, indent=2))
    console.print(f"  sampled {len(sampled)} papers (seed={SEED})")

    # Load framework records for sampled papers
    records: dict[str, ContributionRecord] = {}
    for p in sampled:
        rec_path = records_dir / f"{_safe_id(p.paper_id)}.json"
        if not rec_path.exists():
            continue
        try:
            records[p.paper_id] = ContributionRecord.model_validate_json(rec_path.read_text(encoding="utf-8"))
        except Exception:
            continue
    console.print(f"  loaded {len(records)} framework records of {len(sampled)} sampled")

    # Enumerate errors
    all_errors: list[tuple[str, dict[str, Any]]] = []
    paper_text_lookup: dict[str, str] = {}
    for pid, rec in records.items():
        gold = by_id[pid].gold
        errs = enumerate_errors(rec, gold)
        for e in errs:
            all_errors.append((pid, e))
        paper_text_lookup[pid] = by_id[pid].full_text
    console.print(f"  enumerated {len(all_errors)} per-field errors across {len(records)} papers")

    # Show error-type breakdown
    from collections import Counter
    et_count = Counter((e["field"], e["error_type"]) for _, e in all_errors)
    for (f, t), n in sorted(et_count.items()):
        console.print(f"    {f}/{t}: {n}")

    # Categorize each error
    asyncio.run(_categorize_all(or_client, all_errors, paper_text_lookup,
                                output_dir, concurrency))


async def _categorize_all(or_client, all_errors, paper_text_lookup,
                          output_dir: Path, concurrency: int):
    sem = asyncio.Semaphore(concurrency)
    t0 = time.perf_counter()
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                  MofNCompleteColumn(), TimeElapsedColumn(), console=console) as progress:
        task_id = progress.add_task("categorize", total=len(all_errors))
        results: list[CategorizationResult] = await asyncio.gather(*(
            _categorize_one(or_client, paper_text_lookup[pid], err, pid, sem, progress, task_id)
            for pid, err in all_errors
        ))
    elapsed = time.perf_counter() - t0
    total_cost = sum(r.cost_usd for r in results)
    n_err = sum(1 for r in results if r.error)
    console.print(f"  categorized {len(results)} errors  llm_failed={n_err}  cost=${total_cost:.4f}  wall={elapsed:.0f}s")

    # Save raw categorizations
    raw_path = output_dir / "raw_categorizations.jsonl"
    with raw_path.open("w") as f:
        for r in results:
            f.write(json.dumps({
                "paper_id": r.paper_id,
                "field": r.field,
                "error_type": r.error_type,
                "framework_value": r.framework_value,
                "missed_gold_value": r.missed_gold_value,
                "gold_values": r.gold_values,
                "category": r.category,
                "reason": r.reason,
                "tokens_in": r.tokens_in,
                "tokens_out": r.tokens_out,
                "cost_usd": r.cost_usd,
                "error": r.error,
            }) + "\n")
    console.print(f"  wrote {raw_path}")

    # Aggregate
    from collections import Counter
    overall = Counter()
    by_field: dict[str, Counter] = {}
    by_field_total: dict[str, int] = {}
    cat_to_field_indicators: dict[int, dict[str, list[int]]] = {c: {} for c in CATEGORIES}
    overall_indicators: dict[int, list[int]] = {c: [] for c in CATEGORIES}
    for r in results:
        if r.category is None:
            continue
        overall[r.category] += 1
        by_field.setdefault(r.field, Counter())[r.category] += 1
        by_field_total[r.field] = by_field_total.get(r.field, 0) + 1
        for c in CATEGORIES:
            overall_indicators[c].append(1 if r.category == c else 0)
            cat_to_field_indicators[c].setdefault(r.field, []).append(1 if r.category == c else 0)

    total = sum(overall.values())
    # Per-category overall % + bootstrap CI
    overall_pct: dict[int, dict] = {}
    for c in CATEGORIES:
        ind = overall_indicators[c]
        mean, lo, hi = _bootstrap_pct_ci(ind, total)
        overall_pct[c] = {"category_id": c, "category_name": CATEGORIES[c],
                          "count": overall[c], "pct": mean,
                          "ci_lo": lo, "ci_hi": hi}

    # Per-field × per-category
    by_field_pct: dict[str, dict[int, dict]] = {}
    for f, counter in by_field.items():
        ftotal = by_field_total[f]
        per_cat = {}
        for c in CATEGORIES:
            ind = cat_to_field_indicators[c].get(f, [])
            mean, lo, hi = _bootstrap_pct_ci(ind, ftotal)
            per_cat[c] = {"category_id": c, "category_name": CATEGORIES[c],
                          "count": counter.get(c, 0),
                          "pct": mean, "ci_lo": lo, "ci_hi": hi}
        by_field_pct[f] = {"total_errors": ftotal, "per_category": per_cat}

    result_doc = {
        "experiment": "v9_error_analysis",
        "seed": SEED,
        "n_resamples_bootstrap": N_BOOT,
        "categorizer_model": CATEGORIZER_MODEL,
        "sample_size_requested": SAMPLE_SIZE,
        "papers_with_records": len(set(r.paper_id for r in results)),
        "total_errors_enumerated": len(results),
        "total_errors_categorized": total,
        "categorizer_total_cost_usd": total_cost,
        "categorizer_wall_time_seconds": elapsed,
        "overall_per_category": overall_pct,
        "per_field": by_field_pct,
    }
    (output_dir / "results.json").write_text(json.dumps(result_doc, indent=2))
    console.print(f"  wrote {output_dir / 'results.json'}")

    # SUMMARY.md
    md_lines = [
        "# v9 — Error analysis (50 SciREX dev papers)\n",
        f"_Sample: {result_doc['papers_with_records']} papers (seed={SEED}); "
        f"{result_doc['total_errors_categorized']} per-field errors categorized via {CATEGORIZER_MODEL}._\n",
        f"_Categorizer cost: ${total_cost:.4f}, wall {elapsed:.0f}s._\n",
        "## Overall error-category distribution\n",
        "| # | Category | Count | % of errors [95% CI] |",
        "|---|---|---:|---:|",
    ]
    sorted_cats = sorted(overall_pct.values(), key=lambda x: -x["pct"])
    for r in sorted_cats:
        ci = f"[{r['ci_lo']*100:.1f}, {r['ci_hi']*100:.1f}]"
        md_lines.append(f"| {r['category_id']} | {r['category_name']} | {r['count']} | **{r['pct']*100:.1f}%** {ci} |")
    md_lines.append("")
    md_lines.append("## Category × field breakdown (%)\n")
    md_lines.append("| Category | method | task | datasets | metrics |")
    md_lines.append("|---|---:|---:|---:|---:|")
    for cid, cname in CATEGORIES.items():
        cells = []
        for f in ("method", "task", "datasets", "metrics"):
            blk = by_field_pct.get(f, {}).get("per_category", {}).get(cid)
            if blk and blk["count"]:
                cells.append(f"{blk['pct']*100:.1f}% ({blk['count']})")
            else:
                cells.append("—")
        md_lines.append(f"| {cid}. {cname} | {' | '.join(cells)} |")
    md_lines.append("")
    md_lines.append("## Field totals\n")
    md_lines.append("| Field | Total errors categorized |")
    md_lines.append("|---|---:|")
    for f in ("method", "task", "datasets", "metrics"):
        md_lines.append(f"| {f} | {by_field_pct.get(f, {}).get('total_errors', 0)} |")
    md_lines.append("")
    (output_dir / "SUMMARY.md").write_text("\n".join(md_lines), encoding="utf-8")
    console.print(f"  wrote {output_dir / 'SUMMARY.md'}")


if __name__ == "__main__":
    app()
