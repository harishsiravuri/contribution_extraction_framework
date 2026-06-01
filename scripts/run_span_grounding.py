"""Phase Y2 — span-grounding accuracy on SciREX dev set, three conditions:
  1. Full multi-agent (Extractor → Critic → Consolidator)
  2. No-critic ablation (Extractor → Consolidator)
  3. Single-LLM baseline

For each condition, runs the pipeline (resume support) and writes
ContributionRecords to <output-dir>/<condition>/by_paper/. Then compares
against SciREX gold spans, reporting per-condition span-grounding
precision / recall / F1 with bootstrap 95% CIs and pairwise
permutation p-values.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from paper1.config import load_config
from paper1.loaders import GoldPaper, load_scirex
from paper1.metrics import bootstrap_ci, paired_permutation_test
from paper1.metrics.span_grounding import per_paper_grounding, per_paper_grounding_resolved
from paper1.openrouter import OpenRouterClient
from paper1.pipeline import Pipeline
from paper1.pipelines import BaselinePipeline, NoCriticPipeline
from paper1.schema import ContributionRecord

app = typer.Typer(no_args_is_help=False, add_completion=False)
console = Console()


@dataclass
class PaperResult:
    paper_id: str
    status: str
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    wall_time_seconds: float = 0.0
    error: str | None = None
    output_file: str | None = None


def _safe_id(paper_id: str) -> str:
    return paper_id.replace(":", "__").replace("/", "_")


def _load(path: Path) -> ContributionRecord | None:
    try:
        return ContributionRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        return None


async def _process(pipe, paper: GoldPaper, by_paper_dir: Path, sem: asyncio.Semaphore, progress, task_id) -> PaperResult:
    safe = _safe_id(paper.paper_id)
    out = by_paper_dir / f"{safe}.json"
    if out.exists():
        progress.advance(task_id)
        return PaperResult(paper_id=paper.paper_id, status="skipped", output_file=str(out))
    if not paper.full_text or len(paper.full_text) < 50:
        progress.advance(task_id)
        return PaperResult(paper_id=paper.paper_id, status="skipped", error="empty text")
    async with sem:
        try:
            rec = await pipe.extract(paper_id=paper.paper_id, paper_text=paper.full_text)
            out.write_text(rec.model_dump_json(indent=2, by_alias=True), encoding="utf-8")
            return PaperResult(
                paper_id=paper.paper_id, status="ok",
                cost_usd=rec.meta.cost_usd, tokens_in=rec.meta.tokens_in,
                tokens_out=rec.meta.tokens_out, wall_time_seconds=rec.meta.wall_time_seconds,
                output_file=str(out),
            )
        except Exception as e:  # noqa
            return PaperResult(paper_id=paper.paper_id, status="error", error=f"{type(e).__name__}: {str(e)[:200]}")
        finally:
            progress.advance(task_id)


@app.command()
def run(
    output_dir: Path = typer.Option(Path("outputs/paper_data_v3/span_grounding"), "--output-dir"),
    n_papers: int = typer.Option(66, "--n-papers"),
    concurrency: int = typer.Option(30, "--concurrency"),
    config_path: Path | None = typer.Option(None, "--config"),
    skip_extraction: bool = typer.Option(False, "--skip-extraction"),
    only: str = typer.Option("full,no_critic,baseline", "--only"),
) -> None:
    cfg = load_config(config_path=config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    papers = [p for p in load_scirex(splits=("dev",)) if p.full_text and len(p.full_text) >= 50][:n_papers]
    console.print(f"[bold]SciREX dev: {len(papers)} papers[/bold]")
    asyncio.run(_main(cfg, papers, output_dir, concurrency, skip_extraction, only.split(",")))


async def _main(cfg, papers, output_dir, concurrency, skip_extraction, conditions):
    client = OpenRouterClient(
        api_key=cfg.api_key, base_url=cfg.base_url,
        timeout_s=cfg.defaults.request_timeout_s, max_retries=cfg.defaults.max_retries,
        retry_backoff_s=cfg.defaults.retry_backoff_s,
    )
    pipes = {
        "full": Pipeline(client, cfg),
        "no_critic": NoCriticPipeline(client, cfg),
        "baseline": BaselinePipeline(client, cfg),
    }
    sem = asyncio.Semaphore(concurrency)

    summaries: dict[str, dict] = {}

    if not skip_extraction:
        for cond in conditions:
            cond = cond.strip()
            if cond not in pipes:
                continue
            by_paper_dir = output_dir / cond / "by_paper"
            by_paper_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            console.print(f"\n[bold cyan]{cond}[/bold cyan]")
            with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), console=console) as progress:
                task_id = progress.add_task(cond, total=len(papers))
                results = await asyncio.gather(*(_process(pipes[cond], p, by_paper_dir, sem, progress, task_id) for p in papers))
            elapsed = time.perf_counter() - t0
            ok = sum(1 for r in results if r.status == "ok")
            err = sum(1 for r in results if r.status == "error")
            skip = sum(1 for r in results if r.status == "skipped")
            cost = sum(r.cost_usd for r in results if r.status == "ok")
            summary = {
                "condition": cond, "wall_time_seconds": elapsed,
                "papers_total": len(results), "papers_ok": ok, "papers_error": err, "papers_skipped": skip,
                "total_cost_usd": cost,
                "per_paper": [asdict(r) for r in results],
            }
            (output_dir / cond / "summary.json").write_text(json.dumps(summary, indent=2))
            summaries[cond] = summary
            console.print(f"  ok={ok} err={err} skip={skip} cost=${cost:.4f}")

    await client.aclose()

    # Span-grounding evaluation
    console.print("\n[bold]Computing span-grounding metrics...[/bold]")
    eval_data: dict[str, dict] = {}
    paper_index = {p.paper_id: p for p in papers}
    for cond in conditions:
        cond = cond.strip()
        by_paper_dir = output_dir / cond / "by_paper"
        if not by_paper_dir.exists():
            continue
        per_paper_rows: list[dict] = []
        per_paper_resolved: list[dict] = []
        for p in papers:
            rec_path = by_paper_dir / f"{_safe_id(p.paper_id)}.json"
            rec = _load(rec_path)
            if rec is None:
                continue
            raw = per_paper_grounding(rec, p)
            raw["paper_id"] = p.paper_id
            per_paper_rows.append(raw)
            res = per_paper_grounding_resolved(rec, p)
            res["paper_id"] = p.paper_id
            per_paper_resolved.append(res)
        if not per_paper_rows:
            continue

        def _agg(rows, key):
            vals = [r[key] for r in rows]
            mean, lo, hi = bootstrap_ci(vals, n_resamples=1000)
            return {"mean": mean, "ci_lo": lo, "ci_hi": hi}

        eval_data[cond] = {
            "n": len(per_paper_rows),
            "raw": {
                "precision": _agg(per_paper_rows, "precision"),
                "recall": _agg(per_paper_rows, "recall"),
                "f1": _agg(per_paper_rows, "f1"),
            },
            "resolved": {
                "precision": _agg(per_paper_resolved, "precision"),
                "recall": _agg(per_paper_resolved, "recall"),
                "f1": _agg(per_paper_resolved, "f1"),
            },
            "per_paper_raw": per_paper_rows,
            "per_paper_resolved": per_paper_resolved,
        }

    # Pairwise permutation tests on F1, restricted to papers present in BOTH conditions
    pairs = [("full", "no_critic"), ("full", "baseline"), ("no_critic", "baseline")]
    tests: dict[str, dict] = {"raw": {}, "resolved": {}}
    for a, b in pairs:
        if a not in eval_data or b not in eval_data:
            continue
        for variant, key in (("raw", "per_paper_raw"), ("resolved", "per_paper_resolved")):
            a_rows = {r["paper_id"]: r["f1"] for r in eval_data[a][key]}
            b_rows = {r["paper_id"]: r["f1"] for r in eval_data[b][key]}
            common = sorted(set(a_rows) & set(b_rows))
            if not common:
                continue
            a_vals = [a_rows[pid] for pid in common]
            b_vals = [b_rows[pid] for pid in common]
            p = paired_permutation_test(a_vals, b_vals, n_permutations=5000)
            tests[variant][f"{a}_vs_{b}"] = {"n_paired": len(common), "p_value": p}

    aggregate = {"by_condition": eval_data, "tests": tests}
    (output_dir / "evaluation.json").write_text(json.dumps(aggregate, indent=2))

    # Markdown report
    lines = ["# Phase Y2 — Span-grounding accuracy on SciREX dev\n"]
    lines.append("A claim grounds correctly if a span of the same entity type "
                 "overlaps a gold span AND the claim's normalized name appears "
                 "as a substring of the gold span's text (or vice versa). "
                 "Recall denominator is the unique set of gold (label, normalized_surface) entities.\n")
    lines.append(
        "We report two variants:\n"
        "- **raw**: uses the LLM-emitted `evidence_span` directly.\n"
        "- **resolved**: ignores the LLM span and resolves each entity name to a char\n"
        "  span via a deterministic case-insensitive string-match in the paper text.\n"
        "  Reflects what the system can do *with* a downstream name→span resolver.\n"
    )

    for variant in ("raw", "resolved"):
        lines.append(f"## {variant.capitalize()} span grounding (95% bootstrap CIs)\n")
        lines.append("| Condition | n | Precision [95% CI] | Recall [95% CI] | F1 [95% CI] |")
        lines.append("|---|---:|---|---|---|")
        for cond in ("full", "no_critic", "baseline"):
            d = eval_data.get(cond)
            if d is None:
                lines.append(f"| {cond} | — | — | — | — |")
                continue
            v = d[variant]
            lines.append(
                f"| {cond} | {d['n']} | "
                f"{v['precision']['mean']:.3f} [{v['precision']['ci_lo']:.3f}, {v['precision']['ci_hi']:.3f}] | "
                f"{v['recall']['mean']:.3f} [{v['recall']['ci_lo']:.3f}, {v['recall']['ci_hi']:.3f}] | "
                f"{v['f1']['mean']:.3f} [{v['f1']['ci_lo']:.3f}, {v['f1']['ci_hi']:.3f}] |"
            )
        lines.append("")
        lines.append(f"### {variant.capitalize()} pairwise paired-permutation p-values on F1\n")
        lines.append("| Comparison | n paired | p-value |")
        lines.append("|---|---:|---:|")
        for k, v in tests.get(variant, {}).items():
            sig = " *" if v["p_value"] < 0.05 else ""
            lines.append(f"| {k} | {v['n_paired']} | {v['p_value']:.4f}{sig} |")
        lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines))
    console.print(f"\n[green]Wrote {output_dir / 'report.md'}[/green]")


if __name__ == "__main__":
    app()
