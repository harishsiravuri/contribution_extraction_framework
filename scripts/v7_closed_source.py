"""E1 — Closed-source extractor comparison (GPT-4o vs DeepSeek Chat).

Sample 25 papers from SciREX dev with seed=42 (subset of the 62 papers
used in the v3 dev evaluation, so we can reuse the existing DeepSeek
multi-agent and DeepSeek single-LLM records for free).

Conditions on the same 25 papers:
  (a) DeepSeek multi-agent     — REUSED from outputs/paper_data_v3/benchmarks/scirex/multi_agent/
  (b) GPT-4o   multi-agent     — NEW, this script (config/models_gpt4o.yaml)
  (c) DeepSeek single-LLM      — REUSED from outputs/paper_data_v3/benchmarks/scirex/baseline/
  (d) GPT-4o   single-LLM      — NEW, this script (config/models_gpt4o.yaml baseline)

Outputs:
  outputs/paper_data_v7/closed_source_comparison/
    sampled_papers.json — 25 sampled paper_ids (seed=42)
    gpt4o_multi/<paper>.json
    gpt4o_baseline/<paper>.json
    deepseek_multi_subset/  — symlinks/copies of the v3 records for the 25 papers
    deepseek_baseline_subset/  — same for baseline
    extraction_summary.json
    results.json — F1 + bootstrap CIs + paired-perm comparisons + per-paper cost
"""

from __future__ import annotations

import asyncio
import json
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn,
)

from paper1.config import load_config
from paper1.loaders import load_scirex
from paper1.openrouter import OpenRouterClient
from paper1.pipeline import Pipeline
from paper1.pipelines.baseline import BaselinePipeline

SEED = 42

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


async def _run_one(pipe, paper, out_dir, sem, progress, task_id):
    safe = _safe_id(paper.paper_id)
    out_path = out_dir / f"{safe}.json"
    if out_path.exists():
        progress.advance(task_id)
        return PaperResult(paper_id=paper.paper_id, status="skipped",
                           output_file=str(out_path))
    async with sem:
        t0 = time.perf_counter()
        try:
            record = await pipe.extract(paper_id=paper.paper_id, paper_text=paper.full_text)
            out_path.write_text(record.model_dump_json(indent=2, by_alias=True), encoding="utf-8")
            return PaperResult(
                paper_id=paper.paper_id, status="ok",
                cost_usd=record.meta.cost_usd,
                tokens_in=record.meta.tokens_in,
                tokens_out=record.meta.tokens_out,
                wall_time_seconds=record.meta.wall_time_seconds,
                output_file=str(out_path),
            )
        except Exception as e:  # noqa: BLE001
            return PaperResult(paper_id=paper.paper_id, status="error",
                               error=f"{type(e).__name__}: {str(e)[:200]}")
        finally:
            progress.advance(task_id)


@app.command()
def run(
    output_dir: Path = typer.Option(Path("outputs/paper_data_v7/closed_source_comparison"), "--output-dir"),
    gpt4o_config: Path = typer.Option(Path("config/models_gpt4o.yaml"), "--gpt4o-config"),
    sample_size: int = typer.Option(25, "--sample-size"),
    concurrency: int = typer.Option(5, "--concurrency"),
):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "seed.txt").write_text(f"{SEED}\n")

    # Build the 62-paper dev set as v3 saw it (the 62 papers that have both
    # multi_agent and baseline records on disk).
    dev_papers = [p for p in load_scirex(splits=("dev",)) if p.full_text and len(p.full_text) >= 50][:66]
    v3_multi_dir = Path("outputs/paper_data_v3/benchmarks/scirex/multi_agent")
    v3_base_dir = Path("outputs/paper_data_v3/benchmarks/scirex/baseline")
    v3_ids = set()
    for p in dev_papers:
        safe = _safe_id(p.paper_id)
        if (v3_multi_dir / f"{safe}.json").exists() and (v3_base_dir / f"{safe}.json").exists():
            v3_ids.add(p.paper_id)
    available = [p for p in dev_papers if p.paper_id in v3_ids]
    console.print(f"  {len(available)} dev papers have both v3 multi-agent + baseline records")

    rng = random.Random(SEED)
    sampled = rng.sample(available, k=min(sample_size, len(available)))
    sampled_ids = sorted(p.paper_id for p in sampled)
    (output_dir / "sampled_papers.json").write_text(json.dumps(sampled_ids, indent=2))
    console.print(f"  sampled {len(sampled)} papers (seed={SEED})")

    # Copy DeepSeek records for the sampled subset (no new spend)
    ds_multi_subset = output_dir / "deepseek_multi_subset"
    ds_base_subset = output_dir / "deepseek_baseline_subset"
    ds_multi_subset.mkdir(parents=True, exist_ok=True)
    ds_base_subset.mkdir(parents=True, exist_ok=True)
    for p in sampled:
        safe = _safe_id(p.paper_id)
        shutil.copy2(v3_multi_dir / f"{safe}.json", ds_multi_subset / f"{safe}.json")
        shutil.copy2(v3_base_dir / f"{safe}.json", ds_base_subset / f"{safe}.json")

    # New runs: GPT-4o multi + baseline
    cfg = load_config(config_path=gpt4o_config)
    or_client = OpenRouterClient(
        api_key=cfg.api_key, base_url=cfg.base_url,
        timeout_s=cfg.defaults.request_timeout_s,
        max_retries=cfg.defaults.max_retries,
        retry_backoff_s=cfg.defaults.retry_backoff_s,
        referer=cfg.referer, title=cfg.title,
    )
    multi = Pipeline(or_client, cfg)
    base = BaselinePipeline(or_client, cfg)
    gpt_multi_dir = output_dir / "gpt4o_multi"
    gpt_base_dir = output_dir / "gpt4o_baseline"
    gpt_multi_dir.mkdir(parents=True, exist_ok=True)
    gpt_base_dir.mkdir(parents=True, exist_ok=True)

    asyncio.run(_main(sampled, multi, base, gpt_multi_dir, gpt_base_dir, output_dir, concurrency))


async def _main(sampled, multi, base, gpt_multi_dir, gpt_base_dir, output_dir, concurrency):
    sem = asyncio.Semaphore(concurrency)
    all_results = {}
    for label, pipe, out_d in (
        ("gpt4o_multi", multi, gpt_multi_dir),
        ("gpt4o_baseline", base, gpt_base_dir),
    ):
        t0 = time.perf_counter()
        console.print(f"  [bold]{label}[/bold] extracting on {len(sampled)} papers...")
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                      MofNCompleteColumn(), TimeElapsedColumn(), console=console) as progress:
            task_id = progress.add_task(f"{label}", total=len(sampled))
            results = await asyncio.gather(*(
                _run_one(pipe, p, out_d, sem, progress, task_id) for p in sampled
            ))
        ok = sum(1 for r in results if r.status == "ok")
        err = sum(1 for r in results if r.status == "error")
        skip = sum(1 for r in results if r.status == "skipped")
        cost = sum(r.cost_usd for r in results if r.status == "ok")
        elapsed = time.perf_counter() - t0
        console.print(f"    {label}: ok={ok} err={err} skip={skip} cost=${cost:.4f} wall={elapsed:.1f}s")
        all_results[label] = {
            "papers_total": len(results), "papers_ok": ok, "papers_error": err,
            "papers_skipped": skip, "total_cost_usd": cost,
            "wall_time_seconds": elapsed, "per_paper": [asdict(r) for r in results],
        }
    (output_dir / "extraction_summary.json").write_text(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    app()
