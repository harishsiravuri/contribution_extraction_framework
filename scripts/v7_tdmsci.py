"""E3 — TDMSci cross-corpus evaluation with the default open-weights framework.

Runs the default DeepSeek + Llama 3.3 70B multi-agent pipeline on the
TDMSci test split (Hou et al. 2019 ACL). TDMSci is sentence-level BIO-
tagged for Task / Dataset / Metric (no Method field).

Outputs:
  outputs/paper_data_v7/tdmsci/
    multi_agent/<paper>.json   — per-sentence ContributionRecord
    extraction_summary.json
    results.json — per-field F1 + bootstrap CIs + OOD ECE + confusion summary
    seed.txt
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
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn,
)

from paper1.config import load_config
from paper1.loaders import load_tdmsci
from paper1.openrouter import OpenRouterClient
from paper1.pipeline import Pipeline

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
    output_dir: Path = typer.Option(Path("outputs/paper_data_v7/tdmsci"), "--output-dir"),
    config_path: Path = typer.Option(Path("config/models.yaml"), "--config"),
    max_sentences: int = typer.Option(376, "--max-sentences",
                                      help="Cap on TDMSci test sentences (default: all 376 valid)"),
    concurrency: int = typer.Option(8, "--concurrency"),
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
    multi = Pipeline(or_client, cfg)

    papers = [p for p in load_tdmsci() if "test" in p.paper_id
              and p.full_text and len(p.full_text) >= 50][:max_sentences]
    console.print(f"  TDMSci test sentences (len>=50): {len(papers)}")

    multi_dir = output_dir / "multi_agent"
    multi_dir.mkdir(parents=True, exist_ok=True)

    asyncio.run(_main(papers, multi, multi_dir, output_dir, concurrency))


async def _main(papers, multi, multi_dir, output_dir, concurrency):
    sem = asyncio.Semaphore(concurrency)
    t0 = time.perf_counter()
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                  MofNCompleteColumn(), TimeElapsedColumn(), console=console) as progress:
        task_id = progress.add_task("tdmsci-multi", total=len(papers))
        results = await asyncio.gather(*(
            _run_one(multi, p, multi_dir, sem, progress, task_id) for p in papers
        ))
    ok = sum(1 for r in results if r.status == "ok")
    err = sum(1 for r in results if r.status == "error")
    skip = sum(1 for r in results if r.status == "skipped")
    cost = sum(r.cost_usd for r in results if r.status == "ok")
    elapsed = time.perf_counter() - t0
    console.print(f"  done: ok={ok} err={err} skip={skip} cost=${cost:.4f} wall={elapsed:.1f}s")
    summary = {
        "papers_total": len(results),
        "papers_ok": ok, "papers_error": err, "papers_skipped": skip,
        "total_cost_usd": cost,
        "wall_time_seconds": elapsed,
        "per_paper": [asdict(r) for r in results],
    }
    (output_dir / "extraction_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()
