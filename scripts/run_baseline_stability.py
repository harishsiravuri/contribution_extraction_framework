"""Run only baseline + stability (use this when multi-agent already partially done).

Reads existing outputs/paper_data_v2/pilot_5k/multi_agent/by_paper/ to get
the paper-id list, then runs BaselinePipeline on those + a stability re-run
of the multi-agent pipeline on the first --stability-n paper IDs.
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
from paper1.openrouter import OpenRouterClient
from paper1.pipeline import Pipeline
from paper1.pipelines.baseline import BaselinePipeline

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


@app.command()
def run(
    papers_dir: Path = typer.Option(Path("examples/pilot_corpus_v2"), "--papers-dir"),
    output_dir: Path = typer.Option(Path("outputs/paper_data_v2/pilot_5k"), "--output-dir"),
    concurrency: int = typer.Option(50, "--concurrency"),
    stability_n: int = typer.Option(100, "--stability-n"),
    config_path: Path | None = typer.Option(None, "--config"),
    skip_baseline: bool = typer.Option(False, "--skip-baseline"),
    skip_stability: bool = typer.Option(False, "--skip-stability"),
) -> None:
    cfg = load_config(config_path=config_path)

    # The set of paper IDs we should evaluate baseline/stability on
    multi_dir = output_dir / "multi_agent" / "by_paper"
    completed_ids = sorted(p.stem for p in multi_dir.glob("*.json"))
    console.print(f"[bold]{len(completed_ids)} multi-agent records present[/bold]")

    paper_paths: dict[str, Path] = {}
    for f in papers_dir.glob("*.txt"):
        paper_paths[f.stem] = f
    target_paths = [paper_paths[i] for i in completed_ids if i in paper_paths]
    console.print(f"[bold]{len(target_paths)} matching paper text files[/bold]")

    asyncio.run(_main(cfg, target_paths, output_dir, concurrency, stability_n, skip_baseline, skip_stability))


async def _main(cfg, paper_files, output_dir, concurrency, stability_n, skip_baseline, skip_stability):
    client = OpenRouterClient(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        timeout_s=cfg.defaults.request_timeout_s,
        max_retries=cfg.defaults.max_retries,
        retry_backoff_s=cfg.defaults.retry_backoff_s,
    )
    multi = Pipeline(client, cfg)
    base = BaselinePipeline(client, cfg)
    sem = asyncio.Semaphore(concurrency)

    async def _process(pipe, paper_file: Path, by_paper_dir: Path, force: bool, progress, task_id) -> PaperResult:
        paper_id = paper_file.stem
        out = by_paper_dir / f"{paper_id}.json"
        if out.exists() and not force:
            progress.advance(task_id)
            return PaperResult(paper_id=paper_id, status="skipped", output_file=str(out))
        text = paper_file.read_text(encoding="utf-8")
        async with sem:
            try:
                rec = await pipe.extract(paper_id=paper_id, paper_text=text)
                out.write_text(rec.model_dump_json(indent=2, by_alias=True), encoding="utf-8")
                return PaperResult(
                    paper_id=paper_id,
                    status="ok",
                    cost_usd=rec.meta.cost_usd,
                    tokens_in=rec.meta.tokens_in,
                    tokens_out=rec.meta.tokens_out,
                    wall_time_seconds=rec.meta.wall_time_seconds,
                    output_file=str(out),
                )
            except Exception as e:  # noqa
                return PaperResult(paper_id=paper_id, status="error", error=f"{type(e).__name__}: {str(e)[:200]}")
            finally:
                progress.advance(task_id)

    async def _run_phase(name: str, pipe, paper_files, by_paper_dir, force):
        by_paper_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), console=console) as progress:
            task_id = progress.add_task(name, total=len(paper_files))
            results = await asyncio.gather(*(_process(pipe, p, by_paper_dir, force, progress, task_id) for p in paper_files))
        elapsed = time.perf_counter() - t0
        ok = sum(1 for r in results if r.status == "ok")
        err = sum(1 for r in results if r.status == "error")
        skip = sum(1 for r in results if r.status == "skipped")
        cost = sum(r.cost_usd for r in results if r.status == "ok")
        summary = {
            "phase": name, "wall_time_seconds": elapsed,
            "papers_total": len(results), "papers_ok": ok, "papers_error": err, "papers_skipped": skip,
            "total_cost_usd": cost,
            "total_tokens_in": sum(r.tokens_in for r in results if r.status == "ok"),
            "total_tokens_out": sum(r.tokens_out for r in results if r.status == "ok"),
            "extractor_model": cfg.extractor.model_id, "critic_model": cfg.critic.model_id, "consolidator_model": cfg.consolidator.model_id,
            "concurrency": concurrency,
            "started_at": time.time() - elapsed, "finished_at": time.time(),
            "per_paper": [asdict(r) for r in results],
        }
        out_dir = by_paper_dir.parent
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        console.print(f"[green]{name}[/green]: ok={ok} err={err} skip={skip} cost=${cost:.4f} time={elapsed:.1f}s")

    if not skip_baseline:
        await _run_phase("baseline", base, paper_files, output_dir / "baseline" / "by_paper", force=False)
    if not skip_stability:
        await _run_phase("stability", multi, paper_files[:stability_n], output_dir / "stability" / "by_paper", force=True)

    await client.aclose()


if __name__ == "__main__":
    app()
