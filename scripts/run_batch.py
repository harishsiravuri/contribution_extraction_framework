"""Batch runner — extract from every paper in a directory, in parallel.

Saves per-paper records as JSON + a run summary with cost/timing aggregates.
Resume support: papers whose output JSON already exists are skipped.

Usage:
    python scripts/run_batch.py --papers-dir examples/papers --output-dir outputs/run1
    python scripts/run_batch.py --papers-dir examples/papers --output-dir outputs/run1 --concurrency 5
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
from rich.table import Table

from paper1.config import load_config
from paper1.openrouter import OpenRouterClient
from paper1.pipeline import Pipeline

app = typer.Typer(no_args_is_help=False, add_completion=False)
console = Console()


@dataclass
class PaperResult:
    paper_id: str
    status: str  # "ok" | "error" | "skipped"
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    wall_time_seconds: float = 0.0
    error: str | None = None
    output_file: str | None = None


@dataclass
class RunSummary:
    started_at: float
    finished_at: float
    wall_time_seconds: float
    concurrency: int
    papers_total: int
    papers_ok: int
    papers_error: int
    papers_skipped: int
    total_cost_usd: float
    total_tokens_in: int
    total_tokens_out: int
    extractor_model: str
    critic_model: str
    consolidator_model: str
    per_paper: list[dict]


@app.command()
def run(
    papers_dir: Path = typer.Option(..., "--papers-dir", exists=True, file_okay=False, dir_okay=True),
    output_dir: Path = typer.Option(..., "--output-dir"),
    concurrency: int = typer.Option(10, "--concurrency", help="Max papers in flight at once"),
    config_path: Path | None = typer.Option(None, "--config"),
    glob_pattern: str = typer.Option("*.txt", "--glob", help="Pattern for paper files in --papers-dir"),
) -> None:
    """Run the multi-agent pipeline on every paper file in --papers-dir."""

    cfg = load_config(config_path=config_path)

    paper_files = sorted(papers_dir.glob(glob_pattern))
    if not paper_files:
        console.print(f"[red]No files matching {glob_pattern} in {papers_dir}[/red]")
        raise typer.Exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    by_paper_dir = output_dir / "by_paper"
    by_paper_dir.mkdir(exist_ok=True)

    console.print(
        f"[bold]Processing {len(paper_files)} papers with concurrency={concurrency}[/bold]"
    )
    console.print(f"  Extractor:    {cfg.extractor.model_id}")
    console.print(f"  Critic:       {cfg.critic.model_id}")
    console.print(f"  Consolidator: {cfg.consolidator.model_id}")
    console.print(f"  Outputs:      {output_dir}\n")

    asyncio.run(
        _run_async(cfg, paper_files, by_paper_dir, output_dir, concurrency)
    )


async def _run_async(
    cfg, paper_files: list[Path], by_paper_dir: Path, output_dir: Path, concurrency: int
) -> None:
    client = OpenRouterClient(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        timeout_s=cfg.defaults.request_timeout_s,
        max_retries=cfg.defaults.max_retries,
        retry_backoff_s=cfg.defaults.retry_backoff_s,
        referer=cfg.referer,
        title=cfg.title,
    )
    pipe = Pipeline(client, cfg)
    sem = asyncio.Semaphore(concurrency)

    async def _process(paper_file: Path, progress: Progress, task_id) -> PaperResult:
        paper_id = paper_file.stem
        output_file = by_paper_dir / f"{paper_id}.json"
        if output_file.exists():
            progress.advance(task_id)
            return PaperResult(paper_id=paper_id, status="skipped", output_file=str(output_file))

        paper_text = paper_file.read_text(encoding="utf-8")
        async with sem:
            try:
                record = await pipe.extract(paper_id=paper_id, paper_text=paper_text)
                output_file.write_text(
                    record.model_dump_json(indent=2, by_alias=True), encoding="utf-8"
                )
                result = PaperResult(
                    paper_id=paper_id,
                    status="ok",
                    cost_usd=record.meta.cost_usd,
                    tokens_in=record.meta.tokens_in,
                    tokens_out=record.meta.tokens_out,
                    wall_time_seconds=record.meta.wall_time_seconds,
                    output_file=str(output_file),
                )
            except Exception as e:  # noqa: BLE001 — we want to capture every failure type
                result = PaperResult(
                    paper_id=paper_id,
                    status="error",
                    error=f"{type(e).__name__}: {e}",
                )
            finally:
                progress.advance(task_id)
        return result

    started = time.time()
    t0 = time.perf_counter()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Extracting", total=len(paper_files))
        results: list[PaperResult] = await asyncio.gather(
            *(_process(p, progress, task_id) for p in paper_files)
        )

    elapsed = time.perf_counter() - t0
    finished = time.time()
    await client.aclose()

    # Aggregate
    ok = [r for r in results if r.status == "ok"]
    errors = [r for r in results if r.status == "error"]
    skipped = [r for r in results if r.status == "skipped"]

    summary = RunSummary(
        started_at=started,
        finished_at=finished,
        wall_time_seconds=elapsed,
        concurrency=concurrency,
        papers_total=len(results),
        papers_ok=len(ok),
        papers_error=len(errors),
        papers_skipped=len(skipped),
        total_cost_usd=sum(r.cost_usd for r in ok),
        total_tokens_in=sum(r.tokens_in for r in ok),
        total_tokens_out=sum(r.tokens_out for r in ok),
        extractor_model=cfg.extractor.model_id,
        critic_model=cfg.critic.model_id,
        consolidator_model=cfg.consolidator.model_id,
        per_paper=[asdict(r) for r in results],
    )

    summary_file = output_dir / f"run_{int(started)}.json"
    summary_file.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")

    # Console report
    table = Table(title="Run summary", show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Papers (ok / error / skipped)", f"{len(ok)} / {len(errors)} / {len(skipped)}")
    table.add_row("Wall-clock (s)", f"{elapsed:.1f}")
    table.add_row("Total cost (USD)", f"${summary.total_cost_usd:.4f}")
    table.add_row("Tokens in / out", f"{summary.total_tokens_in:,} / {summary.total_tokens_out:,}")
    if ok:
        table.add_row(
            "Avg cost per paper",
            f"${summary.total_cost_usd / len(ok):.4f}",
        )
        table.add_row(
            "Avg wall-time per paper",
            f"{sum(r.wall_time_seconds for r in ok) / len(ok):.1f}s",
        )
    console.print(table)

    if errors:
        console.print("\n[bold red]Errors:[/bold red]")
        for r in errors:
            console.print(f"  [red]{r.paper_id}[/red]: {r.error}")

    console.print(f"\n[dim]Summary written to {summary_file}[/dim]")
    console.print(f"[dim]Per-paper outputs in {by_paper_dir}/[/dim]")


if __name__ == "__main__":
    app()
