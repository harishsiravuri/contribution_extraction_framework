"""Phase 6 — re-run multi-agent with a frontier extractor on benchmark subsets.

Uses config/models_frontier.yaml (Opus 4.6 extractor, Llama critic + consolidator).
Stays small (default: TDMSci only) to control cost.
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
from paper1.loaders import load_nlp_tdms, load_scirex, load_tdmsci
from paper1.openrouter import OpenRouterClient
from paper1.pipeline import Pipeline
from scripts.run_benchmarks import _evaluate, _process_paper, _safe_id  # type: ignore

app = typer.Typer(no_args_is_help=False, add_completion=False)
console = Console()


@app.command()
def run(
    output_dir: Path = typer.Option(Path("outputs/paper_data/phase6_frontier"), "--output-dir"),
    benchmarks: str = typer.Option("tdmsci", "--benchmarks", help="comma-separated"),
    max_per_benchmark: int = typer.Option(20, "--max-per-benchmark"),
    concurrency: int = typer.Option(5, "--concurrency"),
    config_path: Path = typer.Option(Path("config/models_frontier.yaml"), "--config"),
) -> None:
    cfg = load_config(config_path=config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    bench_funcs = {
        "scirex": lambda: load_scirex(splits=("dev",)),
        "tdmsci": load_tdmsci,
        "nlp_tdms": load_nlp_tdms,
    }
    selected = [b for b in benchmarks.split(",") if b in bench_funcs]
    asyncio.run(_main(cfg, selected, bench_funcs, output_dir, concurrency, max_per_benchmark))


async def _main(cfg, selected, bench_funcs, output_dir, concurrency, max_per_benchmark) -> None:
    client = OpenRouterClient(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        timeout_s=cfg.defaults.request_timeout_s,
        max_retries=cfg.defaults.max_retries,
        retry_backoff_s=cfg.defaults.retry_backoff_s,
    )
    pipe = Pipeline(client, cfg)
    total_cost = 0.0
    summaries = {}

    for benchmark in selected:
        console.print(f"\n[bold]Frontier on {benchmark}[/bold]")
        papers = [p for p in bench_funcs[benchmark]() if p.full_text and len(p.full_text) >= 50][:max_per_benchmark]
        console.print(f"  papers: {len(papers)}")
        bench_dir = output_dir / benchmark / "multi_agent"
        bench_dir.mkdir(parents=True, exist_ok=True)
        sem = asyncio.Semaphore(concurrency)
        t0 = time.perf_counter()
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), console=console) as progress:
            task_id = progress.add_task(f"frontier/{benchmark}", total=len(papers))
            results = await asyncio.gather(*(_process_paper(pipe, p, bench_dir, sem, progress, task_id) for p in papers))
        cost = sum(r.cost_usd for r in results if r.status == "ok")
        total_cost += cost
        summary = {
            "benchmark": benchmark,
            "papers_total": len(results),
            "papers_ok": sum(1 for r in results if r.status == "ok"),
            "papers_error": sum(1 for r in results if r.status == "error"),
            "papers_skipped": sum(1 for r in results if r.status == "skipped"),
            "total_cost_usd": cost,
            "wall_time_seconds": time.perf_counter() - t0,
            "extractor_model": cfg.extractor.model_id,
            "per_paper": [asdict(r) for r in results],
        }
        (output_dir / benchmark / "summary.json").write_text(json.dumps(summary, indent=2))
        summaries[benchmark] = summary

        # Evaluation: compare frontier multi vs Phase 4 baseline
        base_dir = Path("outputs/paper_data/phase4_benchmarks") / benchmark / "baseline"
        if base_dir.exists():
            ev = _evaluate(benchmark, papers, bench_dir, base_dir)
            (output_dir / benchmark / "evaluation.json").write_text(json.dumps(ev, indent=2))

    await client.aclose()

    lines = ["# Phase 6 — Frontier extractor ablation\n",
             f"_Extractor: {cfg.extractor.model_id} (Critic & Consolidator unchanged)_\n",
             f"_Total spend: ${total_cost:.4f}_\n",
             "## Per-benchmark F1 (frontier multi-agent vs open-weights baseline from Phase 4)\n",
             "| Benchmark | Field | Frontier F1 | Baseline (Phase 4) F1 | n |",
             "|---|---|---:|---:|---:|"]
    for benchmark in selected:
        ev_path = output_dir / benchmark / "evaluation.json"
        if not ev_path.exists():
            continue
        ev = json.loads(ev_path.read_text())
        for f in ("methods", "tasks", "datasets", "metrics"):
            row = ev["fields"].get(f)
            if not row:
                continue
            lines.append(f"| {benchmark} | {f} | {row['multi_agent']['f1']:.3f} | {row['baseline']['f1']:.3f} | {row['n']} |")
        if ev.get("triples"):
            t = ev["triples"]
            lines.append(f"| {benchmark} | (T,D,M) triple | {t['multi_agent']['f1']:.3f} | {t['baseline']['f1']:.3f} | {t['n']} |")
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    console.print(f"[green]Wrote {output_dir / 'report.md'}[/green]")


if __name__ == "__main__":
    app()
