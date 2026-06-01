"""Pilot 1k experiment orchestrator.

Runs three phases on ~1,000 arXiv abstracts:
  (a) multi-agent Pipeline
  (b) single-LLM BaselinePipeline (same model as the multi-agent's extractor)
  (c) multi-agent re-run on the first 100 papers (for stability measurement)

Then computes Jaccard stability between (a) and (c), aggregates cost / wall-time
/ coverage / sample comparisons, and writes outputs/pilot/report.md.

Hard fails if cumulative spend exceeds $50 between phases.

Usage:
    python scripts/run_experiment.py
    python scripts/run_experiment.py --concurrency 10
    python scripts/run_experiment.py --papers-dir examples/pilot_corpus --output-dir outputs/pilot
"""

from __future__ import annotations

import asyncio
import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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

from paper1.agents.extractor import ExtractorParseError
from paper1.agents.single_llm import SingleLLMParseError
from paper1.agents.consolidator import ConsolidatorParseError
from paper1.agents.critic import CriticParseError
from paper1.config import load_config
from paper1.metrics.stability import jaccard_stability
from paper1.openrouter import OpenRouterClient, OpenRouterAPIError
from paper1.pipeline import Pipeline
from paper1.pipelines.baseline import BaselinePipeline
from paper1.schema import ContributionRecord

app = typer.Typer(no_args_is_help=False, add_completion=False)
console = Console()
log = logging.getLogger(__name__)

PARSE_ERRORS: tuple[type[Exception], ...] = (
    ExtractorParseError,
    SingleLLMParseError,
    ConsolidatorParseError,
    CriticParseError,
    OpenRouterAPIError,
)

SPEND_HARD_CAP_USD = 50.0


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
class PhaseSummary:
    phase: str
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


async def _run_phase(
    *,
    phase_name: str,
    pipe,  # Pipeline | BaselinePipeline (duck-typed)
    paper_files: list[Path],
    out_dir: Path,
    concurrency: int,
    cfg,
    force: bool = False,
) -> PhaseSummary:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_paper_dir = out_dir / "by_paper"
    by_paper_dir.mkdir(exist_ok=True)

    sem = asyncio.Semaphore(concurrency)

    async def _process(paper_file: Path, progress: Progress, task_id) -> PaperResult:
        paper_id = paper_file.stem
        out_path = by_paper_dir / f"{paper_id}.json"
        if out_path.exists() and not force:
            progress.advance(task_id)
            return PaperResult(
                paper_id=paper_id, status="skipped", output_file=str(out_path)
            )

        paper_text = paper_file.read_text(encoding="utf-8")
        async with sem:
            try:
                record = await pipe.extract(paper_id=paper_id, paper_text=paper_text)
                out_path.write_text(
                    record.model_dump_json(indent=2, by_alias=True), encoding="utf-8"
                )
                result = PaperResult(
                    paper_id=paper_id,
                    status="ok",
                    cost_usd=record.meta.cost_usd,
                    tokens_in=record.meta.tokens_in,
                    tokens_out=record.meta.tokens_out,
                    wall_time_seconds=record.meta.wall_time_seconds,
                    output_file=str(out_path),
                )
            except Exception as e:  # noqa: BLE001
                result = PaperResult(
                    paper_id=paper_id,
                    status="error",
                    error=f"{type(e).__name__}: {str(e)[:300]}",
                )
            finally:
                progress.advance(task_id)
        return result

    started = time.time()
    t0 = time.perf_counter()

    with Progress(
        SpinnerColumn(),
        TextColumn(f"[progress.description]{{task.description}}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task(f"{phase_name}", total=len(paper_files))
        results: list[PaperResult] = await asyncio.gather(
            *(_process(p, progress, task_id) for p in paper_files)
        )

    elapsed = time.perf_counter() - t0
    finished = time.time()

    ok = [r for r in results if r.status == "ok"]
    errors = [r for r in results if r.status == "error"]
    skipped = [r for r in results if r.status == "skipped"]

    summary = PhaseSummary(
        phase=phase_name,
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

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    console.print(
        f"[green]{phase_name}[/green]: {len(ok)} ok / {len(errors)} err / {len(skipped)} skip — "
        f"${summary.total_cost_usd:.4f} in {elapsed:.1f}s"
    )
    return summary


def _load_record(path: Path) -> ContributionRecord | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return ContributionRecord.model_validate(data)
    except Exception as e:
        console.print(f"[yellow]could not load {path}: {e}[/yellow]")
        return None


def _coverage_for_dir(by_paper_dir: Path) -> dict[str, float]:
    """Per-field % of records where the field is non-null / non-empty."""
    files = sorted(by_paper_dir.glob("*.json"))
    if not files:
        return {}
    counts = {"method.name": 0, "task.name": 0, "datasets": 0, "metrics": 0, "claim_strength": 0}
    total = 0
    for f in files:
        rec = _load_record(f)
        if rec is None or not rec.contributions:
            total += 1
            continue
        # Aggregate across contributions: count the record as "has X" if any contribution does
        c0 = rec.contributions[0]
        if c0.method.name:
            counts["method.name"] += 1
        if c0.task.name:
            counts["task.name"] += 1
        if any(ds.name for ds in c0.datasets):
            counts["datasets"] += 1
        if any(m.name for m in c0.metrics):
            counts["metrics"] += 1
        if c0.claim_strength is not None:
            counts["claim_strength"] += 1
        total += 1
    if total == 0:
        return {}
    return {k: 100.0 * v / total for k, v in counts.items()}


def _check_spend(running_total: float) -> None:
    if running_total > SPEND_HARD_CAP_USD:
        raise SystemExit(
            f"\n[ABORT] cumulative spend ${running_total:.2f} exceeded hard cap "
            f"${SPEND_HARD_CAP_USD:.2f}"
        )


@app.command()
def run(
    papers_dir: Path = typer.Option(Path("examples/pilot_corpus"), "--papers-dir"),
    output_dir: Path = typer.Option(Path("outputs/pilot"), "--output-dir"),
    concurrency: int = typer.Option(10, "--concurrency"),
    config_path: Path | None = typer.Option(None, "--config"),
    stability_n: int = typer.Option(100, "--stability-n", help="How many papers to re-run for stability"),
    max_papers: int | None = typer.Option(None, "--max-papers", help="Cap the corpus to first N papers"),
    client_kind: str = typer.Option("openrouter", "--client", help="Inference router: 'openrouter' (default) or 'together'"),
) -> None:
    """Run the full pilot 1k experiment (multi-agent, baseline, stability re-run)."""

    cfg = load_config(config_path=config_path)
    paper_files = sorted(papers_dir.glob("*.txt"))
    if max_papers is not None:
        paper_files = paper_files[:max_papers]
    if not paper_files:
        console.print(f"[red]No .txt files in {papers_dir}[/red]")
        raise typer.Exit(1)
    console.print(f"[bold]Papers: {len(paper_files)}[/bold] in {papers_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    asyncio.run(_main(cfg, paper_files, output_dir, concurrency, stability_n, client_kind))


async def _main(cfg, paper_files: list[Path], output_dir: Path, concurrency: int, stability_n: int, client_kind: str = "openrouter") -> None:
    multi_dir = output_dir / "multi_agent"
    base_dir = output_dir / "baseline"
    stab_dir = output_dir / "stability"

    or_client = OpenRouterClient(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        timeout_s=cfg.defaults.request_timeout_s,
        max_retries=cfg.defaults.max_retries,
        retry_backoff_s=cfg.defaults.retry_backoff_s,
        referer=cfg.referer,
        title=cfg.title,
    )
    if client_kind == "together":
        import os
        from paper1.together_client import RoutingClient, TogetherClient
        tg_key = os.environ.get("TOGETHER_API_KEY", "")
        if not tg_key:
            raise RuntimeError("--client together requires TOGETHER_API_KEY in .env")
        tg_client = TogetherClient(api_key=tg_key, timeout_s=cfg.defaults.request_timeout_s)
        client = RoutingClient(or_client, tg_client, together_models={cfg.extractor.model_id})
    else:
        client = or_client
    multi_pipe = Pipeline(client, cfg)
    base_pipe = BaselinePipeline(client, cfg)

    cumulative_cost = 0.0

    # Phase A: multi-agent on full corpus
    console.print("\n[bold cyan]Phase A — multi-agent on full corpus[/bold cyan]")
    multi_summary = await _run_phase(
        phase_name="multi_agent",
        pipe=multi_pipe,
        paper_files=paper_files,
        out_dir=multi_dir,
        concurrency=concurrency,
        cfg=cfg,
    )
    cumulative_cost += multi_summary.total_cost_usd
    _check_spend(cumulative_cost)

    # Phase B: baseline on full corpus
    console.print("\n[bold cyan]Phase B — single-LLM baseline on full corpus[/bold cyan]")
    base_summary = await _run_phase(
        phase_name="baseline",
        pipe=base_pipe,
        paper_files=paper_files,
        out_dir=base_dir,
        concurrency=concurrency,
        cfg=cfg,
    )
    cumulative_cost += base_summary.total_cost_usd
    _check_spend(cumulative_cost)

    # Phase C: re-run multi-agent on first stability_n papers (force re-run)
    console.print(
        f"\n[bold cyan]Phase C — multi-agent stability re-run on first {stability_n}[/bold cyan]"
    )
    stability_files = paper_files[:stability_n]
    stab_summary = await _run_phase(
        phase_name="stability",
        pipe=multi_pipe,
        paper_files=stability_files,
        out_dir=stab_dir,
        concurrency=concurrency,
        cfg=cfg,
        force=True,
    )
    cumulative_cost += stab_summary.total_cost_usd
    _check_spend(cumulative_cost)

    await client.aclose()

    # Compute stability between multi-agent (Phase A) and stability re-run (Phase C)
    console.print("\n[bold]Computing Jaccard stability...[/bold]")
    stability_scores: list[dict[str, Any]] = []
    per_field_lists: dict[str, list[float]] = {
        "method.name": [], "task.name": [], "datasets": [], "metrics": [], "claim_strength": []
    }
    overall_scores: list[float] = []

    for pf in stability_files:
        pid = pf.stem
        a_path = multi_dir / "by_paper" / f"{pid}.json"
        c_path = stab_dir / "by_paper" / f"{pid}.json"
        if not (a_path.exists() and c_path.exists()):
            continue
        a = _load_record(a_path)
        c = _load_record(c_path)
        if a is None or c is None:
            continue
        s = jaccard_stability(a, c)
        s["paper_id"] = pid
        stability_scores.append(s)
        overall_scores.append(s["overall"])
        for k, v in s["per_field"].items():
            per_field_lists[k].append(v)

    stability_agg = {
        "n": len(overall_scores),
        "overall_mean": statistics.mean(overall_scores) if overall_scores else 0.0,
        "overall_std": statistics.pstdev(overall_scores) if len(overall_scores) > 1 else 0.0,
        "per_field_mean": {
            k: (statistics.mean(v) if v else 0.0) for k, v in per_field_lists.items()
        },
        "per_field_std": {
            k: (statistics.pstdev(v) if len(v) > 1 else 0.0) for k, v in per_field_lists.items()
        },
        "per_paper": stability_scores,
    }
    (output_dir / "stability_scores.json").write_text(
        json.dumps(stability_agg, indent=2), encoding="utf-8"
    )

    # Coverage
    console.print("[bold]Computing coverage...[/bold]")
    coverage_multi = _coverage_for_dir(multi_dir / "by_paper")
    coverage_base = _coverage_for_dir(base_dir / "by_paper")

    # Sample comparisons: pick 5 papers where multi-agent and baseline diverge most
    console.print("[bold]Picking sample comparisons...[/bold]")
    divergences: list[tuple[float, str, ContributionRecord, ContributionRecord]] = []
    for pf in paper_files:
        pid = pf.stem
        a_path = multi_dir / "by_paper" / f"{pid}.json"
        b_path = base_dir / "by_paper" / f"{pid}.json"
        if not (a_path.exists() and b_path.exists()):
            continue
        a = _load_record(a_path)
        b = _load_record(b_path)
        if a is None or b is None:
            continue
        s = jaccard_stability(a, b)
        # Lower Jaccard = greater divergence
        divergences.append((s["overall"], pid, a, b))
    divergences.sort(key=lambda t: t[0])
    samples = divergences[:5]

    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    for div_score, pid, a, b in samples:
        (samples_dir / f"{pid}.json").write_text(
            json.dumps(
                {
                    "paper_id": pid,
                    "divergence_jaccard": div_score,
                    "multi_agent": a.model_dump(by_alias=True),
                    "baseline": b.model_dump(by_alias=True),
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    # Errors
    errors: list[dict[str, str]] = []
    for sm in (multi_summary, base_summary, stab_summary):
        for r in sm.per_paper:
            if r["status"] == "error":
                errors.append({"phase": sm.phase, "paper_id": r["paper_id"], "error": r["error"] or ""})

    # Build report.md
    console.print("[bold]Writing report.md...[/bold]")
    report = _build_report(
        multi_summary=multi_summary,
        base_summary=base_summary,
        stab_summary=stab_summary,
        stability_agg=stability_agg,
        coverage_multi=coverage_multi,
        coverage_base=coverage_base,
        samples=samples,
        errors=errors,
        cumulative_cost=cumulative_cost,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    console.print(f"\n[bold green]Done.[/bold green] Total cost: ${cumulative_cost:.4f}")
    console.print(f"Report: {output_dir / 'report.md'}")


def _fmt_pct(x: float) -> str:
    return f"{x:.1f}%"


def _summarize_contribution(rec: ContributionRecord) -> str:
    if not rec.contributions:
        return "(no contributions)"
    c = rec.contributions[0]
    parts = []
    if c.method.name:
        parts.append(f"method={c.method.name}")
    if c.task.name:
        parts.append(f"task={c.task.name}")
    ds = [d.name for d in c.datasets if d.name]
    if ds:
        parts.append(f"datasets=[{', '.join(ds[:3])}]")
    ms = [m.name for m in c.metrics if m.name]
    if ms:
        parts.append(f"metrics=[{', '.join(ms[:3])}]")
    if c.claim_strength:
        parts.append(f"claim={c.claim_strength}")
    return "; ".join(parts) or "(empty)"


def _build_report(
    *,
    multi_summary: PhaseSummary,
    base_summary: PhaseSummary,
    stab_summary: PhaseSummary,
    stability_agg: dict[str, Any],
    coverage_multi: dict[str, float],
    coverage_base: dict[str, float],
    samples: list[tuple[float, str, ContributionRecord, ContributionRecord]],
    errors: list[dict[str, str]],
    cumulative_cost: float,
) -> str:
    lines: list[str] = []
    lines.append("# Pilot 1k Experiment Report\n")
    lines.append(f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_\n")

    lines.append("## Headline\n")
    lines.append(
        f"- **Stability (Jaccard, n={stability_agg['n']}):** "
        f"{stability_agg['overall_mean']:.3f} ± {stability_agg['overall_std']:.3f}"
    )
    total_papers = multi_summary.papers_total
    err_count = sum(1 for _ in errors)
    err_rate = (err_count / max(1, total_papers * 3)) * 100
    lines.append(f"- **Total cost (3 phases):** ${cumulative_cost:.4f}")
    lines.append(
        f"- **Wall-time:** multi-agent {multi_summary.wall_time_seconds:.1f}s, "
        f"baseline {base_summary.wall_time_seconds:.1f}s, "
        f"stability {stab_summary.wall_time_seconds:.1f}s"
    )
    lines.append(
        f"- **Error rate:** {err_count} errors across all phases ({err_rate:.1f}% of calls)\n"
    )

    lines.append("## Phase summaries\n")
    lines.append("| Phase | Papers ok | Errors | Skipped | Cost (USD) | Wall-time (s) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for sm in (multi_summary, base_summary, stab_summary):
        lines.append(
            f"| {sm.phase} | {sm.papers_ok} | {sm.papers_error} | {sm.papers_skipped} | "
            f"${sm.total_cost_usd:.4f} | {sm.wall_time_seconds:.1f} |"
        )
    lines.append("")

    lines.append("## Coverage comparison (% of papers with non-null field)\n")
    lines.append("| Field | Multi-agent | Baseline |")
    lines.append("|---|---:|---:|")
    for k in ("method.name", "task.name", "datasets", "metrics", "claim_strength"):
        m = coverage_multi.get(k, 0.0)
        b = coverage_base.get(k, 0.0)
        lines.append(f"| {k} | {_fmt_pct(m)} | {_fmt_pct(b)} |")
    lines.append("")

    lines.append("## Cost breakdown\n")
    lines.append("| Phase | Tokens in | Tokens out | Cost (USD) |")
    lines.append("|---|---:|---:|---:|")
    for sm in (multi_summary, base_summary, stab_summary):
        lines.append(
            f"| {sm.phase} | {sm.total_tokens_in:,} | {sm.total_tokens_out:,} | "
            f"${sm.total_cost_usd:.4f} |"
        )
    lines.append(f"| **TOTAL** | | | **${cumulative_cost:.4f}** |\n")

    lines.append("## Per-field stability (multi-agent run A vs. multi-agent re-run C)\n")
    lines.append("| Field | Mean Jaccard | Std |")
    lines.append("|---|---:|---:|")
    for k in ("method.name", "task.name", "datasets", "metrics", "claim_strength"):
        m = stability_agg["per_field_mean"].get(k, 0.0)
        s = stability_agg["per_field_std"].get(k, 0.0)
        lines.append(f"| {k} | {m:.3f} | {s:.3f} |")
    lines.append(
        f"| **overall** | **{stability_agg['overall_mean']:.3f}** | **{stability_agg['overall_std']:.3f}** |\n"
    )

    lines.append("## Sample comparisons (5 papers where multi-agent & baseline diverge most)\n")
    for div_score, pid, a, b in samples:
        lines.append(f"### `{pid}` (Jaccard overlap = {div_score:.2f})\n")
        lines.append(f"- **Multi-agent:** {_summarize_contribution(a)}")
        lines.append(f"- **Baseline:**    {_summarize_contribution(b)}\n")

    lines.append("## Errors encountered\n")
    if not errors:
        lines.append("_No errors._\n")
    else:
        lines.append("| Phase | Paper ID | Error |")
        lines.append("|---|---|---|")
        for e in errors:
            err = e["error"].replace("|", "\\|")[:200]
            lines.append(f"| {e['phase']} | {e['paper_id']} | {err} |")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    app()
