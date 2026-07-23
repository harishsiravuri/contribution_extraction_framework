"""v11 Condition A — Single-LLM baseline on SciREX test.

Same prompt as the multi-agent Extractor (config/prompts/extractor.md).
Single extractor call per paper at temperature 0. No Critic, no Consolidator.

Isolates: whether the dev-split multi-agent gap over the single-LLM
baseline persists on the held-out test split.

Output: outputs/paper_data_v11/baseline_test/
  records/<paper>.json — one ContributionRecord per paper
  extraction_summary.json
  run.log (via stdout redirect)
  seed.txt (42)
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn,
)

from paper1.agents.extractor import ExtractorAgent
from paper1.config import load_config, load_prompt
from paper1.loaders import load_scirex
from paper1.openrouter import OpenRouterClient
from paper1.schema import ContributionRecord, ContributionUnit, MetricEntity, NamedEntity, RunMeta

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

COST_CAP_USD = 10.0
LOG_EVERY = 10  # papers

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


def _draft_to_record(paper_id: str, draft, cfg, tokens_in: int, tokens_out: int,
                     cost_usd: float, wall: float) -> ContributionRecord:
    """Wrap a raw ExtractorDraft into a ContributionRecord (no critic/consolidator)."""
    contribs: list[ContributionUnit] = []
    for d in draft.contributions:
        method = NamedEntity(
            name=(d.method.name if d.method else None),
            canonical_id=(d.method.canonical_id if d.method else None),
            evidence_span=(d.method.evidence_span if d.method else None),
            confidence=1.0,
        )
        task = NamedEntity(
            name=(d.task.name if d.task else None),
            canonical_id=(d.task.canonical_id if d.task else None),
            evidence_span=(d.task.evidence_span if d.task else None),
            confidence=1.0,
        )
        datasets = [
            NamedEntity(name=ds.name, canonical_id=ds.canonical_id,
                        evidence_span=ds.evidence_span, confidence=1.0)
            for ds in d.datasets if ds.name
        ]
        metrics = [
            MetricEntity(name=m.name, value=m.value, unit=m.unit,
                         evidence_span=m.evidence_span, confidence=1.0)
            for m in d.metrics if m.name
        ]
        contribs.append(ContributionUnit(
            method=method, task=task, datasets=datasets, metrics=metrics,
            claim_strength=d.claim_strength,
            comparison_targets=list(d.comparison_targets or []),
            self_consistency=1.0,  # single sample, no voting
        ))
    return ContributionRecord(
        paper_id=paper_id,
        contributions=contribs,
        _meta=RunMeta(
            extractor_model=cfg.extractor.model_id,
            critic_model="DISABLED (Condition A: single-LLM baseline, extractor prompt @ t=0)",
            consolidator_model="DISABLED (Condition A: single-LLM baseline)",
            tokens_in=tokens_in, tokens_out=tokens_out,
            cost_usd=cost_usd, wall_time_seconds=wall,
            voting_samples=1,
        ),
    )


async def _run_one(extractor, paper, cfg, records_dir, sem, progress, task_id,
                   spend_state: dict) -> PaperResult:
    safe = _safe_id(paper.paper_id)
    out_path = records_dir / f"{safe}.json"
    if out_path.exists():
        progress.advance(task_id)
        return PaperResult(paper_id=paper.paper_id, status="skipped", output_file=str(out_path))
    if spend_state["stopped"]:
        progress.advance(task_id)
        return PaperResult(paper_id=paper.paper_id, status="skipped",
                           error="cost cap tripped before this paper")
    async with sem:
        if spend_state["stopped"]:
            progress.advance(task_id)
            return PaperResult(paper_id=paper.paper_id, status="skipped",
                               error="cost cap tripped before this paper")
        t0 = time.perf_counter()
        try:
            draft, completion = await extractor.run_one(
                paper_id=paper.paper_id,
                paper_text=paper.full_text,
                retrieval_bundle="",
                temperature=0.0,
            )
            cost_usd = ((completion.tokens_in / 1e6) * cfg.extractor.price_in_per_m
                        + (completion.tokens_out / 1e6) * cfg.extractor.price_out_per_m)
            spend_state["total_cost"] += cost_usd
            spend_state["done"] += 1
            wall = time.perf_counter() - t0
            record = _draft_to_record(paper.paper_id, draft, cfg,
                                      completion.tokens_in, completion.tokens_out,
                                      cost_usd, wall)
            out_path.write_text(record.model_dump_json(indent=2, by_alias=True), encoding="utf-8")
            # Periodic checkpoint
            if spend_state["done"] % LOG_EVERY == 0:
                projected = (spend_state["total_cost"] / spend_state["done"]) * spend_state["total_papers"]
                console.print(f"    [checkpoint] papers_done={spend_state['done']}  "
                              f"cost_so_far=${spend_state['total_cost']:.4f}  "
                              f"projected_total=${projected:.4f}")
                if projected > COST_CAP_USD:
                    console.print(f"[red]COST CAP TRIPPED: projected ${projected:.4f} > cap ${COST_CAP_USD}[/red]")
                    spend_state["stopped"] = True
            return PaperResult(
                paper_id=paper.paper_id, status="ok",
                cost_usd=cost_usd, tokens_in=completion.tokens_in,
                tokens_out=completion.tokens_out, wall_time_seconds=wall,
                output_file=str(out_path),
            )
        except Exception as e:  # noqa: BLE001
            return PaperResult(paper_id=paper.paper_id, status="error",
                               error=f"{type(e).__name__}: {str(e)[:200]}")
        finally:
            progress.advance(task_id)


@app.command()
def run(
    output_dir: Path = typer.Option(Path("outputs/paper_data_v11/baseline_test"), "--output-dir"),
    config_path: Path = typer.Option(Path("config/models.yaml"), "--config"),
    concurrency: int = typer.Option(5, "--concurrency"),
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
    # KEY POINT: use extractor.md, NOT single_llm.md, per user spec
    prompt = load_prompt("extractor", cfg.prompts_dir)
    extractor = ExtractorAgent(or_client, cfg.extractor, prompt)

    papers = [p for p in load_scirex(splits=("test",)) if p.full_text and len(p.full_text) >= 50][:66]
    console.print(f"Condition A single-LLM: {len(papers)} SciREX test papers, "
                  f"extractor prompt (extractor.md) @ t=0.0, concurrency={concurrency}")
    console.print(f"  extractor: {cfg.extractor.model_id}")
    console.print(f"  cost cap: ${COST_CAP_USD}")

    records_dir = output_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    asyncio.run(_main(papers, extractor, cfg, records_dir, output_dir, concurrency))


async def _main(papers, extractor, cfg, records_dir, output_dir, concurrency):
    sem = asyncio.Semaphore(concurrency)
    spend_state = {"total_cost": 0.0, "done": 0, "total_papers": len(papers), "stopped": False}
    t0 = time.perf_counter()
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                  MofNCompleteColumn(), TimeElapsedColumn(), console=console) as progress:
        task_id = progress.add_task("condition-A", total=len(papers))
        results = await asyncio.gather(*(
            _run_one(extractor, p, cfg, records_dir, sem, progress, task_id, spend_state)
            for p in papers
        ))
    ok = sum(1 for r in results if r.status == "ok")
    err = sum(1 for r in results if r.status == "error")
    skip = sum(1 for r in results if r.status == "skipped")
    cost = sum(r.cost_usd for r in results if r.status == "ok")
    tokens_in = sum(r.tokens_in for r in results if r.status == "ok")
    tokens_out = sum(r.tokens_out for r in results if r.status == "ok")
    elapsed = time.perf_counter() - t0
    console.print(f"[bold]Condition A done: ok={ok} err={err} skip={skip} "
                  f"cost=${cost:.4f} wall={elapsed:.1f}s "
                  f"tokens_in={tokens_in} tokens_out={tokens_out} calls={ok}[/bold]")
    summary = {
        "condition": "A_single_llm_baseline",
        "prompt": "config/prompts/extractor.md",
        "temperature": 0.0,
        "extractor_model": "deepseek/deepseek-chat",
        "seed": SEED,
        "papers_total": len(results),
        "papers_ok": ok, "papers_error": err, "papers_skipped": skip,
        "total_cost_usd": cost,
        "total_tokens_in": tokens_in,
        "total_tokens_out": tokens_out,
        "llm_calls": ok,
        "wall_time_seconds": elapsed,
        "cost_cap_tripped": spend_state["stopped"],
        "per_paper": [asdict(r) for r in results],
    }
    (output_dir / "extraction_summary.json").write_text(json.dumps(summary, indent=2))
    console.print(f"  wrote {output_dir / 'extraction_summary.json'}")


if __name__ == "__main__":
    app()
