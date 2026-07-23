"""v11 Condition B — Cost-matched self-consistency baseline on SciREX test.

Same prompt as the multi-agent Extractor (config/prompts/extractor.md).
Five extractor samples per paper at temperatures [0.0, 0.3, 0.3, 0.7, 0.7],
then majority-vote via paper1.voting.vote() over the ExtractorDraft objects.
No Critic, no Consolidator.

Matches the multi-agent pipeline on total LLM calls per paper (3 extractor
+ 1 critic + 1 consolidator = 5 → 5 extractor samples here). Isolates the
multi-agent structure from raw compute.

Output: outputs/paper_data_v11/selfconsistency_test/
  records/<paper>.json — one ContributionRecord per paper (post-vote)
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
from paper1.voting import vote

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Temperature schedule for self-consistency: 5 samples matching the
# multi-agent pipeline's 5 total LLM calls per paper (3 extractor + critic +
# consolidator). Same base temps as the pipeline (0, 0.3, 0.7), with an
# extra repeat of each non-zero temp so mid-temp mass mirrors the pipeline.
SC_TEMPERATURES = [0.0, 0.3, 0.3, 0.7, 0.7]

COST_CAP_USD = 10.0
LOG_EVERY = 10

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
    n_samples: int = 0


def _safe_id(paper_id: str) -> str:
    return paper_id.replace(":", "__").replace("/", "_")


def _voted_to_record(paper_id: str, voted, cfg, tokens_in: int, tokens_out: int,
                     cost_usd: float, wall: float, n_samples: int) -> ContributionRecord:
    """Wrap voted contributions into a ContributionRecord."""
    contribs: list[ContributionUnit] = []
    for c, sc in voted:
        method = NamedEntity(
            name=c.method.name if c.method else None,
            canonical_id=c.method.canonical_id if c.method else None,
            evidence_span=c.method.evidence_span if c.method else None,
            confidence=1.0,
        )
        task = NamedEntity(
            name=c.task.name if c.task else None,
            canonical_id=c.task.canonical_id if c.task else None,
            evidence_span=c.task.evidence_span if c.task else None,
            confidence=1.0,
        )
        datasets = [
            NamedEntity(name=d.name, canonical_id=d.canonical_id,
                        evidence_span=d.evidence_span, confidence=1.0)
            for d in c.datasets if d.name
        ]
        metrics = [
            MetricEntity(name=m.name, value=m.value, unit=m.unit,
                         evidence_span=m.evidence_span, confidence=1.0)
            for m in c.metrics if m.name
        ]
        contribs.append(ContributionUnit(
            method=method, task=task, datasets=datasets, metrics=metrics,
            claim_strength=c.claim_strength,
            comparison_targets=list(c.comparison_targets or []),
            self_consistency=float(sc),
        ))
    return ContributionRecord(
        paper_id=paper_id,
        contributions=contribs,
        _meta=RunMeta(
            extractor_model=cfg.extractor.model_id,
            critic_model=f"DISABLED (Condition B: self-consistency vote of {n_samples} extractor samples)",
            consolidator_model="DISABLED (Condition B: self-consistency baseline)",
            tokens_in=tokens_in, tokens_out=tokens_out,
            cost_usd=cost_usd, wall_time_seconds=wall,
            voting_samples=n_samples,
        ),
    )


async def _run_one(extractor, paper, cfg, records_dir, sem, progress, task_id,
                   spend_state: dict) -> PaperResult:
    safe = _safe_id(paper.paper_id)
    out_path = records_dir / f"{safe}.json"
    if out_path.exists():
        progress.advance(task_id)
        return PaperResult(paper_id=paper.paper_id, status="skipped",
                           output_file=str(out_path))
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
            # Fire all 5 temperature samples concurrently for this paper
            sample_tasks = [
                extractor.run_one(
                    paper_id=paper.paper_id,
                    paper_text=paper.full_text,
                    retrieval_bundle="",
                    temperature=t,
                )
                for t in SC_TEMPERATURES
            ]
            sample_results = await asyncio.gather(*sample_tasks)
            drafts = [d for d, _ in sample_results]
            tokens_in = sum(c.tokens_in for _, c in sample_results)
            tokens_out = sum(c.tokens_out for _, c in sample_results)
            cost_usd = sum(
                (c.tokens_in / 1e6) * cfg.extractor.price_in_per_m
                + (c.tokens_out / 1e6) * cfg.extractor.price_out_per_m
                for _, c in sample_results
            )
            spend_state["total_cost"] += cost_usd
            spend_state["done"] += 1
            wall = time.perf_counter() - t0

            # Majority-vote via the same paper1.voting.vote used inside the pipeline
            voted = vote(drafts)

            record = _voted_to_record(paper.paper_id, voted, cfg,
                                      tokens_in, tokens_out, cost_usd, wall,
                                      n_samples=len(drafts))
            out_path.write_text(record.model_dump_json(indent=2, by_alias=True), encoding="utf-8")

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
                cost_usd=cost_usd, tokens_in=tokens_in, tokens_out=tokens_out,
                wall_time_seconds=wall, output_file=str(out_path),
                n_samples=len(drafts),
            )
        except Exception as e:  # noqa: BLE001
            return PaperResult(paper_id=paper.paper_id, status="error",
                               error=f"{type(e).__name__}: {str(e)[:200]}")
        finally:
            progress.advance(task_id)


@app.command()
def run(
    output_dir: Path = typer.Option(Path("outputs/paper_data_v11/selfconsistency_test"), "--output-dir"),
    config_path: Path = typer.Option(Path("config/models.yaml"), "--config"),
    concurrency: int = typer.Option(3, "--concurrency"),
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
    prompt = load_prompt("extractor", cfg.prompts_dir)
    extractor = ExtractorAgent(or_client, cfg.extractor, prompt)

    papers = [p for p in load_scirex(splits=("test",)) if p.full_text and len(p.full_text) >= 50][:66]
    console.print(f"Condition B self-consistency: {len(papers)} SciREX test papers, "
                  f"extractor prompt (extractor.md) × {len(SC_TEMPERATURES)} samples "
                  f"@ temperatures={SC_TEMPERATURES}, concurrency={concurrency}")
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
        task_id = progress.add_task("condition-B", total=len(papers))
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
    llm_calls = sum(r.n_samples for r in results if r.status == "ok")
    elapsed = time.perf_counter() - t0
    console.print(f"[bold]Condition B done: ok={ok} err={err} skip={skip} "
                  f"cost=${cost:.4f} wall={elapsed:.1f}s "
                  f"tokens_in={tokens_in} tokens_out={tokens_out} calls={llm_calls}[/bold]")
    summary = {
        "condition": "B_self_consistency_baseline",
        "prompt": "config/prompts/extractor.md",
        "temperatures": SC_TEMPERATURES,
        "samples_per_paper": len(SC_TEMPERATURES),
        "extractor_model": "deepseek/deepseek-chat",
        "seed": SEED,
        "papers_total": len(results),
        "papers_ok": ok, "papers_error": err, "papers_skipped": skip,
        "total_cost_usd": cost,
        "total_tokens_in": tokens_in,
        "total_tokens_out": tokens_out,
        "llm_calls": llm_calls,
        "wall_time_seconds": elapsed,
        "cost_cap_tripped": spend_state["stopped"],
        "per_paper": [asdict(r) for r in results],
    }
    (output_dir / "extraction_summary.json").write_text(json.dumps(summary, indent=2))
    console.print(f"  wrote {output_dir / 'extraction_summary.json'}")


if __name__ == "__main__":
    app()
