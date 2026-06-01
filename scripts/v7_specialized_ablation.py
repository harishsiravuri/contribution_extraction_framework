"""E4 — Specialized-framework component ablation on SciREX test (n=66).

Uses the existing LoRA-fine-tuned Llama 3.1 70B Instruct extractor
(config/models_ft_v6_70b.yaml). Two ablations:
  (a) voting_off    — extractor at t=0 only (single sample), then
                       Critic + Consolidator as usual.
  (b) critic_off    — extractor at 3 temperatures + voting on,
                       Consolidator receives placeholder all-SUPPORTED
                       verdicts (Critic call skipped).

Reuses the existing full-specialized records from
outputs/paper_data_v6/benchmarks_ft_70b_test/scirex/multi_agent/ for the
'full' baseline (no new spend on that arm).

Outputs:
  outputs/paper_data_v7/specialized_ablation/
    voting_off/<paper>.json
    critic_off/<paper>.json
    extraction_summary.json
    results.json — per-field F1 + bootstrap CIs + paired-perm
                   vs the full-specialized v6 records
    seed.txt
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn,
)

from paper1.agents import ConsolidatorAgent, CriticAgent, ExtractorAgent
from paper1.config import load_config, load_prompt
from paper1.loaders import load_scirex
from paper1.openrouter import OpenRouterClient
from paper1.schema import (
    ContributionRecord, CriticOutput, CriticVerdict,
    ExtractorDraft, RunMeta,
)
from paper1.together_client import RoutingClient, TogetherClient
from paper1.voting import vote, _norm

SEED = 42
random.seed(SEED)

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


def _make_placeholder_critic(voted_draft: ExtractorDraft) -> CriticOutput:
    verdicts: list[CriticVerdict] = []
    for i, c in enumerate(voted_draft.contributions):
        if c.method and c.method.name:
            verdicts.append(CriticVerdict(contribution_index=i, field_path="method.name",
                                          verdict="SUPPORTED", reason="critic-off placeholder"))
        if c.task and c.task.name:
            verdicts.append(CriticVerdict(contribution_index=i, field_path="task.name",
                                          verdict="SUPPORTED", reason="critic-off placeholder"))
        for di, d in enumerate(c.datasets):
            if d.name:
                verdicts.append(CriticVerdict(contribution_index=i, field_path=f"datasets[{di}].name",
                                              verdict="SUPPORTED", reason="critic-off placeholder"))
        for mi, m in enumerate(c.metrics):
            if m.name:
                verdicts.append(CriticVerdict(contribution_index=i, field_path=f"metrics[{mi}].name",
                                              verdict="SUPPORTED", reason="critic-off placeholder"))
    return CriticOutput(verdicts=verdicts, overall_summary="Critic disabled (E4 critic-off ablation).")


async def _voting_off(extractor, critic, consolidator, cfg, paper, out_path):
    """Voting-off: single extractor sample at t=0, then Critic + Consolidator."""
    t0 = time.perf_counter()
    # Single extractor at t=0
    draft, ext_completion = await extractor.run_one(
        paper_id=paper.paper_id, paper_text=paper.full_text,
        retrieval_bundle="", temperature=0.0,
    )
    drafts = [draft]
    tokens_in = ext_completion.tokens_in
    tokens_out = ext_completion.tokens_out
    cost_usd = (ext_completion.tokens_in / 1e6) * cfg.extractor.price_in_per_m \
             + (ext_completion.tokens_out / 1e6) * cfg.extractor.price_out_per_m

    # Stage 3: Critic on the single draft
    critic_out, critic_completion = await critic.run(
        paper_id=paper.paper_id, paper_text=paper.full_text, draft=draft,
    )
    tokens_in += critic_completion.tokens_in
    tokens_out += critic_completion.tokens_out
    cost_usd += (critic_completion.tokens_in / 1e6) * cfg.critic.price_in_per_m \
              + (critic_completion.tokens_out / 1e6) * cfg.critic.price_out_per_m

    record, cons_completion = await consolidator.run(
        paper_id=paper.paper_id, drafts=drafts, critic=critic_out,
    )
    tokens_in += cons_completion.tokens_in
    tokens_out += cons_completion.tokens_out
    cost_usd += (cons_completion.tokens_in / 1e6) * cfg.consolidator.price_in_per_m \
              + (cons_completion.tokens_out / 1e6) * cfg.consolidator.price_out_per_m

    record.meta = RunMeta(
        extractor_model=cfg.extractor.model_id,
        critic_model=cfg.critic.model_id,
        consolidator_model=cfg.consolidator.model_id,
        tokens_in=tokens_in, tokens_out=tokens_out,
        cost_usd=cost_usd, wall_time_seconds=time.perf_counter() - t0,
        voting_samples=1,
    )
    out_path.write_text(record.model_dump_json(indent=2, by_alias=True), encoding="utf-8")
    return cost_usd, tokens_in, tokens_out


async def _critic_off(extractor, consolidator, cfg, paper, out_path):
    """Critic-off: 3 extractor samples + voting → placeholder verdicts → Consolidator."""
    t0 = time.perf_counter()
    ext_results = await extractor.run_voting(
        paper_id=paper.paper_id, paper_text=paper.full_text, retrieval_bundle="",
    )
    drafts = [d for d, _ in ext_results]
    tokens_in = sum(c.tokens_in for _, c in ext_results)
    tokens_out = sum(c.tokens_out for _, c in ext_results)
    cost_usd = sum(
        (c.tokens_in / 1e6) * cfg.extractor.price_in_per_m
        + (c.tokens_out / 1e6) * cfg.extractor.price_out_per_m
        for _, c in ext_results
    )
    voted = vote(drafts)
    voted_draft = ExtractorDraft(contributions=[c for c, _ in voted])
    placeholder = _make_placeholder_critic(voted_draft)
    record, cons_completion = await consolidator.run(
        paper_id=paper.paper_id, drafts=drafts, critic=placeholder,
    )
    tokens_in += cons_completion.tokens_in
    tokens_out += cons_completion.tokens_out
    cost_usd += (cons_completion.tokens_in / 1e6) * cfg.consolidator.price_in_per_m \
              + (cons_completion.tokens_out / 1e6) * cfg.consolidator.price_out_per_m
    for i, contribution in enumerate(record.contributions):
        if contribution.self_consistency == 0.0 and i < len(voted):
            contribution.self_consistency = voted[i][1]
    record.meta = RunMeta(
        extractor_model=cfg.extractor.model_id,
        critic_model="DISABLED (E4 critic-off ablation)",
        consolidator_model=cfg.consolidator.model_id,
        tokens_in=tokens_in, tokens_out=tokens_out,
        cost_usd=cost_usd, wall_time_seconds=time.perf_counter() - t0,
        voting_samples=len(drafts),
    )
    out_path.write_text(record.model_dump_json(indent=2, by_alias=True), encoding="utf-8")
    return cost_usd, tokens_in, tokens_out


async def _process(condition: str, paper, extractor, critic, consolidator,
                   cfg, out_dir, sem, progress, task_id):
    safe = _safe_id(paper.paper_id)
    out_path = out_dir / f"{safe}.json"
    if out_path.exists():
        progress.advance(task_id)
        return PaperResult(paper_id=paper.paper_id, status="skipped", output_file=str(out_path))
    async with sem:
        try:
            if condition == "voting_off":
                cost, ti, to = await _voting_off(extractor, critic, consolidator, cfg, paper, out_path)
            elif condition == "critic_off":
                cost, ti, to = await _critic_off(extractor, consolidator, cfg, paper, out_path)
            else:
                raise ValueError(f"unknown condition {condition}")
            return PaperResult(paper_id=paper.paper_id, status="ok",
                               cost_usd=cost, tokens_in=ti, tokens_out=to,
                               output_file=str(out_path))
        except Exception as e:  # noqa: BLE001
            return PaperResult(paper_id=paper.paper_id, status="error",
                               error=f"{type(e).__name__}: {str(e)[:200]}")
        finally:
            progress.advance(task_id)


@app.command()
def run(
    output_dir: Path = typer.Option(Path("outputs/paper_data_v7/specialized_ablation"), "--output-dir"),
    config_path: Path = typer.Option(Path("config/models_ft_v6_70b.yaml"), "--config"),
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
    tg_key = os.environ.get("TOGETHER_API_KEY", "")
    if not tg_key:
        raise RuntimeError("TOGETHER_API_KEY required for E4")
    tg_client = TogetherClient(api_key=tg_key, timeout_s=cfg.defaults.request_timeout_s)
    client = RoutingClient(or_client, tg_client, together_models={cfg.extractor.model_id})
    console.print(f"  routing extractor {cfg.extractor.model_id} via Together")

    extractor = ExtractorAgent(client, cfg.extractor, load_prompt("extractor", cfg.prompts_dir))
    critic = CriticAgent(client, cfg.critic, load_prompt("critic", cfg.prompts_dir))
    consolidator = ConsolidatorAgent(client, cfg.consolidator, load_prompt("consolidator", cfg.prompts_dir))

    papers = [p for p in load_scirex(splits=("test",)) if p.full_text and len(p.full_text) >= 50][:66]
    console.print(f"  SciREX test papers (n=66 cap): {len(papers)}")

    asyncio.run(_main(papers, extractor, critic, consolidator, cfg, output_dir, concurrency))


async def _main(papers, extractor, critic, consolidator, cfg, output_dir, concurrency):
    sem = asyncio.Semaphore(concurrency)
    all_results = {}
    for condition in ("voting_off", "critic_off"):
        out_d = output_dir / condition
        out_d.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        console.print(f"  [bold]{condition}[/bold] extracting on {len(papers)} papers...")
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                      MofNCompleteColumn(), TimeElapsedColumn(), console=console) as progress:
            task_id = progress.add_task(condition, total=len(papers))
            results = await asyncio.gather(*(
                _process(condition, p, extractor, critic, consolidator, cfg,
                         out_d, sem, progress, task_id) for p in papers
            ))
        ok = sum(1 for r in results if r.status == "ok")
        err = sum(1 for r in results if r.status == "error")
        skip = sum(1 for r in results if r.status == "skipped")
        cost = sum(r.cost_usd for r in results if r.status == "ok")
        elapsed = time.perf_counter() - t0
        console.print(f"    {condition}: ok={ok} err={err} skip={skip} cost=${cost:.4f} wall={elapsed:.1f}s")
        all_results[condition] = {
            "papers_total": len(results), "papers_ok": ok, "papers_error": err,
            "papers_skipped": skip, "total_cost_usd": cost,
            "wall_time_seconds": elapsed, "per_paper": [asdict(r) for r in results],
        }
    (output_dir / "extraction_summary.json").write_text(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    app()
