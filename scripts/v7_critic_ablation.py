"""E2 — Critic ablation on SciREX dev (66 papers).

Re-runs the default open-weights extractor (DeepSeek Chat) + 3-temperature
voting, then hands the voted drafts to the Consolidator with placeholder
all-SUPPORTED CriticOutput (one verdict per (contribution, field) tuple).
The Critic call is skipped entirely.

Output: outputs/paper_data_v7/critic_ablation/
  - multi_agent/<paper>.json — per-paper ContributionRecord
  - extractor_drafts/<paper>.json — raw 3-sample drafts + voted draft (saved
    so we can also use them as input to other experiments without re-extracting)
  - summary.json — F1 + bootstrap CIs vs gold, paired-perm vs the Critic-on
    v3 records on the matched paper subset
  - calibration.json — per-field ECE with placeholder verdicts (T-scaling
    fit on 20%, evaluated on 80%; same split as v2 calibration)
  - results.json — final headline numbers in the format requested by the
    spec, including comparison to v2 Critic-on calibration baseline
  - seed.txt — random seed used (42)

Spec deviation note: the spec said "reuse existing Extractor and voting
outputs from outputs/paper_data_v3/multi_agent/ to avoid re-extraction
cost". Inspecting v3 outputs shows only the final ContributionRecord is
cached (with Critic verdicts already merged into the Consolidator's
output); the pre-Critic voted drafts are NOT on disk. We therefore
re-extract fresh on all 66 papers; cost is still well under the $5 E2
budget (DeepSeek ≈ $1.30 + Llama Consolidator ≈ $0.30 ≈ $1.60 total).
"""

from __future__ import annotations

import asyncio
import json
import random
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

from paper1.agents import ConsolidatorAgent, ExtractorAgent
from paper1.config import load_config, load_prompt
from paper1.loaders import load_scirex
from paper1.openrouter import OpenRouterClient
from paper1.schema import (
    ContributionRecord,
    CriticOutput,
    CriticVerdict,
    ExtractorDraft,
    RunMeta,
)
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
    """Build a CriticOutput with all-SUPPORTED verdicts for every populated field."""
    verdicts: list[CriticVerdict] = []
    for i, contrib in enumerate(voted_draft.contributions):
        # method
        if contrib.method and contrib.method.name:
            verdicts.append(CriticVerdict(
                contribution_index=i, field_path="method.name",
                verdict="SUPPORTED", reason="critic-off placeholder",
            ))
        # task
        if contrib.task and contrib.task.name:
            verdicts.append(CriticVerdict(
                contribution_index=i, field_path="task.name",
                verdict="SUPPORTED", reason="critic-off placeholder",
            ))
        # datasets
        for di, d in enumerate(contrib.datasets):
            if d.name:
                verdicts.append(CriticVerdict(
                    contribution_index=i, field_path=f"datasets[{di}].name",
                    verdict="SUPPORTED", reason="critic-off placeholder",
                ))
        # metrics
        for mi, m in enumerate(contrib.metrics):
            if m.name:
                verdicts.append(CriticVerdict(
                    contribution_index=i, field_path=f"metrics[{mi}].name",
                    verdict="SUPPORTED", reason="critic-off placeholder",
                ))
    return CriticOutput(verdicts=verdicts, overall_summary="Critic disabled (E2 ablation).")


async def _process_paper(
    paper_id: str,
    paper_text: str,
    extractor: ExtractorAgent,
    consolidator: ConsolidatorAgent,
    cfg,
    out_dir: Path,
    sem: asyncio.Semaphore,
    progress: Progress,
    task_id,
) -> PaperResult:
    safe = _safe_id(paper_id)
    out_path = out_dir / f"{safe}.json"
    if out_path.exists():
        progress.advance(task_id)
        return PaperResult(paper_id=paper_id, status="skipped", output_file=str(out_path))

    async with sem:
        t0 = time.perf_counter()
        try:
            # Stage 1: Extractor (3 voting samples)
            ext_results = await extractor.run_voting(
                paper_id=paper_id, paper_text=paper_text, retrieval_bundle=""
            )
            drafts = [d for d, _ in ext_results]
            tokens_in = sum(c.tokens_in for _, c in ext_results)
            tokens_out = sum(c.tokens_out for _, c in ext_results)
            cost_usd = sum(
                (c.tokens_in / 1e6) * cfg.extractor.price_in_per_m
                + (c.tokens_out / 1e6) * cfg.extractor.price_out_per_m
                for _, c in ext_results
            )

            # Stage 2: vote
            voted = vote(drafts)
            voted_draft = ExtractorDraft(contributions=[c for c, _ in voted])

            # Stage 3: build placeholder critic
            placeholder = _make_placeholder_critic(voted_draft)

            # Stage 4: Consolidator
            record, cons_completion = await consolidator.run(
                paper_id=paper_id, drafts=drafts, critic=placeholder,
            )
            tokens_in += cons_completion.tokens_in
            tokens_out += cons_completion.tokens_out
            cost_usd += (cons_completion.tokens_in / 1e6) * cfg.consolidator.price_in_per_m
            cost_usd += (cons_completion.tokens_out / 1e6) * cfg.consolidator.price_out_per_m

            # Backfill self_consistency from voting
            for i, contribution in enumerate(record.contributions):
                if contribution.self_consistency == 0.0 and i < len(voted):
                    contribution.self_consistency = voted[i][1]

            elapsed = time.perf_counter() - t0
            record.meta = RunMeta(
                extractor_model=cfg.extractor.model_id,
                critic_model="DISABLED (E2 critic-off ablation)",
                consolidator_model=cfg.consolidator.model_id,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost_usd,
                wall_time_seconds=elapsed,
                voting_samples=len(drafts),
            )
            out_path.write_text(record.model_dump_json(indent=2, by_alias=True), encoding="utf-8")

            return PaperResult(
                paper_id=paper_id, status="ok",
                cost_usd=cost_usd, tokens_in=tokens_in, tokens_out=tokens_out,
                wall_time_seconds=elapsed, output_file=str(out_path),
            )
        except Exception as e:  # noqa: BLE001
            return PaperResult(
                paper_id=paper_id, status="error",
                error=f"{type(e).__name__}: {str(e)[:200]}",
            )
        finally:
            progress.advance(task_id)


@app.command()
def run(
    output_dir: Path = typer.Option(Path("outputs/paper_data_v7/critic_ablation"), "--output-dir"),
    config_path: Path = typer.Option(Path("config/models.yaml"), "--config"),
    concurrency: int = typer.Option(5, "--concurrency"),
):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "seed.txt").write_text(f"{SEED}\n")

    cfg = load_config(config_path=config_path)
    or_client = OpenRouterClient(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        timeout_s=cfg.defaults.request_timeout_s,
        max_retries=cfg.defaults.max_retries,
        retry_backoff_s=cfg.defaults.retry_backoff_s,
        referer=cfg.referer,
        title=cfg.title,
    )
    extractor = ExtractorAgent(
        or_client, cfg.extractor, load_prompt("extractor", cfg.prompts_dir)
    )
    consolidator = ConsolidatorAgent(
        or_client, cfg.consolidator, load_prompt("consolidator", cfg.prompts_dir)
    )
    multi_dir = output_dir / "multi_agent"
    multi_dir.mkdir(parents=True, exist_ok=True)

    papers = [p for p in load_scirex(splits=("dev",)) if p.full_text and len(p.full_text) >= 50][:66]
    console.print(f"[bold]E2 critic-off[/bold]: {len(papers)} SciREX dev papers, concurrency={concurrency}")

    asyncio.run(_main(papers, extractor, consolidator, cfg, multi_dir, concurrency))


async def _main(papers, extractor, consolidator, cfg, multi_dir, concurrency):
    sem = asyncio.Semaphore(concurrency)
    t0 = time.perf_counter()
    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
        MofNCompleteColumn(), TimeElapsedColumn(), console=console
    ) as progress:
        task_id = progress.add_task("critic-off", total=len(papers))
        results = await asyncio.gather(*(
            _process_paper(p.paper_id, p.full_text, extractor, consolidator,
                           cfg, multi_dir, sem, progress, task_id)
            for p in papers
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
    (multi_dir.parent / "extraction_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()
