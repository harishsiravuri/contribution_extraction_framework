"""Pipeline orchestrator: Extractor (3 voting samples) → Critic → Consolidator.

All three agents run via the same OpenRouter client. The Extractor's three
samples run concurrently. After they all complete, the Critic runs on the
self-consistency-voted draft, then the Consolidator produces the final record.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from paper1.agents import ConsolidatorAgent, CriticAgent, ExtractorAgent
from paper1.config import Config, load_prompt
from paper1.openrouter import OpenRouterClientProtocol
from paper1.schema import ContributionRecord, ExtractorDraft, RunMeta
from paper1.voting import vote

log = logging.getLogger(__name__)


@dataclass
class PipelineCostTracker:
    """Tracks per-call token usage and dollar cost."""

    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0

    def add(self, tokens_in: int, tokens_out: int, price_in_per_m: float, price_out_per_m: float) -> None:
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.cost_usd += (tokens_in / 1_000_000.0) * price_in_per_m + (
            tokens_out / 1_000_000.0
        ) * price_out_per_m


class Pipeline:
    """Owns the three agents and runs them end-to-end on a single paper."""

    def __init__(self, client: OpenRouterClientProtocol, config: Config) -> None:
        self._client = client
        self._config = config

        self._extractor = ExtractorAgent(
            client, config.extractor, load_prompt("extractor", config.prompts_dir)
        )
        self._critic = CriticAgent(
            client, config.critic, load_prompt("critic", config.prompts_dir)
        )
        self._consolidator = ConsolidatorAgent(
            client, config.consolidator, load_prompt("consolidator", config.prompts_dir)
        )

    async def extract(
        self,
        *,
        paper_id: str,
        paper_text: str,
        retrieval_bundle: str = "",
    ) -> ContributionRecord:
        """Run the full pipeline on one paper."""

        t0 = time.perf_counter()
        cost = PipelineCostTracker()

        # Stage 1: Extractor (parallel temperature samples)
        log.info("[%s] Extractor: running %d voting samples", paper_id, len(self._config.extractor.temperatures or (0.0,)))
        ext_results = await self._extractor.run_voting(
            paper_id=paper_id,
            paper_text=paper_text,
            retrieval_bundle=retrieval_bundle,
        )
        drafts: list[ExtractorDraft] = [d for d, _ in ext_results]
        for _, completion in ext_results:
            cost.add(
                completion.tokens_in,
                completion.tokens_out,
                self._config.extractor.price_in_per_m,
                self._config.extractor.price_out_per_m,
            )

        # Stage 2 (cheap pre-vote): self-consistency voting
        voted = vote(drafts)
        # Build a "voted draft" we can hand to the Critic
        voted_draft = ExtractorDraft(contributions=[c for c, _ in voted])
        log.info("[%s] Voting collapsed %d drafts into %d contribution(s)", paper_id, len(drafts), len(voted))

        # Stage 3: Critic on the voted draft
        log.info("[%s] Critic: verifying voted draft", paper_id)
        critic_out, critic_completion = await self._critic.run(
            paper_id=paper_id,
            paper_text=paper_text,
            draft=voted_draft,
        )
        cost.add(
            critic_completion.tokens_in,
            critic_completion.tokens_out,
            self._config.critic.price_in_per_m,
            self._config.critic.price_out_per_m,
        )

        # Stage 4: Consolidator
        log.info("[%s] Consolidator: producing final record", paper_id)
        record, consolidator_completion = await self._consolidator.run(
            paper_id=paper_id,
            drafts=drafts,
            critic=critic_out,
        )
        cost.add(
            consolidator_completion.tokens_in,
            consolidator_completion.tokens_out,
            self._config.consolidator.price_in_per_m,
            self._config.consolidator.price_out_per_m,
        )

        # Backfill self_consistency from voting if Consolidator omitted it
        for i, contribution in enumerate(record.contributions):
            if contribution.self_consistency == 0.0 and i < len(voted):
                contribution.self_consistency = voted[i][1]

        # Attach meta
        elapsed = time.perf_counter() - t0
        record.meta = RunMeta(
            extractor_model=self._config.extractor.model_id,
            critic_model=self._config.critic.model_id,
            consolidator_model=self._config.consolidator.model_id,
            tokens_in=cost.tokens_in,
            tokens_out=cost.tokens_out,
            cost_usd=cost.cost_usd,
            wall_time_seconds=elapsed,
            voting_samples=len(drafts),
        )

        log.info(
            "[%s] Done in %.1fs ($%.4f, %d in / %d out tokens)",
            paper_id,
            elapsed,
            cost.cost_usd,
            cost.tokens_in,
            cost.tokens_out,
        )
        return record

    async def extract_many(
        self,
        papers: list[tuple[str, str, str]],  # (paper_id, paper_text, retrieval_bundle)
        *,
        max_concurrency: int | None = None,
    ) -> list[ContributionRecord]:
        """Run the pipeline on a batch of papers concurrently.

        max_concurrency caps how many papers are in flight at once. Defaults to
        config.concurrency.papers_in_flight.
        """

        max_concurrency = max_concurrency or self._config.concurrency.papers_in_flight
        sem = asyncio.Semaphore(max_concurrency)

        async def _bounded(pid: str, text: str, bundle: str) -> ContributionRecord:
            async with sem:
                return await self.extract(
                    paper_id=pid, paper_text=text, retrieval_bundle=bundle
                )

        return await asyncio.gather(
            *(_bounded(pid, text, bundle) for pid, text, bundle in papers)
        )
