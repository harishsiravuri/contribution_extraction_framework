"""No-critic ablation pipeline.

Same as the full multi-agent pipeline EXCEPT that the Critic stage is skipped:
  Extractor (3 voting samples) → voting → Consolidator
The Consolidator receives an empty CriticOutput so its prompt template still
parses, but no actual critic verdicts are sent.

Used for the Phase Y2 span-grounding ablation: if Full > NoCritic > Baseline,
the critic is contributing to span grounding.
"""

from __future__ import annotations

import asyncio
import logging
import time

from paper1.agents import ConsolidatorAgent, ExtractorAgent
from paper1.config import Config, load_prompt
from paper1.openrouter import OpenRouterClientProtocol
from paper1.pipeline import PipelineCostTracker
from paper1.schema import ContributionRecord, CriticOutput, ExtractorDraft, RunMeta
from paper1.voting import vote

log = logging.getLogger(__name__)


class NoCriticPipeline:
    """Extractor (3 samples) → vote → Consolidator. No critic in between."""

    def __init__(self, client: OpenRouterClientProtocol, config: Config) -> None:
        self._client = client
        self._config = config
        self._extractor = ExtractorAgent(
            client, config.extractor, load_prompt("extractor", config.prompts_dir)
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
        t0 = time.perf_counter()
        cost = PipelineCostTracker()

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

        voted = vote(drafts)
        log.info(
            "[%s] (no-critic) voting collapsed %d drafts into %d contribution(s)",
            paper_id, len(drafts), len(voted),
        )

        # Empty critic output (preserves the consolidator prompt's expected shape).
        critic_out = CriticOutput()

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

        for i, contribution in enumerate(record.contributions):
            if contribution.self_consistency == 0.0 and i < len(voted):
                contribution.self_consistency = voted[i][1]

        elapsed = time.perf_counter() - t0
        record.meta = RunMeta(
            extractor_model=self._config.extractor.model_id,
            critic_model="(none)",
            consolidator_model=self._config.consolidator.model_id,
            tokens_in=cost.tokens_in,
            tokens_out=cost.tokens_out,
            cost_usd=cost.cost_usd,
            wall_time_seconds=elapsed,
            voting_samples=len(drafts),
        )
        return record

    async def extract_many(
        self,
        papers: list[tuple[str, str, str]],
        *,
        max_concurrency: int | None = None,
    ) -> list[ContributionRecord]:
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
