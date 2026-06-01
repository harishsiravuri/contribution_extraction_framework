"""Single-LLM baseline pipeline.

Mirrors the .extract() / .extract_many() interface of paper1.pipeline.Pipeline,
but uses one SingleLLMAgent call per paper (no critic, no voting, no
consolidation). Tracks tokens/cost in the same shape so the runner can
aggregate uniformly.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from paper1.agents.single_llm import SingleLLMAgent
from paper1.config import Config, load_prompt
from paper1.openrouter import OpenRouterClientProtocol
from paper1.pipeline import PipelineCostTracker
from paper1.schema import ContributionRecord, RunMeta

log = logging.getLogger(__name__)


def _load_baseline_prompt(prompts_dir: Path) -> str:
    return load_prompt("single_llm", prompts_dir)


class BaselinePipeline:
    """Single-LLM baseline. Same interface as Pipeline."""

    def __init__(self, client: OpenRouterClientProtocol, config: Config) -> None:
        self._client = client
        self._config = config
        prompt = _load_baseline_prompt(config.prompts_dir)
        # Use the extractor's model config so the comparison is "same model, fewer agents".
        self._agent = SingleLLMAgent(client, config.extractor, prompt)

    async def extract(
        self,
        *,
        paper_id: str,
        paper_text: str,
        retrieval_bundle: str = "",
    ) -> ContributionRecord:
        t0 = time.perf_counter()
        cost = PipelineCostTracker()

        record, completion = await self._agent.run(
            paper_id=paper_id,
            paper_text=paper_text,
            retrieval_bundle=retrieval_bundle,
        )
        cost.add(
            completion.tokens_in,
            completion.tokens_out,
            self._config.extractor.price_in_per_m,
            self._config.extractor.price_out_per_m,
        )

        elapsed = time.perf_counter() - t0
        record.meta = RunMeta(
            extractor_model=self._config.extractor.model_id,
            critic_model="(none)",
            consolidator_model="(none)",
            tokens_in=cost.tokens_in,
            tokens_out=cost.tokens_out,
            cost_usd=cost.cost_usd,
            wall_time_seconds=elapsed,
            voting_samples=1,
        )

        log.info(
            "[%s] baseline done in %.1fs ($%.4f, %d in / %d out)",
            paper_id,
            elapsed,
            cost.cost_usd,
            cost.tokens_in,
            cost.tokens_out,
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
