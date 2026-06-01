"""Consolidator agent.

Takes the (already self-consistency-voted) extractor drafts plus the critic's
verdicts and produces the final ContributionRecord. Uses an LLM call to do
semantic merging (canonical-form preference, name disambiguation) that the
plain voting layer can't.
"""

from __future__ import annotations

import json

from paper1.agents.extractor import _split_prompt
from paper1.config import AgentModelConfig
from paper1.openrouter import (
    CompletionResult,
    OpenRouterClientProtocol,
    parse_json_response,
)
from paper1.schema import ContributionRecord, CriticOutput, ExtractorDraft


class ConsolidatorAgent:
    """Wraps the LLM call for the Consolidator role."""

    def __init__(
        self,
        client: OpenRouterClientProtocol,
        config: AgentModelConfig,
        prompt_template: str,
    ) -> None:
        self._client = client
        self._config = config
        self._system, self._user_template = _split_prompt(prompt_template)

    async def run(
        self,
        *,
        paper_id: str,
        drafts: list[ExtractorDraft],
        critic: CriticOutput,
    ) -> tuple[ContributionRecord, CompletionResult]:
        """Run the Consolidator and parse its output into a ContributionRecord."""

        drafts_json = json.dumps(
            [d.model_dump() for d in drafts], indent=2, ensure_ascii=False
        )
        critic_json = json.dumps(critic.model_dump(), indent=2, ensure_ascii=False)

        user = self._user_template.format(
            paper_id=paper_id,
            drafts_json=drafts_json,
            critic_json=critic_json,
        )

        temperature = self._config.temperature if self._config.temperature is not None else 0.0
        result = await self._client.complete(
            model_id=self._config.model_id,
            system=self._system,
            user=user,
            temperature=temperature,
            max_tokens=self._config.max_tokens,
            top_p=self._config.top_p,
        )

        try:
            data = parse_json_response(result.text)
        except (json.JSONDecodeError, ValueError) as e:
            raise ConsolidatorParseError(
                f"Failed to parse Consolidator JSON output: {e}\nRaw text:\n{result.text[:500]}"
            ) from e

        # The Consolidator's prompt asks for the final-shape object; meta is filled in by Pipeline.
        try:
            # Drop any unknown fields the model may have included
            record = ContributionRecord.model_validate({"paper_id": paper_id, **data})
        except Exception as e:
            raise ConsolidatorParseError(
                f"Consolidator output failed schema validation: {e}\nParsed JSON:\n{data}"
            ) from e

        return record, result


class ConsolidatorParseError(Exception):
    """Raised when the Consolidator's output cannot be parsed/validated."""
