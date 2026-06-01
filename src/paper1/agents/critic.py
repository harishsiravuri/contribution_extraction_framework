"""Critic agent.

Given the paper text and a draft extraction, returns per-field verdicts in
{SUPPORTED, PARTIAL, UNSUPPORTED} with corrected evidence spans where possible.
"""

from __future__ import annotations

import json

from paper1.agents.extractor import _split_prompt  # reuse splitter
from paper1.config import AgentModelConfig
from paper1.openrouter import (
    CompletionResult,
    OpenRouterClientProtocol,
    parse_json_response,
)
from paper1.schema import CriticOutput, ExtractorDraft


class CriticAgent:
    """Wraps the LLM call for the Critic role."""

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
        paper_text: str,
        draft: ExtractorDraft,
    ) -> tuple[CriticOutput, CompletionResult]:
        """Run the Critic on a single draft."""

        draft_json = json.dumps(draft.model_dump(), indent=2, ensure_ascii=False)
        user = self._user_template.format(
            paper_id=paper_id,
            paper_text=paper_text,
            draft_extraction=draft_json,
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
            raise CriticParseError(
                f"Failed to parse Critic JSON output: {e}\nRaw text:\n{result.text[:500]}"
            ) from e

        try:
            critic_out = CriticOutput.model_validate(data)
        except Exception as e:
            raise CriticParseError(
                f"Critic output failed schema validation: {e}\nParsed JSON:\n{data}"
            ) from e

        return critic_out, result


class CriticParseError(Exception):
    """Raised when the Critic's output cannot be parsed/validated."""
