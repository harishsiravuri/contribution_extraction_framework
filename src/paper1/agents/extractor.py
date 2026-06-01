"""Extractor agent.

Reads paper text + retrieval bundle and emits a draft contribution-unit JSON.
Run multiple times at different temperatures to enable self-consistency voting.
"""

from __future__ import annotations

import asyncio
import json

from paper1.config import AgentModelConfig
from paper1.openrouter import (
    CompletionResult,
    OpenRouterClientProtocol,
    parse_json_response,
)
from paper1.schema import ExtractorDraft


def _split_prompt(template: str) -> tuple[str, str]:
    """Split a prompt template into (system, user) by 'SYSTEM:' / 'USER:' markers."""

    if "USER:" not in template:
        raise ValueError("Prompt template missing 'USER:' marker")
    system_part, user_part = template.split("USER:", 1)
    if "SYSTEM:" in system_part:
        system_part = system_part.split("SYSTEM:", 1)[1]
    return system_part.strip(), user_part.strip()


class ExtractorAgent:
    """Wraps the LLM call for the Extractor role."""

    def __init__(
        self,
        client: OpenRouterClientProtocol,
        config: AgentModelConfig,
        prompt_template: str,
    ) -> None:
        self._client = client
        self._config = config
        self._system, self._user_template = _split_prompt(prompt_template)

    async def run_one(
        self,
        *,
        paper_id: str,
        paper_text: str,
        retrieval_bundle: str = "",
        temperature: float | None = None,
    ) -> tuple[ExtractorDraft, CompletionResult]:
        """Run the Extractor once at the given temperature."""

        if temperature is None:
            # Default to first temperature in the voting list
            temps = self._config.temperatures or (0.0,)
            temperature = temps[0]

        user = self._user_template.format(
            paper_id=paper_id,
            paper_text=paper_text,
            retrieval_bundle=retrieval_bundle or "(none)",
        )

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
            raise ExtractorParseError(
                f"Failed to parse Extractor JSON output: {e}\nRaw text:\n{result.text[:500]}"
            ) from e

        try:
            draft = ExtractorDraft.model_validate(data)
        except Exception as e:
            raise ExtractorParseError(
                f"Extractor output failed schema validation: {e}\nParsed JSON:\n{data}"
            ) from e

        return draft, result

    async def run_voting(
        self,
        *,
        paper_id: str,
        paper_text: str,
        retrieval_bundle: str = "",
    ) -> list[tuple[ExtractorDraft, CompletionResult]]:
        """Run the Extractor at every configured temperature, in parallel."""

        temps = self._config.temperatures or (0.0,)
        tasks = [
            self.run_one(
                paper_id=paper_id,
                paper_text=paper_text,
                retrieval_bundle=retrieval_bundle,
                temperature=t,
            )
            for t in temps
        ]
        return await asyncio.gather(*tasks)


class ExtractorParseError(Exception):
    """Raised when the Extractor's output cannot be parsed/validated."""
