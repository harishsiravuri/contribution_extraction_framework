"""Single-LLM baseline agent.

Same prompt interface as ExtractorAgent (paper_id, paper_text, retrieval_bundle)
but produces a final-shape ContributionRecord in one LLM call — no critic, no
voting, no consolidation. Used for the "is multi-agent buying us anything?"
comparison.
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
from paper1.schema import ContributionRecord


class SingleLLMAgent:
    """One LLM call → final ContributionRecord."""

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
        retrieval_bundle: str = "",
        temperature: float | None = None,
    ) -> tuple[ContributionRecord, CompletionResult]:
        if temperature is None:
            # Use first temperature in voting list, or 0.0
            temps = self._config.temperatures
            if temps:
                temperature = temps[0]
            elif self._config.temperature is not None:
                temperature = self._config.temperature
            else:
                temperature = 0.0

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
            raise SingleLLMParseError(
                f"Failed to parse SingleLLM JSON output: {e}\nRaw text:\n{result.text[:500]}"
            ) from e

        # Default fields the baseline can't populate
        for c in data.get("contributions", []):
            c.setdefault("self_consistency", 0.5)
            c.setdefault(
                "critic_verdict",
                {
                    "method": "PARTIAL",
                    "task": "PARTIAL",
                    "datasets": "PARTIAL",
                    "metrics": "PARTIAL",
                },
            )

        try:
            record = ContributionRecord.model_validate({"paper_id": paper_id, **data})
        except Exception as e:
            raise SingleLLMParseError(
                f"SingleLLM output failed schema validation: {e}\nParsed JSON:\n{data}"
            ) from e

        return record, result


class SingleLLMParseError(Exception):
    """Raised when the SingleLLM agent's output cannot be parsed/validated."""
