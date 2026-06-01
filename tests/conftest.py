"""Shared test fixtures, including a MockOpenRouterClient.

The mock returns canned responses keyed by model_id so the full pipeline can
run end-to-end with no network access and no API key.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from paper1.config import (
    AgentModelConfig,
    Concurrency,
    Config,
    Defaults,
    DEFAULT_PROMPTS_DIR,
)
from paper1.openrouter import CompletionResult, OpenRouterClientProtocol


class MockOpenRouterClient:
    """Returns pre-canned responses based on model_id and (optionally) call index."""

    def __init__(self, responses: dict[str, list[str] | Callable[[int], str]]):
        self._responses = responses
        self._call_counts: dict[str, int] = {}

    async def complete(
        self,
        *,
        model_id: str,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
        top_p: float = 0.95,
    ) -> CompletionResult:
        idx = self._call_counts.get(model_id, 0)
        self._call_counts[model_id] = idx + 1

        spec = self._responses.get(model_id)
        if spec is None:
            raise RuntimeError(f"MockOpenRouterClient: no response configured for {model_id}")
        if callable(spec):
            text = spec(idx)
        else:
            text = spec[idx % len(spec)]

        # Synthetic token counts: rough proxy
        tokens_in = max(1, len(user) // 4)
        tokens_out = max(1, len(text) // 4)

        return CompletionResult(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model_id=model_id,
            raw={"choices": [{"message": {"content": text}}]},
        )

    async def aclose(self) -> None:
        pass


@pytest.fixture
def fake_extractor_response() -> str:
    """A minimal but schema-valid Extractor JSON response."""

    return json.dumps(
        {
            "contributions": [
                {
                    "method": {
                        "name": "BERT",
                        "canonical_id": "pwc:bert",
                        "evidence_span": {"start": 100, "end": 130},
                    },
                    "task": {
                        "name": "question answering",
                        "canonical_id": None,
                        "evidence_span": {"start": 50, "end": 80},
                    },
                    "datasets": [
                        {
                            "name": "SQuAD",
                            "canonical_id": "pwc:squad",
                            "evidence_span": {"start": 200, "end": 215},
                        }
                    ],
                    "metrics": [
                        {
                            "name": "F1",
                            "value": 88.5,
                            "unit": "F1",
                            "evidence_span": {"start": 300, "end": 320},
                        }
                    ],
                    "claim_strength": "improves",
                    "comparison_targets": ["openalex:W2742617811"],
                    "notes": None,
                }
            ]
        }
    )


@pytest.fixture
def fake_critic_response() -> str:
    return json.dumps(
        {
            "verdicts": [
                {
                    "contribution_index": 0,
                    "field_path": "method.name",
                    "verdict": "SUPPORTED",
                    "reason": "Method clearly named in section 3.",
                    "corrected_evidence_span": None,
                },
                {
                    "contribution_index": 0,
                    "field_path": "datasets[0].name",
                    "verdict": "SUPPORTED",
                    "reason": "SQuAD mentioned in experiments.",
                    "corrected_evidence_span": None,
                },
                {
                    "contribution_index": 0,
                    "field_path": "metrics[0].value",
                    "verdict": "PARTIAL",
                    "reason": "Value present but unit ambiguous.",
                    "corrected_evidence_span": None,
                },
            ],
            "overall_summary": "Extraction is broadly correct with one partial match.",
        }
    )


@pytest.fixture
def fake_consolidator_response() -> str:
    return json.dumps(
        {
            "contributions": [
                {
                    "method": {
                        "name": "BERT",
                        "canonical_id": "pwc:bert",
                        "evidence_span": {"start": 100, "end": 130},
                        "confidence": 1.0,
                    },
                    "task": {
                        "name": "question answering",
                        "canonical_id": None,
                        "evidence_span": {"start": 50, "end": 80},
                        "confidence": 1.0,
                    },
                    "datasets": [
                        {
                            "name": "SQuAD",
                            "canonical_id": "pwc:squad",
                            "evidence_span": {"start": 200, "end": 215},
                            "confidence": 1.0,
                        }
                    ],
                    "metrics": [
                        {
                            "name": "F1",
                            "value": 88.5,
                            "unit": "F1",
                            "evidence_span": {"start": 300, "end": 320},
                            "confidence": 0.67,
                        }
                    ],
                    "claim_strength": "improves",
                    "comparison_targets": ["openalex:W2742617811"],
                    "self_consistency": 0.92,
                    "critic_verdict": {
                        "method": "SUPPORTED",
                        "task": "SUPPORTED",
                        "datasets": "SUPPORTED",
                        "metrics": "PARTIAL",
                    },
                }
            ]
        }
    )


@pytest.fixture
def mock_client(
    fake_extractor_response: str,
    fake_critic_response: str,
    fake_consolidator_response: str,
) -> MockOpenRouterClient:
    # Critic and consolidator share a model in the production config; the mock
    # alternates between fake_critic_response and fake_consolidator_response based
    # on call index so end-to-end tests still verify both agents.
    llama_responses: list[str] = []
    for _ in range(10):
        llama_responses.append(fake_critic_response)
        llama_responses.append(fake_consolidator_response)

    return MockOpenRouterClient(
        {
            "deepseek/deepseek-chat": [fake_extractor_response] * 10,
            "meta-llama/llama-3.3-70b-instruct": llama_responses,
        }
    )


@pytest.fixture
def test_config(tmp_path) -> Config:
    """A minimal Config that points at the real prompt templates."""

    return Config(
        extractor=AgentModelConfig(
            model_id="deepseek/deepseek-chat",
            price_in_per_m=0.32,
            price_out_per_m=0.89,
            max_tokens=2000,
            top_p=0.95,
            temperatures=(0.0, 0.3, 0.7),
            temperature=None,
        ),
        critic=AgentModelConfig(
            model_id="meta-llama/llama-3.3-70b-instruct",
            price_in_per_m=0.59,
            price_out_per_m=0.79,
            max_tokens=1500,
            top_p=0.95,
            temperatures=None,
            temperature=0.0,
        ),
        consolidator=AgentModelConfig(
            model_id="meta-llama/llama-3.3-70b-instruct",
            price_in_per_m=0.59,
            price_out_per_m=0.79,
            max_tokens=3000,
            top_p=0.95,
            temperatures=None,
            temperature=0.0,
        ),
        defaults=Defaults(request_timeout_s=90.0, max_retries=2, retry_backoff_s=0.1),
        concurrency=Concurrency(papers_in_flight=4, per_provider_max=2),
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        referer=None,
        title=None,
        prompts_dir=DEFAULT_PROMPTS_DIR,
    )


def _ensure_implements_protocol() -> None:
    # Static check that MockOpenRouterClient satisfies the protocol
    _: OpenRouterClientProtocol = MockOpenRouterClient({})


_ensure_implements_protocol()
