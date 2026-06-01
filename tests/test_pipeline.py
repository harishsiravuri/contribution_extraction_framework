"""End-to-end pipeline test using the mock OpenRouter client."""

from __future__ import annotations

import pytest

from paper1.pipeline import Pipeline


@pytest.mark.asyncio
async def test_pipeline_end_to_end(mock_client, test_config):
    pipe = Pipeline(mock_client, test_config)
    record = await pipe.extract(
        paper_id="openalex:demo",
        paper_text="A short paper text. We propose BERT. We evaluate on SQuAD. We achieve F1 88.5.",
        retrieval_bundle="(none)",
    )

    assert record.paper_id == "openalex:demo"
    assert len(record.contributions) == 1
    contrib = record.contributions[0]
    assert contrib.method.name == "BERT"
    assert any(d.name == "SQuAD" for d in contrib.datasets)
    assert any(m.name == "F1" for m in contrib.metrics)
    assert contrib.claim_strength == "improves"
    assert contrib.critic_verdict.method == "SUPPORTED"

    # Meta should be populated
    assert record.meta.extractor_model == test_config.extractor.model_id
    assert record.meta.tokens_in > 0
    assert record.meta.tokens_out > 0
    assert record.meta.wall_time_seconds >= 0
    assert record.meta.voting_samples == 3


@pytest.mark.asyncio
async def test_pipeline_extract_many_runs_concurrently(mock_client, test_config):
    pipe = Pipeline(mock_client, test_config)
    papers = [
        (f"openalex:p{i}", "Paper text mentioning BERT and SQuAD with F1 88.5.", "")
        for i in range(5)
    ]
    records = await pipe.extract_many(papers, max_concurrency=3)
    assert len(records) == 5
    paper_ids = {r.paper_id for r in records}
    assert paper_ids == {f"openalex:p{i}" for i in range(5)}
