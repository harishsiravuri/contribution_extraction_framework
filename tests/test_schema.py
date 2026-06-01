"""Schema round-trip tests."""

from __future__ import annotations

import json

from paper1.schema import (
    ContributionRecord,
    ContributionUnit,
    EvidenceSpan,
    ExtractorDraft,
    NamedEntity,
    RunMeta,
)


def test_evidence_span_validation():
    span = EvidenceSpan(start=10, end=42)
    assert span.start == 10
    assert span.end == 42


def test_contribution_record_round_trip():
    record = ContributionRecord(
        paper_id="openalex:W123",
        contributions=[
            ContributionUnit(
                method=NamedEntity(name="BERT", canonical_id="pwc:bert", confidence=0.9),
                claim_strength="improves",
                self_consistency=0.95,
            )
        ],
        _meta=RunMeta(
            extractor_model="x",
            critic_model="y",
            consolidator_model="z",
            tokens_in=100,
            tokens_out=50,
            cost_usd=0.001,
            wall_time_seconds=1.2,
        ),
    )
    j = record.model_dump_json(by_alias=True)
    parsed = json.loads(j)
    assert parsed["paper_id"] == "openalex:W123"
    assert parsed["contributions"][0]["method"]["name"] == "BERT"
    assert parsed["_meta"]["extractor_model"] == "x"

    back = ContributionRecord.model_validate(parsed)
    assert back.paper_id == record.paper_id


def test_extractor_draft_validates_minimal_json():
    """The Extractor's prompt produces this loose shape; schema must accept it."""

    raw = {
        "contributions": [
            {
                "method": {"name": "Foo", "canonical_id": None, "evidence_span": None},
                "task": {"name": None, "canonical_id": None, "evidence_span": None},
                "datasets": [],
                "metrics": [],
                "claim_strength": None,
                "comparison_targets": [],
                "notes": "could not determine claim strength",
            }
        ]
    }
    draft = ExtractorDraft.model_validate(raw)
    assert len(draft.contributions) == 1
    assert draft.contributions[0].method.name == "Foo"
    assert draft.contributions[0].claim_strength is None
