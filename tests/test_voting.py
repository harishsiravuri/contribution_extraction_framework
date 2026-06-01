"""Voting logic tests."""

from __future__ import annotations

from paper1.schema import (
    EvidenceSpan,
    ExtractorDraft,
    ExtractorDraftContribution,
    ExtractorDraftEntity,
    ExtractorDraftMetric,
)
from paper1.voting import _norm, vote


def test_norm_handles_punctuation_and_case():
    assert _norm("BERT") == _norm("bert")
    assert _norm("BERT-base") == _norm("bert base")
    assert _norm("  BERT  model.") == _norm("bert model")
    assert _norm(None) is None
    assert _norm("") is None


def _draft(method_name: str, dataset_name: str | None = None) -> ExtractorDraft:
    datasets = (
        [ExtractorDraftEntity(name=dataset_name)] if dataset_name else []
    )
    return ExtractorDraft(
        contributions=[
            ExtractorDraftContribution(
                method=ExtractorDraftEntity(name=method_name),
                datasets=datasets,
                metrics=[
                    ExtractorDraftMetric(
                        name="F1",
                        value=88.5,
                        unit="F1",
                        evidence_span=EvidenceSpan(start=10, end=20),
                    )
                ],
                claim_strength="improves",
            )
        ]
    )


def test_majority_vote_picks_consistent_method():
    drafts = [_draft("BERT"), _draft("BERT"), _draft("RoBERTa")]
    result = vote(drafts)
    assert len(result) == 1
    contribution, confidence = result[0]
    assert contribution.method.name and contribution.method.name.lower() == "bert"
    # Self-consistency is averaged across fields; method-level vote was 2/3
    assert 0.0 < confidence <= 1.0


def test_unanimous_vote_returns_high_confidence():
    drafts = [_draft("BERT", "SQuAD") for _ in range(3)]
    result = vote(drafts)
    contribution, confidence = result[0]
    assert contribution.method.name == "BERT"
    assert confidence > 0.9


def test_disagreement_lowers_confidence():
    drafts = [
        _draft("BERT", "SQuAD"),
        _draft("RoBERTa", "GLUE"),
        _draft("ELECTRA", "RACE"),
    ]
    result = vote(drafts)
    contribution, confidence = result[0]
    # No two drafts agree; method confidence is 1/3
    assert confidence < 0.6


def test_empty_drafts_returns_empty():
    assert vote([]) == []


def test_metric_voting_aggregates_by_name():
    drafts = [
        ExtractorDraft(
            contributions=[
                ExtractorDraftContribution(
                    method=ExtractorDraftEntity(name="BERT"),
                    metrics=[
                        ExtractorDraftMetric(name="F1", value=88.5, unit="F1"),
                        ExtractorDraftMetric(name="EM", value=80.0, unit="EM"),
                    ],
                )
            ]
        ),
        ExtractorDraft(
            contributions=[
                ExtractorDraftContribution(
                    method=ExtractorDraftEntity(name="BERT"),
                    metrics=[
                        ExtractorDraftMetric(name="F1", value=88.5, unit="F1"),
                    ],
                )
            ]
        ),
    ]
    result = vote(drafts)
    contribution, _ = result[0]
    metric_names = {m.name for m in contribution.metrics}
    assert metric_names == {"F1", "EM"}
