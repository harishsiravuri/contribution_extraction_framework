"""Self-consistency voting over multiple Extractor drafts.

For each field, take the value most-frequently produced across drafts.
For free-text fields (method/task/dataset names), match by case-folded
normalized strings (lightweight surface canonicalization). The Consolidator
agent does the heavier semantic merging downstream — this is just the cheap
first pass that surfaces obvious agreement.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence

from paper1.schema import (
    ExtractorDraft,
    ExtractorDraftContribution,
    ExtractorDraftEntity,
    ExtractorDraftMetric,
)

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def _norm(s: str | None) -> str | None:
    """Lightweight name normalization for vote counting."""

    if s is None:
        return None
    s = s.strip().lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s)
    return s.strip() or None


def _vote_majority(values: Sequence[str | None]) -> tuple[str | None, float]:
    """Return (most-common-value, agreement-fraction).

    Agreement is the fraction of non-None inputs that match the winner.
    If all inputs are None, returns (None, 0.0).
    """

    non_null = [v for v in values if v is not None]
    if not non_null:
        return None, 0.0
    counter = Counter(non_null)
    winner, count = counter.most_common(1)[0]
    return winner, count / len(non_null)


def _pick_canonical_form(values: Iterable[str | None], norm: str) -> str | None:
    """Among input strings whose normalized form == norm, return the longest one."""

    matching = [v for v in values if v is not None and _norm(v) == norm]
    if not matching:
        return None
    # Prefer the longest surface form (more informative)
    return max(matching, key=len)


def _vote_entity(entities: Sequence[ExtractorDraftEntity]) -> tuple[ExtractorDraftEntity, float]:
    """Vote on a NamedEntity across drafts. Returns (winner, confidence)."""

    names = [e.name for e in entities]
    norms = [_norm(n) for n in names]
    winning_norm, agreement = _vote_majority(norms)
    if winning_norm is None:
        return ExtractorDraftEntity(), 0.0

    surface = _pick_canonical_form(names, winning_norm)

    # Pick a canonical_id from any draft that agrees on the name
    canonical_ids = [
        e.canonical_id
        for e in entities
        if e.canonical_id is not None and _norm(e.name) == winning_norm
    ]
    canonical_id = canonical_ids[0] if canonical_ids else None

    # Pick the most common evidence span among agreeing drafts
    spans = [
        (e.evidence_span.start, e.evidence_span.end)
        for e in entities
        if e.evidence_span is not None and _norm(e.name) == winning_norm
    ]
    span = None
    if spans:
        span_counter = Counter(spans)
        winning_span, _ = span_counter.most_common(1)[0]
        from paper1.schema import EvidenceSpan

        span = EvidenceSpan(start=winning_span[0], end=winning_span[1])

    return (
        ExtractorDraftEntity(name=surface, canonical_id=canonical_id, evidence_span=span),
        agreement,
    )


def _vote_metrics(
    metric_lists: Sequence[Sequence[ExtractorDraftMetric]],
) -> list[tuple[ExtractorDraftMetric, float]]:
    """Vote on metric lists by metric name."""

    # Group by normalized metric name
    by_name: dict[str, list[ExtractorDraftMetric]] = {}
    n_drafts = len(metric_lists)
    for draft_metrics in metric_lists:
        seen_in_draft: set[str] = set()
        for m in draft_metrics:
            key = _norm(m.name)
            if key is None or key in seen_in_draft:
                continue
            seen_in_draft.add(key)
            by_name.setdefault(key, []).append(m)

    out: list[tuple[ExtractorDraftMetric, float]] = []
    for key, ms in by_name.items():
        # Confidence = fraction of drafts that mentioned this metric
        confidence = len(ms) / n_drafts
        # Pick the most common (name, value, unit) among the drafts
        triples = Counter((m.name.strip(), m.value, m.unit) for m in ms)
        (winning_name, winning_val, winning_unit), _ = triples.most_common(1)[0]
        # Pick the most common evidence span
        spans = [
            (m.evidence_span.start, m.evidence_span.end)
            for m in ms
            if m.evidence_span is not None
        ]
        span = None
        if spans:
            from paper1.schema import EvidenceSpan

            (s, e), _ = Counter(spans).most_common(1)[0]
            span = EvidenceSpan(start=s, end=e)
        out.append(
            (
                ExtractorDraftMetric(
                    name=winning_name, value=winning_val, unit=winning_unit, evidence_span=span
                ),
                confidence,
            )
        )
    return out


def vote(drafts: Sequence[ExtractorDraft]) -> list[tuple[ExtractorDraftContribution, float]]:
    """Run self-consistency voting across drafts.

    Returns one consolidated contribution per "row" in the contributions list,
    aligned by index across drafts. (Sophisticated alignment is the
    Consolidator's job; this just bins by position.)
    """

    if not drafts:
        return []

    # Align by index; pad shorter drafts implicitly via getter
    max_len = max(len(d.contributions) for d in drafts)
    out: list[tuple[ExtractorDraftContribution, float]] = []
    for i in range(max_len):
        slice_: list[ExtractorDraftContribution] = []
        for d in drafts:
            if i < len(d.contributions):
                slice_.append(d.contributions[i])
        if not slice_:
            continue

        method, m_conf = _vote_entity([c.method for c in slice_])
        task, t_conf = _vote_entity([c.task for c in slice_])

        # For datasets, treat each draft's list as a set of entities and aggregate
        all_datasets: list[ExtractorDraftEntity] = [
            ds for c in slice_ for ds in c.datasets
        ]
        # Group datasets by normalized name
        ds_groups: dict[str, list[ExtractorDraftEntity]] = {}
        for ds in all_datasets:
            key = _norm(ds.name)
            if key is None:
                continue
            ds_groups.setdefault(key, []).append(ds)
        n_drafts = len(slice_)
        consolidated_datasets: list[ExtractorDraftEntity] = []
        ds_confidences: list[float] = []
        for _, ds_list in ds_groups.items():
            # Confidence = fraction of drafts that mentioned this dataset
            distinct_drafts = len(ds_list)
            ds_conf = distinct_drafts / n_drafts
            entity, _ = _vote_entity(ds_list)
            consolidated_datasets.append(entity)
            ds_confidences.append(ds_conf)

        metric_votes = _vote_metrics([c.metrics for c in slice_])
        consolidated_metrics = [m for m, _ in metric_votes]
        metric_confidences = [c for _, c in metric_votes]

        # Claim strength: majority vote
        cs_values = [c.claim_strength for c in slice_]
        cs_winner, cs_agreement = _vote_majority(cs_values)

        # Comparison targets: take the union if mentioned by ≥ half of drafts
        target_counts: Counter[str] = Counter()
        for c in slice_:
            for t in c.comparison_targets:
                target_counts[t] += 1
        threshold = max(1, n_drafts // 2)
        comparison_targets = [t for t, n in target_counts.items() if n >= threshold]

        consolidated = ExtractorDraftContribution(
            method=method,
            task=task,
            datasets=consolidated_datasets,
            metrics=consolidated_metrics,
            claim_strength=cs_winner,  # type: ignore[arg-type]
            comparison_targets=comparison_targets,
        )

        # Self-consistency = mean of per-field agreement
        per_field = [m_conf, t_conf, cs_agreement] + ds_confidences + metric_confidences
        per_field = [c for c in per_field if c > 0.0]
        self_consistency = sum(per_field) / len(per_field) if per_field else 0.0

        out.append((consolidated, self_consistency))

    return out
