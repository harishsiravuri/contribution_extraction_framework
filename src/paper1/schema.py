"""Pydantic schemas for the multi-agent extraction pipeline.

These models define the contract between the agents and the rest of the system.
The output schema (ContributionRecord) is what downstream papers (Papers 2-5) will
consume — treat changes here as breaking.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Output schema (what the Consolidator produces — the public contract)
# ---------------------------------------------------------------------------


class EvidenceSpan(BaseModel):
    """A character-level span in the paper text supporting a claim."""

    model_config = ConfigDict(extra="forbid")

    start: int = Field(ge=0, description="Inclusive start char offset (0-indexed)")
    end: int = Field(ge=0, description="Exclusive end char offset")


class NamedEntity(BaseModel):
    """A named scientific entity (Method, Task, Dataset)."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    canonical_id: str | None = Field(
        default=None,
        description="Canonical ID, e.g. 'pwc:bert', 'orkg:R12345', 'openalex:T10001'",
    )
    evidence_span: EvidenceSpan | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class MetricEntity(BaseModel):
    """A metric value reported in the paper, e.g. F1 = 0.84."""

    model_config = ConfigDict(extra="forbid")

    name: str
    value: float | None = None
    unit: str | None = None
    evidence_span: EvidenceSpan | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


ClaimStrength = Literal["improves", "comparable", "novel", "applies"]
CriticLabel = Literal["SUPPORTED", "PARTIAL", "UNSUPPORTED"]


class CriticVerdictSummary(BaseModel):
    """Per-field critic verdict, rolled up to the contribution level."""

    model_config = ConfigDict(extra="forbid")

    method: CriticLabel | None = None
    task: CriticLabel | None = None
    datasets: CriticLabel | None = None
    metrics: CriticLabel | None = None

    @field_validator("method", "task", "datasets", "metrics", mode="before")
    @classmethod
    def _coerce_unknown_label(cls, v):
        if v is None:
            return None
        if isinstance(v, str) and v not in ("SUPPORTED", "PARTIAL", "UNSUPPORTED"):
            # Llama occasionally emits "NOT_APPLICABLE", "N/A" etc. — treat as missing.
            return None
        return v


class ContributionUnit(BaseModel):
    """One contribution claim made by a paper."""

    model_config = ConfigDict(extra="forbid")

    method: NamedEntity = Field(default_factory=NamedEntity)
    task: NamedEntity = Field(default_factory=NamedEntity)
    datasets: list[NamedEntity] = Field(default_factory=list)
    metrics: list[MetricEntity] = Field(default_factory=list)
    claim_strength: ClaimStrength | None = None
    comparison_targets: list[str] = Field(
        default_factory=list, description="OpenAlex/DOI references this contribution compares against"
    )
    self_consistency: float = Field(default=0.0, ge=0.0, le=1.0)
    critic_verdict: CriticVerdictSummary = Field(default_factory=CriticVerdictSummary)


class RunMeta(BaseModel):
    """Provenance — which models were used, how many tokens, how long."""

    model_config = ConfigDict(extra="forbid")

    extractor_model: str
    critic_model: str
    consolidator_model: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    wall_time_seconds: float = 0.0
    voting_samples: int = 3


class ContributionRecord(BaseModel):
    """The full per-paper output of the pipeline."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    contributions: list[ContributionUnit] = Field(default_factory=list)
    meta: RunMeta = Field(alias="_meta", default=None)  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Internal schemas (intermediate agent outputs — not the public contract)
# ---------------------------------------------------------------------------


class ExtractorDraftEntity(BaseModel):
    """Loosely-typed entity from the Extractor (no confidence yet)."""

    model_config = ConfigDict(extra="ignore")

    name: str | None = None
    canonical_id: str | None = None
    evidence_span: EvidenceSpan | None = None


class ExtractorDraftMetric(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str | None = None
    value: float | None = None
    unit: str | None = None
    evidence_span: EvidenceSpan | None = None


class ExtractorDraftContribution(BaseModel):
    """One contribution as produced by the Extractor (pre-Critic)."""

    model_config = ConfigDict(extra="ignore")

    method: ExtractorDraftEntity = Field(default_factory=ExtractorDraftEntity)
    task: ExtractorDraftEntity = Field(default_factory=ExtractorDraftEntity)
    datasets: list[ExtractorDraftEntity] = Field(default_factory=list)
    metrics: list[ExtractorDraftMetric] = Field(default_factory=list)
    claim_strength: ClaimStrength | None = None
    comparison_targets: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("method", "task", mode="before")
    @classmethod
    def _none_to_empty_entity(cls, v):
        # Extractors sometimes emit `null` for absent fields; treat as empty entity.
        if v is None:
            return {}
        return v

    @field_validator("datasets", "metrics", mode="before")
    @classmethod
    def _none_to_empty_list(cls, v):
        if v is None:
            return []
        return v

    @field_validator("claim_strength", mode="before")
    @classmethod
    def _coerce_unknown_claim_strength(cls, v):
        # Extractors sometimes emit "extends", "applies-to", etc. that aren't in our 4 literals.
        if v is None:
            return None
        if isinstance(v, str) and v not in ("improves", "comparable", "novel", "applies"):
            return None
        return v

    @field_validator("comparison_targets", mode="before")
    @classmethod
    def _none_to_empty_list_targets(cls, v):
        if v is None:
            return []
        return v


class ExtractorDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")

    contributions: list[ExtractorDraftContribution] = Field(default_factory=list)


class CriticVerdict(BaseModel):
    model_config = ConfigDict(extra="ignore")

    contribution_index: int
    field_path: str
    verdict: CriticLabel
    reason: str = ""
    corrected_evidence_span: EvidenceSpan | None = None


class CriticOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    verdicts: list[CriticVerdict] = Field(default_factory=list)
    overall_summary: str = ""
