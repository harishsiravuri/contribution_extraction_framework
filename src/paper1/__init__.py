"""Paper 1: Multi-Agent Contribution Extraction with Verifiable Grounding.

A three-agent pipeline (Extractor / Critic / Consolidator) that reads a scientific
paper and emits a structured contribution record, accessed through OpenRouter.
"""

__version__ = "0.1.0"

from paper1.schema import (
    ContributionRecord,
    ContributionUnit,
    EvidenceSpan,
    NamedEntity,
    MetricEntity,
    CriticVerdictSummary,
)

__all__ = [
    "ContributionRecord",
    "ContributionUnit",
    "EvidenceSpan",
    "NamedEntity",
    "MetricEntity",
    "CriticVerdictSummary",
    "__version__",
]
