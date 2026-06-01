"""Post-processing pass that snaps extracted entities to the SciREX ontology.

Applied as a final step before scoring against SciREX gold; not part of the
generic multi-agent pipeline (it would over-constrain non-SciREX corpora).
"""

from __future__ import annotations

from paper1.ontology import resolve_to_ontology
from paper1.schema import ContributionRecord


def _snap(name: str | None, kind: str) -> tuple[str | None, str | None]:
    """Return (resolved_name, canonical_id) — both may be None."""
    if not name:
        return name, None
    canon = resolve_to_ontology(name, kind)
    if canon is None:
        return name, None
    return canon, f"scirex:{canon}"


def apply_ontology_grounding(rec: ContributionRecord) -> ContributionRecord:
    """Mutate-in-place `rec`'s entity names to the closest SciREX-ontology form.

    Mutates and returns the same record. Empty / unmatched entities are
    left untouched.
    """
    for c in rec.contributions:
        if c.method.name:
            new, cid = _snap(c.method.name, "method")
            c.method.name = new
            if cid:
                c.method.canonical_id = cid
        if c.task.name:
            new, cid = _snap(c.task.name, "task")
            c.task.name = new
            if cid:
                c.task.canonical_id = cid
        for d in c.datasets:
            if d.name:
                new, cid = _snap(d.name, "dataset")
                d.name = new
                if cid:
                    d.canonical_id = cid
        for m in c.metrics:
            if m.name:
                new, _cid = _snap(m.name, "metric")
                m.name = new
    return rec
