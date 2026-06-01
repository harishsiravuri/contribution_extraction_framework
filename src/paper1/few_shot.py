"""Few-shot example selection from the SciREX train split.

Picks `n` short, information-dense examples (all four entity types populated,
plus at least one (T,D,M) triple) and renders each as a (truncated paper text
→ expected JSON) pair suitable for splicing into the extractor prompt.

The selection is reproducible: we order candidates by full-text length and
take the first `n` that satisfy the density predicate, tie-broken by paper-id.
The chosen example IDs are written to outputs/paper_data_v4/few_shot_examples.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from paper1.loaders import GoldPaper, load_scirex_train

DEFAULT_TEXT_BUDGET = 3000  # chars per example
DEFAULT_N = 3
DEFAULT_SEED = 42  # not strictly needed since selection is deterministic


def _is_dense(p: GoldPaper) -> bool:
    g = p.gold
    return bool(g.methods and g.tasks and g.datasets and g.metrics and g.triples)


def select_few_shot_examples(
    n: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
    text_budget: int = DEFAULT_TEXT_BUDGET,
) -> list[GoldPaper]:
    """Pick `n` short, dense SciREX-train examples (deterministic)."""
    train = load_scirex_train()
    candidates = [p for p in train if _is_dense(p)]
    candidates.sort(key=lambda p: (len(p.full_text), p.paper_id))
    return candidates[:n]


def _expected_record_json(p: GoldPaper) -> str:
    """Render the gold answer as a ContributionRecord-shaped JSON for the prompt.

    We emit ONE contribution per (task, dataset, metric) triple. The method
    field is the first canonical method name from the gold (SciREX rarely
    has triple-level method binding); we hard-code it across all triples.
    Evidence spans are deliberately omitted in examples — the model should
    still emit char offsets at inference, but training-time gold spans don't
    align across train/dev (different documents).
    """
    g = p.gold
    methods = sorted(g.methods)
    method_name = methods[0] if methods else None
    contributions = []
    for t, d, m in sorted(g.triples):
        contributions.append({
            "method": {"name": method_name, "canonical_id": None, "evidence_span": None},
            "task": {"name": t, "canonical_id": None, "evidence_span": None},
            "datasets": [{"name": d, "canonical_id": None, "evidence_span": None}],
            "metrics": [{"name": m, "value": None, "unit": None, "evidence_span": None}],
            "claim_strength": "improves",
            "comparison_targets": [],
            "notes": None,
        })
    return json.dumps({"contributions": contributions}, indent=2)


def format_example(p: GoldPaper, text_budget: int = DEFAULT_TEXT_BUDGET) -> str:
    """Render a single (paper-excerpt, expected-JSON) example."""
    text = p.full_text[:text_budget]
    if len(p.full_text) > text_budget:
        text += " ..."
    return (
        f"### Example: paper {p.paper_id}\n"
        f"Paper text (truncated to {text_budget} chars):\n"
        f"---\n{text}\n---\n"
        f"Expected JSON:\n```json\n{_expected_record_json(p)}\n```\n"
    )


def render_examples_block(
    n: int = DEFAULT_N, text_budget: int = DEFAULT_TEXT_BUDGET
) -> str:
    examples = select_few_shot_examples(n=n, text_budget=text_budget)
    parts = ["## Examples\n",
             "Below are real SciREX papers and the structured contribution "
             "records experts annotated for them. Use them as a guide for the "
             "binding rule, the level of detail, and the canonical-name style.\n"]
    for i, p in enumerate(examples, 1):
        parts.append(f"### Example {i}\n")
        parts.append(format_example(p, text_budget=text_budget))
    return "\n".join(parts)


def save_chosen_ids(
    out_path: Path,
    n: int = DEFAULT_N,
    text_budget: int = DEFAULT_TEXT_BUDGET,
) -> dict:
    examples = select_few_shot_examples(n=n, text_budget=text_budget)
    record = {
        "n": n,
        "text_budget_chars": text_budget,
        "selection_criterion": "shortest SciREX-train papers with all 4 entity types populated and ≥ 1 (T,D,M) triple",
        "ids": [p.paper_id for p in examples],
        "lengths": [len(p.full_text) for p in examples],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))
    return record
