"""SciREX canonical-entity ontology grounding.

Walk the SciREX train split, collect every canonical (lowercase) entity name
per type into a sorted list. At inference, fuzzy-match an extracted entity
against the ontology with rapidfuzz token_sort_ratio; if the best match is
≥ a threshold, snap to the ontology's canonical surface form.

Cached on first build; rebuild by deleting the JSON file.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from rapidfuzz import fuzz, process

from paper1.loaders import load_scirex_train

DEFAULT_ONTOLOGY_PATH = Path("outputs/paper_data_v4/scirex_ontology.json")
KIND_FIELD = {
    "method": "methods",
    "task": "tasks",
    "dataset": "datasets",
    "metric": "metrics",
}


def build_scirex_ontology(out_path: Path = DEFAULT_ONTOLOGY_PATH) -> dict[str, list[str]]:
    """Walk the SciREX train split and collect every canonical entity name per type.

    Returns the ontology dict and writes it to `out_path`.
    """
    train = load_scirex_train()
    onto: dict[str, set[str]] = {k: set() for k in KIND_FIELD.values()}
    for p in train:
        onto["methods"].update(p.gold.methods)
        onto["tasks"].update(p.gold.tasks)
        onto["datasets"].update(p.gold.datasets)
        onto["metrics"].update(p.gold.metrics)
    sorted_onto = {k: sorted(v) for k, v in onto.items()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sorted_onto, indent=2))
    return sorted_onto


@lru_cache(maxsize=1)
def _load_ontology(path_str: str) -> dict[str, list[str]]:
    p = Path(path_str)
    if not p.exists():
        return build_scirex_ontology(p)
    return json.loads(p.read_text())


def resolve_to_ontology(
    name: str | None,
    kind: str,
    threshold: int = 85,
    ontology_path: Path = DEFAULT_ONTOLOGY_PATH,
) -> str | None:
    """Return the closest canonical name in the ontology for `kind`, or None.

    `kind` ∈ {"method", "task", "dataset", "metric"}. Match uses rapidfuzz's
    token_sort_ratio (handles word reordering and minor variation). Threshold
    is an integer 0–100; default 85 gives high precision.
    """
    if not name or kind not in KIND_FIELD:
        return None
    onto = _load_ontology(str(ontology_path))
    pool = onto.get(KIND_FIELD[kind], [])
    if not pool:
        return None
    needle = name.strip().lower()
    if not needle:
        return None
    best = process.extractOne(needle, pool, scorer=fuzz.token_sort_ratio)
    if best is None:
        return None
    match, score, _ = best
    if score < threshold:
        return None
    return match
