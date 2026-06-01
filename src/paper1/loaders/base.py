"""Common interface for gold-benchmark loaders."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GoldContribution:
    """One gold contribution from a benchmark.

    Fields are flat sets of normalized strings (lowercased). Loaders should
    populate whichever subset their benchmark provides; missing fields are empty.

    `gold_spans` (when present) is a list of (start_char, end_char, label, surface)
    tuples — character offsets into the GoldPaper.full_text, label in
    {Method, Task, Dataset, Metric}, surface as it appears in the text. Used for
    Phase Y2 span-grounding accuracy. Optional; loaders that don't have spans
    just leave it empty.
    """

    methods: set[str] = field(default_factory=set)
    tasks: set[str] = field(default_factory=set)
    datasets: set[str] = field(default_factory=set)
    metrics: set[str] = field(default_factory=set)
    triples: set[tuple[str, str, str]] = field(
        default_factory=set,
        metadata={"doc": "(task, dataset, metric) triples"},
    )
    gold_spans: list[tuple[int, int, str, str]] = field(default_factory=list)


@dataclass
class GoldPaper:
    paper_id: str
    full_text: str
    benchmark: str
    gold: GoldContribution
    title: str | None = None
