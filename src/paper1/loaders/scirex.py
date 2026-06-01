"""SciREX loader (Jain et al. 2020, ACL).

Source: https://github.com/allenai/SciREX
License: Apache 2.0

The SciREX release JSON-Lines have these per-doc keys:
- doc_id, words (token list), sentences, sections,
- ner: [start_tok, end_tok, label] with labels in {Method, Task, Material, Metric}
  ("Material" is SciREX's name for Dataset.)
- coref: {canonical_name: [[start_tok, end_tok], ...]}
- n_ary_relations: list of {Method, Task, Material, Metric, score}

For the GoldPaper interface we flatten:
- gold methods = the set of canonical method names from coref keys appearing
  in any n-ary relation
- gold datasets = same for Material
- gold tasks = same for Task
- gold metrics = same for Metric
- triples = (task, dataset, metric) over n-ary relations
- full_text = the words joined by spaces

Token-level offsets are kept implicit (we don't compute char spans here — the
agents see joined-word text and produce their own char spans).
"""

from __future__ import annotations

import json
from pathlib import Path

from paper1.loaders.base import GoldContribution, GoldPaper

DEFAULT_DIR = Path("data/raw/scirex/scirex_dataset/release_data")


def _norm(s: str) -> str:
    return s.strip().lower().replace("_", " ")


def _parse_doc(doc: dict) -> GoldPaper:
    words = doc.get("words", [])
    full_text = " ".join(words)
    relations = doc.get("n_ary_relations", []) or []

    # token-index → start_char of that token in the joined-with-space full_text
    tok_starts: list[int] = []
    cursor = 0
    for w in words:
        tok_starts.append(cursor)
        cursor += len(w) + 1  # +1 for the space after each token

    # token-index → end_char (exclusive)
    def char_span(start_tok: int, end_tok: int) -> tuple[int, int]:
        # end_tok is exclusive in SciREX format
        if end_tok <= start_tok or end_tok > len(words):
            return (0, 0)
        s = tok_starts[start_tok]
        last = end_tok - 1
        e = tok_starts[last] + len(words[last])
        return (s, e)

    # Build gold spans from NER (Method/Task/Dataset/Metric)
    label_map = {"Method": "Method", "Task": "Task", "Material": "Dataset", "Metric": "Metric"}
    gold_spans: list[tuple[int, int, str, str]] = []
    for span in doc.get("ner", []) or []:
        if len(span) < 3:
            continue
        st, et, lab = span[0], span[1], span[2]
        if lab not in label_map:
            continue
        s, e = char_span(int(st), int(et))
        if e <= s:
            continue
        gold_spans.append((s, e, label_map[lab], full_text[s:e]))

    methods: set[str] = set()
    tasks: set[str] = set()
    datasets: set[str] = set()
    metrics: set[str] = set()
    triples: set[tuple[str, str, str]] = set()
    for rel in relations:
        m = rel.get("Method")
        t = rel.get("Task")
        d = rel.get("Material")
        mt = rel.get("Metric")
        if m:
            methods.add(_norm(m))
        if t:
            tasks.add(_norm(t))
        if d:
            datasets.add(_norm(d))
        if mt:
            metrics.add(_norm(mt))
        if t and d and mt:
            triples.add((_norm(t), _norm(d), _norm(mt)))

    return GoldPaper(
        paper_id=f"scirex:{doc['doc_id']}",
        full_text=full_text,
        benchmark="SciREX",
        gold=GoldContribution(
            methods=methods,
            tasks=tasks,
            datasets=datasets,
            metrics=metrics,
            triples=triples,
            gold_spans=gold_spans,
        ),
    )


def load_scirex(
    data_dir: Path = DEFAULT_DIR, splits: tuple[str, ...] = ("train", "dev", "test")
) -> list[GoldPaper]:
    out: list[GoldPaper] = []
    for split in splits:
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                out.append(_parse_doc(doc))
    return out


def load_scirex_train(data_dir: Path = DEFAULT_DIR) -> list[GoldPaper]:
    """Convenience wrapper for the official SciREX train split (~306 papers)."""
    return load_scirex(data_dir=data_dir, splits=("train",))
