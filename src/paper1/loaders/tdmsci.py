"""TDMSci loader (Hou et al. 2019, ACL).

Source: https://github.com/IBM/science-result-extractor (the
data/TDMSci/conllFormat/ subdir)
License: Apache 2.0 (per the parent IBM repo)

The corpus is sentence-level BIO-tagged with B-/I- on TASK, DATASET, METRIC.
We treat each sentence as one GoldPaper because TDMSci itself is annotated at
sentence granularity. The original paper reports F1 over (T, D, M) entity sets
extracted per sentence.
"""

from __future__ import annotations

from pathlib import Path

from paper1.loaders.base import GoldContribution, GoldPaper

DEFAULT_DIR = Path("data/raw/science-result-extractor/data/TDMSci/conllFormat")
SPLITS = ("train_1500_v2.conll", "test_500_v2.conll")


def _norm(s: str) -> str:
    return s.strip().lower()


def _read_conll(path: Path) -> list[list[tuple[str, str]]]:
    """Return list of sentences, each a list of (token, label)."""
    sentences: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                if cur:
                    sentences.append(cur)
                    cur = []
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            cur.append((parts[0], parts[2]))
    if cur:
        sentences.append(cur)
    return sentences


def _extract_entities(tagged: list[tuple[str, str]]) -> tuple[set[str], set[str], set[str]]:
    tasks, datasets, metrics = set(), set(), set()
    cur: list[str] = []
    cur_type: str | None = None
    for tok, lab in tagged + [("", "O")]:
        if lab.startswith("B-"):
            if cur and cur_type:
                _push(cur, cur_type, tasks, datasets, metrics)
            cur = [tok]
            cur_type = lab[2:]
        elif lab.startswith("I-") and cur_type == lab[2:]:
            cur.append(tok)
        else:
            if cur and cur_type:
                _push(cur, cur_type, tasks, datasets, metrics)
            cur, cur_type = [], None
    return tasks, datasets, metrics


def _push(
    toks: list[str], typ: str, tasks: set[str], datasets: set[str], metrics: set[str]
) -> None:
    name = _norm(" ".join(toks))
    if not name:
        return
    if typ == "TASK":
        tasks.add(name)
    elif typ == "DATASET":
        datasets.add(name)
    elif typ == "METRIC":
        metrics.add(name)


def load_tdmsci(data_dir: Path = DEFAULT_DIR) -> list[GoldPaper]:
    out: list[GoldPaper] = []
    idx = 0
    for fname in SPLITS:
        path = data_dir / fname
        if not path.exists():
            continue
        sentences = _read_conll(path)
        for sent in sentences:
            tasks, datasets, metrics = _extract_entities(sent)
            if not (tasks or datasets or metrics):
                continue
            text = " ".join(t for t, _ in sent)
            paper_id = f"tdmsci:{fname.removesuffix('.conll')}:{idx:05d}"
            idx += 1
            out.append(
                GoldPaper(
                    paper_id=paper_id,
                    full_text=text,
                    benchmark="TDMSci",
                    gold=GoldContribution(
                        tasks=tasks, datasets=datasets, metrics=metrics
                    ),
                )
            )
    return out
