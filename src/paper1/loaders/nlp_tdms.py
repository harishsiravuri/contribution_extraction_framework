"""NLP-TDMS loader (Hou et al. 2019; Mondal et al. 2021).

Source: https://github.com/IBM/science-result-extractor (data/NLP-TDMS/)
License: Apache 2.0

The annotation file resultsAnnotation.tsv has rows of the form:
    <pdf_filename>\\t<task#dataset#metric#score>$<task#dataset#metric#score>...

We construct one GoldPaper per PDF, using full text (when available) or just
the filename + tuples as the input. Most papers in this corpus have a parallel
.txt extraction in pdfFile_txt/ — we use that when it exists.
"""

from __future__ import annotations

from pathlib import Path

from paper1.loaders.base import GoldContribution, GoldPaper

DEFAULT_ROOT = Path("data/raw/science-result-extractor/data/NLP-TDMS")


def _norm(s: str) -> str:
    return s.strip().lower().replace("_", " ")


def load_nlp_tdms(root: Path = DEFAULT_ROOT) -> list[GoldPaper]:
    annotations = root / "annotations" / "resultsAnnotation.tsv"
    if not annotations.exists():
        return []

    txt_dir = root / "pdfFile_txt"

    out: list[GoldPaper] = []
    with annotations.open("r", encoding="utf-8") as f:
        for raw in f:
            row = raw.rstrip("\n")
            if not row:
                continue
            parts = row.split("\t")
            if len(parts) < 2:
                continue
            pdf_name = parts[0]
            tuples_blob = parts[1]
            tuples = [t for t in tuples_blob.split("$") if t.strip()]

            tasks, datasets, metrics = set(), set(), set()
            triples: set[tuple[str, str, str]] = set()
            for t in tuples:
                fields = t.split("#")
                if len(fields) < 3:
                    continue
                task, dataset, metric = fields[:3]
                if task and task != "-":
                    tasks.add(_norm(task))
                if dataset and dataset != "-":
                    datasets.add(_norm(dataset))
                if metric and metric != "-":
                    metrics.add(_norm(metric))
                if all(f and f != "-" for f in (task, dataset, metric)):
                    triples.add((_norm(task), _norm(dataset), _norm(metric)))

            # Find a matching txt file (pdfFile_txt has the same stems)
            stem = Path(pdf_name).stem
            full_text = ""
            for cand in (txt_dir / f"{stem}.txt", txt_dir / f"{pdf_name}.txt"):
                if cand.exists():
                    try:
                        full_text = cand.read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        full_text = ""
                    break
            if not full_text:
                # Fall back to just the PDF name as a stand-in (loader skips
                # papers with no full-text — see filter below)
                full_text = ""

            paper_id = f"nlp_tdms:{stem}"
            out.append(
                GoldPaper(
                    paper_id=paper_id,
                    full_text=full_text,
                    benchmark="NLP-TDMS",
                    gold=GoldContribution(
                        tasks=tasks,
                        datasets=datasets,
                        metrics=metrics,
                        triples=triples,
                    ),
                )
            )
    return out
