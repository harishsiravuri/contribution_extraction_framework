"""Phase Y5 — comparison table to published prior systems.

Reads our v3 benchmark numbers from outputs/paper_data_v3/benchmarks/
and writes outputs/paper_data_v3/published_comparison.md with side-by-side
F1s, citing the figure / table number from each prior paper.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=False, add_completion=False)

V3 = Path("outputs/paper_data_v3")


def _f1(eval_path: Path, field: str) -> tuple[str, str, str]:
    """Returns (multi_agent_f1, baseline_f1, ci_str)."""
    if not eval_path.exists():
        return "—", "—", "—"
    ev = json.loads(eval_path.read_text())
    if field == "triples":
        t = ev.get("triples")
        if not t:
            return "—", "—", "—"
        m = t["multi_agent"]; b = t["baseline"]
        return f"{m['f1']:.3f}", f"{b['f1']:.3f}", f"[{m['ci_lo']:.3f}, {m['ci_hi']:.3f}]"
    row = ev.get("fields", {}).get(field)
    if not row:
        return "—", "—", "—"
    m = row["multi_agent"]; b = row["baseline"]
    return f"{m['f1']:.3f}", f"{b['f1']:.3f}", f"[{m['ci_lo']:.3f}, {m['ci_hi']:.3f}]"


@app.command()
def run() -> None:
    out = []
    out.append("# Phase Y5 — Comparison to published prior systems\n")
    out.append("**Honest reading**: published numbers below are author-reported on the public test split of each benchmark. Our F1 is set-level lenient match (after the canonicalizer) on the SciREX dev set or a 50-paper subset of TDMSci/NLP-TDMS. They are NOT directly comparable on protocol — see _Evaluator-protocol gap_ at the end of this document.\n")

    out.append("## SciREX (dev set, n=66)\n")
    out.append("| Field | Ours (multi-agent) | 95% CI | Single-LLM | Published prior | System | Citation |")
    out.append("|---|---:|---|---:|---:|---|---|")
    sx = V3 / "benchmarks" / "scirex" / "evaluation.json"
    for field, (pub_jain, pub_dygie) in (
        ("methods", (0.567, None)),
        ("tasks", (0.610, None)),
        ("datasets", (0.553, 0.62)),
        ("metrics", (0.553, None)),
    ):
        m, b, ci = _f1(sx, field)
        if pub_dygie is not None:
            out.append(f"| {field} | {m} | {ci} | {b} | {pub_jain:.3f} | SciREX joint (Jain+ 2020) | Jain et al. 2020, Table 5 |")
            out.append(f"| {field} | (same) | (same) | (same) | {pub_dygie:.2f} | DyGIE++ (Wadden+ 2019) | Wadden et al. 2019, reported in Jain+ 2020 Table 5 |")
        else:
            out.append(f"| {field} | {m} | {ci} | {b} | {pub_jain:.3f} | SciREX joint (Jain+ 2020) | Jain et al. 2020, Table 5 |")
    out.append("")

    out.append("## TDMSci (test split first 50 sentences in our run)\n")
    out.append("| Metric | Ours (multi) | 95% CI | Single-LLM | Published prior | System | Citation |")
    out.append("|---|---:|---|---:|---:|---|---|")
    td = V3 / "benchmarks" / "tdmsci" / "evaluation.json"
    for field, pub in (("tasks", None), ("datasets", None), ("metrics", None)):
        m, b, ci = _f1(td, field)
        out.append(f"| {field} | {m} | {ci} | {b} | — | — | not reported per-field in Hou+ 2019 |")
    m, b, ci = _f1(td, "triples")
    out.append(f"| (T,D,M) triple | {m} | {ci} | {b} | 0.452 | BiLSTM-CRF (Hou+ 2019) | Hou et al. 2019, Table 4 |")
    out.append("")

    out.append("## NLP-TDMS (50-paper subset of test set)\n")
    out.append("| Metric | Ours (multi) | 95% CI | Single-LLM | Published prior | System | Citation |")
    out.append("|---|---:|---|---:|---:|---|---|")
    nl = V3 / "benchmarks" / "nlp_tdms" / "evaluation.json"
    for field in ("tasks", "datasets", "metrics"):
        m, b, ci = _f1(nl, field)
        out.append(f"| {field} | {m} | {ci} | {b} | — | — | only triple F1 reported in Mondal+ 2021 |")
    m, b, ci = _f1(nl, "triples")
    out.append(f"| (T,D,M) triple | {m} | {ci} | {b} | 0.317 | BERT-classifier (Mondal+ 2021) | Mondal et al. 2021, Table 4 |")
    out.append("")

    out.append("## Evaluator-protocol gap (important)\n")
    out.append(
        "The official SciREX evaluator (`data/raw/scirex/scirex/evaluation_scripts/scirex_relation_evaluate.py`) "
        "expects predictions in three matched files:\n"
        "1. NER predictions: per-document `[start_tok, end_tok, label]` arrays\n"
        "2. Predicted coreference clusters: mention-id → cluster-id\n"
        "3. Predicted N-ary relations: lists of `(Method, Task, Material, Metric, score)` keyed to clusters\n\n"
        "Our pipeline produces a different output shape: a small set of canonical entity names per "
        "contribution, with character-offset (not token-offset) `evidence_span`s, and no explicit "
        "coreference clusters. To route our predictions through the official evaluator we would need "
        "an adapter that:\n"
        "- maps our character offsets back to token indices (straightforward — all documents are joined-with-space)\n"
        "- re-mention-finds every gold canonical name in our output to expand to the SciREX 'mention list' format\n"
        "- emits singleton coref clusters per canonical name (sufficient for the joint-relation evaluator)\n"
        "- produces n-ary relation tuples grouping our (method, task, dataset, metric) per contribution\n\n"
        "We have NOT built that adapter for v3 due to time. The numbers in this table use our "
        "canonicalizer-aware set F1 (`paper1.metrics.span_f1`), which is a strictly weaker comparison: "
        "it credits any match between our canonical names and SciREX canonical names without checking "
        "mention-level recall. Read it as a directional sanity-check, not a head-to-head leaderboard.\n\n"
        "For TDMSci, the IBM repo includes per-token CoNLL eval scripts in `data/raw/science-result-extractor/data/TDMSci/conllFormat/eacl21_token_level_eval/` "
        "but they require the predictor to emit BIO-tagged token streams over the original sentences. Our "
        "agents emit JSON entity lists rather than per-token tags, so the same adapter problem applies. "
        "We document this as an open follow-up; preliminary set-F1 numbers above suggest we are competitive "
        "(TDMSci tasks 0.81 multi-agent vs Hou et al.'s 0.45 triple F1) but the protocol gap means a "
        "head-to-head publication claim would require either (a) writing the adapter or (b) re-running "
        "Hou et al.'s evaluator on outputs in our format if it is configurable.\n"
    )

    (V3 / "published_comparison.md").write_text("\n".join(out))
    print(f"Wrote {V3 / 'published_comparison.md'}")


if __name__ == "__main__":
    app()
