"""Phase Y3 — critic-suppression precision/recall on SciREX gold.

For each SciREX paper present in BOTH the full and no_critic conditions of
the Y2 run, compare the two records to identify critic-suppressed
extractions. Score them against gold to compute critic_precision (what
fraction of suppressions were genuinely wrong) and critic_recall (what
fraction of truly-wrong full-system extractions were caught).
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from paper1.loaders import load_scirex
from paper1.metrics.critic_validation import aggregate, critic_validation
from paper1.schema import ContributionRecord

app = typer.Typer(no_args_is_help=False, add_completion=False)
console = Console()


def _load(path: Path) -> ContributionRecord | None:
    if not path.exists():
        return None
    try:
        return ContributionRecord.model_validate_json(path.read_text())
    except Exception:
        return None


def _safe_id(paper_id: str) -> str:
    return paper_id.replace(":", "__").replace("/", "_")


@app.command()
def run(
    full_dir: Path = typer.Option(Path("outputs/paper_data_v3/span_grounding/full/by_paper"), "--full-dir"),
    no_critic_dir: Path = typer.Option(Path("outputs/paper_data_v3/span_grounding/no_critic/by_paper"), "--no-critic-dir"),
    output_dir: Path = typer.Option(Path("outputs/paper_data_v3/critic_analysis"), "--output-dir"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    papers = {p.paper_id: p for p in load_scirex(splits=("dev",))}
    per_paper: list[dict] = []
    false_supp_examples: list[dict] = []
    correct_supp_examples: list[dict] = []
    n_pairs = 0
    for pid, p in papers.items():
        full_path = full_dir / f"{_safe_id(pid)}.json"
        nc_path = no_critic_dir / f"{_safe_id(pid)}.json"
        full = _load(full_path)
        nc = _load(nc_path)
        if full is None or nc is None:
            continue
        n_pairs += 1
        result = critic_validation(full, nc, p)
        per_paper.append(result)
        for f, d in result.items():
            if d["correct_suppression"]:
                correct_supp_examples.append({"paper_id": pid, "field": f, **d})
            elif d["false_suppression"]:
                false_supp_examples.append({"paper_id": pid, "field": f, **d})

    agg = aggregate(per_paper)
    out = {
        "n_papers": n_pairs,
        "aggregate": agg,
        "false_suppression_examples": false_supp_examples[:20],
        "correct_suppression_examples": correct_supp_examples[:20],
    }
    (output_dir / "critic_validation.json").write_text(json.dumps(out, indent=2))

    lines = ["# Phase Y3 — Critic-suppression validation\n"]
    lines.append(f"_n papers compared (full ∩ no_critic): **{n_pairs}**_\n")
    ov = agg["overall"]
    lines.append("## Overall\n")
    lines.append(f"- Critic suppressions detected (full=null, no_critic=name, verdict=UNSUPPORTED): **{ov['n_critic_suppressions']}**")
    lines.append(f"- Of those, **truly wrong** (no gold match): {ov['correct_supp']}")
    lines.append(f"- Of those, **wrongly suppressed** (gold did contain it): {ov['false_supp']}")
    lines.append(f"- Truly-wrong extractions the critic missed (full retained, no gold match): {ov['missed_supp']}")
    lines.append(f"- Implicit suppressions (full=null, no_critic=name, verdict ≠ UNSUPPORTED): {ov['implicit_supp']}\n")
    if ov["critic_precision"] is not None:
        lines.append(f"**Critic precision (UNSUPPORTED ∩ truly_wrong / UNSUPPORTED): {ov['critic_precision']:.3f}**")
    if ov["critic_recall"] is not None:
        lines.append(f"**Critic recall (UNSUPPORTED ∩ truly_wrong / all_truly_wrong): {ov['critic_recall']:.3f}**\n")

    lines.append("## By field\n")
    lines.append("| Field | suppressions | correct | false | missed | precision | recall |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for f, d in agg["by_field"].items():
        prec = f"{d['critic_precision']:.3f}" if d["critic_precision"] is not None else "—"
        rec = f"{d['critic_recall']:.3f}" if d["critic_recall"] is not None else "—"
        lines.append(f"| {f} | {d['n_critic_suppressions']} | {d['correct_supp']} | {d['false_supp']} | {d['missed_supp']} | {prec} | {rec} |")
    lines.append("")
    lines.append("## Sample false-suppressions (critic killed a correct extraction)\n")
    if not false_supp_examples:
        lines.append("_None observed in this run._")
    else:
        for x in false_supp_examples[:5]:
            lines.append(f"- `{x['paper_id']}` field=`{x['field']}` no_critic_name=`{x['no_critic_name']}` (gold has match)")
    (output_dir / "report.md").write_text("\n".join(lines))
    console.print(f"[green]Wrote {output_dir / 'report.md'}[/green]")


if __name__ == "__main__":
    app()
