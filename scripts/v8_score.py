"""Score E6 (binding ablation) and tidy v8 summaries.

E6: paired-permutation per-field F1 vs v3 default-deployment on same 30 papers.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import typer
from rich.console import Console

from paper1.loaders import load_scirex
from paper1.metrics import bootstrap_ci, paired_permutation_test
from paper1.metrics.span_f1 import set_f1
from paper1.schema import ContributionRecord
from paper1.voting import _norm

SEED = 42
N_BOOT = 1000
N_PERM = 10000

app = typer.Typer(no_args_is_help=False, add_completion=False)
console = Console()


def _safe_id(paper_id: str) -> str:
    return paper_id.replace(":", "__").replace("/", "_")


def _load_dir(d: Path) -> dict[str, ContributionRecord]:
    out: dict[str, ContributionRecord] = {}
    for f in d.glob("*.json"):
        try:
            rec = ContributionRecord.model_validate_json(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        pid = f.stem.replace("__", ":")
        out[pid] = rec
    return out


def _sets(rec: ContributionRecord) -> dict[str, set[str]]:
    out = {f: set() for f in ("methods", "tasks", "datasets", "metrics")}
    for c in rec.contributions:
        if (n := _norm(c.method.name)): out["methods"].add(n)
        if (n := _norm(c.task.name)): out["tasks"].add(n)
        for d in c.datasets:
            if (n := _norm(d.name)): out["datasets"].add(n)
        for m in c.metrics:
            if (n := _norm(m.name)): out["metrics"].add(n)
    return out


@app.command()
def binding_ablation():
    base = Path("outputs/paper_data_v8/binding_ablation")
    sampled_ids = set(json.loads((base / "sampled_papers.json").read_text()))
    papers = [p for p in load_scirex(splits=("dev",)) if p.paper_id in sampled_ids]
    console.print(f"  sampled: {len(papers)}")

    no_binding = _load_dir(base / "multi_agent")
    default = _load_dir(Path("outputs/paper_data_v3/benchmarks/scirex/multi_agent"))
    console.print(f"  no_binding records: {len(no_binding)}  v3 default available: {len(default)}")

    KIND = {"methods": "method", "tasks": "task", "datasets": "dataset", "metrics": "metric"}
    out_fields: dict = {}
    for f in ("methods", "tasks", "datasets", "metrics"):
        a_scores, b_scores = [], []
        for p in papers:
            ra = no_binding.get(p.paper_id)
            rb = default.get(p.paper_id)
            if ra is None or rb is None:
                continue
            gold = getattr(p.gold, f)
            if not gold:
                continue
            a_set = _sets(ra)[f]
            b_set = _sets(rb)[f]
            a_scores.append(set_f1(a_set, gold, lenient=True, kind=KIND[f])["f1"])
            b_scores.append(set_f1(b_set, gold, lenient=True, kind=KIND[f])["f1"])
        if not a_scores:
            out_fields[f] = None
            continue
        a_rng = np.random.default_rng(SEED)
        b_rng = np.random.default_rng(SEED)
        p_rng = np.random.default_rng(SEED)
        a_mean, a_lo, a_hi = bootstrap_ci(a_scores, n_resamples=N_BOOT, rng=a_rng)
        b_mean, b_lo, b_hi = bootstrap_ci(b_scores, n_resamples=N_BOOT, rng=b_rng)
        pval = paired_permutation_test(a_scores, b_scores, n_permutations=N_PERM, rng=p_rng)
        out_fields[f] = {
            "n": len(a_scores),
            "no_binding_f1": a_mean,
            "no_binding_ci_lo": a_lo, "no_binding_ci_hi": a_hi,
            "default_binding_f1": b_mean,
            "default_binding_ci_lo": b_lo, "default_binding_ci_hi": b_hi,
            "delta_no_binding_minus_default": a_mean - b_mean,
            "p_value_paired_perm": pval,
        }

    # Contributions-per-paper distribution comparison
    def _mean_cpp(recs):
        cs = [len(r.contributions) for r in recs.values()]
        return float(np.mean(cs)) if cs else 0.0, cs
    mean_nb, dist_nb = _mean_cpp(no_binding)
    mean_def, dist_def = _mean_cpp({k: default[k] for k in no_binding if k in default})

    extr = json.loads((base / "extraction_summary.json").read_text())
    result = {
        "experiment": "E6_binding_ablation",
        "seed": SEED,
        "n_resamples_bootstrap": N_BOOT,
        "n_permutations_paired": N_PERM,
        "sample_size_requested": 30,
        "papers_no_binding_ok": len(no_binding),
        "f1_per_field": out_fields,
        "mean_contributions_per_paper_no_binding": mean_nb,
        "mean_contributions_per_paper_default_on_same_papers": mean_def,
        "no_binding_total_cost_usd": extr.get("total_cost_usd"),
    }
    out_path = base / "results.json"
    out_path.write_text(json.dumps(result, indent=2))
    console.print(f"[green]Wrote {out_path}[/green]")


if __name__ == "__main__":
    app()
