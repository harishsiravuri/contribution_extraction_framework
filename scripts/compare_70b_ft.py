"""Paired comparisons for v6 70B FT extractor on SciREX dev.

Computes 70B-FT vs each prior system on the *intersection* of papers where both
systems have a successful prediction:
  - 70B FT vs paired single-LLM baseline (same FT 70B, fewer agents)
  - 70B FT vs 8B FT (v6 8B multi-agent on SciREX dev)
  - 70B FT vs v3 zero-shot multi-agent (Llama 3.3 70B, no fine-tuning)

For each field reports F1, 95% bootstrap CI, paired permutation p-value.
Writes JSON to outputs/paper_data_v6/benchmarks_ft_70b/scirex/comparisons.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from paper1.loaders import GoldPaper, load_scirex
from paper1.metrics import bootstrap_ci, paired_permutation_test
from paper1.metrics.span_f1 import set_f1
from paper1.metrics.triple_f1 import triple_f1
from paper1.schema import ContributionRecord
from paper1.voting import _norm

FIELDS = ("methods", "tasks", "datasets", "metrics")
KIND = {"methods": "method", "tasks": "task", "datasets": "dataset", "metrics": "metric"}


def _safe_id(paper_id: str) -> str:
    return paper_id.replace(":", "__").replace("/", "_")


def _load(path: Path) -> ContributionRecord | None:
    if not path.exists():
        return None
    try:
        return ContributionRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _sets(rec: ContributionRecord) -> dict[str, set[str]]:
    out = {f: set() for f in FIELDS}
    for c in rec.contributions:
        if (n := _norm(c.method.name)):
            out["methods"].add(n)
        if (n := _norm(c.task.name)):
            out["tasks"].add(n)
        for d in c.datasets:
            if (n := _norm(d.name)):
                out["datasets"].add(n)
        for m in c.metrics:
            if (n := _norm(m.name)):
                out["metrics"].add(n)
    return out


def _triples(rec: ContributionRecord) -> set[tuple[str, str, str]]:
    out: set[tuple[str, str, str]] = set()
    for c in rec.contributions:
        nt = _norm(c.task.name)
        if not nt:
            continue
        for d in c.datasets:
            nd = _norm(d.name)
            if not nd:
                continue
            for m in c.metrics:
                nm = _norm(m.name)
                if nm:
                    out.add((nt, nd, nm))
    return out


@dataclass
class FieldStats:
    n: int
    a_f1: float
    b_f1: float
    a_ci: tuple[float, float]
    b_ci: tuple[float, float]
    p_value: float
    delta: float


def _paired_field(
    a_recs: dict[str, ContributionRecord],
    b_recs: dict[str, ContributionRecord],
    papers: list[GoldPaper],
    field: str,
) -> FieldStats | None:
    a_scores: list[float] = []
    b_scores: list[float] = []
    kind = KIND[field]
    for p in papers:
        a = a_recs.get(p.paper_id)
        b = b_recs.get(p.paper_id)
        if a is None or b is None:
            continue
        gold = getattr(p.gold, field)
        if not gold:
            continue
        a_sets = _sets(a)
        b_sets = _sets(b)
        a_scores.append(set_f1(a_sets[field], gold, lenient=True, kind=kind)["f1"])
        b_scores.append(set_f1(b_sets[field], gold, lenient=True, kind=kind)["f1"])
    if not a_scores:
        return None
    a_mean, a_lo, a_hi = bootstrap_ci(a_scores)
    b_mean, b_lo, b_hi = bootstrap_ci(b_scores)
    p = paired_permutation_test(a_scores, b_scores)
    return FieldStats(
        n=len(a_scores),
        a_f1=a_mean,
        b_f1=b_mean,
        a_ci=(a_lo, a_hi),
        b_ci=(b_lo, b_hi),
        p_value=p,
        delta=a_mean - b_mean,
    )


def _paired_triple(
    a_recs: dict[str, ContributionRecord],
    b_recs: dict[str, ContributionRecord],
    papers: list[GoldPaper],
) -> FieldStats | None:
    a_scores: list[float] = []
    b_scores: list[float] = []
    for p in papers:
        a = a_recs.get(p.paper_id)
        b = b_recs.get(p.paper_id)
        if a is None or b is None:
            continue
        if not p.gold.triples:
            continue
        a_scores.append(triple_f1(_triples(a), p.gold.triples)["f1"])
        b_scores.append(triple_f1(_triples(b), p.gold.triples)["f1"])
    if not a_scores:
        return None
    a_mean, a_lo, a_hi = bootstrap_ci(a_scores)
    b_mean, b_lo, b_hi = bootstrap_ci(b_scores)
    p = paired_permutation_test(a_scores, b_scores)
    return FieldStats(
        n=len(a_scores),
        a_f1=a_mean,
        b_f1=b_mean,
        a_ci=(a_lo, a_hi),
        b_ci=(b_lo, b_hi),
        p_value=p,
        delta=a_mean - b_mean,
    )


def _load_dir(d: Path) -> dict[str, ContributionRecord]:
    out: dict[str, ContributionRecord] = {}
    if not d.exists():
        return out
    for f in d.glob("*.json"):
        rec = _load(f)
        if rec is None:
            continue
        # Filename is `scirex__<sha>.json` → paper_id `scirex:<sha>`
        pid = f.stem.replace("scirex__", "scirex:")
        out[pid] = rec
    return out


def _to_dict(s: FieldStats | None, label_a: str, label_b: str) -> dict | None:
    if s is None:
        return None
    return {
        "n": s.n,
        label_a: {"f1": s.a_f1, "ci_lo": s.a_ci[0], "ci_hi": s.a_ci[1]},
        label_b: {"f1": s.b_f1, "ci_lo": s.b_ci[0], "ci_hi": s.b_ci[1]},
        "delta": s.delta,
        "p_value": s.p_value,
    }


def main() -> None:
    papers = [p for p in load_scirex(splits=("dev",)) if p.full_text and len(p.full_text) >= 50][:66]
    print(f"papers: {len(papers)}")

    # Load all systems
    ft70 = _load_dir(Path("outputs/paper_data_v6/benchmarks_ft_70b/scirex/multi_agent"))
    ft70_base = _load_dir(Path("outputs/paper_data_v6/benchmarks_ft_70b/scirex/baseline"))
    ft8 = _load_dir(Path("outputs/paper_data_v6/benchmarks_ft_8b/scirex/multi_agent"))
    v3 = _load_dir(Path("outputs/paper_data_v3/benchmarks/scirex/multi_agent"))
    print(f"70B FT multi: {len(ft70)}  70B FT baseline: {len(ft70_base)}  8B FT multi: {len(ft8)}  v3 multi: {len(v3)}")

    out: dict = {"benchmark": "scirex"}

    # Comparison 1: 70B FT multi vs 70B FT baseline (same model, fewer agents)
    block: dict = {}
    for f in FIELDS:
        s = _paired_field(ft70, ft70_base, papers, f)
        block[f] = _to_dict(s, "ft_70b_multi", "ft_70b_baseline")
    block["triple"] = _to_dict(_paired_triple(ft70, ft70_base, papers), "ft_70b_multi", "ft_70b_baseline")
    out["vs_baseline_70b"] = block

    # Comparison 2: 70B FT vs 8B FT
    block = {}
    for f in FIELDS:
        s = _paired_field(ft70, ft8, papers, f)
        block[f] = _to_dict(s, "ft_70b", "ft_8b")
    block["triple"] = _to_dict(_paired_triple(ft70, ft8, papers), "ft_70b", "ft_8b")
    out["vs_8b_ft"] = block

    # Comparison 3: 70B FT vs v3 zero-shot multi-agent
    block = {}
    for f in FIELDS:
        s = _paired_field(ft70, v3, papers, f)
        block[f] = _to_dict(s, "ft_70b", "v3_zeroshot")
    block["triple"] = _to_dict(_paired_triple(ft70, v3, papers), "ft_70b", "v3_zeroshot")
    out["vs_v3_zeroshot"] = block

    out_path = Path("outputs/paper_data_v6/benchmarks_ft_70b/scirex/comparisons.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")

    # Pretty-print summary
    def _row(label: str, s: dict | None) -> str:
        if s is None:
            return f"  {label}: <no overlap>"
        a_label = next(k for k in s if k not in ("n", "delta", "p_value"))
        b_label = next(k for k in s if k not in ("n", "delta", "p_value", a_label))
        sig = "***" if s["p_value"] < 0.001 else ("**" if s["p_value"] < 0.01 else ("*" if s["p_value"] < 0.05 else ""))
        return f"  {label:>10s}  n={s['n']:3d}  {a_label}={s[a_label]['f1']:.3f}  {b_label}={s[b_label]['f1']:.3f}  Δ={s['delta']:+.3f}  p={s['p_value']:.4f} {sig}"

    for cmp_name in ("vs_baseline_70b", "vs_8b_ft", "vs_v3_zeroshot"):
        print(f"\n=== {cmp_name} ===")
        for f in (*FIELDS, "triple"):
            print(_row(f, out[cmp_name][f]))


if __name__ == "__main__":
    main()
