"""Run multi-agent + baseline pipelines on gold benchmarks.

For each loaded benchmark (SciREX, TDMSci, NLP-TDMS):
  - Run multi-agent Pipeline on every paper (resume support)
  - Run BaselinePipeline on every paper
  - Compute the appropriate F1 (span-F1 for SciREX, triple-F1 for TDM datasets)
  - Bootstrap 95% CI on F1
  - Paired permutation test multi-agent vs baseline

Saves per-benchmark records and a summary report.

Usage:
    python scripts/run_benchmarks.py --output-dir outputs/paper_data/phase4_benchmarks
    python scripts/run_benchmarks.py --benchmarks scirex --max-per-benchmark 50
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from paper1.config import load_config
from paper1.loaders import GoldPaper, load_nlp_tdms, load_scirex, load_tdmsci
from paper1.metrics import bootstrap_ci, paired_permutation_test
from paper1.metrics.span_f1 import set_f1
from paper1.metrics.triple_f1 import triple_f1
from paper1.openrouter import OpenRouterClient
from paper1.pipeline import Pipeline
from paper1.pipelines.baseline import BaselinePipeline
from paper1.schema import ContributionRecord
from paper1.voting import _norm

app = typer.Typer(no_args_is_help=False, add_completion=False)
console = Console()


@dataclass
class PaperResult:
    paper_id: str
    status: str
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    wall_time_seconds: float = 0.0
    error: str | None = None
    output_file: str | None = None


def _safe_id(paper_id: str) -> str:
    return paper_id.replace(":", "__").replace("/", "_")


async def _process_paper(
    pipe,  # Pipeline | BaselinePipeline
    paper: GoldPaper,
    out_dir: Path,
    sem: asyncio.Semaphore,
    progress: Progress,
    task_id,
) -> PaperResult:
    safe = _safe_id(paper.paper_id)
    out_path = out_dir / f"{safe}.json"
    if out_path.exists():
        progress.advance(task_id)
        return PaperResult(paper_id=paper.paper_id, status="skipped", output_file=str(out_path))
    if not paper.full_text or len(paper.full_text) < 50:
        progress.advance(task_id)
        return PaperResult(paper_id=paper.paper_id, status="skipped", error="empty full_text")

    async with sem:
        try:
            record = await pipe.extract(paper_id=paper.paper_id, paper_text=paper.full_text)
            out_path.write_text(record.model_dump_json(indent=2, by_alias=True), encoding="utf-8")
            return PaperResult(
                paper_id=paper.paper_id,
                status="ok",
                cost_usd=record.meta.cost_usd,
                tokens_in=record.meta.tokens_in,
                tokens_out=record.meta.tokens_out,
                wall_time_seconds=record.meta.wall_time_seconds,
                output_file=str(out_path),
            )
        except Exception as e:  # noqa: BLE001
            return PaperResult(
                paper_id=paper.paper_id,
                status="error",
                error=f"{type(e).__name__}: {str(e)[:200]}",
            )
        finally:
            progress.advance(task_id)


def _extract_pred_sets(rec: ContributionRecord) -> dict[str, set[str]]:
    """Flatten a ContributionRecord into per-field sets."""
    methods, tasks, datasets, metrics = set(), set(), set(), set()
    for c in rec.contributions:
        n = _norm(c.method.name)
        if n:
            methods.add(n)
        n = _norm(c.task.name)
        if n:
            tasks.add(n)
        for d in c.datasets:
            n = _norm(d.name)
            if n:
                datasets.add(n)
        for m in c.metrics:
            n = _norm(m.name)
            if n:
                metrics.add(n)
    return {"methods": methods, "tasks": tasks, "datasets": datasets, "metrics": metrics}


def _extract_pred_triples(rec: ContributionRecord) -> set[tuple[str, str, str]]:
    out: set[tuple[str, str, str]] = set()
    for c in rec.contributions:
        nt = _norm(c.task.name)
        for d in c.datasets:
            nd = _norm(d.name)
            if not nd or not nt:
                continue
            for m in c.metrics:
                nm = _norm(m.name)
                if nm:
                    out.add((nt, nd, nm))
    return out


def _load_record(path: Path) -> ContributionRecord | None:
    try:
        return ContributionRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _evaluate(
    benchmark: str,
    papers: list[GoldPaper],
    multi_dir: Path,
    base_dir: Path,
    apply_ontology: bool = False,
) -> dict[str, Any]:
    """Compute per-paper F1 for both pipelines, plus aggregates."""
    fields = ("methods", "tasks", "datasets", "metrics")
    multi_per_field: dict[str, list[float]] = {f: [] for f in fields}
    base_per_field: dict[str, list[float]] = {f: [] for f in fields}
    multi_triples: list[float] = []
    base_triples: list[float] = []

    paired_field: dict[str, list[tuple[float, float]]] = {f: [] for f in fields}

    for paper in papers:
        safe = _safe_id(paper.paper_id)
        m_path = multi_dir / f"{safe}.json"
        b_path = base_dir / f"{safe}.json"
        if not (m_path.exists() and b_path.exists()):
            continue
        m_rec = _load_record(m_path)
        b_rec = _load_record(b_path)
        if m_rec is None or b_rec is None:
            continue

        if apply_ontology and benchmark == "scirex":
            from paper1.postprocess import apply_ontology_grounding
            m_rec = apply_ontology_grounding(m_rec)
            b_rec = apply_ontology_grounding(b_rec)

        m_sets = _extract_pred_sets(m_rec)
        b_sets = _extract_pred_sets(b_rec)
        gold_sets = {
            "methods": paper.gold.methods,
            "tasks": paper.gold.tasks,
            "datasets": paper.gold.datasets,
            "metrics": paper.gold.metrics,
        }

        for f in fields:
            if not gold_sets[f]:
                continue
            kind = {"methods": "method", "tasks": "task", "datasets": "dataset", "metrics": "metric"}[f]
            mf1 = set_f1(m_sets[f], gold_sets[f], lenient=True, kind=kind)["f1"]
            bf1 = set_f1(b_sets[f], gold_sets[f], lenient=True, kind=kind)["f1"]
            multi_per_field[f].append(mf1)
            base_per_field[f].append(bf1)
            paired_field[f].append((mf1, bf1))

        if paper.gold.triples:
            m_tri = triple_f1(_extract_pred_triples(m_rec), paper.gold.triples)["f1"]
            b_tri = triple_f1(_extract_pred_triples(b_rec), paper.gold.triples)["f1"]
            multi_triples.append(m_tri)
            base_triples.append(b_tri)

    # Aggregate
    out: dict[str, Any] = {"benchmark": benchmark, "fields": {}, "triples": None}
    for f in fields:
        vals_m = multi_per_field[f]
        vals_b = base_per_field[f]
        if not vals_m:
            out["fields"][f] = None
            continue
        m_mean, m_lo, m_hi = bootstrap_ci(vals_m, n_resamples=1000)
        b_mean, b_lo, b_hi = bootstrap_ci(vals_b, n_resamples=1000)
        a_paired = [a for a, _ in paired_field[f]]
        b_paired = [b for _, b in paired_field[f]]
        p = paired_permutation_test(a_paired, b_paired, n_permutations=5000)
        out["fields"][f] = {
            "n": len(vals_m),
            "multi_agent": {"f1": m_mean, "ci_lo": m_lo, "ci_hi": m_hi},
            "baseline": {"f1": b_mean, "ci_lo": b_lo, "ci_hi": b_hi},
            "p_value": p,
        }
    if multi_triples:
        m_mean, m_lo, m_hi = bootstrap_ci(multi_triples, n_resamples=1000)
        b_mean, b_lo, b_hi = bootstrap_ci(base_triples, n_resamples=1000)
        p = paired_permutation_test(multi_triples, base_triples, n_permutations=5000)
        out["triples"] = {
            "n": len(multi_triples),
            "multi_agent": {"f1": m_mean, "ci_lo": m_lo, "ci_hi": m_hi},
            "baseline": {"f1": b_mean, "ci_lo": b_lo, "ci_hi": b_hi},
            "p_value": p,
        }
    return out


@app.command()
def run(
    output_dir: Path = typer.Option(Path("outputs/paper_data/phase4_benchmarks"), "--output-dir"),
    benchmarks: str = typer.Option("scirex,tdmsci,nlp_tdms", "--benchmarks"),
    max_per_benchmark: int | None = typer.Option(None, "--max-per-benchmark"),
    concurrency: int = typer.Option(50, "--concurrency"),
    config_path: Path | None = typer.Option(None, "--config"),
    skip_extraction: bool = typer.Option(False, "--skip-extraction", help="Only re-evaluate from cached outputs"),
    prompt_path: Path | None = typer.Option(None, "--prompt", help="Path to a custom extractor prompt (e.g. extractor_fewshot.md)"),
    apply_ontology: bool = typer.Option(False, "--apply-ontology", help="Snap extracted entities to the SciREX ontology before scoring"),
    client_kind: str = typer.Option("openrouter", "--client", help="Inference router: 'openrouter' (default) or 'together' (route extractor to Together)"),
    scirex_split: str = typer.Option("dev", "--scirex-split", help="SciREX split for evaluation: dev (default) or test"),
) -> None:
    cfg = load_config(config_path=config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if scirex_split not in ("dev", "test"):
        console.print(f"[red]Invalid --scirex-split: {scirex_split} (use 'dev' or 'test')[/red]")
        raise typer.Exit(1)
    # SciREX: standard evaluation uses the dev split (66 papers); use --scirex-split test for the
    # held-out test split (also 66 papers) — head-to-head with Jain 2020's reported numbers.
    bench_funcs = {
        "scirex": lambda: load_scirex(splits=(scirex_split,)),
        "tdmsci": load_tdmsci,
        "nlp_tdms": load_nlp_tdms,
    }
    selected = [b for b in benchmarks.split(",") if b.strip() in bench_funcs]
    if not selected:
        console.print("[red]No valid benchmarks selected[/red]")
        raise typer.Exit(1)

    asyncio.run(_main(cfg, selected, bench_funcs, output_dir, concurrency, max_per_benchmark, skip_extraction, prompt_path, apply_ontology, client_kind))


async def _main(cfg, selected: list[str], bench_funcs: dict, output_dir: Path, concurrency: int, max_per_benchmark: int | None, skip_extraction: bool, prompt_path: Path | None = None, apply_ontology: bool = False, client_kind: str = "openrouter") -> None:
    or_client = OpenRouterClient(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        timeout_s=cfg.defaults.request_timeout_s,
        max_retries=cfg.defaults.max_retries,
        retry_backoff_s=cfg.defaults.retry_backoff_s,
        referer=cfg.referer,
        title=cfg.title,
    )
    if client_kind == "together":
        import os
        from paper1.together_client import RoutingClient, TogetherClient
        tg_key = os.environ.get("TOGETHER_API_KEY", "")
        if not tg_key:
            raise RuntimeError("--client together requires TOGETHER_API_KEY in .env")
        tg_client = TogetherClient(api_key=tg_key, timeout_s=cfg.defaults.request_timeout_s)
        # Route the extractor model_id through Together; everything else stays on OpenRouter.
        client = RoutingClient(or_client, tg_client, together_models={cfg.extractor.model_id})
        console.print(f"[bold]Routing extractor[/bold] {cfg.extractor.model_id} [bold]via Together[/bold]; critic+consolidator via OpenRouter")
    else:
        client = or_client
    if prompt_path is not None:
        from paper1.pipelines.few_shot import make_few_shot_pipeline
        multi = make_few_shot_pipeline(client, cfg, prompt_path)
        console.print(f"[bold]Using custom extractor prompt:[/bold] {prompt_path}")
    else:
        multi = Pipeline(client, cfg)
    base = BaselinePipeline(client, cfg)

    all_results: dict[str, dict[str, Any]] = {}
    total_cost = 0.0

    for benchmark in selected:
        console.print(f"\n[bold cyan]Benchmark: {benchmark}[/bold cyan]")
        papers = bench_funcs[benchmark]()
        # Filter to those with usable text and only first max_per_benchmark
        papers = [p for p in papers if p.full_text and len(p.full_text) >= 50]
        if max_per_benchmark is not None:
            papers = papers[:max_per_benchmark]
        console.print(f"  papers (with text): {len(papers)}")

        bench_dir = output_dir / benchmark
        multi_dir = bench_dir / "multi_agent"
        base_dir = bench_dir / "baseline"
        multi_dir.mkdir(parents=True, exist_ok=True)
        base_dir.mkdir(parents=True, exist_ok=True)

        if not skip_extraction:
            sem = asyncio.Semaphore(concurrency)
            for phase_name, pipe, out_d in (
                ("multi_agent", multi, multi_dir),
                ("baseline", base, base_dir),
            ):
                t0 = time.perf_counter()
                console.print(f"  [bold]{phase_name}[/bold] extracting...")
                with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), console=console) as progress:
                    task_id = progress.add_task(f"{benchmark}/{phase_name}", total=len(papers))
                    results = await asyncio.gather(*(_process_paper(pipe, p, out_d, sem, progress, task_id) for p in papers))
                phase_cost = sum(r.cost_usd for r in results if r.status == "ok")
                total_cost += phase_cost
                ok = sum(1 for r in results if r.status == "ok")
                err = sum(1 for r in results if r.status == "error")
                skip = sum(1 for r in results if r.status == "skipped")
                summary = {
                    "benchmark": benchmark,
                    "phase": phase_name,
                    "papers_total": len(results),
                    "papers_ok": ok,
                    "papers_error": err,
                    "papers_skipped": skip,
                    "total_cost_usd": phase_cost,
                    "wall_time_seconds": time.perf_counter() - t0,
                    "per_paper": [asdict(r) for r in results],
                }
                (bench_dir / f"{phase_name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
                console.print(f"    {phase_name}: ok={ok} err={err} skip={skip} ${phase_cost:.4f}")

        console.print(f"  [bold]evaluating[/bold]...")
        eval_result = _evaluate(benchmark, papers, multi_dir, base_dir, apply_ontology=apply_ontology)
        (bench_dir / "evaluation.json").write_text(json.dumps(eval_result, indent=2), encoding="utf-8")
        all_results[benchmark] = eval_result

    await client.aclose()

    # Summary report
    lines = ["# Phase 4 — Benchmark evaluation\n"]
    lines.append(f"_Total OpenRouter spend (this phase): ${total_cost:.4f}_\n")

    # Detailed per-benchmark per-field tables
    lines.append("## Per-field F1 (lenient set match) with bootstrap 95% CIs and paired permutation p-values\n")
    for benchmark, ev in all_results.items():
        lines.append(f"### {benchmark}\n")
        lines.append("| Field | n | Multi-agent F1 [95% CI] | Baseline F1 [95% CI] | p (multi vs base) |")
        lines.append("|---|---:|---|---|---:|")
        for f in ("methods", "tasks", "datasets", "metrics"):
            row = ev["fields"].get(f)
            if row is None:
                lines.append(f"| {f} | — | — | — | — |")
                continue
            sig = " *" if row["p_value"] < 0.05 else ""
            lines.append(
                f"| {f} | {row['n']} | "
                f"{row['multi_agent']['f1']:.3f} [{row['multi_agent']['ci_lo']:.3f}, {row['multi_agent']['ci_hi']:.3f}] | "
                f"{row['baseline']['f1']:.3f} [{row['baseline']['ci_lo']:.3f}, {row['baseline']['ci_hi']:.3f}] | "
                f"{row['p_value']:.4f}{sig} |"
            )
        if ev["triples"]:
            t = ev["triples"]
            sig = " *" if t["p_value"] < 0.05 else ""
            lines.append(
                f"| **(T,D,M) triple** | {t['n']} | "
                f"{t['multi_agent']['f1']:.3f} [{t['multi_agent']['ci_lo']:.3f}, {t['multi_agent']['ci_hi']:.3f}] | "
                f"{t['baseline']['f1']:.3f} [{t['baseline']['ci_lo']:.3f}, {t['baseline']['ci_hi']:.3f}] | "
                f"{t['p_value']:.4f}{sig} |"
            )
        lines.append("")

    # Comparison vs published
    lines.append("## Comparison to published prior work\n")
    lines.append(
        "Published numbers below are author-reported on the *public test split* of "
        "each benchmark — not directly comparable to our F1 (we report on whatever "
        "subset we ran, set-level lenient F1 after the multi-agent pipeline). "
        "Use as a directional sanity-check, not as a head-to-head leaderboard.\n"
    )
    lines.append("| Benchmark | Field | Our F1 | Published | System | Source |")
    lines.append("|---|---|---:|---:|---|---|")

    def _our_f1(bench: str, field: str) -> str:
        ev = all_results.get(bench)
        if not ev:
            return "—"
        if field == "triples":
            t = ev.get("triples")
            return f"{t['multi_agent']['f1']:.3f}" if t else "—"
        row = ev.get("fields", {}).get(field)
        return f"{row['multi_agent']['f1']:.3f}" if row else "—"

    lines.append(
        f"| SciREX | methods (entity F1) | {_our_f1('scirex','methods')} | 0.567 | "
        "SciREX joint model | Jain et al. 2020, ACL |"
    )
    lines.append(
        f"| SciREX | tasks (entity F1) | {_our_f1('scirex','tasks')} | 0.610 | "
        "SciREX joint model | Jain et al. 2020, ACL |"
    )
    lines.append(
        f"| SciREX | datasets (entity F1) | {_our_f1('scirex','datasets')} | 0.553 | "
        "SciREX joint model | Jain et al. 2020, ACL |"
    )
    lines.append(
        f"| SciREX | metrics (entity F1) | {_our_f1('scirex','metrics')} | 0.553 | "
        "SciREX joint model | Jain et al. 2020, ACL |"
    )
    lines.append(
        f"| SciREX | datasets (entity F1) | {_our_f1('scirex','datasets')} | ~0.62 | "
        "DyGIE++ | Wadden et al. 2019, EMNLP |"
    )
    lines.append(
        f"| TDMSci | (T,D,M) triple F1 | {_our_f1('tdmsci','triples')} | 0.452 | "
        "Hou et al. 2019 BiLSTM-CRF | Hou et al. 2019, ACL |"
    )
    lines.append(
        f"| NLP-TDMS | (T,D,M) triple F1 | {_our_f1('nlp_tdms','triples')} | 0.317 | "
        "BERT-classifier baseline | Mondal et al. 2021 |"
    )
    lines.append("")

    (output_dir / "results_table.md").write_text("\n".join(lines), encoding="utf-8")
    console.print(f"\n[green]Wrote {output_dir / 'results_table.md'}[/green]")


if __name__ == "__main__":
    app()
