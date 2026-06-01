"""E7 — Deployment case study on 100 recent arXiv NLP papers.

Reads the fetched papers from outputs/paper_data_v8/deployment_case_study/papers/,
runs the default open-weights framework, then computes usage statistics.

Output: outputs/paper_data_v8/deployment_case_study/
  records/<arxiv_id>.json
  extraction_summary.json
  results.json — mean contribs/paper, top-20 methods/datasets/metrics,
                 self_consistency histogram, mean per-field calibrated conf,
                 total wall + cost
  examples.md — three illustrative paper-level outputs
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn,
)

from paper1.config import load_config
from paper1.metrics.temperature_scaling import apply_temperature
from paper1.openrouter import OpenRouterClient
from paper1.pipeline import Pipeline
from paper1.schema import ContributionRecord
from paper1.voting import _norm

SEED = 42

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


async def _run_one(pipe, paper_id, paper_text, out_dir, sem, progress, task_id):
    safe = _safe_id(paper_id)
    out_path = out_dir / f"{safe}.json"
    if out_path.exists():
        progress.advance(task_id)
        return PaperResult(paper_id=paper_id, status="skipped", output_file=str(out_path))
    async with sem:
        try:
            record = await pipe.extract(paper_id=paper_id, paper_text=paper_text)
            out_path.write_text(record.model_dump_json(indent=2, by_alias=True), encoding="utf-8")
            return PaperResult(
                paper_id=paper_id, status="ok",
                cost_usd=record.meta.cost_usd,
                tokens_in=record.meta.tokens_in,
                tokens_out=record.meta.tokens_out,
                wall_time_seconds=record.meta.wall_time_seconds,
                output_file=str(out_path),
            )
        except Exception as e:  # noqa: BLE001
            return PaperResult(paper_id=paper_id, status="error",
                               error=f"{type(e).__name__}: {str(e)[:200]}")
        finally:
            progress.advance(task_id)


@app.command()
def extract(
    papers_dir: Path = typer.Option(Path("outputs/paper_data_v8/deployment_case_study/papers"), "--papers-dir"),
    records_dir: Path = typer.Option(Path("outputs/paper_data_v8/deployment_case_study/records"), "--records-dir"),
    summary_path: Path = typer.Option(Path("outputs/paper_data_v8/deployment_case_study/extraction_summary.json"), "--summary-path"),
    config_path: Path = typer.Option(Path("config/models.yaml"), "--config"),
    concurrency: int = typer.Option(10, "--concurrency"),
):
    records_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(config_path=config_path)
    or_client = OpenRouterClient(
        api_key=cfg.api_key, base_url=cfg.base_url,
        timeout_s=cfg.defaults.request_timeout_s,
        max_retries=cfg.defaults.max_retries,
        retry_backoff_s=cfg.defaults.retry_backoff_s,
        referer=cfg.referer, title=cfg.title,
    )
    multi = Pipeline(or_client, cfg)

    paper_files = sorted(papers_dir.glob("arxiv_*.txt"))
    console.print(f"  papers in dir: {len(paper_files)}")
    asyncio.run(_main(paper_files, multi, records_dir, summary_path, concurrency))


async def _main(paper_files, multi, records_dir, summary_path, concurrency):
    sem = asyncio.Semaphore(concurrency)
    t0 = time.perf_counter()
    inputs = []
    for f in paper_files:
        arxiv_id = f.stem.removeprefix("arxiv_")
        text = f.read_text(encoding="utf-8")
        inputs.append((f"arxiv:{arxiv_id}", text))
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                  MofNCompleteColumn(), TimeElapsedColumn(), console=console) as progress:
        task_id = progress.add_task("case-study", total=len(inputs))
        results = await asyncio.gather(*(
            _run_one(multi, pid, txt, records_dir, sem, progress, task_id) for pid, txt in inputs
        ))
    ok = sum(1 for r in results if r.status == "ok")
    err = sum(1 for r in results if r.status == "error")
    skip = sum(1 for r in results if r.status == "skipped")
    cost = sum(r.cost_usd for r in results if r.status == "ok")
    elapsed = time.perf_counter() - t0
    console.print(f"  done: ok={ok} err={err} skip={skip} cost=${cost:.4f} wall={elapsed:.1f}s")
    summary = {
        "papers_total": len(results), "papers_ok": ok, "papers_error": err, "papers_skipped": skip,
        "total_cost_usd": cost, "wall_time_seconds": elapsed,
        "per_paper": [asdict(r) for r in results],
    }
    summary_path.write_text(json.dumps(summary, indent=2))


@app.command()
def analyze(
    records_dir: Path = typer.Option(Path("outputs/paper_data_v8/deployment_case_study/records"), "--records-dir"),
    summary_path: Path = typer.Option(Path("outputs/paper_data_v8/deployment_case_study/extraction_summary.json"), "--summary-path"),
    calib_path: Path = typer.Option(Path("outputs/paper_data_v2/calibration/calibration.json"), "--calib-path"),
    output_dir: Path = typer.Option(Path("outputs/paper_data_v8/deployment_case_study"), "--output-dir"),
):
    """Build results.json + examples.md from extracted records."""
    records: list[ContributionRecord] = []
    for f in sorted(records_dir.glob("*.json")):
        try:
            records.append(ContributionRecord.model_validate_json(f.read_text(encoding="utf-8")))
        except Exception:
            continue
    console.print(f"  loaded {len(records)} records")

    # Mean contributions per paper
    contribs_per_paper = [len(r.contributions) for r in records]
    mean_contribs = float(np.mean(contribs_per_paper)) if contribs_per_paper else 0.0

    # Self_consistency histogram (all populated fields)
    confs: list[float] = []
    for r in records:
        for c in r.contributions:
            confs.append(c.self_consistency)
    edges = list(np.linspace(0.0, 1.0, 11))
    hist, _ = np.histogram(confs, bins=edges)
    hist_rows = [{"bin_lo": edges[i], "bin_hi": edges[i+1], "n": int(hist[i])}
                 for i in range(len(hist))]

    # Mean per-field calibrated confidence
    T_per_field: dict[str, float | None] = {}
    if calib_path.exists():
        c = json.loads(calib_path.read_text())["per_field"]
        for f in ("method.name", "task.name", "datasets", "metrics"):
            T_per_field[f] = (c.get(f) or {}).get("T")

    # All confidences are global (per contribution self_consistency, not per-field)
    # Apply each field's T to that global confidence to get "field-conditioned" calibrated estimates
    mean_cal_conf = {}
    if confs:
        for f, T in T_per_field.items():
            if T:
                scaled = apply_temperature(confs, T)
                mean_cal_conf[f] = float(np.mean(scaled))
            else:
                mean_cal_conf[f] = None
    mean_cal_conf["uncalibrated"] = float(np.mean(confs)) if confs else 0.0

    # Top 20 methods / datasets / metrics
    methods, datasets, metrics_c = Counter(), Counter(), Counter()
    for r in records:
        for c in r.contributions:
            if c.method and c.method.name:
                n = _norm(c.method.name)
                if n:
                    methods[n] += 1
            for d in c.datasets:
                if d.name:
                    n = _norm(d.name)
                    if n:
                        datasets[n] += 1
            for m in c.metrics:
                if m.name:
                    n = _norm(m.name)
                    if n:
                        metrics_c[n] += 1
    top20 = lambda c: [{"name": k, "count": v} for k, v in c.most_common(20)]

    # Wall + cost
    summary = json.loads(summary_path.read_text())

    # Three representative examples — pick records that:
    #  - have >=2 contributions, AND
    #  - have at least one populated dataset/metric pair
    # Sort by total mentions desc and pick first 3 distinct papers
    def _score(r: ContributionRecord) -> int:
        s = 0
        for c in r.contributions:
            if c.method and c.method.name:
                s += 1
            if c.task and c.task.name:
                s += 1
            s += len([d for d in c.datasets if d.name])
            s += len([m for m in c.metrics if m.name])
        return s
    sorted_recs = sorted(records, key=_score, reverse=True)
    examples = []
    for r in sorted_recs[:3]:
        examples.append({
            "paper_id": r.paper_id,
            "n_contributions": len(r.contributions),
            "record": json.loads(r.model_dump_json(by_alias=True)),
        })

    # Write examples.md
    md_lines = ["# E7 — Three illustrative case-study extractions\n"]
    for i, ex in enumerate(examples, 1):
        md_lines.append(f"## Example {i}: `{ex['paper_id']}`")
        md_lines.append(f"\n_Number of contribution records emitted: **{ex['n_contributions']}**_\n")
        for j, c in enumerate(ex["record"]["contributions"]):
            md_lines.append(f"### Contribution {j+1}")
            method_n = (c.get("method") or {}).get("name")
            task_n = (c.get("task") or {}).get("name")
            md_lines.append(f"- **Method:** {method_n}")
            md_lines.append(f"- **Task:** {task_n}")
            ds = ", ".join(d.get("name", "?") for d in (c.get("datasets") or []))
            md_lines.append(f"- **Datasets:** {ds or '(none)'}")
            ms = ", ".join(f"{m.get('name','?')}" + (f"={m.get('value')}" if m.get('value') is not None else "")
                           for m in (c.get("metrics") or []))
            md_lines.append(f"- **Metrics:** {ms or '(none)'}")
            md_lines.append(f"- **Claim strength:** {c.get('claim_strength')}")
            md_lines.append(f"- **Self-consistency:** {c.get('self_consistency', 0):.2f}")
            md_lines.append("")
    (output_dir / "examples.md").write_text("\n".join(md_lines), encoding="utf-8")
    console.print(f"  wrote {output_dir / 'examples.md'}")

    result = {
        "experiment": "E7_deployment_case_study",
        "seed": SEED,
        "papers_total": summary.get("papers_total"),
        "papers_ok": summary.get("papers_ok"),
        "papers_error": summary.get("papers_error"),
        "wall_time_seconds": summary.get("wall_time_seconds"),
        "total_cost_usd": summary.get("total_cost_usd"),
        "cost_per_paper_usd": summary.get("total_cost_usd", 0.0) / max(summary.get("papers_ok") or 1, 1),
        "mean_contributions_per_paper": mean_contribs,
        "contributions_per_paper_distribution": dict(Counter(contribs_per_paper).most_common()),
        "self_consistency_histogram": hist_rows,
        "mean_calibrated_confidence_per_field_applied": mean_cal_conf,
        "scirex_T_per_field": T_per_field,
        "top20_methods": top20(methods),
        "top20_datasets": top20(datasets),
        "top20_metrics": top20(metrics_c),
        "n_distinct_methods": len(methods),
        "n_distinct_datasets": len(datasets),
        "n_distinct_metrics": len(metrics_c),
        "examples": [{"paper_id": e["paper_id"], "n_contributions": e["n_contributions"]} for e in examples],
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2))
    console.print(f"  wrote {output_dir / 'results.json'}")


if __name__ == "__main__":
    app()
