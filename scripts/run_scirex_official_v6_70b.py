"""Run the official SciREX evaluator on the v6 70B FT multi-agent predictions."""

from __future__ import annotations

import json
from pathlib import Path

from paper1.loaders import load_scirex
from paper1.metrics.scirex_official import (
    build_prediction_files,
    parse_evaluator_output,
    run_official_evaluator,
)
from paper1.schema import ContributionRecord


def _load_dir(d: Path) -> dict[str, ContributionRecord]:
    out: dict[str, ContributionRecord] = {}
    for f in d.glob("*.json"):
        try:
            rec = ContributionRecord.model_validate_json(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        pid = f.stem.replace("scirex__", "scirex:")
        out[pid] = rec
    return out


def main() -> None:
    papers = [p for p in load_scirex(splits=("dev",)) if p.full_text and len(p.full_text) >= 50][:66]
    records = _load_dir(Path("outputs/paper_data_v6/benchmarks_ft_70b/scirex/multi_agent"))
    print(f"papers={len(papers)} records={len(records)}")

    out_dir = Path("outputs/paper_data_v6/scirex_official_pred_v6_70b")
    info = build_prediction_files(records=records, gold_papers=papers, out_dir=out_dir)
    print(f"wrote pred files: {info}")

    proc = run_official_evaluator(pred_dir=out_dir, gold_split="dev")
    print(f"evaluator returncode: {proc['returncode']}")
    if proc["returncode"] != 0:
        print("STDERR:")
        print(proc["stderr"])
    print("STDOUT:")
    print(proc["stdout"])

    metrics = parse_evaluator_output(proc["stdout"])
    out_path = Path("outputs/paper_data_v6/scirex_official_eval_v6_70b.json")
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nparsed metrics → {out_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
