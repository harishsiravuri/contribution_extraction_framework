"""Build a Together-AI-format fine-tune dataset from SciREX train.

Each example pairs the rendered extractor prompt (system + user with the paper
text inserted) with the gold ContributionRecord JSON as the target completion.
The "binding rule" is enforced at construction time: one contribution per
(method, task, dataset, metric) gold tuple.

Output format (one JSON object per line, Together's chat-format expectation):
    {"messages": [
        {"role": "system",    "content": <system>},
        {"role": "user",      "content": <user with paper text>},
        {"role": "assistant", "content": <gold record JSON>}
    ]}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from paper1.agents.extractor import _split_prompt
from paper1.config import load_prompt
from paper1.loaders import GoldPaper, load_scirex_train


# Together's per-example token cap for fine-tuning is ~16K. We use 4-char/token
# as a quick estimator and exclude any example with a rendered chat that would
# exceed this; truncation is forbidden because it corrupts the supervision.
TOGETHER_TOKEN_LIMIT = 16000


@dataclass
class BuiltExample:
    paper_id: str
    chars_total: int
    est_tokens: int
    n_contributions: int


def _gold_record_json(p: GoldPaper) -> str:
    """Render the gold answer as the assistant's expected JSON."""
    g = p.gold
    methods = sorted(g.methods)
    method_name = methods[0] if methods else None
    contributions = []
    for t, d, m in sorted(g.triples):
        contributions.append({
            "method": {"name": method_name, "canonical_id": None, "evidence_span": None},
            "task": {"name": t, "canonical_id": None, "evidence_span": None},
            "datasets": [{"name": d, "canonical_id": None, "evidence_span": None}],
            "metrics": [{"name": m, "value": None, "unit": None, "evidence_span": None}],
            "claim_strength": "improves",
            "comparison_targets": [],
            "notes": None,
        })
    if not contributions:
        # Fall back to a single contribution with whatever's available
        contributions.append({
            "method": {"name": method_name, "canonical_id": None, "evidence_span": None},
            "task": {"name": (sorted(g.tasks) or [None])[0], "canonical_id": None, "evidence_span": None},
            "datasets": [{"name": d, "canonical_id": None, "evidence_span": None} for d in sorted(g.datasets)],
            "metrics": [{"name": m, "value": None, "unit": None, "evidence_span": None} for m in sorted(g.metrics)],
            "claim_strength": "improves",
            "comparison_targets": [],
            "notes": None,
        })
    return json.dumps({"contributions": contributions}, indent=2)


def _render_user(user_template: str, p: GoldPaper) -> str:
    return user_template.format(
        paper_id=p.paper_id,
        paper_text=p.full_text,
        retrieval_bundle="(none)",
    )


def build_messages(p: GoldPaper, system: str, user_template: str) -> dict:
    user = _render_user(user_template, p)
    target = _gold_record_json(p)
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": target},
        ]
    }


def build_dataset(
    out_path: Path = Path("outputs/paper_data_v5/scirex_finetune_train.jsonl"),
    val_out_path: Path = Path("outputs/paper_data_v5/scirex_finetune_val.jsonl"),
    val_frac: float = 0.10,
    seed: int = 42,
) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    template = load_prompt("extractor")
    system, user_template = _split_prompt(template)

    train = load_scirex_train()
    built: list[BuiltExample] = []
    written: list[dict] = []
    excluded: list[dict] = []
    for p in train:
        msgs = build_messages(p, system, user_template)
        chars = sum(len(m["content"]) for m in msgs["messages"])
        est_tokens = chars // 4
        n_contrib = len(json.loads(msgs["messages"][-1]["content"])["contributions"])
        if est_tokens > TOGETHER_TOKEN_LIMIT:
            excluded.append({"paper_id": p.paper_id, "est_tokens": est_tokens})
            continue
        built.append(
            BuiltExample(paper_id=p.paper_id, chars_total=chars, est_tokens=est_tokens, n_contributions=n_contrib)
        )
        written.append(msgs)

    # Deterministic split — no SciREX dev contamination since we train only
    # on the train split; val is a held-out 10% of train.
    import random
    rng = random.Random(seed)
    indices = list(range(len(written)))
    rng.shuffle(indices)
    n_val = max(1, int(len(written) * val_frac))
    val_idx = set(indices[:n_val])
    train_lines = [written[i] for i in range(len(written)) if i not in val_idx]
    val_lines = [written[i] for i in range(len(written)) if i in val_idx]

    with out_path.open("w", encoding="utf-8") as f:
        for ex in train_lines:
            f.write(json.dumps(ex, ensure_ascii=False))
            f.write("\n")
    with val_out_path.open("w", encoding="utf-8") as f:
        for ex in val_lines:
            f.write(json.dumps(ex, ensure_ascii=False))
            f.write("\n")

    chars = [b.chars_total for b in built]
    toks = [b.est_tokens for b in built]
    contrib = [b.n_contributions for b in built]
    chars.sort(); toks.sort(); contrib.sort()
    p95 = lambda xs: xs[max(0, int(0.95 * len(xs)) - 1)]
    p50 = lambda xs: xs[len(xs) // 2]
    summary = {
        "n_built": len(built),
        "n_excluded_over_token_limit": len(excluded),
        "excluded": excluded[:10],
        "n_train": len(train_lines),
        "n_val": len(val_lines),
        "val_frac": val_frac,
        "seed": seed,
        "chars_per_example": {"min": min(chars), "p50": p50(chars), "p95": p95(chars), "max": max(chars), "mean": sum(chars) // len(chars)},
        "est_tokens_per_example": {"min": min(toks), "p50": p50(toks), "p95": p95(toks), "max": max(toks), "mean": sum(toks) // len(toks)},
        "contributions_per_example": {"min": min(contrib), "p50": p50(contrib), "p95": p95(contrib), "max": max(contrib)},
        "train_jsonl": str(out_path),
        "val_jsonl": str(val_out_path),
    }
    (out_path.parent / "scirex_finetune_dataset_meta.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    s = build_dataset()
    print(json.dumps(s, indent=2))
