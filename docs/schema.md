# ContributionRecord schema

The framework's output schema is a Pydantic model defined in
[`src/paper1/schema.py`](../src/paper1/schema.py). Below is the JSON
Schema description plus a worked example.

## JSON Schema (informal)

```jsonc
{
  "paper_id": "string — opaque paper identifier, e.g. 'scirex:<sha>' or 'arxiv:<id>'",

  "contributions": [
    {
      "method": {
        "name":             "string | null   — canonical method name",
        "canonical_id":     "string | null   — optional KB identifier",
        "evidence_span":    {"start": "int", "end": "int"} | null,
        "confidence":       "float in [0, 1] — per-field confidence (post-vote)"
      },
      "task": {
        "name":             "string | null",
        "canonical_id":     "string | null",
        "evidence_span":    {"start": "int", "end": "int"} | null,
        "confidence":       "float in [0, 1]"
      },
      "datasets": [
        {
          "name":           "string",
          "canonical_id":   "string | null",
          "evidence_span":  {"start": "int", "end": "int"} | null,
          "confidence":     "float in [0, 1]"
        }
      ],
      "metrics": [
        {
          "name":           "string",
          "value":          "number | null",
          "unit":           "string | null",
          "evidence_span":  {"start": "int", "end": "int"} | null,
          "confidence":     "float in [0, 1]"
        }
      ],

      "claim_strength":      "'improves' | 'comparable' | 'novel' | 'applies' | null",
      "comparison_targets":  ["string"],
      "self_consistency":    "float in [0, 1] — agreement across the 3 Extractor temperature votes",
      "critic_verdict": {
        "method":   "'SUPPORTED' | 'PARTIAL' | 'UNSUPPORTED' | null",
        "task":     "'SUPPORTED' | 'PARTIAL' | 'UNSUPPORTED' | null",
        "datasets": "'SUPPORTED' | 'PARTIAL' | 'UNSUPPORTED' | null",
        "metrics":  "'SUPPORTED' | 'PARTIAL' | 'UNSUPPORTED' | null"
      }
    }
  ],

  "_meta": {
    "extractor_model":     "string — OpenRouter or Together model_id",
    "critic_model":        "string",
    "consolidator_model":  "string",
    "tokens_in":           "int",
    "tokens_out":          "int",
    "cost_usd":            "float",
    "wall_time_seconds":   "float",
    "voting_samples":      "int — always 3 under the default config"
  }
}
```

## Binding rule

A single `contribution` bundles **one** (method × task × dataset × metric)
tuple. If a paper reports the same method on multiple
(task, dataset, metric) combinations, the framework emits one
`contribution` per combination — not one big contribution with parallel
lists.

This is enforced by the Extractor prompt at
[`config/prompts/extractor.md`](../config/prompts/extractor.md). The
ablation prompt that disables this rule is at
`config/prompts/extractor_no_binding.md`; the ablation result is in
`artifacts/paper_data_v8/binding_ablation/results.json` and shows that
the rule does not measurably affect per-field set-F1 — its value is
schema cleanliness for downstream relational consumers, not extraction
quality.

## Worked example

This is the first contribution from
`artifacts/paper_data_v3/benchmarks/scirex/multi_agent/scirex__007ff2ca5f297b04636699ce4d01ca6d6f21dc77.json`,
truncated for readability:

```json
{
  "paper_id": "scirex:007ff2ca5f297b04636699ce4d01ca6d6f21dc77",
  "contributions": [
    {
      "method":   {"name": "aESIM", "evidence_span": {"start": 99, "end": 104}, "confidence": 1.0},
      "task":     {"name": "natural language inference", "evidence_span": {"start": 224, "end": 248}, "confidence": 1.0},
      "datasets": [
        {"name": "SNLI",     "evidence_span": {"start": 505, "end": 509}, "confidence": 0.333},
        {"name": "MultiNLI", "evidence_span": {"start": 565, "end": 573}, "confidence": 0.333}
      ],
      "metrics":  [
        {"name": "accuracy", "value": 88.1,  "unit": "%", "evidence_span": {"start": 514, "end": 521}, "confidence": 0.5},
        {"name": "accuracy", "value": 88.01, "unit": "%", "evidence_span": {"start": 604, "end": 611}, "confidence": 0.5}
      ],
      "claim_strength":     "improves",
      "comparison_targets": ["ESIM"],
      "self_consistency":   0.875,
      "critic_verdict": {
        "method":   "SUPPORTED",
        "task":     "SUPPORTED",
        "datasets": "SUPPORTED",
        "metrics":  "SUPPORTED"
      }
    }
  ],
  "_meta": {
    "extractor_model":    "deepseek/deepseek-chat",
    "critic_model":       "meta-llama/llama-3.3-70b-instruct",
    "consolidator_model": "meta-llama/llama-3.3-70b-instruct",
    "tokens_in": 12421, "tokens_out": 1893,
    "cost_usd": 0.00536, "wall_time_seconds": 18.7,
    "voting_samples": 3
  }
}
```

## Notes on `self_consistency`

`self_consistency` is the agreement (Jaccard-like) across the three
Extractor temperature votes. The empirical mapping from
`self_consistency` to extraction precision is shown in
`artifacts/paper_data_v8/threshold_analysis/results.json`:

- At `self_consistency >= 0.9`, per-field precision is **0.64-0.82**
  (vs 0.39-0.57 at the unfiltered base rate), with ~18% coverage.
- For pipelines that need high-precision extractions, filter at
  `self_consistency >= 0.9` and accept the coverage hit.
- Temperature-scaled confidence (`scripts/run_calibration_v2.py`,
  `paper_data_v2/calibration/calibration.json`) is calibrated for
  Expected Calibration Error, not for precision-thresholding — the
  fitted T values compress most confidences toward base rate, so they
  are **not** useful as a precision filter.
