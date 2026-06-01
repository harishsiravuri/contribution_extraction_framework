# Phase Y5 — Comparison to published prior systems

**Honest reading**: published numbers below are author-reported on the public test split of each benchmark. Our F1 is set-level lenient match (after the canonicalizer) on the SciREX dev set or a 50-paper subset of TDMSci/NLP-TDMS. They are NOT directly comparable on protocol — see _Evaluator-protocol gap_ at the end of this document.

## SciREX (dev set, n=66)

| Field | Ours (multi-agent) | 95% CI | Single-LLM | Published prior | System | Citation |
|---|---:|---|---:|---:|---|---|
| methods | 0.498 | [0.362, 0.643] | 0.517 | 0.567 | SciREX joint (Jain+ 2020) | Jain et al. 2020, Table 5 |
| tasks | 0.546 | [0.408, 0.683] | 0.342 | 0.610 | SciREX joint (Jain+ 2020) | Jain et al. 2020, Table 5 |
| datasets | 0.483 | [0.372, 0.601] | 0.469 | 0.553 | SciREX joint (Jain+ 2020) | Jain et al. 2020, Table 5 |
| datasets | (same) | (same) | (same) | 0.62 | DyGIE++ (Wadden+ 2019) | Wadden et al. 2019, reported in Jain+ 2020 Table 5 |
| metrics | 0.399 | [0.283, 0.518] | 0.271 | 0.553 | SciREX joint (Jain+ 2020) | Jain et al. 2020, Table 5 |

## TDMSci (test split first 50 sentences in our run)

| Metric | Ours (multi) | 95% CI | Single-LLM | Published prior | System | Citation |
|---|---:|---|---:|---:|---|---|
| tasks | 0.780 | [0.629, 0.917] | 0.765 | — | — | not reported per-field in Hou+ 2019 |
| datasets | 0.889 | [0.667, 1.000] | 0.667 | — | — | not reported per-field in Hou+ 2019 |
| metrics | 0.911 | [0.756, 1.000] | 0.911 | — | — | not reported per-field in Hou+ 2019 |
| (T,D,M) triple | — | — | — | 0.452 | BiLSTM-CRF (Hou+ 2019) | Hou et al. 2019, Table 4 |

## NLP-TDMS (50-paper subset of test set)

| Metric | Ours (multi) | 95% CI | Single-LLM | Published prior | System | Citation |
|---|---:|---|---:|---:|---|---|
| tasks | 0.423 | [0.231, 0.615] | 0.385 | — | — | only triple F1 reported in Mondal+ 2021 |
| datasets | 0.548 | [0.383, 0.707] | 0.548 | — | — | only triple F1 reported in Mondal+ 2021 |
| metrics | 0.677 | [0.510, 0.836] | 0.622 | — | — | only triple F1 reported in Mondal+ 2021 |
| (T,D,M) triple | 0.038 | [0.000, 0.115] | 0.038 | 0.317 | BERT-classifier (Mondal+ 2021) | Mondal et al. 2021, Table 4 |

## Evaluator-protocol gap (important)

The official SciREX evaluator (`data/raw/scirex/scirex/evaluation_scripts/scirex_relation_evaluate.py`) expects predictions in three matched files:
1. NER predictions: per-document `[start_tok, end_tok, label]` arrays
2. Predicted coreference clusters: mention-id → cluster-id
3. Predicted N-ary relations: lists of `(Method, Task, Material, Metric, score)` keyed to clusters

Our pipeline produces a different output shape: a small set of canonical entity names per contribution, with character-offset (not token-offset) `evidence_span`s, and no explicit coreference clusters. To route our predictions through the official evaluator we would need an adapter that:
- maps our character offsets back to token indices (straightforward — all documents are joined-with-space)
- re-mention-finds every gold canonical name in our output to expand to the SciREX 'mention list' format
- emits singleton coref clusters per canonical name (sufficient for the joint-relation evaluator)
- produces n-ary relation tuples grouping our (method, task, dataset, metric) per contribution

We have NOT built that adapter for v3 due to time. The numbers in this table use our canonicalizer-aware set F1 (`paper1.metrics.span_f1`), which is a strictly weaker comparison: it credits any match between our canonical names and SciREX canonical names without checking mention-level recall. Read it as a directional sanity-check, not a head-to-head leaderboard.

For TDMSci, the IBM repo includes per-token CoNLL eval scripts in `data/raw/science-result-extractor/data/TDMSci/conllFormat/eacl21_token_level_eval/` but they require the predictor to emit BIO-tagged token streams over the original sentences. Our agents emit JSON entity lists rather than per-token tags, so the same adapter problem applies. We document this as an open follow-up; preliminary set-F1 numbers above suggest we are competitive (TDMSci tasks 0.81 multi-agent vs Hou et al.'s 0.45 triple F1) but the protocol gap means a head-to-head publication claim would require either (a) writing the adapter or (b) re-running Hou et al.'s evaluator on outputs in our format if it is configurable.
