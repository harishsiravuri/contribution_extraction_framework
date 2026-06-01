# Phase 4 — Benchmark evaluation

_Total OpenRouter spend (this phase): $0.4568_

## Per-field F1 (lenient set match) with bootstrap 95% CIs and paired permutation p-values

### scirex

| Field | n | Multi-agent F1 [95% CI] | Baseline F1 [95% CI] | p (multi vs base) |
|---|---:|---|---|---:|
| methods | 62 | 0.484 [0.367, 0.597] | 0.496 [0.367, 0.610] | 0.6429 |
| tasks | 62 | 0.559 [0.452, 0.661] | 0.349 [0.247, 0.457] | 0.0004 * |
| datasets | 62 | 0.459 [0.378, 0.551] | 0.436 [0.348, 0.525] | 0.4549 |
| metrics | 62 | 0.423 [0.335, 0.515] | 0.307 [0.216, 0.404] | 0.0002 * |
| **(T,D,M) triple** | 62 | 0.036 [0.010, 0.071] | 0.010 [0.000, 0.029] | 0.0292 * |

## Comparison to published prior work

Published numbers below are author-reported on the *public test split* of each benchmark — not directly comparable to our F1 (we report on whatever subset we ran, set-level lenient F1 after the multi-agent pipeline). Use as a directional sanity-check, not as a head-to-head leaderboard.

| Benchmark | Field | Our F1 | Published | System | Source |
|---|---|---:|---:|---|---|
| SciREX | methods (entity F1) | 0.484 | 0.567 | SciREX joint model | Jain et al. 2020, ACL |
| SciREX | tasks (entity F1) | 0.559 | 0.610 | SciREX joint model | Jain et al. 2020, ACL |
| SciREX | datasets (entity F1) | 0.459 | 0.553 | SciREX joint model | Jain et al. 2020, ACL |
| SciREX | metrics (entity F1) | 0.423 | 0.553 | SciREX joint model | Jain et al. 2020, ACL |
| SciREX | datasets (entity F1) | 0.459 | ~0.62 | DyGIE++ | Wadden et al. 2019, EMNLP |
| TDMSci | (T,D,M) triple F1 | — | 0.452 | Hou et al. 2019 BiLSTM-CRF | Hou et al. 2019, ACL |
| NLP-TDMS | (T,D,M) triple F1 | — | 0.317 | BERT-classifier baseline | Mondal et al. 2021 |
