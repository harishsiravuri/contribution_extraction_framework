# Phase 4 — Benchmark evaluation

_Total OpenRouter spend (this phase): $0.1995_

## Per-field F1 (lenient set match) with bootstrap 95% CIs and paired permutation p-values

### scirex

| Field | n | Multi-agent F1 [95% CI] | Baseline F1 [95% CI] | p (multi vs base) |
|---|---:|---|---|---:|
| methods | 65 | 0.582 [0.464, 0.692] | 0.474 [0.349, 0.587] | 0.0398 * |
| tasks | 65 | 0.768 [0.678, 0.859] | 0.684 [0.576, 0.786] | 0.0302 * |
| datasets | 65 | 0.527 [0.433, 0.622] | 0.554 [0.455, 0.649] | 0.4861 |
| metrics | 65 | 0.639 [0.541, 0.732] | 0.608 [0.507, 0.701] | 0.4179 |
| **(T,D,M) triple** | 65 | 0.162 [0.095, 0.235] | 0.145 [0.080, 0.213] | 0.5603 |

## Comparison to published prior work

Published numbers below are author-reported on the *public test split* of each benchmark — not directly comparable to our F1 (we report on whatever subset we ran, set-level lenient F1 after the multi-agent pipeline). Use as a directional sanity-check, not as a head-to-head leaderboard.

| Benchmark | Field | Our F1 | Published | System | Source |
|---|---|---:|---:|---|---|
| SciREX | methods (entity F1) | 0.582 | 0.567 | SciREX joint model | Jain et al. 2020, ACL |
| SciREX | tasks (entity F1) | 0.768 | 0.610 | SciREX joint model | Jain et al. 2020, ACL |
| SciREX | datasets (entity F1) | 0.527 | 0.553 | SciREX joint model | Jain et al. 2020, ACL |
| SciREX | metrics (entity F1) | 0.639 | 0.553 | SciREX joint model | Jain et al. 2020, ACL |
| SciREX | datasets (entity F1) | 0.527 | ~0.62 | DyGIE++ | Wadden et al. 2019, EMNLP |
| TDMSci | (T,D,M) triple F1 | — | 0.452 | Hou et al. 2019 BiLSTM-CRF | Hou et al. 2019, ACL |
| NLP-TDMS | (T,D,M) triple F1 | — | 0.317 | BERT-classifier baseline | Mondal et al. 2021 |
