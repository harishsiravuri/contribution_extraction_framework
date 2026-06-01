# Phase 4 — Benchmark evaluation

_Total OpenRouter spend (this phase): $0.1860_

## Per-field F1 (lenient set match) with bootstrap 95% CIs and paired permutation p-values

### scirex

| Field | n | Multi-agent F1 [95% CI] | Baseline F1 [95% CI] | p (multi vs base) |
|---|---:|---|---|---:|
| methods | 59 | 0.590 [0.460, 0.709] | 0.540 [0.415, 0.664] | 0.1832 |
| tasks | 59 | 0.627 [0.520, 0.746] | 0.554 [0.444, 0.667] | 0.1334 |
| datasets | 59 | 0.536 [0.435, 0.636] | 0.464 [0.364, 0.562] | 0.0300 * |
| metrics | 59 | 0.459 [0.359, 0.561] | 0.476 [0.373, 0.586] | 0.7049 |
| **(T,D,M) triple** | 59 | 0.148 [0.082, 0.221] | 0.103 [0.048, 0.162] | 0.1282 |

## Comparison to published prior work

Published numbers below are author-reported on the *public test split* of each benchmark — not directly comparable to our F1 (we report on whatever subset we ran, set-level lenient F1 after the multi-agent pipeline). Use as a directional sanity-check, not as a head-to-head leaderboard.

| Benchmark | Field | Our F1 | Published | System | Source |
|---|---|---:|---:|---|---|
| SciREX | methods (entity F1) | 0.590 | 0.567 | SciREX joint model | Jain et al. 2020, ACL |
| SciREX | tasks (entity F1) | 0.627 | 0.610 | SciREX joint model | Jain et al. 2020, ACL |
| SciREX | datasets (entity F1) | 0.536 | 0.553 | SciREX joint model | Jain et al. 2020, ACL |
| SciREX | metrics (entity F1) | 0.459 | 0.553 | SciREX joint model | Jain et al. 2020, ACL |
| SciREX | datasets (entity F1) | 0.536 | ~0.62 | DyGIE++ | Wadden et al. 2019, EMNLP |
| TDMSci | (T,D,M) triple F1 | — | 0.452 | Hou et al. 2019 BiLSTM-CRF | Hou et al. 2019, ACL |
| NLP-TDMS | (T,D,M) triple F1 | — | 0.317 | BERT-classifier baseline | Mondal et al. 2021 |
