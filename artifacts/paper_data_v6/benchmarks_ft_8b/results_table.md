# Phase 4 — Benchmark evaluation

_Total OpenRouter spend (this phase): $0.0000_

## Per-field F1 (lenient set match) with bootstrap 95% CIs and paired permutation p-values

### scirex

| Field | n | Multi-agent F1 [95% CI] | Baseline F1 [95% CI] | p (multi vs base) |
|---|---:|---|---|---:|
| methods | 30 | 0.511 [0.333, 0.689] | 0.428 [0.267, 0.600] | 0.3645 |
| tasks | 30 | 0.793 [0.671, 0.904] | 0.344 [0.189, 0.511] | 0.0006 * |
| datasets | 30 | 0.517 [0.372, 0.657] | 0.415 [0.268, 0.571] | 0.2124 |
| metrics | 30 | 0.474 [0.322, 0.612] | 0.367 [0.233, 0.506] | 0.0656 |
| **(T,D,M) triple** | 30 | 0.159 [0.067, 0.267] | 0.010 [0.000, 0.038] | 0.0054 * |

## Comparison to published prior work

Published numbers below are author-reported on the *public test split* of each benchmark — not directly comparable to our F1 (we report on whatever subset we ran, set-level lenient F1 after the multi-agent pipeline). Use as a directional sanity-check, not as a head-to-head leaderboard.

| Benchmark | Field | Our F1 | Published | System | Source |
|---|---|---:|---:|---|---|
| SciREX | methods (entity F1) | 0.511 | 0.567 | SciREX joint model | Jain et al. 2020, ACL |
| SciREX | tasks (entity F1) | 0.793 | 0.610 | SciREX joint model | Jain et al. 2020, ACL |
| SciREX | datasets (entity F1) | 0.517 | 0.553 | SciREX joint model | Jain et al. 2020, ACL |
| SciREX | metrics (entity F1) | 0.474 | 0.553 | SciREX joint model | Jain et al. 2020, ACL |
| SciREX | datasets (entity F1) | 0.517 | ~0.62 | DyGIE++ | Wadden et al. 2019, EMNLP |
| TDMSci | (T,D,M) triple F1 | — | 0.452 | Hou et al. 2019 BiLSTM-CRF | Hou et al. 2019, ACL |
| NLP-TDMS | (T,D,M) triple F1 | — | 0.317 | BERT-classifier baseline | Mondal et al. 2021 |
