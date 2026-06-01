# Phase 4 — Benchmark evaluation

_Total OpenRouter spend (this phase): $1.4823_

## Per-field F1 (lenient set match) with bootstrap 95% CIs and paired permutation p-values

### scirex

| Field | n | Multi-agent F1 [95% CI] | Baseline F1 [95% CI] | p (multi vs base) |
|---|---:|---|---|---:|
| methods | 64 | 0.548 [0.440, 0.666] | 0.544 [0.427, 0.661] | 0.9418 |
| tasks | 64 | 0.548 [0.434, 0.653] | 0.448 [0.325, 0.570] | 0.0250 * |
| datasets | 64 | 0.496 [0.411, 0.582] | 0.482 [0.402, 0.571] | 0.3969 |
| metrics | 64 | 0.512 [0.416, 0.609] | 0.398 [0.310, 0.492] | 0.0002 * |
| **(T,D,M) triple** | 64 | 0.049 [0.022, 0.082] | 0.022 [0.003, 0.051] | 0.0460 * |

## Comparison to published prior work

Published numbers below are author-reported on the *public test split* of each benchmark — not directly comparable to our F1 (we report on whatever subset we ran, set-level lenient F1 after the multi-agent pipeline). Use as a directional sanity-check, not as a head-to-head leaderboard.

| Benchmark | Field | Our F1 | Published | System | Source |
|---|---|---:|---:|---|---|
| SciREX | methods (entity F1) | 0.548 | 0.567 | SciREX joint model | Jain et al. 2020, ACL |
| SciREX | tasks (entity F1) | 0.548 | 0.610 | SciREX joint model | Jain et al. 2020, ACL |
| SciREX | datasets (entity F1) | 0.496 | 0.553 | SciREX joint model | Jain et al. 2020, ACL |
| SciREX | metrics (entity F1) | 0.512 | 0.553 | SciREX joint model | Jain et al. 2020, ACL |
| SciREX | datasets (entity F1) | 0.496 | ~0.62 | DyGIE++ | Wadden et al. 2019, EMNLP |
| TDMSci | (T,D,M) triple F1 | — | 0.452 | Hou et al. 2019 BiLSTM-CRF | Hou et al. 2019, ACL |
| NLP-TDMS | (T,D,M) triple F1 | — | 0.317 | BERT-classifier baseline | Mondal et al. 2021 |
