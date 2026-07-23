# v11 — Matched-budget baselines on SciREX test

_Comparing the multi-agent framework (v10) against two same-extractor baselines on the SciREX test split. All three systems use `config/prompts/extractor.md` and `deepseek/deepseek-chat` via OpenRouter. Seed = 42. Bootstrap 95% CIs from 1000 resamples; paired permutation p-values from 10000 permutations, Bonferroni-corrected across 4 fields per comparison._

## Compute / cost per condition

| Condition | Papers ok | LLM calls | Tokens in | Tokens out | OpenRouter spend | Wall time |
|---|---:|---:|---:|---:|---:|---:|
| Multi-agent (v10) | 64 | ~320 | — | — | $1.2966 | 2100s |
| A: Single-LLM baseline | 66 | 66 | 504537 | 56421 | $0.2117 | 380s |
| B: Self-consistency | 65 | 325 | 2485535 | 278828 | $1.0435 | 843s |

**v11 total OpenRouter spend (Conditions A + B): $1.2552.**

## Per-field F1 (standalone, bootstrap 95% CI)

| Field   | Multi-agent (v10) F1 [95% CI]     | Condition A F1 [95% CI]        | Condition B F1 [95% CI]        |
|---------|------------------------------------|---------------------------------|---------------------------------|
| Method  | 0.548 [0.435, 0.662] (n=64) | 0.553 [0.441, 0.658] (n=66) | 0.543 [0.424, 0.656] (n=65) |
| Task    | 0.548 [0.442, 0.656] (n=64) | 0.559 [0.457, 0.664] (n=66) | 0.580 [0.476, 0.687] (n=65) |
| Dataset | 0.496 [0.411, 0.582] (n=64) | 0.469 [0.380, 0.555] (n=66) | 0.497 [0.409, 0.584] (n=65) |
| Metric  | 0.512 [0.413, 0.610] (n=64) | 0.553 [0.458, 0.638] (n=66) | 0.548 [0.455, 0.641] (n=65) |
| Triple* | 0.049 [0.022, 0.084] (n=64) | 0.054 [0.022, 0.093] (n=66) | 0.056 [0.025, 0.092] (n=65) |

_*Triple = (Task, Dataset, Metric) joint F1, exploratory._

## Paired-permutation p-values (Bonferroni across 4 fields)

### Multi-agent vs Condition A (single-LLM)

| Field | n paired | Δ (x − y) | p (raw) | p (Bonferroni) |
|---|---:|---:|---:|---:|
| Method  | 64 | -0.006 | 0.8312 | **1.0000**  |
| Task    | 64 | -0.014 | 0.5973 | **1.0000**  |
| Dataset | 64 | +0.028 | 0.1152 | **0.4608**  |
| Metric  | 64 | -0.040 | 0.1997 | **0.7987**  |
| Triple* | 64 | +0.001 | 0.9059 | — (not Bonferroni-adjusted with fields) |

### Multi-agent vs Condition B (self-consistency)

| Field | n paired | Δ (x − y) | p (raw) | p (Bonferroni) |
|---|---:|---:|---:|---:|
| Method  | 63 | -0.003 | 0.9179 | **1.0000**  |
| Task    | 63 | -0.026 | 0.4545 | **1.0000**  |
| Dataset | 63 | +0.007 | 0.3779 | **1.0000**  |
| Metric  | 63 | -0.043 | 0.1877 | **0.7507**  |
| Triple* | 63 | -0.002 | 0.8117 | — (not Bonferroni-adjusted with fields) |

### Condition B (self-consistency) vs Condition A (single-LLM)

| Field | n paired | Δ (x − y) | p (raw) | p (Bonferroni) |
|---|---:|---:|---:|---:|
| Method  | 65 | -0.003 | 0.7508 | **1.0000**  |
| Task    | 65 | +0.012 | 0.6246 | **1.0000**  |
| Dataset | 65 | +0.021 | 0.3275 | **1.0000**  |
| Metric  | 65 | -0.003 | 0.8842 | **1.0000**  |
| Triple* | 65 | +0.001 | 0.8769 | — (not Bonferroni-adjusted with fields) |

## Interpretation

**Reviewer 4 persistence check:** the dev-split multi-agent > single-LLM advantage does NOT reach Bonferroni-corrected significance on any per-field F1 on the held-out test split. Raw p-values below; the direction of the effect (Δ column) still favours multi-agent on tasks/metrics on dev; test narrows or reverses the gap depending on the field.

**Advisor Priority 1 (cost-matched compute) check:** at ~equal LLM-call budget, the multi-agent structure does NOT produce a Bonferroni-significant per-field F1 improvement over the cost-matched self-consistency baseline. See the Δ column for the direction and magnitude of the (statistically insignificant) differences per field.
