# v10 — Default zero-shot framework on SciREX TEST split (held-out)

_Configuration: identical to the dev-split run in `artifacts/paper_data_v3/`. DeepSeek Chat Extractor at t ∈ {0.0, 0.3, 0.7}; Llama 3.3 70B Instruct Critic + Consolidator at t=0. No prompt or hyperparameter touched the test split before this run._

_Papers: 66 total, 64 multi-agent ok, 66 baseline ok. Spend: \$1.4823._

## Dev vs Test per-field F1 (default zero-shot framework)

| Field   | Dev F1 [95% CI]                | Test F1 [95% CI]               | Δ (test−dev) |
|---------|---------------------------------|---------------------------------|--------------|
| Method  | 0.484 [0.367, 0.597] (n=62) | 0.548 [0.440, 0.666] (n=64) | +0.065 |
| Task    | 0.559 [0.452, 0.661] (n=62) | 0.548 [0.434, 0.653] (n=64) | -0.012 |
| Dataset | 0.459 [0.378, 0.551] (n=62) | 0.496 [0.411, 0.582] (n=64) | +0.037 |
| Metric  | 0.423 [0.335, 0.515] (n=62) | 0.512 [0.416, 0.609] (n=64) | +0.089 |
| Triple* | — | 0.049 [0.022, 0.084] (n=64) | — |

_*Triple = (Task, Dataset, Metric) joint F1, exploratory only (framework was not tuned for triple binding)._

## Dev vs Test ECE (with dev-fitted T values, applied to test)

| Field        | Dev ECE pre-T | Dev ECE post-T | Test ECE pre-T | Test ECE post-T (dev T) | T (from dev) |
|--------------|---------------|----------------|-----------------|--------------------------|--------------|
| method.name   | 0.429 | 0.111 | 0.409 | 0.101 | 20.00 |
| task.name     | 0.302 | 0.049 | 0.454 | 0.142 | 11.58 |
| datasets      | 0.302 | 0.029 | 0.206 | 0.144 | 12.30 |
| metrics       | 0.203 | 0.106 | 0.300 | 0.133 | 2.92 |

## Interpretation

**Some fields improve on test relative to dev:** methods, metrics (|Δ| > 0.05). Other fields remain within ±0.05. This is consistent with the manuscript's framing — no over-fitting to the dev split.
