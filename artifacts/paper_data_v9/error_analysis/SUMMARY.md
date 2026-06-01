# v9 — Error analysis (50 SciREX dev papers)

_Sample: 45 papers (seed=42); 385 per-field errors categorized via meta-llama/llama-3.3-70b-instruct._

_Categorizer cost: $0.2819, wall 59s._

## Overall error-category distribution

| # | Category | Count | % of errors [95% CI] |
|---|---|---:|---:|
| 4 | Schema-coverage gap | 240 | **62.3%** [57.4, 67.0] |
| 1 | Passing-mention method | 96 | **24.9%** [20.5, 29.4] |
| 3 | Ambiguous task description | 35 | **9.1%** [6.2, 12.5] |
| 2 | Shared-baseline binding error | 10 | **2.6%** [1.0, 4.2] |
| 5 | Other / annotation disagreement | 4 | **1.0%** [0.3, 2.1] |

## Category × field breakdown (%)

| Category | method | task | datasets | metrics |
|---|---:|---:|---:|---:|
| 1. Passing-mention method | 91.3% (42) | 12.5% (5) | 22.8% (42) | 6.1% (7) |
| 2. Shared-baseline binding error | — | — | 5.4% (10) | — |
| 3. Ambiguous task description | — | 87.5% (35) | — | — |
| 4. Schema-coverage gap | 8.7% (4) | — | 71.7% (132) | 90.4% (104) |
| 5. Other / annotation disagreement | — | — | — | 3.5% (4) |

## Field totals

| Field | Total errors categorized |
|---|---:|
| method | 46 |
| task | 40 |
| datasets | 184 |
| metrics | 115 |
