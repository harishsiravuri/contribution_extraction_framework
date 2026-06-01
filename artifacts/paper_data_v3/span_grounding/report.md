# Phase Y2 — Span-grounding accuracy on SciREX dev

A claim grounds correctly if a span of the same entity type overlaps a gold span AND the claim's normalized name appears as a substring of the gold span's text (or vice versa). Recall denominator is the unique set of gold (label, normalized_surface) entities.

We report two variants:
- **raw**: uses the LLM-emitted `evidence_span` directly.
- **resolved**: ignores the LLM span and resolves each entity name to a char
  span via a deterministic case-insensitive string-match in the paper text.
  Reflects what the system can do *with* a downstream name→span resolver.

## Raw span grounding (95% bootstrap CIs)

| Condition | n | Precision [95% CI] | Recall [95% CI] | F1 [95% CI] |
|---|---:|---|---|---|
| full | 61 | 0.063 [0.041, 0.086] | 0.003 [0.002, 0.004] | 0.005 [0.003, 0.007] |
| no_critic | 59 | 0.119 [0.083, 0.157] | 0.006 [0.004, 0.008] | 0.011 [0.007, 0.015] |
| baseline | 65 | 0.168 [0.124, 0.221] | 0.010 [0.007, 0.013] | 0.018 [0.012, 0.024] |

### Raw pairwise paired-permutation p-values on F1

| Comparison | n paired | p-value |
|---|---:|---:|
| full_vs_no_critic | 54 | 0.0124 * |
| full_vs_baseline | 60 | 0.0002 * |
| no_critic_vs_baseline | 58 | 0.0032 * |

## Resolved span grounding (95% bootstrap CIs)

| Condition | n | Precision [95% CI] | Recall [95% CI] | F1 [95% CI] |
|---|---:|---|---|---|
| full | 61 | 0.686 [0.632, 0.741] | 0.023 [0.019, 0.026] | 0.043 [0.037, 0.050] |
| no_critic | 59 | 0.715 [0.657, 0.771] | 0.022 [0.020, 0.025] | 0.043 [0.038, 0.048] |
| baseline | 65 | 0.787 [0.725, 0.846] | 0.019 [0.017, 0.022] | 0.037 [0.032, 0.042] |

### Resolved pairwise paired-permutation p-values on F1

| Comparison | n paired | p-value |
|---|---:|---:|
| full_vs_no_critic | 54 | 0.2492 |
| full_vs_baseline | 60 | 0.0006 * |
| no_critic_vs_baseline | 58 | 0.0130 * |
