# Phase Y3 — Critic-suppression validation

_n papers compared (full ∩ no_critic): **54**_

## Overall

- Critic suppressions detected (full=null, no_critic=name, verdict=UNSUPPORTED): **2**
- Of those, **truly wrong** (no gold match): 1
- Of those, **wrongly suppressed** (gold did contain it): 1
- Truly-wrong extractions the critic missed (full retained, no gold match): 116
- Implicit suppressions (full=null, no_critic=name, verdict ≠ UNSUPPORTED): 0

**Critic precision (UNSUPPORTED ∩ truly_wrong / UNSUPPORTED): 0.500**
**Critic recall (UNSUPPORTED ∩ truly_wrong / all_truly_wrong): 0.009**

## By field

| Field | suppressions | correct | false | missed | precision | recall |
|---|---:|---:|---:|---:|---:|---:|
| method.name | 0 | 0 | 0 | 30 | — | 0.000 |
| task.name | 0 | 0 | 0 | 28 | — | 0.000 |
| datasets | 0 | 0 | 0 | 33 | — | 0.000 |
| metrics | 2 | 1 | 1 | 25 | 0.500 | 0.038 |

## Sample false-suppressions (critic killed a correct extraction)

- `scirex:33bcc97b605f00145098d095be2841a1fa6b9a95` field=`metrics` no_critic_name=`accuracy` (gold has match)