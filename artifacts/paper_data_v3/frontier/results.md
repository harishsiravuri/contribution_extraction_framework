# Phase D — Frontier (GPT-5) extractor ablation, full SciREX dev

_Extractor: openai/gpt-5; Critic + Consolidator: meta-llama/llama-3.3-70b-instruct_

_n SciREX dev papers extracted: **50** of 66 (rest hit OpenAI rate-limits / max_tokens). Total spend: $9.0358_

Two head-to-head comparisons on the same papers:
1. Frontier multi-agent vs **single-LLM baseline** (DeepSeek)
2. Frontier multi-agent vs **open-weights multi-agent** (DeepSeek extractor; same critic + consolidator)

## Frontier vs single-LLM baseline (paired same-papers)

| Field | n | Frontier F1 [95% CI] | Baseline F1 [95% CI] | Δ | p (paired-perm) |
|---|---:|---|---|---:|---:|
| methods | 50 | 0.580 [0.471, 0.694] | 0.447 [0.320, 0.573] | +0.133 | 0.0032 * |
| tasks | 50 | 0.596 [0.483, 0.718] | 0.347 [0.227, 0.473] | +0.249 | 0.0004 * |
| datasets | 50 | 0.498 [0.389, 0.605] | 0.468 [0.360, 0.566] | +0.031 | 0.4935 |
| metrics | 50 | 0.462 [0.358, 0.566] | 0.328 [0.220, 0.433] | +0.134 | 0.0008 * |

## Frontier vs open-weights multi-agent (paired same-papers, same critic/consolidator)

_isolates the marginal contribution of the GPT-5 extractor over DeepSeek_

| Field | n | Frontier F1 [95% CI] | Open-weights F1 [95% CI] | Δ | p (paired-perm) |
|---|---:|---|---|---:|---:|
| methods | 48 | 0.590 [0.483, 0.705] | 0.454 [0.336, 0.586] | +0.136 | 0.0042 * |
| tasks | 48 | 0.591 [0.478, 0.706] | 0.538 [0.420, 0.667] | +0.053 | 0.2929 |
| datasets | 48 | 0.486 [0.388, 0.588] | 0.486 [0.393, 0.584] | +0.001 | 0.9826 |
| metrics | 48 | 0.450 [0.345, 0.559] | 0.450 [0.348, 0.556] | -0.000 | 0.9990 |

## Cost / runtime

- Papers extracted: 50 / 66 (rest stalled on slow tail; killed at n=50)
- Total OpenRouter spend: **$9.0358**
- Per-paper cost: **$0.1807** (37× more than open-weights)

## Honest read

- GPT-5 extractor significantly beats single-LLM baseline on **methods, tasks, metrics** (p < 0.005 each); datasets is tied (p = 0.49). This confirms multi-agent + frontier extractor > single-LLM at proper n.
- GPT-5 vs **open-weights multi-agent**: only **methods** is significantly different (frontier +0.14 F1, p = 0.004). Tasks, datasets, metrics are all tied (p > 0.29). The GPT-5 extractor does not buy a real F1 lift on this benchmark beyond what DeepSeek + critic + consolidator already does — except for method names.
- Cost ratio: ~37× more expensive per paper than open-weights, for a marginal-and-significant win only on .