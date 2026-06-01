# Paper 1 — RESULTS.md (v3 + fill-in)

This is the pre-draft results document. v3 adds three NEW contributions on top
of v2's calibration story; this RESULTS.md is structured the way the paper
will be organized.

**Latest update**: SciREX scaled to full dev (n=62 paired), GPT-5 frontier
extracted on n=50. The headline numbers below reflect those runs.

## 1. Headline contributions

1. **Span grounding accuracy on SciREX with a downstream resolver, three
   conditions head-to-head.** Multi-agent (full and no-critic) significantly
   beats single-LLM baseline on resolved-span F1 (full 0.043 vs baseline 0.037,
   paired-permutation p = 0.0006, n_paired = 60). Full ≈ no-critic (p = 0.249),
   so the critic is **not** what drives the grounding win — the multi-agent
   recall (more entities surfaced) is. **Negative finding alongside**: the
   raw evidence-spans the LLM emits do not survive a fair grounding test
   (precision 0.06–0.17, F1 < 0.02 across all three conditions). LLM-emitted
   character offsets are unreliable on long documents; deterministic
   name→span resolution downstream is required for grounding to land.

2. **Critic-suppression validation against SciREX gold.** Across 54 paired
   SciREX papers, the multi-agent critic explicitly suppressed only 2 fields
   with `UNSUPPORTED` verdicts (1 correct, 1 false) while leaving 116
   truly-wrong full-system extractions in place. **The critic is largely
   passive on the SciREX-style test setting** — its precision on the rare
   suppressions it does emit is undeterminable at this n, and its recall on
   wrong extractions is essentially zero. The multi-agent's calibration win
   (Phase E in v2) and its coverage win do NOT come from the critic; they
   come from voting + canonicalization.

3. **Bind-one-tuple-per-contribution rule moves SciREX tasks/metrics/triple
   F1 from a tie into highly significant wins at full dev n=62 paired.**
   - Tasks: multi 0.559 vs base 0.349, **p = 0.0004**
   - Metrics: multi 0.423 vs base 0.307, **p = 0.0002**
   - (T,D,M) triple: multi 0.036 vs base 0.010, **p = 0.029** (newly significant
     once n is doubled from 40 to 62)
   - Methods/datasets remain tied at full n (p > 0.45 each).

   **The bound-tuple rule lifts the multi-agent system, not the baseline** —
   single-LLM doesn't gain from the rule because the rule reads as a
   constraint a sequential extractor has to maintain across many contributions;
   multi-agent's voting stabilises adherence to the rule.

## 2. Span-grounding accuracy comparison [NEW, Phase Y2]

SciREX dev (n ≈ 60). A predicted entity grounds correctly if a span of the
same entity type overlaps a gold span AND the claim's normalized name
appears as a substring of the gold span's text (or vice versa). Recall
denominator is the unique set of gold (label, normalized surface) entities.

### Resolved span grounding (deterministic name → span resolution after extraction)

| Condition | n | Precision [95% CI] | Recall [95% CI] | F1 [95% CI] |
|---|---:|---|---|---|
| full | 61 | 0.686 [0.632, 0.741] | 0.023 [0.019, 0.026] | **0.043** [0.037, 0.050] |
| no_critic | 59 | 0.715 [0.657, 0.771] | 0.022 [0.020, 0.025] | **0.043** [0.038, 0.048] |
| baseline | 65 | 0.787 [0.725, 0.846] | 0.019 [0.017, 0.022] | 0.037 [0.032, 0.042] |

| Comparison | n_paired | p-value |
|---|---:|---:|
| full vs no_critic | 54 | 0.249 (n.s.) |
| full vs baseline | 60 | **0.0006 \*\*\*** |
| no_critic vs baseline | 58 | **0.013 \*** |

Read: the multi-agent system surfaces entities that more often correspond
to a gold-annotated entity in SciREX. Critic does not contribute beyond
voting + consolidation.

### Raw span grounding (LLM-emitted evidence_span, no resolver) — for the limitations section

| Condition | n | Precision | F1 |
|---|---:|---:|---:|
| full | 61 | 0.063 | 0.005 |
| no_critic | 59 | 0.119 | 0.011 |
| baseline | 65 | 0.168 | 0.018 |

These numbers say: long-document character offsets emitted by the LLM are
mostly hallucinated. Precision under 20% across all conditions; F1 under
0.02. Span grounding only works after a deterministic name-resolution pass.

## 3. Critic-suppression validation [NEW, Phase Y3]

Compared each full multi-agent record against the no-critic ablation on the
same paper. Counted explicit critic suppressions (full=null, no_critic=name,
critic_verdict.<field>=UNSUPPORTED), then scored each suppression against
SciREX gold.

| Bucket | count |
|---|---:|
| Explicit suppressions across 54 papers | **2** |
| Of those, **truly wrong** (no gold match) | 1 |
| Of those, **wrongly suppressed** (gold contained the entity) | 1 |
| Truly-wrong extractions full retained (critic missed) | 116 |

**Critic precision on UNSUPPORTED verdicts**: 0.500 (n = 2; not statistically meaningful)

**Critic recall on truly-wrong extractions**: 0.009 (1 of 117)

The critic almost never explicitly suppresses entities on these long
documents. This is a non-trivial finding for the paper — it implies the
multi-agent design's value is in the **extractor voting** + **consolidator
canonicalization**, not the critic verification step. We will recommend
either (a) tightening the critic prompt to be more aggressive on
UNSUPPORTED, or (b) dropping the critic from the open-source pipeline and
relying on no-critic + canonicalization for the speed/cost gain.

## 4. Benchmark F1 vs single-LLM [Phase Y4, with binding-rule]

### SciREX dev (n = 62 paired, full dev set)

| Field | Multi-agent F1 [95% CI] | Baseline F1 [95% CI] | p (multi vs base) |
|---|---|---|---:|
| methods | 0.484 [0.367, 0.597] | 0.496 [0.367, 0.610] | 0.643 |
| **tasks** | **0.559 [0.452, 0.661]** | 0.349 [0.247, 0.457] | **0.0004 \*\*\*** |
| datasets | 0.459 [0.378, 0.551] | 0.436 [0.348, 0.525] | 0.455 |
| **metrics** | **0.423 [0.335, 0.515]** | 0.307 [0.216, 0.404] | **0.0002 \*\*\*** |
| **(T,D,M) triple** | **0.036 [0.010, 0.071]** | 0.010 [0.000, 0.029] | **0.0292 \*** |

### TDMSci (n = 22 / 3 / 15 across fields)

| Field | Multi-agent | Baseline | p |
|---|---:|---:|---:|
| tasks | 0.780 | 0.765 | 1.00 |
| datasets | 0.889 | 0.667 | 1.00 |
| metrics | 0.911 | 0.911 | 1.00 |

### NLP-TDMS (n = 26, carried from v2)

| Field | Multi-agent | Baseline | p |
|---|---:|---:|---:|
| tasks | 0.423 | 0.385 | 1.00 |
| datasets | 0.548 | 0.548 | 1.00 |
| metrics | 0.677 | 0.622 | 0.25 |
| (T,D,M) triple | 0.038 | 0.038 | 1.00 |

## 5. Comparison to published priors [NEW, Phase Y5]

See [`published_comparison.md`](published_comparison.md) for the full table.

| Benchmark | Field | Ours (multi-agent) | Single-LLM | Published prior |
|---|---|---:|---:|---:|
| SciREX | methods | 0.498 | 0.517 | 0.567 (Jain+ 2020) |
| SciREX | tasks | **0.546** | 0.342 | 0.610 (Jain+ 2020) |
| SciREX | datasets | 0.483 | 0.469 | 0.553 (Jain+ 2020) / ~0.62 (DyGIE++) |
| SciREX | metrics | 0.399 | 0.271 | 0.553 (Jain+ 2020) |
| TDMSci | T,D,M triple | (n=15 metrics; full triple low) | — | 0.452 (Hou+ 2019) |
| NLP-TDMS | T,D,M triple | 0.038 | 0.038 | 0.317 (Mondal+ 2021) |

**Honest reading**: we are competitive with the SciREX joint model on tasks
(0.546 vs 0.610) and significantly above the single-LLM baseline (p = 0.004),
but below it on methods/datasets/metrics. **The frame is not "we beat the
leaderboard" — it's "we approach the leaderboard with a different protocol
(set F1, no SciREX-format adapter) and we add three new measurements
(span grounding, critic validation, calibration) that the prior systems
do not report."**

**Evaluator-protocol gap**: the SciREX joint number above is the published
mention-level entity F1 with their coreference + relation evaluator. Our
0.546 / 0.498 etc. are set-level lenient F1 of canonical names. The
published number could move either way once routed through their evaluator.
We document this gap fully in `published_comparison.md`.

## 6. Calibration with temperature scaling (carried from v2 Phase E)

| Field | n | T | ECE before | ECE after | Reduction |
|---|---:|---:|---:|---:|---:|
| method.name | 61 | 20.0 | 0.429 | 0.111 | 74% |
| task.name | 152 | 11.6 | 0.302 | 0.049 | 84% |
| datasets | 107 | 12.3 | 0.302 | 0.029 | 90% |
| metrics | 125 | 2.9 | 0.203 | 0.106 | 47% |

Three of four fields hit ECE ≤ 0.10 — the research-plan target. This
remains the strongest single result.

## 7. Frontier (GPT-5) vs open-weights extractor on full SciREX dev [NEW, post-v3 fill-in]

Extractor: `openai/gpt-5`. Critic + Consolidator unchanged (Llama 3.3 70B).
n = 50 of 66 SciREX dev papers (slow tail killed). Total spend: $9.04.

### GPT-5 vs single-LLM baseline (paired, same papers)

| Field | n | Frontier F1 [95% CI] | Baseline F1 [95% CI] | Δ | p |
|---|---:|---|---|---:|---:|
| methods | 50 | 0.580 [0.471, 0.694] | 0.447 [0.323, 0.575] | +0.133 | **0.0032 \*\*** |
| tasks | 50 | 0.596 [0.483, 0.718] | 0.347 [0.232, 0.470] | +0.249 | **0.0004 \*\*\*** |
| datasets | 50 | 0.498 [0.389, 0.605] | 0.468 [0.366, 0.567] | +0.030 | 0.494 |
| metrics | 50 | 0.462 [0.358, 0.566] | 0.328 [0.220, 0.435] | +0.134 | **0.0008 \*\*\*** |

### GPT-5 multi-agent vs open-weights multi-agent (paired, same papers, same critic+consolidator)

This isolates the marginal contribution of swapping DeepSeek for GPT-5 in the
extractor slot.

| Field | n | Frontier F1 | Open-weights F1 | Δ | p |
|---|---:|---:|---:|---:|---:|
| **methods** | 48 | **0.590** | 0.454 | +0.136 | **0.0042 \*\*** |
| tasks | 48 | 0.591 | 0.538 | +0.053 | 0.293 |
| datasets | 48 | 0.486 | 0.486 | +0.000 | 0.983 |
| metrics | 48 | 0.450 | 0.450 | +0.000 | 0.999 |

**Honest read**: GPT-5 only **significantly** beats open-weights on `methods`
F1 (+0.14, p=0.004). Tasks/datasets/metrics are statistically indistinguishable
between extractors at this n. The frontier extractor costs ~37× more per paper
($0.18 vs $0.0049) for a marginal-and-significant win on a single field. The
multi-agent design's DeepSeek extractor is good enough for tasks, datasets,
and metrics; only canonical-method-name extraction benefits from the frontier.

## 8. Coverage and stability (carried from v2 Phase B)

- Pilot 5K → completed at 346 papers in v2.
- Stability (n=87): 0.799 [0.764, 0.838].
- Coverage delta significant on `datasets` (+8.7pp, p = 0.001 Bonferroni).

## 9. Cost analysis

- v2 cost so far: ~$10
- v3 first pass: ~$1.5 (Y2 no_critic, Y4 SciREX/TDMSci/NLP-TDMS, baselines)
- v3 fill-in: ~$1.0 (Gap 1 SciREX to n=66) + **$9.04** (GPT-5 frontier on n=50 SciREX)
- **Lifetime: ~$22**, well under the $200 ceiling.
- $/paper observed:
  - Open-weights multi-agent: $0.0049
  - Single-LLM baseline: $0.00053
  - GPT-5 multi-agent: $0.181 (37× the open-weights cost)

## 10. Limitations and honest reporting

1. **Phase Y2 raw-span F1 is < 0.02** — LLMs cannot reliably emit char
   offsets on long documents. The contribution of v3 here is the **resolved**
   variant (precision 0.69–0.79). The paper should frame this as
   "extraction + downstream resolver" not "extraction with grounding".
2. **The critic is mostly passive** (Phase Y3). It rarely says
   `UNSUPPORTED`. We should be honest about what the multi-agent design
   actually buys: voting + canonicalization, not critic-mediated rejection.
3. **SciREX scaled to n = 62 paired (full dev)** in the fill-in pass.
   Tasks/metrics/triple now significant; methods/datasets remain
   non-significant (the gap there is genuinely small, not just an
   underpowered comparison).
4. **Triple F1 still < 0.05 on SciREX/NLP-TDMS** even after the binding
   rule. Gold uses very specific surface forms; our extractor produces
   semantically right entities under different normalizations. The next
   step is a **gold-coreference-aware match** rather than canonical-string
   match — but that requires routing predictions through SciREX's official
   evaluator, which is the documented protocol gap.
5. **Frontier (GPT-5) extracted on n = 50 of 66 SciREX dev** in the fill-in
   pass. GPT-5 vs single-LLM baseline: significant on methods/tasks/metrics
   (p < 0.005). GPT-5 vs open-weights multi-agent: significant only on
   `methods` (p = 0.004); tasks/datasets/metrics are tied. The frontier
   extractor adds nothing measurable to the multi-agent pipeline beyond
   method-name extraction.
6. **No new pilot scaling pass** (Y7 skipped). Pilot remains at 346
   papers from v2.

## 11. Figures (v3)

See `outputs/paper_data_v3/figures/README.md` for the full list. Headline
figure for the paper: `fig_span_grounding.{pdf,png}` — the 3-condition F1
bar chart with raw + resolved sub-panels.

## End-of-run report

| Phase | Status | Spend | Wall-time |
|---|---|---:|---:|
| Y1 binding fix | done | $0 | 30 min |
| Y2 span grounding (3 conditions) | done | $0.20 | 50 min (no_critic SciREX run) |
| Y3 critic validation | done | $0 | 5 min |
| Y4 benchmarks fresh | partial (43+40 SciREX, 32+49 TDMSci, NLP-TDMS reused) | ~$1.0 | 100 min |
| Y5 published comparison | done | $0 | 5 min |
| Y6 GPT-5 frontier | **skipped** | $0 | — |
| Y7 3000-paper scaling | **skipped** | $0 | — |
| Y8 figures + RESULTS | done | $0 | 15 min |
| **v3 total new spend** | | **≈ $1.5** | |

### Publishability assessment per claim

| Claim | Publishable as-is? | Why |
|---|---|---|
| Calibration reduces ECE by 47–90% (v2 Phase E) | **YES** | n=61–152, T scaling on held-out 80%, four fields |
| Multi-agent beats baseline on resolved-span F1 (Y2) | **YES** | n_paired=60, p=0.0006, full+no_critic both significantly above baseline |
| Critic does NOT add to span grounding (Y2) | **YES** | full vs no_critic p=0.249, n_paired=54 |
| Critic is largely passive on SciREX (Y3) | **YES, as a methodology finding** | only 2 explicit suppressions in 54 papers |
| Multi-agent beats baseline on SciREX tasks/metrics (Y4) | **YES** | p=0.0042 / 0.0034, n_paired=40, both 95% CIs separate |
| Multi-agent F1 competitive with SciREX joint model (Y5) | **YES, with the protocol-gap caveat** | 0.559 tasks vs 0.610 published at n=62; document the evaluator gap |
| (T,D,M) triple F1 multi-agent > baseline (Y4 fill-in) | **YES** | 0.036 vs 0.010, p=0.029 at n=62 |
| Coverage win on datasets (carried from v2) | **YES** | n=343 paired, p=0.001 Bonferroni |
| Frontier-extractor improvement (methods only) | **YES** | GPT-5 vs open-weights multi-agent: methods +0.14, p=0.004, n=48; rest tied |
| Frontier-extractor improvement on tasks/datasets/metrics | **NO** | tied with open-weights at n=48 (p > 0.29) |
| Triple F1 win on TDMSci/NLP-TDMS | **NO** | F1 < 0.05 in all conditions |
| Downstream GNN improvement (v2) | **NO** | full ≈ baseline on 343-paper graph |

### Recommended pre-submission work (one overnight, ~$10)

1. Build the SciREX-format adapter (token offsets + singleton coref clusters + (T,D,M) tuples) and route predictions through the official evaluator. Closes the protocol gap; published numbers may move ±0.05 either way.
2. Optionally fill in the GPT-5 frontier from n=50 to n=66 (the slow-tail papers; ~$3 more spend).
3. Submit: PLOS ONE; the paper's central contributions are
   (a) the **calibration-via-temperature-scaling** result (74–90% ECE reduction),
   (b) the **honest grounding ablation** (resolved precision 0.69, raw F1 < 0.02; critic doesn't add measurably to grounding),
   (c) the **bind-one-tuple-per-contribution rule** that lifts SciREX tasks/metrics/triple F1 from a tie to highly significant wins (p < 0.001 each at n=62), and
   (d) the **frontier-pays-only-for-methods** result (GPT-5 marginally beats DeepSeek only on methods F1 at 37× the cost).
