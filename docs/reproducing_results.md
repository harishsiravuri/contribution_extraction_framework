# Reproducing the results in the manuscript

Every table and figure in the paper maps to a script under `scripts/` that
produced it and an artifact under `artifacts/` that holds the numbers.

All scripts seed their randomness with `42` and write `seed.txt` next to
their output. Bootstrap 95% CIs use 1,000 resamples; paired permutation
tests use 10,000 permutations.

## Headline tables (SciREX)

| Manuscript element                             | Script                                              | Artifact                                                                       |
|-----------------------------------------------|-----------------------------------------------------|---------------------------------------------------------------------------------|
| Table 1: dev F1, multi-agent vs single-LLM    | `scripts/run_benchmarks.py --benchmarks scirex`     | `artifacts/paper_data_v3/benchmarks/scirex/evaluation.json`                     |
| Table 2: test F1, specialized 70B FT vs Jain  | `scripts/run_benchmarks.py --benchmarks scirex --scirex-split test --config config/models_ft_v6_70b.yaml --client together` | `artifacts/paper_data_v6/benchmarks_ft_70b_test/scirex/evaluation.json` |
| Table 3: calibration ECE before/after T-scaling | `scripts/run_calibration_v2.py`                   | `artifacts/paper_data_v2/calibration/calibration.json` *(see note below)*       |
| Table 4: span-grounding ablation              | `scripts/run_span_grounding.py`                     | `artifacts/paper_data_v3/span_grounding/`                                       |
| Table 5: official SciREX evaluator (test)     | `scripts/run_scirex_official_v6_70b_test.py`        | `artifacts/paper_data_v6/scirex_official_eval_v6_70b_test.json`                 |

*Note: the calibration baseline numbers were generated in `paper_data_v2`.
We release `paper_data_v3` onwards in this repo; if you need to re-fit
the v2 calibration, point `scripts/run_calibration_v2.py` at any
multi-agent record directory.*

## Ablations and analyses (v7, v8)

| Manuscript element                             | Script                                                                                                                          | Artifact                                                                       |
|-----------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| Closed-source extractor (GPT-4o vs DeepSeek)  | `scripts/v7_closed_source.py`                                                                                                   | `artifacts/paper_data_v7/closed_source_comparison/results.json`                |
| Critic ablation (Critic-off vs Critic-on)     | `scripts/v7_critic_ablation.py`                                                                                                 | `artifacts/paper_data_v7/critic_ablation/results.json`                         |
| TDMSci cross-corpus evaluation                | `scripts/v7_tdmsci.py`                                                                                                          | `artifacts/paper_data_v7/tdmsci/results.json`                                  |
| Specialized framework component ablation      | `scripts/v7_specialized_ablation.py`                                                                                            | `artifacts/paper_data_v7/specialized_ablation/results.json`                    |
| Precision @ confidence threshold              | `scripts/v8_threshold_analysis.py`                                                                                              | `artifacts/paper_data_v8/threshold_analysis/results.json`                      |
| Binding-rule ablation                         | `scripts/v8_binding_ablation.py` + `scripts/v8_score.py`                                                                        | `artifacts/paper_data_v8/binding_ablation/results.json`                        |
| Deployment case study (100 recent arXiv)      | `scripts/v8_fetch_recent.py` (free) + `scripts/v8_case_study.py extract` + `analyze`                                            | `artifacts/paper_data_v8/deployment_case_study/results.json` + `examples.md`   |
| Quantitative error analysis (50 papers)       | `scripts/v9_error_analysis.py`                                                                                                  | `artifacts/paper_data_v9/error_analysis/results.json` + `SUMMARY.md`           |

## Figures (manuscript)

| Figure | Script                                                       | Output                                                                    |
|--------|--------------------------------------------------------------|----------------------------------------------------------------------------|
| Fig 1 (architecture)                  | `scripts/figures/build_manuscript_figures.py` | `artifacts/figures_out/fig_architecture.{png,pdf}` *(if regenerated)*     |
| Fig 2 (per-field F1)                  | `scripts/figures/build_manuscript_figures.py` | `artifacts/figures_out/fig_benchmark_f1.{png,pdf}`                         |
| Fig 3 (reliability diagram)           | `scripts/figures/build_manuscript_figures.py` | `artifacts/figures_out/fig_reliability_diagram.{png,pdf}`                  |
| Fig 4 (cost vs Task F1, v1)           | `scripts/figures/build_manuscript_figures.py` | `artifacts/figures_out/fig_cost_vs_quality.{png,pdf}`                      |
| Fig 4 (cost vs Task F1, v2 w/ GPT-4o) | `scripts/figures/build_fig_cost_vs_quality_v2.py`             | `artifacts/figures_out/fig_cost_vs_quality_v2.{png,pdf}`                   |
| Precision-vs-coverage (v8 supplement) | `scripts/v8_threshold_analysis.py`                            | `artifacts/paper_data_v8/threshold_analysis/fig_precision_vs_coverage.{png,pdf}` |

(`artifacts/figures_out/` is populated by re-running the figure scripts;
the manuscript-final versions are included for direct comparison.)

## Cost summary (full reproduction)

| Phase | Approx LLM cost | Approx wall-clock |
|-------|---------------:|-------------------:|
| Dev-split default open-weights (v3, n=62)            | $0.30   | 10 min  |
| Test-split specialized 70B FT (v6, n=66)             | $25     | 90 min  |
| Closed-source GPT-4o comparison (v7 E1, n=25)        | $3      | 10 min  |
| Critic ablation (v7 E2, n=66 dev)                    | $0.70   | 11 min  |
| TDMSci cross-corpus (v7 E3, n=376 sentences)         | $1      | 25 min  |
| Specialized ablation (v7 E4, n=66 test, two arms)    | $21     | 40 min  |
| Threshold analysis (v8 E5)                           | $0      | <1 min  |
| Binding ablation (v8 E6, n=30)                       | $0.50   | 9 min   |
| Case study (v8 E7, n=100 recent arXiv)               | $0.70   | 14 min  |
| Error analysis (v9, 385 categorizations)             | $0.30   | 1 min   |
| **TOTAL**                                            | **≈$53**| **≈3.5 h**|

## Reproducibility caveats

- LLM outputs are stochastic at temperatures > 0. The Extractor runs at
  `temperatures: [0.0, 0.3, 0.7]`; the t=0 sample is deterministic per
  provider, but the t=0.3 and t=0.7 samples will vary across runs. This is
  the source of non-determinism in the framework's output; voting collapses
  it. Expect per-paper F1 to vary by ±0.03 across reruns.
- Model identifiers (`config/models.yaml` etc.) are pinned to specific
  OpenRouter snapshots as of 2026-05. If a provider deprecates a snapshot,
  update the `model_id` field to the current canonical ID.
- The Together-AI 70B dedicated endpoint must be started by hand from the
  Together console; `scripts/wait_for_endpoint.sh` polls until ready.
