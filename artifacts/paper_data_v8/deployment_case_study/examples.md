# Deployment case study — illustrative extractions

These three examples are drawn from the 100-paper arXiv NLP case study
(see the Deployment Case Study subsection in the paper for the full
summary statistics). They were selected to illustrate the framework's
behavior across three common extraction patterns. Each example is the
verbatim final `ContributionRecord` emitted by the default
open-weights deployment (DeepSeek Chat Extractor, Llama 3.3 70B
Instruct Critic and Consolidator) on the paper text retrieved from
arXiv. No post-hoc editing was performed.

The paper IDs are preserved so a reviewer can fetch the source PDFs
from arXiv and verify the extracted claims against the source.

The full per-paper extraction records, including Critic verdicts,
evidence spans, and self-consistency scores for every field, are
stored in `outputs/paper_data_v8/deployment_case_study/records/`. The
91 successfully extracted papers from the 100-paper sample are all
there; these three were selected as representative across the three
patterns described below.


## Example 1 — Multi-method benchmark enumeration

**Source paper:** `arxiv:2605.30162`

**What this illustrates:** When a paper benchmarks several methods on
a single task with a single metric, the binding rule produces one
contribution record per (method, task, metric) combination. The
framework correctly emits nine distinct records and assigns the
appropriate refusal-rate value to each. Self-consistency is 1.00
across the record set because the three temperature-sampled Extractor
drafts agreed on every field after voting.

| # | Method               | Task                | Metric                          | Self-consistency |
|---|----------------------|---------------------|---------------------------------|------------------|
| 1 | Gemma 2 2B-IT        | biosecurity refusal | refusal rate = 0.0              | 1.00             |
| 2 | Gemma 4 E2B-IT       | biosecurity refusal | refusal rate = 65.0             | 1.00             |
| 3 | Gemma 4 E2B-IT       | biosecurity refusal | refusal rate = 0.0              | 1.00             |
| 4 | Gemma 2 2B-IT        | biosecurity refusal | refusal rate = 0.0              | 1.00             |
| 5 | Gemma 4 E2B-IT       | biosecurity refusal | refusal rate = 0.0              | 1.00             |
| 6 | Qwen 2.5 1.5B        | biosecurity refusal | refusal rate = 83.0             | 1.00             |
| 7 | Phi-3-mini           | biosecurity refusal | refusal rate = 87.0             | 1.00             |
| 8 | Llama 3.2 1B         | biosecurity refusal | refusal rate = 61.0             | 1.00             |
| 9 | sparse autoencoder   | biosecurity refusal | divergence score D = 0.647      | 1.00             |

The `datasets` field is null on every record because the paper measures
refusal on a curated probe set rather than on a named benchmark. The
Critic returned SUPPORTED for every field that was emitted.


## Example 2 — Method-comparison extraction with a Critic correction

**Source paper:** `arxiv:2605.28363`

**What this illustrates:** The framework correctly distinguishes two
methods evaluated on the same task and dataset, and the Critic agent
flags one field as UNSUPPORTED. The third record retains the
(method, task, dataset) triple but the Consolidator drops its metrics
field because the Critic could not verify any specific metric value
against the paper. This is the framework's verification mechanism
acting at the field level rather than dropping a whole record.

| # | Method                          | Task                       | Dataset                          | Metric                       | Critic verdict           | Self-consistency |
|---|---------------------------------|----------------------------|----------------------------------|------------------------------|--------------------------|------------------|
| 1 | PubMedBERT                      | causal relation extraction | PubMedCausal                     | F1 = 0.7391                  | all SUPPORTED            | 0.50             |
| 2 | DeepSeek-R1-32B                 | causal relation extraction | PubMedCausal                     | Cosine Pair F1 = 0.6765      | all SUPPORTED            | 0.50             |
| 3 | PubMedCausal-trained encoders   | causal relation extraction | external causal relation datasets | (dropped)                    | metrics UNSUPPORTED      | 0.50             |

The two named methods (PubMedBERT, DeepSeek-R1-32B) are benchmarked on
the same dataset with different metric variants, and the framework
retains the distinction rather than collapsing the records. The lower
self-consistency (0.50) reflects genuine disagreement between
Extractor drafts on whether the third method should be emitted; only
two of three drafts proposed it, and the Critic's UNSUPPORTED verdict
on the metrics field caused the Consolidator to drop those values
while retaining the method, task, and dataset.


## Example 3 — Specialized method across three tasks

**Source paper:** `arxiv:2605.28669`

**What this illustrates:** When a single named method is evaluated on
multiple distinct tasks with corresponding datasets and metrics, the
binding rule emits one record per (task, dataset) pair. All three
records share the same Method (ACROS) but differ on the other three
fields. Two of three records reach self-consistency 1.00; the middle
record drops to 0.83 because one Extractor draft missed one of the
two metrics reported for the lexical-steering task.

| # | Method | Task                        | Dataset       | Metric                                           | Self-consistency |
|---|--------|-----------------------------|---------------|--------------------------------------------------|------------------|
| 1 | ACROS  | word-sense disambiguation   | Raganato ALL  | F1 = 64.95                                       | 1.00             |
| 2 | ACROS  | lexical steering            | CoInCo        | KL (qualitative); positive shifts recovery = 90.0| 0.83             |
| 3 | ACROS  | cross-lingual adaptation    | FLORES        | R@1 = 0.988; PPL = 7.94                          | 1.00             |

The KL metric in record 2 carries a null numerical value because the
paper reports KL qualitatively (as a divergence direction) rather than
as a single number; the framework correctly emits the metric name
without fabricating a value.
