# `config/` — model and prompt configurations

## Model configs (`*.yaml`)

Each YAML file pins exact model identifiers for the three agents plus
default request settings. The Pydantic loader is
[`src/paper1/config.py`](../src/paper1/config.py); the schema is:

```yaml
extractor:                       # The Extractor agent
  model_id: "openrouter-id"      # e.g. "deepseek/deepseek-chat"
  price_in_per_m: float          # USD per million input tokens (for cost tracking)
  price_out_per_m: float
  temperatures: [0.0, 0.3, 0.7]  # The three self-consistency-voting samples
  max_tokens: 4000
  top_p: 0.95

critic:                          # The Critic agent
  model_id: "openrouter-id"      # e.g. "meta-llama/llama-3.3-70b-instruct"
  price_in_per_m: float
  price_out_per_m: float
  temperature: 0.0
  max_tokens: 8000
  top_p: 0.95

consolidator:                    # The Consolidator agent
  model_id: "openrouter-id"
  price_in_per_m: float
  price_out_per_m: float
  temperature: 0.0
  max_tokens: 4000
  top_p: 0.95

defaults:                        # HTTP-client defaults
  request_timeout_s: 600
  max_retries: 4
  retry_backoff_s: 2.0

concurrency:                     # In-flight cap for batch processing
  papers_in_flight: 5
  per_provider_max: 5
```

### Released configs

| File                              | Purpose                                                                             |
|-----------------------------------|-------------------------------------------------------------------------------------|
| `models.yaml`                     | Default open-weights deployment: DeepSeek Chat + Llama 3.3 70B (×2).                |
| `models_frontier.yaml`            | Frontier-extractor ablation: Claude Opus replaces DeepSeek; Critic/Consolidator unchanged. |
| `models_frontier_gpt5.yaml`       | Same shape with GPT-5 (placeholder, current as of 2026-05).                         |
| `models_gpt4o.yaml`               | v7 closed-source comparison: GPT-4o-2024-11-20 replaces DeepSeek.                   |
| `models_ft.yaml`                  | v5 fine-tune (8B, smaller LoRA rank).                                               |
| `models_ft_v6.yaml`               | v6 8B fine-tune extractor (LoRA r64, 8 epochs).                                     |
| `models_ft_v6_70b.yaml`           | v6 70B fine-tune extractor; routed through Together AI dedicated endpoint.          |

**Pricing fields.** `price_in_per_m` / `price_out_per_m` are USD per
million tokens as advertised by the provider at run-time. Update them if
provider pricing changes; the framework only uses them to compute
`cost_usd` in the `_meta` block for accounting and they don't affect
extraction quality.

## Prompts (`config/prompts/*.md`)

Each prompt file is a single Markdown file with a `SYSTEM:` block and a
`USER:` block separated by the literal `SYSTEM:` and `USER:` markers; the
loader in [`src/paper1/agents/extractor.py::_split_prompt`](../src/paper1/agents/extractor.py)
parses these.

| File                          | Role / paper section                                                       |
|-------------------------------|-----------------------------------------------------------------------------|
| `extractor.md`                | Default Extractor prompt with the one-tuple-per-contribution binding rule.  |
| `critic.md`                   | Critic prompt — verifies each field of the voted draft against paper text.  |
| `consolidator.md`             | Consolidator prompt — merges 3 drafts + critic verdicts → final record.     |
| `extractor_fewshot.md`        | Few-shot variant for low-resource benchmarks.                               |
| `extractor_no_binding.md`     | v8 E6 ablation — disables the binding rule, allows parallel-list contributions. |
| `single_llm.md`               | Used by `BaselinePipeline` for the single-LLM baseline comparison.          |

The prompts are released under CC-BY-4.0 (see `../LICENSE`). Reuse is
welcome with attribution; please cite the paper (see `../CITATION.cff`).

## Customising

To swap one agent for a different model without writing code, copy
`models.yaml` to a new file, replace the relevant `model_id` and pricing,
then pass `--config path/to/your.yaml` to any of the scripts in `../scripts/`.

To write a new Extractor prompt without modifying the shipped one, drop
your `.md` file under `config/prompts/`, and pass it to the existing
benchmark driver:

```bash
uv run python scripts/run_benchmarks.py \
    --benchmarks scirex \
    --config config/models.yaml \
    --prompt config/prompts/your_extractor.md \
    --output-dir /tmp/your_ablation
```

This is how the v8 binding ablation worked — see
`scripts/v8_binding_ablation.py`.
