# contribution_extraction_framework

A three-agent (Extractor + Critic + Consolidator) pipeline for extracting
**(Method, Task, Dataset, Metric)** tuples from scientific papers, with
self-consistency voting across three temperature samples and temperature-scaled
confidence calibration. The default deployment uses open-weights LLMs
(DeepSeek Chat as Extractor, Meta Llama 3.3 70B Instruct as Critic and
Consolidator) at roughly **$0.005 per paper** through OpenRouter; a specialized
variant uses a LoRA fine-tuned Llama 3.1 70B Instruct extractor that beats the
SciREX 2020 published numbers on 3 of 4 fields on the held-out test split.

This repository accompanies the ICMLA 2026 submission. It contains the full
pipeline source code, agent prompts, model configs, all experimental result
artifacts referenced in the paper, and the scripts that produced them.

---

## Quickstart (5 minutes)

```bash
git clone https://github.com/harishsiravuri/contribution_extraction_framework.git
cd contribution_extraction_framework

# 1. Set up the Python env with uv (https://docs.astral.sh/uv/)
uv sync

# 2. Provide your OpenRouter API key
cp .env.example .env
# Edit .env and replace OPENROUTER_API_KEY=sk-or-v1-REPLACE_ME with a real key.
# Get one at https://openrouter.ai/keys.

# 3. Run the existing test suite
uv run python -m pytest tests/ -q
# Expect: 11 passed

# 4. Run a 5-paper smoke test on SciREX dev. This costs ~$0.02 of OpenRouter
#    credit. You need the SciREX gold release data placed at
#    data/raw/scirex/scirex_dataset/release_data/dev.jsonl first (see below).
uv run python scripts/run_benchmarks.py \
    --benchmarks scirex --max-per-benchmark 5 \
    --config config/models.yaml \
    --output-dir /tmp/smoke_test
```

See [docs/quickstart.md](docs/quickstart.md) for a longer guide.

---

## What this repository releases

| Area                         | Contents                                                                |
|------------------------------|-------------------------------------------------------------------------|
| **Pipeline source**          | `src/paper1/` — `pipeline.py`, `voting.py`, three agents, schema.       |
| **Agent prompts**            | `config/prompts/` — extractor, critic, consolidator, plus `extractor_no_binding.md` (the v8 ablation variant) and `extractor_fewshot.md`. |
| **Model configurations**     | `config/*.yaml` — default open-weights, frontier-extractor ablation, 8B/70B fine-tuned, GPT-4o ablation. |
| **All paper artifacts**      | `artifacts/paper_data_v3` through `v9` — every per-paper ContributionRecord, evaluation.json, calibration.json, comparison.json that the manuscript cites. |
| **Analysis scripts**         | `scripts/` — every script that produced a number or figure in the paper. The `scripts/v7_*.py`, `v8_*.py`, `v9_*.py` drivers correspond to the experiments of the same name. |
| **Figure generators**        | `scripts/figures/build_manuscript_figures.py`, `build_fig_cost_vs_quality_v2.py`. |
| **Tests**                    | `tests/` — 11 unit tests covering voting determinism, schema validation, and the pipeline's three-agent flow with a mock client. |

See [docs/reproducing_results.md](docs/reproducing_results.md) for the
table/figure → script → artifact mapping.

## What is NOT included

- **The SciREX paper texts and gold annotations.** Apache-2.0 licensed but
  redistribution is outside this repo's scope. Obtain from
  [allenai/SciREX](https://github.com/allenai/SciREX) and extract
  `release_data.tar.gz` to `data/raw/scirex/scirex_dataset/release_data/`.
- **The TDMSci CoNLL files.** Obtain from
  [IBM/science-result-extractor](https://github.com/IBM/science-result-extractor)
  and place under `data/raw/science-result-extractor/data/TDMSci/conllFormat/`.
- **The 100 arXiv case-study paper texts.** Re-fetch via
  `scripts/v8_fetch_recent.py` (free arXiv API; we sample the same 100 with
  `seed=42`).
- **API keys.** `.env` is gitignored; copy `.env.example` to `.env` and fill
  in your own.

---

## Citation

A machine-readable record is in [CITATION.cff](CITATION.cff). The
preferred BibTeX form (please update with the final publication metadata
once available):

```bibtex
@inproceedings{siravuri2026contribution,
  title     = {Multi-Agent Contribution Extraction with Verifiable Grounding},
  author    = {Siravuri, Harish},
  booktitle = {International Conference on Machine Learning and Applications (ICMLA)},
  year      = {2026},
  note      = {Code and data available at \url{https://github.com/harishsiravuri/contribution_extraction_framework}}
}
```

## Documentation

- [docs/quickstart.md](docs/quickstart.md) — 5-minute end-to-end run.
- [docs/reproducing_results.md](docs/reproducing_results.md) — every table
  and figure in the paper mapped to the script that produced it and the
  artifact that holds the numbers.
- [docs/schema.md](docs/schema.md) — the `ContributionRecord` JSON Schema
  with a worked example.
- [config/README.md](config/README.md) — the model-config and prompt-loader
  schema.

## License

- Code (`src/`, `tests/`, `scripts/`) — **MIT License**, see [LICENSE](LICENSE).
- Prompts (`config/prompts/`), model configs (`config/*.yaml`), and result
  artifacts (`artifacts/`) — **CC-BY-4.0**.

Attribution to third-party dependencies and external services is in
[NOTICE](NOTICE).

