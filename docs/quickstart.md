# Quickstart

A 5-minute end-to-end reproduction of the smallest meaningful experiment
in the paper (the 5-paper SciREX-dev smoke test). Costs ~$0.02 of
OpenRouter credit.

## Prerequisites

- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- An [OpenRouter](https://openrouter.ai/) API key (free signup; ~$5 credit
  goes a long way at our prices)
- (Optional, only for the specialized framework results) A
  [Together AI](https://api.together.ai/) API key

## 1. Install

```bash
git clone https://github.com/<your-handle>/contribution_extraction_framework.git
cd contribution_extraction_framework
uv sync
```

If you do not use uv, `pip install -e ".[dev]"` works equivalently.

## 2. Set your API key

```bash
cp .env.example .env
# Open .env in your editor and replace
#   OPENROUTER_API_KEY=sk-or-v1-REPLACE_ME
# with your real key from https://openrouter.ai/keys.
```

## 3. Run the test suite

```bash
uv run python -m pytest tests/ -q
```

You should see `11 passed`. These are pure-unit tests (mocked LLM
calls) — no API key required, no money spent.

## 4. Obtain SciREX dev data (one-time, free)

```bash
mkdir -p data/raw/scirex
cd data/raw/scirex
git clone https://github.com/allenai/SciREX.git
tar -xzf SciREX/scirex_dataset/release_data.tar.gz -C SciREX/scirex_dataset/
ln -s SciREX/scirex_dataset .
cd ../../..
```

The loader at `src/paper1/loaders/scirex.py` expects
`data/raw/scirex/scirex_dataset/release_data/dev.jsonl`.

## 5. 5-paper smoke test on SciREX dev

```bash
uv run python scripts/run_benchmarks.py \
    --benchmarks scirex \
    --max-per-benchmark 5 \
    --config config/models.yaml \
    --output-dir /tmp/smoke_test
```

Expected wall-clock: ~2 minutes (concurrency 5 × ~15s per paper).
Expected cost: ~$0.02 of OpenRouter credit.

After it finishes:

- `/tmp/smoke_test/scirex/multi_agent/*.json` — 5 ContributionRecord JSONs.
- `/tmp/smoke_test/scirex/baseline/*.json` — 5 single-LLM-baseline records.
- `/tmp/smoke_test/scirex/evaluation.json` — F1 per field with bootstrap
  CIs and paired-permutation p-values vs the single-LLM baseline.

## 6. Inspect a record

```bash
cat /tmp/smoke_test/scirex/multi_agent/*.json | head -50
```

You should see the `ContributionRecord` shape documented in
[docs/schema.md](schema.md).

## What next

- For the full 66-paper SciREX dev evaluation (cost ~$0.30, ~10 min):
  drop `--max-per-benchmark`.
- For the SciREX test split: add `--scirex-split test`.
- For the specialized fine-tuned 70B framework (cost ~$10 + ~$25 of
  Together AI dedicated-endpoint time): see
  [docs/reproducing_results.md](reproducing_results.md) section "v6
  specialized framework".
- For the v7 / v8 / v9 ablation and case-study experiments: each has its
  own `scripts/v{N}_*.py` driver documented in `reproducing_results.md`.
