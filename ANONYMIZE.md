# Checklist: Anonymizing the repo before uploading to anonymous.4open.science

Before uploading this repo for anonymous-review distribution, redact the following.

## 1. API keys (already gitignored, but verify)

- `.env` — gitignored; verify with `git ls-files | grep -i env`. The repo includes only `.env.example` with placeholder values.
- No keys should appear in any file under `config/`, `scripts/`, or `src/`.

## 2. Personal identifiers in `pyproject.toml`

Edit `pyproject.toml`:
```toml
# Before
authors = [{name = "Harish Siravuri", email = "harish.siravuri@gmail.com"}]
description = "Multi-Agent Contribution Extraction with Verifiable Grounding (Paper 1 of Harish Siravuri's PhD dissertation)"

# After (for anonymous upload)
authors = [{name = "Anonymous", email = "anonymous@example.com"}]
description = "Multi-Agent Contribution Extraction with Verifiable Grounding"
```

## 3. Together AI fine-tune model ID

The fine-tuned model ID `harishsiravuri_e088/Meta-Llama-3.1-70B-Instruct-Reference-scirex-v6-70b-7cecddd1`
includes a Together-account-derived username. Edit `config/models_ft_v6_70b.yaml`:

```yaml
# Before
extractor:
  model_id: "harishsiravuri_e088/Meta-Llama-3.1-70B-Instruct-Reference-scirex-v6-70b-7cecddd1"

# After
extractor:
  model_id: "<your-account>/Meta-Llama-3.1-70B-Instruct-Reference-scirex-ft"
  # Reproduce by running scripts/finetune_together.py with the released JSONL
  # dataset and the hyperparameters in src/paper1/finetune/together_finetune.py.
```

## 4. Git commit author

When initializing the anonymous-upload git repo on your Mac:

```bash
git config user.email "anonymous@example.com"
git config user.name "Anonymous"
```

…before any `git commit`. Or rewrite history with `git filter-repo` if commits already exist.

## 5. Verify nothing else identifies you

```bash
grep -ri "harish" .              # should return nothing
grep -ri "siravuri" .            # should return nothing
grep -ri "harishsiravuri_e088" . # should return nothing
git log --format='%an %ae'       # should show only "Anonymous"
```

## 6. Upload

Use [anonymous.4open.science](https://anonymous.4open.science) for double-blind-review-friendly hosting.
Reviewers see code without a GitHub username.

After acceptance, revert these redactions and push to a regular GitHub repo with your name.
