"""Few-shot pipeline: standard multi-agent Pipeline, but the Extractor's
prompt is loaded from a caller-supplied path rather than the default
`extractor.md` in the prompts dir.

This is a thin wrapper that reuses Pipeline's machinery (Critic, Consolidator,
voting, cost tracking, retries) and swaps only the ExtractorAgent's prompt.
"""

from __future__ import annotations

from pathlib import Path

from paper1.agents import ExtractorAgent
from paper1.config import Config
from paper1.openrouter import OpenRouterClientProtocol
from paper1.pipeline import Pipeline


def make_few_shot_pipeline(
    client: OpenRouterClientProtocol,
    config: Config,
    extractor_prompt_path: Path,
) -> Pipeline:
    """Construct a Pipeline whose ExtractorAgent uses the prompt at
    `extractor_prompt_path` instead of the default extractor.md."""
    pipe = Pipeline(client, config)
    prompt = Path(extractor_prompt_path).read_text(encoding="utf-8")
    # Replace the extractor agent in place — preserves the rest of the pipeline.
    pipe._extractor = ExtractorAgent(client, config.extractor, prompt)
    return pipe
