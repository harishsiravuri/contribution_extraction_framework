"""Convenience script: run the pipeline on the bundled example paper.

Usage:
    python scripts/extract_one.py
    python scripts/extract_one.py path/to/your_paper.txt
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from paper1.config import load_config
from paper1.openrouter import OpenRouterClient
from paper1.pipeline import Pipeline


async def main(paper_path: Path, paper_id: str) -> None:
    cfg = load_config()
    paper_text = paper_path.read_text(encoding="utf-8")

    client = OpenRouterClient(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        timeout_s=cfg.defaults.request_timeout_s,
        max_retries=cfg.defaults.max_retries,
        retry_backoff_s=cfg.defaults.retry_backoff_s,
        referer=cfg.referer,
        title=cfg.title,
    )
    try:
        pipe = Pipeline(client, cfg)
        record = await pipe.extract(paper_id=paper_id, paper_text=paper_text)
    finally:
        await client.aclose()

    print(record.model_dump_json(indent=2, by_alias=True))


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    default_paper = here.parent / "examples" / "sample_paper.txt"
    paper_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_paper
    paper_id = sys.argv[2] if len(sys.argv) > 2 else f"local:{paper_path.stem}"
    asyncio.run(main(paper_path, paper_id))
