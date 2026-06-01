"""Command-line interface."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler

from paper1.config import load_config
from paper1.openrouter import OpenRouterClient
from paper1.pipeline import Pipeline

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Paper 1 CLI")
console = Console()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )


@app.command()
def extract(
    paper: Path = typer.Option(..., exists=True, readable=True, help="Path to paper text file (UTF-8)"),
    paper_id: str = typer.Option(..., help="Paper ID, e.g. openalex:W12345 or arxiv:2401.01234"),
    output: Path | None = typer.Option(None, help="Where to write the JSON record (default: stdout)"),
    retrieval: Path | None = typer.Option(None, help="Optional retrieval bundle text file"),
    config_path: Path | None = typer.Option(None, "--config", help="Path to models.yaml"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Run the multi-agent extraction pipeline on one paper."""

    _setup_logging(verbose)
    cfg = load_config(config_path=config_path)

    paper_text = paper.read_text(encoding="utf-8")
    retrieval_bundle = retrieval.read_text(encoding="utf-8") if retrieval else ""

    async def _run() -> None:
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
            record = await pipe.extract(
                paper_id=paper_id,
                paper_text=paper_text,
                retrieval_bundle=retrieval_bundle,
            )
        finally:
            await client.aclose()

        record_json = record.model_dump_json(indent=2, by_alias=True)
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(record_json, encoding="utf-8")
            console.print(f"[green]Wrote[/green] {output}")
        else:
            console.print_json(record_json)

        console.print(
            f"[dim]Cost: ${record.meta.cost_usd:.4f} · "
            f"{record.meta.tokens_in} in / {record.meta.tokens_out} out tokens · "
            f"{record.meta.wall_time_seconds:.1f}s[/dim]"
        )

    asyncio.run(_run())


@app.command()
def show_config(
    config_path: Path | None = typer.Option(None, "--config", help="Path to models.yaml"),
) -> None:
    """Print the resolved configuration (with the API key redacted)."""

    cfg = load_config(config_path=config_path, require_api_key=False)
    redacted = {
        "extractor": cfg.extractor.__dict__,
        "critic": cfg.critic.__dict__,
        "consolidator": cfg.consolidator.__dict__,
        "defaults": cfg.defaults.__dict__,
        "concurrency": cfg.concurrency.__dict__,
        "base_url": cfg.base_url,
        "api_key_set": bool(cfg.api_key),
    }
    console.print_json(json.dumps(redacted, default=str))


if __name__ == "__main__":
    app()
