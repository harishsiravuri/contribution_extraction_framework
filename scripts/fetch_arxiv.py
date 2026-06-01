"""Fetch arXiv paper metadata + abstracts and save them as .txt files.

For the 10-paper smoke test we use abstracts only (small, fast, no PDF parsing).
For the real evaluation later we'll switch to S2ORC full text.

Usage:
    python scripts/fetch_arxiv.py --ids-file examples/arxiv_ids.txt --output-dir examples/papers
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import httpx
import typer
from rich.console import Console

ARXIV_API = "https://export.arxiv.org/api/query"
console = Console()
app = typer.Typer(no_args_is_help=False, add_completion=False)


def _read_ids(path: Path) -> list[str]:
    out: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def _parse_atom(xml: str) -> list[dict[str, str]]:
    """Lightweight regex parser for the Atom feed arXiv returns.

    Avoids adding an XML lib dependency; the format is simple enough.
    """

    entries = re.findall(r"<entry>(.+?)</entry>", xml, flags=re.DOTALL)
    out: list[dict[str, str]] = []
    for entry in entries:
        id_match = re.search(r"<id>http://arxiv\.org/abs/([^<]+?)</id>", entry)
        title_match = re.search(r"<title>(.+?)</title>", entry, flags=re.DOTALL)
        summary_match = re.search(r"<summary>(.+?)</summary>", entry, flags=re.DOTALL)
        if not (id_match and title_match and summary_match):
            continue
        arxiv_id_with_v = id_match.group(1).strip()
        # strip "v1", "v2", ... suffix
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id_with_v)
        title = re.sub(r"\s+", " ", title_match.group(1).strip())
        abstract = re.sub(r"\s+", " ", summary_match.group(1).strip())
        out.append({"arxiv_id": arxiv_id, "title": title, "abstract": abstract})
    return out


async def _fetch_all(ids: list[str]) -> list[dict[str, str]]:
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        resp = await client.get(
            ARXIV_API,
            params={"id_list": ",".join(ids), "max_results": str(len(ids))},
        )
        resp.raise_for_status()
        return _parse_atom(resp.text)


@app.command()
def fetch(
    ids_file: Path = typer.Option(
        Path("examples/arxiv_ids.txt"), "--ids-file", exists=True, readable=True
    ),
    output_dir: Path = typer.Option(Path("examples/papers"), "--output-dir"),
) -> None:
    """Fetch arXiv abstracts for every ID in `ids_file` and save to `output_dir`."""

    ids = _read_ids(ids_file)
    console.print(f"[bold]Fetching {len(ids)} arXiv papers...[/bold]")
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = asyncio.run(_fetch_all(ids))
    fetched_ids = {e["arxiv_id"] for e in entries}
    missing = [i for i in ids if i not in fetched_ids]
    if missing:
        console.print(f"[yellow]Warning:[/yellow] no entry returned for: {missing}")

    for e in entries:
        safe_id = e["arxiv_id"].replace("/", "_")
        text = f"Title: {e['title']}\n\nAbstract: {e['abstract']}\n"
        path = output_dir / f"arxiv_{safe_id}.txt"
        path.write_text(text, encoding="utf-8")
        console.print(f"  [green]saved[/green] {path.name}  ({len(e['abstract'])} chars)")

    console.print(f"\n[bold]Done.[/bold] {len(entries)} files in {output_dir}")


if __name__ == "__main__":
    app()
