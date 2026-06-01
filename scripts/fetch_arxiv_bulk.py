"""Bulk arXiv abstract fetcher for the Pilot 1k experiment.

Fetches 1,000 arXiv abstracts from cs.LG, cs.CL, cs.CV (>= 2022-01-01),
sorted by submission date descending. Writes one .txt per paper to
examples/pilot_corpus/.

Usage:
    python scripts/fetch_arxiv_bulk.py
    python scripts/fetch_arxiv_bulk.py --target 1000 --output-dir examples/pilot_corpus
"""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path

import httpx
import typer
from rich.console import Console

ARXIV_API = "https://export.arxiv.org/api/query"
console = Console()
app = typer.Typer(no_args_is_help=False, add_completion=False)

PAGE_SIZE = 100  # smaller pages are friendlier on arXiv's API
DELAY_S = 3.0


def _parse_atom(xml: str) -> list[dict[str, str]]:
    entries = re.findall(r"<entry>(.+?)</entry>", xml, flags=re.DOTALL)
    out: list[dict[str, str]] = []
    for entry in entries:
        id_match = re.search(r"<id>http[s]?://arxiv\.org/abs/([^<]+?)</id>", entry)
        title_match = re.search(r"<title>(.+?)</title>", entry, flags=re.DOTALL)
        summary_match = re.search(r"<summary>(.+?)</summary>", entry, flags=re.DOTALL)
        published_match = re.search(r"<published>([^<]+)</published>", entry)
        if not (id_match and title_match and summary_match):
            continue
        arxiv_id_with_v = id_match.group(1).strip()
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id_with_v)
        title = re.sub(r"\s+", " ", title_match.group(1).strip())
        abstract = re.sub(r"\s+", " ", summary_match.group(1).strip())
        published = published_match.group(1).strip() if published_match else ""
        out.append(
            {"arxiv_id": arxiv_id, "title": title, "abstract": abstract, "published": published}
        )
    return out


async def _fetch_page(client: httpx.AsyncClient, search_query: str, start: int) -> list[dict[str, str]]:
    params = {
        "search_query": search_query,
        "start": str(start),
        "max_results": str(PAGE_SIZE),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    resp = await client.get(ARXIV_API, params=params)
    resp.raise_for_status()
    return _parse_atom(resp.text)


@app.command()
def fetch(
    target: int = typer.Option(1000, "--target", help="How many papers to fetch"),
    output_dir: Path = typer.Option(
        Path("examples/pilot_corpus"), "--output-dir"
    ),
    date_from: str = typer.Option("202201010000", "--date-from", help="arXiv submitted-date lower bound (YYYYMMDDHHMM)"),
    date_to: str = typer.Option("202612312359", "--date-to", help="arXiv submitted-date upper bound"),
) -> None:
    """Fetch ~target arXiv abstracts from cs.LG/cs.CL/cs.CV in a date window."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # arXiv API: cat:cs.LG OR cat:cs.CL OR cat:cs.CV plus a submitted-date filter
    search_query = (
        "(cat:cs.LG OR cat:cs.CL OR cat:cs.CV) "
        f"AND submittedDate:[{date_from} TO {date_to}]"
    )

    saved: set[str] = {p.stem.removeprefix("arxiv_") for p in output_dir.glob("arxiv_*.txt")}
    if len(saved) >= target:
        console.print(
            f"[green]Already have {len(saved)} papers in {output_dir}[/green]"
        )
        return

    asyncio.run(_run(search_query, output_dir, target, saved))


async def _run(search_query: str, output_dir: Path, target: int, saved: set[str]) -> None:
    async with httpx.AsyncClient(
        timeout=60.0, follow_redirects=True, headers={"User-Agent": "paper1-pilot/0.1"}
    ) as client:
        start = 0
        attempts_with_no_progress = 0
        while len(saved) < target:
            console.print(
                f"[dim]page start={start}, have {len(saved)} / {target}[/dim]"
            )
            try:
                entries = await _fetch_page(client, search_query, start)
            except httpx.HTTPError as e:
                console.print(f"[yellow]http error at start={start}: {e}; sleeping 10s[/yellow]")
                await asyncio.sleep(10)
                continue

            if not entries:
                attempts_with_no_progress += 1
                if attempts_with_no_progress >= 3:
                    console.print("[yellow]no more results from arXiv; stopping[/yellow]")
                    break
                await asyncio.sleep(DELAY_S)
                start += PAGE_SIZE
                continue

            new_in_page = 0
            for e in entries:
                if len(saved) >= target:
                    break
                aid = e["arxiv_id"]
                if aid in saved:
                    continue
                safe = aid.replace("/", "_")
                text = f"Title: {e['title']}\n\nAbstract: {e['abstract']}\n"
                (output_dir / f"arxiv_{safe}.txt").write_text(text, encoding="utf-8")
                saved.add(aid)
                new_in_page += 1

            if new_in_page == 0:
                attempts_with_no_progress += 1
            else:
                attempts_with_no_progress = 0

            start += PAGE_SIZE
            if len(saved) < target:
                await asyncio.sleep(DELAY_S)

    console.print(f"[bold green]Saved {len(saved)} papers to {output_dir}[/bold green]")


if __name__ == "__main__":
    app()
