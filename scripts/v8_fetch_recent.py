"""Fetch 100 recent (2025-01-01 to 2026-05-31) cs.CL/cs.LG arXiv papers
uniformly at random within the window for E7 case study.

Uses arxiv API (free, no key). Saves to outputs/paper_data_v8/deployment_case_study/papers/
as one .txt per arxiv id with `Title: ...\n\nAbstract: ...` format.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from pathlib import Path

import httpx
import typer
from rich.console import Console

SEED = 42
ARXIV_API = "https://export.arxiv.org/api/query"
console = Console()
app = typer.Typer(no_args_is_help=False, add_completion=False)

PAGE_SIZE = 100
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
        out.append({"arxiv_id": arxiv_id, "title": title, "abstract": abstract, "published": published})
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
    target: int = typer.Option(100, "--target"),
    candidate_pool: int = typer.Option(600, "--candidate-pool", help="Fetch this many candidates then uniform-sample target from them"),
    output_dir: Path = typer.Option(Path("outputs/paper_data_v8/deployment_case_study/papers"), "--output-dir"),
    date_from: str = typer.Option("202501010000", "--date-from"),
    date_to: str = typer.Option("202605312359", "--date-to"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir.parent / "seed.txt").write_text(f"{SEED}\n")
    search_query = (
        "(cat:cs.CL OR cat:cs.LG) "
        f"AND submittedDate:[{date_from} TO {date_to}]"
    )
    console.print(f"  search: {search_query}")
    console.print(f"  target={target} pool={candidate_pool}")
    asyncio.run(_run(search_query, output_dir, target, candidate_pool))


async def _run(search_query: str, output_dir: Path, target: int, candidate_pool: int) -> None:
    candidates: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    async with httpx.AsyncClient(
        timeout=60.0, follow_redirects=True, headers={"User-Agent": "paper1-v8-case-study/0.1"}
    ) as client:
        start = 0
        empty_streak = 0
        while len(candidates) < candidate_pool:
            console.print(f"[dim]  page start={start}, have {len(candidates)}/{candidate_pool}[/dim]")
            try:
                entries = await _fetch_page(client, search_query, start)
            except httpx.HTTPError as e:
                console.print(f"[yellow]  http err: {e}; sleeping 10s[/yellow]")
                await asyncio.sleep(10)
                continue
            if not entries:
                empty_streak += 1
                if empty_streak >= 3:
                    console.print("[yellow]  no more results; stopping early[/yellow]")
                    break
            else:
                empty_streak = 0
            for e in entries:
                if e["arxiv_id"] in seen_ids:
                    continue
                # Re-verify the published date is in our window (safety; arxiv API can be loose)
                pub = e.get("published", "")
                if pub and pub[:10] < "2025-01-01":
                    continue
                if pub and pub[:10] > "2026-05-31":
                    continue
                candidates.append(e)
                seen_ids.add(e["arxiv_id"])
                if len(candidates) >= candidate_pool:
                    break
            start += PAGE_SIZE
            if len(candidates) < candidate_pool:
                await asyncio.sleep(DELAY_S)
    console.print(f"  total candidates: {len(candidates)}")

    rng = random.Random(SEED)
    sample = rng.sample(candidates, k=min(target, len(candidates)))
    console.print(f"  uniformly sampled {len(sample)} papers (seed={SEED})")

    # Save each as Title + Abstract format
    for e in sample:
        safe = e["arxiv_id"].replace("/", "_")
        text = f"Title: {e['title']}\n\nAbstract: {e['abstract']}\n"
        (output_dir / f"arxiv_{safe}.txt").write_text(text, encoding="utf-8")
    # Metadata index
    meta = [{"arxiv_id": e["arxiv_id"], "title": e["title"], "published": e.get("published", "")}
            for e in sample]
    (output_dir.parent / "sampled_papers.json").write_text(json.dumps(meta, indent=2))
    console.print(f"[green]  wrote {len(sample)} papers to {output_dir}[/green]")


if __name__ == "__main__":
    app()
