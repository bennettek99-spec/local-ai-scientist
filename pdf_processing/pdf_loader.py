"""Download and locally cache arXiv PDFs.

Downloads are asynchronous (via httpx) so the pipeline can fetch several papers
concurrently. Files are stored under ``data/papers`` keyed by arXiv id and are
skipped if already present.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from config.settings import settings
from core.models import Paper
from utils.logging_config import get_logger

logger = get_logger(__name__)

_USER_AGENT = "LocalAIScientist/1.0 (personal research assistant)"


def _safe_filename(arxiv_id: str) -> str:
    """Turn an arXiv id into a filesystem-safe filename."""
    return arxiv_id.replace("/", "_") + ".pdf"


def local_pdf_path(arxiv_id: str) -> Path:
    """Return the canonical local path for a paper's PDF."""
    return settings.papers_dir / _safe_filename(arxiv_id)


async def download_pdf(
    paper: Paper,
    client: httpx.AsyncClient | None = None,
    force: bool = False,
) -> Path | None:
    """Download a single paper's PDF, returning the local path (or None).

    Skips the download if the file already exists unless ``force`` is set.
    """
    if not paper.pdf_url:
        logger.warning("Paper %s has no pdf_url; skipping download.", paper.arxiv_id)
        return None

    destination = local_pdf_path(paper.arxiv_id)
    if destination.exists() and not force and destination.stat().st_size > 0:
        logger.debug("PDF already cached: %s", destination.name)
        return destination

    owns_client = client is None
    client = client or httpx.AsyncClient(
        timeout=httpx.Timeout(60.0),
        headers={"User-Agent": _USER_AGENT},
        follow_redirects=True,
    )
    try:
        logger.info("Downloading PDF for %s", paper.arxiv_id)
        response = await client.get(paper.pdf_url)
        response.raise_for_status()
        destination.write_bytes(response.content)
        logger.info("Saved %s (%d KB)", destination.name, len(response.content) // 1024)
        return destination
    except httpx.HTTPError as exc:
        logger.error("Failed to download %s: %s", paper.arxiv_id, exc)
        return None
    finally:
        if owns_client:
            await client.aclose()


async def download_many(
    papers: list[Paper],
    max_concurrent: int | None = None,
    force: bool = False,
) -> dict[str, Path | None]:
    """Download multiple PDFs concurrently with a bounded semaphore.

    Returns a mapping of arXiv id -> local path (None for failures).
    """
    max_concurrent = max_concurrent or settings.max_concurrent_jobs
    semaphore = asyncio.Semaphore(max_concurrent)
    results: dict[str, Path | None] = {}

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(60.0),
        headers={"User-Agent": _USER_AGENT},
        follow_redirects=True,
    ) as client:

        async def _worker(paper: Paper) -> None:
            async with semaphore:
                results[paper.arxiv_id] = await download_pdf(
                    paper, client=client, force=force
                )

        await asyncio.gather(*(_worker(p) for p in papers))

    return results
