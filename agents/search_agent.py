"""Search agent: discover recent papers on arXiv.

Wraps the ``arxiv`` Python package, mapping the user's research fields to arXiv
category codes and returning :class:`Paper` objects.
"""

from __future__ import annotations

from datetime import datetime, timezone

import arxiv

from config.settings import settings
from core.models import Paper
from utils.logging_config import get_logger

logger = get_logger(__name__)


class SearchAgent:
    """Query arXiv for recent papers in configured fields or by free text."""

    def __init__(self, page_size: int = 50, delay_seconds: float = 3.0) -> None:
        # A shared client with arXiv-friendly rate limiting.
        self._client = arxiv.Client(
            page_size=page_size,
            delay_seconds=delay_seconds,
            num_retries=3,
        )

    def search_field(self, field: str, max_results: int | None = None) -> list[Paper]:
        """Search a single human-friendly field (e.g. ``"Paleogenetics"``)."""
        max_results = max_results or settings.max_results_per_field
        categories = settings.categories_for_fields([field]).get(field, [])
        if not categories:
            logger.warning("Unknown field '%s'; no arXiv categories mapped.", field)
            return []

        query = " OR ".join(f"cat:{c}" for c in categories)
        return self._run_search(query, max_results, field=field)

    def search_fields(
        self, fields: list[str] | None = None, max_results: int | None = None
    ) -> list[Paper]:
        """Search several fields and return a de-duplicated list of papers."""
        fields = fields or settings.default_fields
        seen: dict[str, Paper] = {}
        for field in fields:
            for paper in self.search_field(field, max_results=max_results):
                # Keep the first field that surfaced a given paper.
                seen.setdefault(paper.arxiv_id, paper)
        papers = list(seen.values())
        logger.info("Search across %d fields returned %d unique papers", len(fields), len(papers))
        return papers

    def search_query(
        self,
        query: str,
        max_results: int = 20,
        field: str = "",
        by_relevance: bool = False,
    ) -> list[Paper]:
        """Run a free-text arXiv query (advanced syntax supported).

        Set ``by_relevance=True`` to rank by relevance (best for question-driven
        searches); the default ranks by submission date (best for "what's new").
        """
        return self._run_search(
            query, max_results, field=field, by_relevance=by_relevance
        )

    # ----------------------------------------------------------------- helpers
    def _run_search(
        self, query: str, max_results: int, field: str, by_relevance: bool = False
    ) -> list[Paper]:
        logger.info("arXiv search: %s (max=%d, relevance=%s)", query, max_results, by_relevance)
        sort_by = (
            arxiv.SortCriterion.Relevance
            if by_relevance
            else arxiv.SortCriterion.SubmittedDate
        )
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=arxiv.SortOrder.Descending,
        )
        papers: list[Paper] = []
        try:
            for result in self._client.results(search):
                papers.append(self._to_paper(result, field))
        except Exception as exc:  # noqa: BLE001 - network / parsing issues
            logger.error("arXiv search failed for '%s': %s", query, exc)
        return papers

    @staticmethod
    def _to_paper(result: "arxiv.Result", field: str) -> Paper:
        arxiv_id = result.get_short_id()
        published = result.published or datetime.now(timezone.utc)
        return Paper(
            arxiv_id=arxiv_id,
            title=result.title.strip().replace("\n", " "),
            authors=[a.name for a in result.authors],
            abstract=result.summary.strip().replace("\n", " "),
            primary_category=result.primary_category or "",
            categories=list(result.categories),
            field=field,
            published=published,
            updated=result.updated,
            pdf_url=result.pdf_url or "",
            entry_url=result.entry_id or "",
        )
