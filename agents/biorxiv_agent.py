"""bioRxiv Science Assistant: live preprint search + answer, via Europe PMC.

bioRxiv's own API only lists papers by date/DOI, so we keyword-search bioRxiv
through the Europe PMC REST API (free, no auth), which indexes bioRxiv preprints
with full abstracts. This gives much richer coverage than arXiv for biology,
genetics, paleogenetics, and related fields.

Flow per question (mirrors the arXiv Live assistant):
1. The LLM builds precise + broad Europe PMC queries.
2. We search bioRxiv preprints live, trying precise -> broad -> raw.
3. The LLM answers from the fresh abstracts, citing DOIs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import httpx

from core.llm import LLMClient, make_llm_client
from core.llm_base import LLMError
from utils.logging_config import get_logger

logger = get_logger(__name__)

_EUROPE_PMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
_BIORXIV_FILTER = 'PUBLISHER:"bioRxiv"'
_USER_AGENT = "LocalAIScientist/1.0 (personal research assistant)"

_QUERY_SYSTEM = (
    "You convert research questions into Europe PMC search queries. You reply "
    "only with JSON."
)

_QUERY_PROMPT = """\
Produce TWO Europe PMC search queries for the question.

CRITICAL: Europe PMC does NOT enforce plain quoted phrases — they return huge
amounts of noise. You MUST target fields with ABSTRACT:"..." and/or TITLE:"..."
so the phrase is actually required.

- "precise": (ABSTRACT:"core phrase" OR TITLE:"core phrase") optionally AND a
  key qualifier in parentheses (use OR for synonyms).
- "broad": (ABSTRACT:"core phrase" OR TITLE:"core phrase") with no extra AND.

Use ONLY substantive scientific terms (organisms, genes, methods, phenomena).
NEVER include filler words such as recent, latest, new, important, advances,
review, study, paper, work — they wreck relevance. Leave temporal intent OUT
(results are relevance-ranked).

Do NOT add any publisher/source filter; that is added automatically.

Examples:
Q: most important work on Homo heidelbergensis
-> {{"precise": "(ABSTRACT:\\"Homo heidelbergensis\\" OR TITLE:\\"Homo heidelbergensis\\")", "broad": "TITLE:\\"Homo heidelbergensis\\""}}
Q: Neanderthal introgression and the human immune system
-> {{"precise": "ABSTRACT:\\"Neanderthal introgression\\" AND (immune OR immunity)", "broad": "(ABSTRACT:\\"Neanderthal introgression\\" OR TITLE:\\"Neanderthal introgression\\")"}}
Q: methods for authenticating ancient DNA
-> {{"precise": "ABSTRACT:\\"ancient DNA\\" AND (authentication OR validation OR verification)", "broad": "(ABSTRACT:\\"ancient DNA\\" OR TITLE:\\"ancient DNA\\")"}}

Return JSON exactly as: {{"precise": "...", "broad": "..."}}

Question: {question}
"""

_ANSWER_SYSTEM = (
    "You are a research assistant answering from bioRxiv PREPRINTS (not yet "
    "peer-reviewed). Use ONLY the provided abstracts. Cite the DOI in square "
    "brackets after each claim. Note that these are preprints. If the results "
    "don't address the question, say so plainly."
)

_ANSWER_PROMPT = """\
Question: {question}

Fresh bioRxiv preprint results (title + abstract):
{context}

Write a clear, well-structured answer grounded in these abstracts. Cite DOIs in
[brackets]. Remind the reader briefly that bioRxiv papers are preprints.
"""


@dataclass
class BioPaper:
    """A single bioRxiv preprint result."""

    doi: str
    title: str
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    published: str = ""
    url: str = ""


@dataclass
class BioAnswer:
    """Result of a live bioRxiv-backed query."""

    question: str
    query: str
    answer: str
    papers: list[BioPaper] = field(default_factory=list)


def _strip_tags(text: str) -> str:
    """Remove JATS/HTML tags that sometimes appear in Europe PMC abstracts."""
    return re.sub(r"<[^>]+>", "", text or "").strip()


def search_biorxiv(query: str, max_results: int = 6) -> list[BioPaper]:
    """Search bioRxiv preprints via Europe PMC, ranked by relevance."""
    params = {
        "query": f"({query}) AND {_BIORXIV_FILTER}",
        "format": "json",
        "resultType": "core",
        "pageSize": max_results,
    }
    try:
        response = httpx.get(
            _EUROPE_PMC, params=params, timeout=30, headers={"User-Agent": _USER_AGENT}
        )
        response.raise_for_status()
        results = response.json().get("resultList", {}).get("result", [])
    except (httpx.HTTPError, ValueError) as exc:
        logger.error("Europe PMC search failed for '%s': %s", query, exc)
        return []

    papers: list[BioPaper] = []
    for res in results:
        doi = res.get("doi") or res.get("id") or ""
        author_string = res.get("authorString") or ""
        authors = [a.strip() for a in author_string.split(",") if a.strip()]
        papers.append(
            BioPaper(
                doi=doi,
                title=_strip_tags(res.get("title") or "").rstrip("."),
                authors=authors,
                abstract=_strip_tags(res.get("abstractText") or ""),
                published=res.get("firstPublicationDate") or "",
                url=f"https://doi.org/{doi}" if doi else "",
            )
        )
    logger.info("bioRxiv search '%s' -> %d results", query, len(papers))
    return papers


class BioRxivAssistant:
    """Answer questions by searching bioRxiv live and synthesising the results."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or make_llm_client()

    def build_queries(self, question: str) -> list[str]:
        """Return Europe PMC queries to try: precise -> broad -> raw question."""
        queries: list[str] = []
        try:
            data = self.client.generate_json(
                _QUERY_PROMPT.format(question=question), system=_QUERY_SYSTEM
            )
            for key in ("precise", "broad"):
                candidate = str(data.get(key, "")).strip()
                if candidate and candidate not in queries:
                    queries.append(candidate)
        except LLMError as exc:
            logger.warning("bioRxiv query generation failed: %s", exc)
        # Only fall back to the bare question if the model produced nothing —
        # a non-field-targeted query would otherwise reintroduce noise.
        if not queries:
            queries.append(question)
        logger.info("bioRxiv query candidates for '%s': %s", question, queries)
        return queries

    @staticmethod
    def _build_context(papers: list[BioPaper]) -> str:
        blocks: list[str] = []
        for paper in papers:
            authors = ", ".join(paper.authors[:6])
            blocks.append(
                f"[{paper.doi}] {paper.title}\n"
                f"Authors: {authors} ({paper.published})\n"
                f"Abstract: {paper.abstract}"
            )
        return "\n\n---\n\n".join(blocks)

    def answer(self, question: str, max_results: int = 6) -> BioAnswer:
        """Search bioRxiv live for ``question`` and answer from the results."""
        candidates = self.build_queries(question)
        papers: list[BioPaper] = []
        used_query = candidates[0]
        for candidate in candidates:
            papers = search_biorxiv(candidate, max_results=max_results)
            used_query = candidate
            if papers:
                break

        if not papers:
            return BioAnswer(
                question=question,
                query=used_query,
                answer=(
                    "No matching bioRxiv preprints were found, even after "
                    "broadening the search. Try rephrasing the question."
                ),
                papers=[],
            )

        prompt = _ANSWER_PROMPT.format(
            question=question, context=self._build_context(papers)
        )
        try:
            answer_text = self.client.generate(prompt, system=_ANSWER_SYSTEM)
        except LLMError as exc:
            logger.error("bioRxiv answer generation failed: %s", exc)
            answer_text = f"Failed to generate an answer: {exc}"

        return BioAnswer(
            question=question, query=used_query, answer=answer_text, papers=papers
        )
