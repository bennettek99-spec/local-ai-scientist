"""NTRS Assistant: answer questions from the NASA Technical Reports Server.

For each question the LLM builds a keyword query, we search NTRS live
(ntrs.nasa.gov public API, relevance-ranked), and the LLM answers grounded in
the retrieved abstracts. NTRS is strong on aerospace, planetary science,
astrophysics, propulsion, and materials.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx

from core.llm import LLMClient, make_llm_client
from core.llm_base import LLMError
from utils.logging_config import get_logger

logger = get_logger(__name__)

_NTRS_SEARCH = "https://ntrs.nasa.gov/api/citations/search"
_USER_AGENT = "LocalAIScientist/1.0 (personal research assistant)"

_QUERY_SYSTEM = (
    "You convert research questions into effective NASA NTRS keyword searches. "
    "You reply only with JSON."
)

_QUERY_PROMPT = """\
Produce TWO NTRS keyword queries for the question:
- "precise": the core subject with any ACRONYMS EXPANDED to their full names
  (e.g. "HLS" -> "Human Landing System", "TPS" -> "thermal protection system").
  Keep it SHORT — the key noun phrase plus at most one qualifier. Do NOT pile on
  extra descriptive terms; that over-narrows and hurts results.
- "broad": just the single most central term or expanded acronym.

NEVER include filler words like recent, latest, new, advances, review, study,
paper, work — they wreck relevance.

Examples:
Q: Artemis HLS
-> {{"precise": "Artemis Human Landing System", "broad": "Human Landing System"}}
Q: thermal protection for reentry vehicles
-> {{"precise": "thermal protection system reentry", "broad": "thermal protection system"}}
Q: high-entropy alloys for turbines
-> {{"precise": "high-entropy alloy turbine", "broad": "high-entropy alloy"}}

Return JSON exactly as: {{"precise": "...", "broad": "..."}}

Question: {question}
"""

_ANSWER_SYSTEM = (
    "You are a research assistant answering from NASA technical reports (NTRS). "
    "Use ONLY the provided abstracts. Cite the report by its NTRS id in square "
    "brackets, e.g. [20190027435]. These are abstracts, not full reports, so "
    "avoid over-claiming. If the results don't address the question, say so."
)

_ANSWER_PROMPT = """\
Question: {question}

NASA NTRS results (title + abstract):
{context}

Write a clear, well-structured answer grounded in these abstracts. Cite NTRS
ids in [brackets]. Note briefly if the results seem tangential or thin.
"""


@dataclass
class NtrsPaper:
    """A single NASA technical report."""

    id: str
    title: str
    authors: list[str]
    abstract: str
    published: str
    url: str
    center: str = ""
    sti_type: str = ""
    score: float = 0.0


@dataclass
class NtrsAnswer:
    """Result of an NTRS-backed query."""

    question: str
    query: str
    answer: str
    papers: list[NtrsPaper] = field(default_factory=list)


class NtrsAssistant:
    """Answer questions by searching NASA NTRS live and synthesising results."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or make_llm_client()

    def build_queries(self, question: str) -> list[str]:
        """Return NTRS queries to try in order: precise -> broad -> raw question."""
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
            logger.warning("NTRS query generation failed: %s", exc)
        if question not in queries:
            queries.append(question)
        logger.info("NTRS query candidates for '%s': %s", question, queries)
        return queries

    def _search(self, query: str, max_results: int) -> list[NtrsPaper]:
        params = {"q": query, "page.size": max_results}
        try:
            response = httpx.get(
                _NTRS_SEARCH,
                params=params,
                headers={"User-Agent": _USER_AGENT},
                timeout=30,
                follow_redirects=True,
            )
            response.raise_for_status()
            results = response.json().get("results", [])
        except Exception as exc:  # noqa: BLE001 - network/parsing issues
            logger.error("NTRS search failed for '%s': %s", query, exc)
            return []

        papers: list[NtrsPaper] = []
        for res in results:
            authors = [
                aa.get("meta", {}).get("author", {}).get("name")
                for aa in res.get("authorAffiliations", [])
            ]
            authors = [a for a in authors if a]
            center = res.get("center")
            center_name = center.get("name") if isinstance(center, dict) else (center or "")
            pid = str(res.get("id", ""))
            papers.append(
                NtrsPaper(
                    id=pid,
                    title=(res.get("title") or "").strip(),
                    authors=authors,
                    abstract=(res.get("abstract") or "").strip(),
                    published=(res.get("distributionDate") or res.get("created") or "")[:10],
                    url=f"https://ntrs.nasa.gov/citations/{pid}",
                    center=center_name,
                    sti_type=res.get("stiType", ""),
                    score=float(res.get("_meta", {}).get("score", 0.0)),
                )
            )
        return papers

    @staticmethod
    def _build_context(papers: list[NtrsPaper]) -> str:
        blocks: list[str] = []
        for paper in papers:
            if not paper.abstract:
                continue
            authors = ", ".join(paper.authors[:6])
            blocks.append(
                f"[{paper.id}] {paper.title}\n"
                f"Authors: {authors}\nAbstract: {paper.abstract}"
            )
        return "\n\n---\n\n".join(blocks)

    def answer(self, question: str, max_results: int = 6) -> NtrsAnswer:
        """Search NTRS live for ``question`` and answer from the results.

        Runs every query variant (expanded, broad, and the raw question), then
        merges the hits and keeps the best by NTRS relevance score — so an
        over-narrow variant can't crowd out better matches.
        """
        candidates = self.build_queries(question)
        # Fetch a few extra per query so the merge has room to rank.
        fetch_n = max(max_results * 2, 10)
        merged: dict[str, NtrsPaper] = {}
        for candidate in candidates:
            for paper in self._search(candidate, max_results=fetch_n):
                if paper.id and (
                    paper.id not in merged or paper.score > merged[paper.id].score
                ):
                    merged[paper.id] = paper
        papers = sorted(merged.values(), key=lambda p: p.score, reverse=True)[:max_results]
        used_query = " · ".join(candidates)

        if not papers:
            return NtrsAnswer(
                question=question,
                query=used_query,
                answer=(
                    "No matching reports were found on NASA NTRS. Try rephrasing "
                    "toward aerospace, planetary science, propulsion, or materials "
                    "topics, where NTRS coverage is strongest."
                ),
                papers=[],
            )

        context = self._build_context(papers)
        try:
            answer_text = self.client.generate(
                _ANSWER_PROMPT.format(question=question, context=context),
                system=_ANSWER_SYSTEM,
            )
        except LLMError as exc:
            logger.error("NTRS answer generation failed: %s", exc)
            answer_text = f"Failed to generate an answer: {exc}"

        return NtrsAnswer(
            question=question, query=used_query, answer=answer_text, papers=papers
        )
