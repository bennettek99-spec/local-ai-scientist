"""Live Science Assistant: answer questions with a fresh arXiv search per query.

Unlike :class:`QuestionAgent` (which retrieves from the local vector store), this
agent goes out to arXiv for every question:

1. The LLM turns the natural-language question into a focused arXiv query.
2. :class:`SearchAgent` runs that query live against arXiv.
3. The LLM answers the question grounded in the freshly-retrieved abstracts,
   citing arXiv ids.

The answer is based on abstracts (not full text), which keeps it fast and means
the relevant papers need not already be in your library.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agents.search_agent import SearchAgent
from core.llm import LLMClient, make_llm_client
from core.llm_base import LLMError
from core.models import Paper
from utils.logging_config import get_logger

logger = get_logger(__name__)

_QUERY_SYSTEM = (
    "You convert research questions into effective arXiv search queries. You "
    "reply only with JSON."
)

_QUERY_PROMPT = """\
Produce TWO arXiv search queries for the question, so we can fall back if the
precise one finds nothing:
- "precise": field-targeted for on-topic hits — abs:"core phrase" optionally
  AND a key qualifier; use OR (in parentheses) for synonyms. Don't over-constrain.
- "broad": just the 2-4 core keywords, NO field prefixes, NO quotes — maximise
  recall for sparsely-covered topics.

CRITICAL: use ONLY substantive scientific terms. NEVER include filler words like
recent, latest, new, important, advances, review, study, paper, work — they wreck
relevance. Leave temporal words out (results are ranked by relevance).

Examples:
Q: recent methods for removing contamination from ancient DNA samples
-> {{"precise": "abs:\\"ancient DNA\\" AND (abs:contamination OR abs:decontamination)", "broad": "ancient DNA contamination"}}
Q: most important papers about Homo heidelbergensis
-> {{"precise": "abs:\\"Homo heidelbergensis\\"", "broad": "Homo heidelbergensis hominin"}}
Q: breakthroughs in high-entropy alloy design
-> {{"precise": "abs:\\"high-entropy alloy\\" AND (abs:design OR abs:discovery)", "broad": "high-entropy alloy design"}}

Return JSON exactly as: {{"precise": "...", "broad": "..."}}

Question: {question}
"""

_ANSWER_SYSTEM = (
    "You are a research assistant answering with up-to-date arXiv results. Use "
    "ONLY the provided paper abstracts. Cite arXiv ids in square brackets, e.g. "
    "[2401.01234]. These are abstracts, not full papers, so avoid over-claiming. "
    "If the results don't address the question, say so plainly."
)

_ANSWER_PROMPT = """\
Question: {question}

Fresh arXiv results (title + abstract):
{context}

Write a clear, well-structured answer grounded in these abstracts. Cite arXiv
ids in [brackets]. Note briefly if the results seem tangential or thin.
"""


@dataclass
class LiveAnswer:
    """Result of a live arXiv-backed query."""

    question: str
    arxiv_query: str
    answer: str
    papers: list[Paper] = field(default_factory=list)


class LiveScienceAssistant:
    """Answer questions by searching arXiv live and synthesising the results."""

    def __init__(
        self,
        search_agent: SearchAgent | None = None,
        client: LLMClient | None = None,
    ) -> None:
        self.search_agent = search_agent or SearchAgent()
        self.client = client or make_llm_client()

    def build_queries(self, question: str) -> list[str]:
        """Return arXiv queries to try in order: precise -> broad -> raw question.

        Trying progressively broader queries means sparsely-covered topics (e.g.
        much of paleoanthropology) still surface something instead of failing.
        """
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
            logger.warning("Query generation failed: %s", exc)
        # Always keep the raw question as a final fallback.
        if question not in queries:
            queries.append(question)
        logger.info("Live query candidates for '%s': %s", question, queries)
        return queries

    @staticmethod
    def _build_context(papers: list[Paper]) -> str:
        blocks: list[str] = []
        for paper in papers:
            authors = ", ".join(paper.authors[:6])
            blocks.append(
                f"[{paper.arxiv_id}] {paper.title}\n"
                f"Authors: {authors}\n"
                f"Abstract: {paper.abstract}"
            )
        return "\n\n---\n\n".join(blocks)

    def answer(self, question: str, max_results: int = 6) -> LiveAnswer:
        """Search arXiv live for ``question`` and answer from the results.

        Tries progressively broader queries and stops at the first that returns
        results, so on-topic-but-sparse subjects still surface papers.
        """
        candidates = self.build_queries(question)
        papers: list[Paper] = []
        used_query = candidates[0]
        for candidate in candidates:
            papers = self.search_agent.search_query(
                candidate, max_results=max_results, by_relevance=True
            )
            used_query = candidate
            if papers:
                break

        if not papers:
            return LiveAnswer(
                question=question,
                arxiv_query=used_query,
                answer=(
                    "No matching papers were found on arXiv, even after broadening "
                    "the search. Note that arXiv has limited coverage of some "
                    "fields (e.g. much of paleoanthropology and clinical biology) — "
                    "for those, journals, bioRxiv, or PubMed are better sources. "
                    "Try rephrasing toward a more arXiv-covered angle (e.g. the "
                    "genetics or computational-methods side of the topic)."
                ),
                papers=[],
            )

        prompt = _ANSWER_PROMPT.format(
            question=question, context=self._build_context(papers)
        )
        try:
            answer_text = self.client.generate(prompt, system=_ANSWER_SYSTEM)
        except LLMError as exc:
            logger.error("Live answer generation failed: %s", exc)
            answer_text = f"Failed to generate an answer: {exc}"

        return LiveAnswer(
            question=question, arxiv_query=used_query, answer=answer_text, papers=papers
        )
