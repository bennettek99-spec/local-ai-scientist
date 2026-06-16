"""Summary agent: turn raw paper text into a structured analysis via Granite.

Produces a :class:`PaperAnalysis` containing a concise summary, key findings,
important equations, assumptions/limitations, a simplified explanation, related
topics, and suggested future work.
"""

from __future__ import annotations

from config.settings import settings
from core.llm import LLMClient, make_llm_client
from core.llm_base import LLMError
from core.models import Paper, PaperAnalysis
from utils.logging_config import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a meticulous research assistant analysing scientific papers across "
    "physics, astrophysics, materials science, genetics, paleogenetics, "
    "paleoanthropology, and artificial intelligence. You are precise, you never "
    "invent results, and you respond only with valid JSON matching the requested "
    "schema."
)

_ANALYSIS_INSTRUCTIONS = """\
Analyse the following paper and return a JSON object with EXACTLY these keys:

{{
  "summary": "3-5 sentence concise summary of the paper",
  "key_findings": ["bullet point finding", "..."],
  "equations": ["important equation in plain text or LaTeX with a short note", "..."],
  "assumptions_limitations": ["assumption or limitation", "..."],
  "simplified_explanation": "explanation a curious high-school student could follow",
  "related_topics": ["topic or keyword", "..."],
  "future_work": ["suggested next step or open question", "..."]
}}

Rules:
- Base everything strictly on the paper text; if something is absent use an empty list.
- Keep list items short (one sentence each).
- For equations, include the symbol meaning briefly; if there are none, return [].
- Do not include any text outside the JSON object.

Paper title: {title}
Primary category: {category}
Authors: {authors}

Paper text (may be truncated):
\"\"\"
{text}
\"\"\"
"""


class SummaryAgent:
    """Generate structured analyses of papers using Granite via Ollama."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or make_llm_client()

    def _build_prompt(self, paper: Paper, text: str) -> str:
        snippet = text[: settings.max_chars_for_summary]
        # Fall back to the abstract if no full text was extracted.
        if not snippet.strip():
            snippet = paper.abstract
        return _ANALYSIS_INSTRUCTIONS.format(
            title=paper.title,
            category=paper.primary_category,
            authors=", ".join(paper.authors[:10]),
            text=snippet,
        )

    def analyze(self, paper: Paper, text: str) -> PaperAnalysis:
        """Synchronously analyse a paper. Returns an empty analysis on failure."""
        prompt = self._build_prompt(paper, text)
        try:
            data = self.client.generate_json(prompt, system=_SYSTEM_PROMPT)
        except LLMError as exc:
            logger.error("Summary generation failed for %s: %s", paper.arxiv_id, exc)
            return PaperAnalysis()
        return self._to_analysis(data)

    async def aanalyze(self, paper: Paper, text: str) -> PaperAnalysis:
        """Async counterpart of :meth:`analyze` for concurrent pipelines."""
        prompt = self._build_prompt(paper, text)
        try:
            data = await self.client.agenerate_json(prompt, system=_SYSTEM_PROMPT)
        except LLMError as exc:
            logger.error("Summary generation failed for %s: %s", paper.arxiv_id, exc)
            return PaperAnalysis()
        return self._to_analysis(data)

    @staticmethod
    def _to_analysis(data: dict) -> PaperAnalysis:
        """Coerce raw model JSON into a validated PaperAnalysis."""

        def as_list(value) -> list[str]:
            if isinstance(value, list):
                return [str(v).strip() for v in value if str(v).strip()]
            if isinstance(value, str) and value.strip():
                return [value.strip()]
            return []

        return PaperAnalysis(
            summary=str(data.get("summary", "")).strip(),
            key_findings=as_list(data.get("key_findings")),
            equations=as_list(data.get("equations")),
            assumptions_limitations=as_list(data.get("assumptions_limitations")),
            simplified_explanation=str(data.get("simplified_explanation", "")).strip(),
            related_topics=as_list(data.get("related_topics")),
            future_work=as_list(data.get("future_work")),
        )
