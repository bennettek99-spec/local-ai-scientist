"""Question agent: retrieval-augmented Q&A over the paper collection.

Retrieves the most relevant passages from the vector store and asks Granite to
answer using only that context, with inline citations to arXiv ids.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.llm import LLMClient, make_llm_client
from core.llm_base import LLMError
from database.vector_store import VectorStore
from utils.logging_config import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a research assistant answering questions about a personal library "
    "of scientific papers. Answer using ONLY the provided context passages. Cite "
    "the arXiv id in square brackets after each claim, e.g. [2401.01234]. If the "
    "context does not contain the answer, say so plainly rather than guessing."
)

_PROMPT_TEMPLATE = """\
Question: {question}

Context passages:
{context}

Write a clear, well-organised answer grounded in the passages above. Cite arXiv
ids in [brackets]. If the passages are insufficient, say what is missing.
"""


@dataclass
class Answer:
    """An answer plus the sources that informed it."""

    question: str
    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)

    def cited_papers(self) -> list[str]:
        """Unique arXiv ids of the retrieved sources, in order."""
        seen: list[str] = []
        for src in self.sources:
            aid = src.get("metadata", {}).get("arxiv_id")
            if aid and aid not in seen:
                seen.append(aid)
        return seen


class QuestionAgent:
    """Answer natural-language questions via RAG over the vector store."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        client: LLMClient | None = None,
    ) -> None:
        self.vector_store = vector_store or VectorStore()
        self.client = client or make_llm_client()

    def _build_context(self, chunks: list[dict[str, Any]]) -> str:
        blocks: list[str] = []
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            header = f"[{meta.get('arxiv_id', '?')}] {meta.get('title', '')}".strip()
            blocks.append(f"{header}\n{chunk.get('text', '')}")
        return "\n\n---\n\n".join(blocks)

    def ask(
        self, question: str, top_k: int | None = None, field: str | None = None
    ) -> Answer:
        """Answer a question grounded in the retrieved passages."""
        chunks = self.vector_store.query(question, top_k=top_k, field=field)
        if not chunks:
            return Answer(
                question=question,
                answer=(
                    "I couldn't find anything relevant in your library yet. "
                    "Try searching and processing some papers first."
                ),
                sources=[],
            )

        prompt = _PROMPT_TEMPLATE.format(
            question=question, context=self._build_context(chunks)
        )
        try:
            answer_text = self.client.generate(prompt, system=_SYSTEM_PROMPT)
        except LLMError as exc:
            logger.error("Question answering failed: %s", exc)
            answer_text = f"Failed to generate an answer: {exc}"

        return Answer(question=question, answer=answer_text, sources=chunks)
