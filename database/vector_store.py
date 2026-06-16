"""Local vector database for semantic search over paper text.

Wraps a persistent ChromaDB collection that uses a sentence-transformers model
for embeddings. Papers are stored as overlapping chunks so retrieval can return
the most relevant passages rather than whole documents.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions

from config.settings import settings
from core.models import Paper
from pdf_processing.text_extractor import chunk_text
from utils.logging_config import get_logger

logger = get_logger(__name__)

_COLLECTION_NAME = "papers"


class VectorStore:
    """Persistent semantic index over the paper collection."""

    def __init__(
        self,
        persist_dir: Path | None = None,
        embedding_model: str | None = None,
    ) -> None:
        self.persist_dir = persist_dir or settings.embeddings_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model or settings.embedding_model

        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=False),
        )
        # SentenceTransformer model is downloaded once and cached locally.
        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model
        )
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------- write
    def add_paper(self, paper: Paper, text: str) -> int:
        """Chunk ``text`` and add it to the index. Returns the chunk count.

        Existing chunks for the paper are removed first so re-embedding is
        idempotent.
        """
        if not text.strip():
            logger.warning("No text to embed for %s", paper.arxiv_id)
            return 0

        self.remove_paper(paper.arxiv_id)
        chunks = chunk_text(text)
        if not chunks:
            return 0

        ids = [f"{paper.arxiv_id}::{i}" for i in range(len(chunks))]
        metadatas: list[dict[str, Any]] = [
            {
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "field": paper.field,
                "primary_category": paper.primary_category,
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]
        self._collection.add(ids=ids, documents=chunks, metadatas=metadatas)
        logger.info("Embedded %d chunks for %s", len(chunks), paper.arxiv_id)
        return len(chunks)

    def remove_paper(self, arxiv_id: str) -> None:
        """Delete all chunks belonging to a paper."""
        try:
            self._collection.delete(where={"arxiv_id": arxiv_id})
        except Exception as exc:  # noqa: BLE001
            logger.debug("Nothing to delete for %s (%s)", arxiv_id, exc)

    # -------------------------------------------------------------------- read
    def query(
        self,
        question: str,
        top_k: int | None = None,
        field: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the most relevant chunks for a natural-language query.

        Each result is a dict with ``text``, ``metadata`` and ``distance``.
        """
        top_k = top_k or settings.retrieval_top_k
        where = {"field": field} if field else None
        try:
            results = self._collection.query(
                query_texts=[question],
                n_results=top_k,
                where=where,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Vector query failed: %s", exc)
            return []

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        return [
            {"text": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(documents, metadatas, distances)
        ]

    def count(self) -> int:
        try:
            return self._collection.count()
        except Exception:  # noqa: BLE001
            return 0
