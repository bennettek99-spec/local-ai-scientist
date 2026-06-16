"""SQLite-backed metadata store for papers and their analyses.

This is the system of record for *what* papers we have and their processing
state. The vector store (ChromaDB) holds the embeddings; the knowledge graph is
rebuilt from this database on demand.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from config.settings import settings
from core.models import Paper, PaperAnalysis
from utils.logging_config import get_logger

logger = get_logger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id          TEXT PRIMARY KEY,
    title             TEXT NOT NULL,
    authors           TEXT NOT NULL,            -- JSON list
    abstract          TEXT,
    primary_category  TEXT,
    categories        TEXT,                     -- JSON list
    field             TEXT,
    published         TEXT,                     -- ISO 8601
    updated           TEXT,
    pdf_url           TEXT,
    entry_url         TEXT,
    local_pdf_path    TEXT,
    text_path         TEXT,
    analysis          TEXT,                     -- JSON PaperAnalysis or NULL
    summarized        INTEGER DEFAULT 0,
    embedded          INTEGER DEFAULT 0,
    added_at          TEXT
);
CREATE INDEX IF NOT EXISTS idx_papers_field ON papers(field);
CREATE INDEX IF NOT EXISTS idx_papers_published ON papers(published);
"""


class PaperDatabase:
    """Lightweight data-access layer over a single SQLite file."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or settings.db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
        logger.debug("Initialised paper database at %s", self.db_path)

    # ------------------------------------------------------------- write paths
    def upsert_paper(self, paper: Paper) -> None:
        """Insert a paper or update its mutable metadata fields."""
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT arxiv_id FROM papers WHERE arxiv_id = ?", (paper.arxiv_id,)
            ).fetchone()
            if existing:
                conn.execute(
                    """
                    UPDATE papers SET
                        title = ?, authors = ?, abstract = ?, primary_category = ?,
                        categories = ?, field = ?, published = ?, updated = ?,
                        pdf_url = ?, entry_url = ?
                    WHERE arxiv_id = ?
                    """,
                    (
                        paper.title,
                        json.dumps(paper.authors),
                        paper.abstract,
                        paper.primary_category,
                        json.dumps(paper.categories),
                        paper.field,
                        _iso(paper.published),
                        _iso(paper.updated),
                        paper.pdf_url,
                        paper.entry_url,
                        paper.arxiv_id,
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO papers (
                        arxiv_id, title, authors, abstract, primary_category,
                        categories, field, published, updated, pdf_url, entry_url,
                        local_pdf_path, text_path, analysis, summarized, embedded,
                        added_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        paper.arxiv_id,
                        paper.title,
                        json.dumps(paper.authors),
                        paper.abstract,
                        paper.primary_category,
                        json.dumps(paper.categories),
                        paper.field,
                        _iso(paper.published),
                        _iso(paper.updated),
                        paper.pdf_url,
                        paper.entry_url,
                        paper.local_pdf_path,
                        paper.text_path,
                        None,
                        int(paper.summarized),
                        int(paper.embedded),
                        _iso(paper.added_at),
                    ),
                )

    def set_paths(
        self,
        arxiv_id: str,
        local_pdf_path: str | None = None,
        text_path: str | None = None,
    ) -> None:
        """Record the local PDF / extracted-text paths for a paper."""
        sets, params = [], []
        if local_pdf_path is not None:
            sets.append("local_pdf_path = ?")
            params.append(local_pdf_path)
        if text_path is not None:
            sets.append("text_path = ?")
            params.append(text_path)
        if not sets:
            return
        params.append(arxiv_id)
        with self._connect() as conn:
            conn.execute(f"UPDATE papers SET {', '.join(sets)} WHERE arxiv_id = ?", params)

    def save_analysis(self, arxiv_id: str, analysis: PaperAnalysis) -> None:
        """Persist a paper's structured analysis and mark it summarized."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE papers SET analysis = ?, summarized = 1 WHERE arxiv_id = ?",
                (analysis.model_dump_json(), arxiv_id),
            )

    def mark_embedded(self, arxiv_id: str, embedded: bool = True) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE papers SET embedded = ? WHERE arxiv_id = ?",
                (int(embedded), arxiv_id),
            )

    # -------------------------------------------------------------- read paths
    def exists(self, arxiv_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM papers WHERE arxiv_id = ?", (arxiv_id,)
            ).fetchone()
        return row is not None

    def get_paper(self, arxiv_id: str) -> Paper | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)
            ).fetchone()
        return _row_to_paper(row) if row else None

    def get_analysis(self, arxiv_id: str) -> PaperAnalysis | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT analysis FROM papers WHERE arxiv_id = ?", (arxiv_id,)
            ).fetchone()
        if row and row["analysis"]:
            return PaperAnalysis.model_validate_json(row["analysis"])
        return None

    def list_papers(
        self,
        field: str | None = None,
        summarized: bool | None = None,
        embedded: bool | None = None,
        limit: int | None = None,
    ) -> list[Paper]:
        """List papers, optionally filtered by field / processing state."""
        clauses, params = [], []
        if field:
            clauses.append("field = ?")
            params.append(field)
        if summarized is not None:
            clauses.append("summarized = ?")
            params.append(int(summarized))
        if embedded is not None:
            clauses.append("embedded = ?")
            params.append(int(embedded))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM papers {where} ORDER BY published DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_paper(r) for r in rows]

    def papers_since(self, days: int) -> list[Paper]:
        """Papers added to the library within the last ``days`` days."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM papers WHERE added_at >= ? ORDER BY published DESC",
                (cutoff,),
            ).fetchall()
        return [_row_to_paper(r) for r in rows]

    def field_counts(self) -> dict[str, int]:
        """Return a mapping of field -> number of papers."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT field, COUNT(*) AS n FROM papers GROUP BY field"
            ).fetchall()
        return {r["field"] or "Uncategorised": r["n"] for r in rows}

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]

    def ids_by_field(self, field: str) -> list[str]:
        """Return the arXiv ids of all papers in a given field."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT arxiv_id FROM papers WHERE field = ?", (field,)
            ).fetchall()
        return [r["arxiv_id"] for r in rows]

    def delete_paper(self, arxiv_id: str) -> None:
        """Delete a paper's metadata row."""
        with self._connect() as conn:
            conn.execute("DELETE FROM papers WHERE arxiv_id = ?", (arxiv_id,))


# ----------------------------------------------------------------- converters
def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _row_to_paper(row: sqlite3.Row) -> Paper:
    """Build a :class:`Paper` from a database row."""
    published = _parse_dt(row["published"]) or datetime.now(timezone.utc)
    return Paper(
        arxiv_id=row["arxiv_id"],
        title=row["title"],
        authors=json.loads(row["authors"]) if row["authors"] else [],
        abstract=row["abstract"] or "",
        primary_category=row["primary_category"] or "",
        categories=json.loads(row["categories"]) if row["categories"] else [],
        field=row["field"] or "",
        published=published,
        updated=_parse_dt(row["updated"]),
        pdf_url=row["pdf_url"] or "",
        entry_url=row["entry_url"] or "",
        local_pdf_path=row["local_pdf_path"],
        text_path=row["text_path"],
        summarized=bool(row["summarized"]),
        embedded=bool(row["embedded"]),
        added_at=_parse_dt(row["added_at"]) or datetime.now(timezone.utc),
    )
