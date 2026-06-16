"""Extract and clean text from PDF files using PyMuPDF (fitz).

Also provides a simple character-based chunker used when embedding papers into
the vector store.
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF

from config.settings import settings
from utils.logging_config import get_logger

logger = get_logger(__name__)


def extract_text(pdf_path: str | Path, max_pages: int | None = None) -> str:
    """Extract plain text from a PDF.

    Args:
        pdf_path: Path to the PDF file.
        max_pages: Optionally cap the number of pages read (useful for very
            long papers / appendices).

    Returns:
        Cleaned, whitespace-normalised text. Returns an empty string if the
        file cannot be opened.
    """
    path = Path(pdf_path)
    if not path.exists():
        logger.error("PDF not found: %s", path)
        return ""

    try:
        parts: list[str] = []
        with fitz.open(path) as doc:
            page_count = doc.page_count if max_pages is None else min(doc.page_count, max_pages)
            for page_index in range(page_count):
                page = doc.load_page(page_index)
                parts.append(page.get_text("text"))
        text = "\n".join(parts)
        cleaned = _clean_text(text)
        logger.info("Extracted %d chars from %s", len(cleaned), path.name)
        return cleaned
    except Exception as exc:  # noqa: BLE001 - corrupt PDFs are common in the wild
        logger.error("Failed to extract text from %s: %s", path, exc)
        return ""


def save_text(text: str, destination: str | Path) -> Path:
    """Persist extracted text to disk and return the path written."""
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    return dest


def _clean_text(text: str) -> str:
    """Collapse excessive whitespace and strip control characters."""
    # Normalise newlines and remove form-feeds.
    text = text.replace("\x0c", "\n")
    # Join hyphenated line breaks: "evolu-\ntion" -> "evolution".
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Collapse runs of blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of spaces/tabs.
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """Split text into overlapping chunks for embedding.

    Tries to break on paragraph/sentence boundaries near the target size so
    chunks stay semantically coherent.
    """
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            # Prefer a paragraph break, then a sentence break, within the tail.
            window = text[start:end]
            split_at = max(window.rfind("\n\n"), window.rfind(". "))
            if split_at > chunk_size // 2:
                end = start + split_at + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return chunks
