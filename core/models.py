"""Shared data models used across the application.

These pydantic models are the common currency passed between the search agent,
PDF processing, the databases, and the analysis agents.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class Paper(BaseModel):
    """Metadata for a single arXiv paper."""

    arxiv_id: str = Field(..., description="arXiv identifier, e.g. '2401.01234v1'.")
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    primary_category: str = ""
    categories: list[str] = Field(default_factory=list)
    field: str = Field(default="", description="Human-friendly field that surfaced it.")
    published: datetime
    updated: datetime | None = None
    pdf_url: str = ""
    entry_url: str = ""

    # Local processing state.
    local_pdf_path: str | None = None
    text_path: str | None = None
    summarized: bool = False
    embedded: bool = False
    added_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def short_id(self) -> str:
        """arXiv id without a version suffix (``2401.01234v2`` -> ``2401.01234``)."""
        return self.arxiv_id.split("v")[0]

    def citation(self) -> str:
        """Compact human-readable citation string."""
        first_author = self.authors[0] if self.authors else "Unknown"
        suffix = " et al." if len(self.authors) > 1 else ""
        year = self.published.year if self.published else ""
        return f"{first_author}{suffix} ({year}) — {self.title}"


class PaperAnalysis(BaseModel):
    """Structured analysis produced by the summary agent via Granite."""

    summary: str = ""
    key_findings: list[str] = Field(default_factory=list)
    equations: list[str] = Field(default_factory=list)
    assumptions_limitations: list[str] = Field(default_factory=list)
    simplified_explanation: str = ""
    related_topics: list[str] = Field(default_factory=list)
    future_work: list[str] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """True if the model returned nothing usable (e.g. a failed call).

        Used by the pipeline to avoid marking a paper as summarised when the
        analysis is blank, so it gets retried on the next processing run.
        """
        return not (
            self.summary
            or self.key_findings
            or self.equations
            or self.assumptions_limitations
            or self.simplified_explanation
        )

    def to_markdown(self, paper: "Paper | None" = None) -> str:
        """Render the analysis as a readable markdown block."""
        lines: list[str] = []
        if paper is not None:
            lines.append(f"# {paper.title}\n")
            lines.append(f"*{', '.join(paper.authors)}*  ")
            lines.append(f"`{paper.arxiv_id}` — {paper.primary_category}\n")

        def section(title: str, items: list[str]) -> None:
            if items:
                lines.append(f"## {title}")
                lines.extend(f"- {item}" for item in items)
                lines.append("")

        if self.summary:
            lines.append("## Summary")
            lines.append(self.summary + "\n")
        section("Key Findings", self.key_findings)
        section("Important Equations", self.equations)
        section("Assumptions & Limitations", self.assumptions_limitations)
        if self.simplified_explanation:
            lines.append("## Explain It Simply")
            lines.append(self.simplified_explanation + "\n")
        section("Suggested Future Work", self.future_work)
        section("Related Topics", self.related_topics)
        return "\n".join(lines).strip() + "\n"
