"""Report agent: synthesise weekly research digests.

Collects papers added in a recent window, feeds their summaries to Granite, and
produces a markdown report covering major discoveries, interesting papers,
cross-disciplinary connections, and emerging trends.
"""

from __future__ import annotations

from datetime import date

from config.settings import settings
from core.llm import LLMClient, make_llm_client
from core.llm_base import LLMError
from database.paper_database import PaperDatabase
from utils.logging_config import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a scientific editor writing a weekly research digest for a curious, "
    "technically literate reader. You synthesise across disciplines, highlight "
    "what is genuinely novel, and never fabricate papers or findings beyond the "
    "provided material."
)

_REPORT_TEMPLATE = """\
Below are papers added to a personal research library in the last {days} days,
with their summaries and findings. Write a markdown weekly report.

Structure the report with these sections (use ## headings):
1. Major Discoveries — the most significant results this week.
2. Interesting Papers — a curated short list with one-line takes (cite arXiv id).
3. Cross-Disciplinary Connections — links between different fields below.
4. Emerging Trends — patterns or recurring themes across the papers.

Be specific and cite arXiv ids in [brackets]. Keep it tight and skimmable.

Papers:
{papers_block}
"""


class ReportAgent:
    """Generate and persist weekly markdown reports."""

    def __init__(
        self,
        database: PaperDatabase | None = None,
        client: LLMClient | None = None,
    ) -> None:
        self.db = database or PaperDatabase()
        self.client = client or make_llm_client()

    def _papers_block(self, days: int) -> tuple[str, int]:
        papers = self.db.papers_since(days)
        if not papers:
            return "", 0

        # Only include papers that actually have an analysis; abstracts for the
        # rest would bloat the prompt and blow provider token-per-minute limits.
        # Cap the count and trim each entry to keep the single report call small.
        analysed = [(p, self.db.get_analysis(p.arxiv_id)) for p in papers]
        analysed = [(p, a) for p, a in analysed if a is not None]
        analysed = analysed[: settings.report_max_papers]
        if not analysed:
            return "", 0

        blocks: list[str] = []
        for paper, analysis in analysed:
            summary = (analysis.summary or paper.abstract)[:350]
            findings = "; ".join(analysis.key_findings[:2]) if analysis.key_findings else ""
            blocks.append(
                f"- [{paper.arxiv_id}] ({paper.field}) {paper.title}\n"
                f"    Summary: {summary}\n"
                + (f"    Findings: {findings}\n" if findings else "")
            )
        return "\n".join(blocks), len(analysed)

    def generate(self, days: int | None = None, save: bool = True) -> str:
        """Generate a weekly report covering the last ``days`` days.

        Returns the markdown text (also written to ``data/reports`` if ``save``).
        """
        days = days or settings.search_lookback_days
        papers_block, count = self._papers_block(days)
        header = f"# Weekly Research Report — {date.today().isoformat()}\n\n"

        if count == 0:
            body = (
                "_No papers were added in the selected window. Run a search and "
                "process some papers, then regenerate this report._\n"
            )
            report = header + body
        else:
            prompt = _REPORT_TEMPLATE.format(days=days, papers_block=papers_block)
            try:
                body = self.client.generate(prompt, system=_SYSTEM_PROMPT)
            except LLMError as exc:
                logger.error("Report generation failed: %s", exc)
                body = f"_Report generation failed: {exc}_\n"
            report = (
                header
                + f"*Covering {count} papers from the last {days} days.*\n\n"
                + body
                + "\n"
            )

        if save:
            self._save(report)
        return report

    def _save(self, report: str) -> None:
        filename = f"report_{date.today().isoformat()}.md"
        path = settings.reports_dir / filename
        path.write_text(report, encoding="utf-8")
        logger.info("Saved weekly report to %s", path)

    @staticmethod
    def list_reports() -> list[str]:
        """Return saved report filenames, newest first."""
        reports = sorted(settings.reports_dir.glob("report_*.md"), reverse=True)
        return [p.name for p in reports]
