"""High-level orchestration tying every component together.

``ResearchAssistant`` is the single entry point used by both the CLI
(``main.py``) and the Streamlit UI, so behaviour stays consistent across
interfaces.

Typical flow::

    ra = ResearchAssistant()
    ra.search_and_store(fields=["Paleogenetics"], max_results=5)
    ra.process_papers(limit=5)          # download, extract, summarise, embed
    print(ra.ask("What discusses Denisovan DNA?").answer)
    ra.generate_report()
    ra.build_graph()
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from agents.biorxiv_agent import BioAnswer, BioPaper, BioRxivAssistant
from agents.live_agent import LiveAnswer, LiveScienceAssistant
from agents.question_agent import Answer, QuestionAgent
from agents.report_agent import ReportAgent
from agents.search_agent import SearchAgent
from agents.summary_agent import SummaryAgent
from config.settings import settings
from core.llm import make_llm_client
from core.models import Paper
from database.paper_database import PaperDatabase
from database.vector_store import VectorStore
from knowledge_graph.graph_builder import KnowledgeGraphBuilder
from pdf_processing import pdf_loader
from pdf_processing.text_extractor import extract_text, save_text
from utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessResult:
    """Outcome of processing a single paper through the pipeline."""

    arxiv_id: str
    downloaded: bool = False
    extracted: bool = False
    summarized: bool = False
    embedded: bool = False
    error: str | None = None


@dataclass
class ProgressEvent:
    """A live progress update emitted while processing a batch of papers.

    ``stage`` is one of: ``start``, ``downloading``, ``extracting``,
    ``summarizing``, ``embedding``, ``done``, ``error``. ``done``/``total`` count
    *completed* papers, so a UI can render ``done/total`` and a per-paper stage.
    """

    done: int
    total: int
    arxiv_id: str
    title: str
    stage: str
    elapsed: float = 0.0
    error: str | None = None


# A progress callback receives one ProgressEvent per stage transition. It must
# be cheap and non-blocking (the CLI prints a line; Streamlit updates a bar).
ProgressCallback = Callable[[ProgressEvent], None]


class ResearchAssistant:
    """Facade coordinating search, storage, analysis, RAG, reports, and graphs."""

    def __init__(self) -> None:
        settings.ensure_dirs()
        self.settings = settings
        self.db = PaperDatabase()
        self.vector_store = VectorStore()
        self.llm = make_llm_client()
        self.search_agent = SearchAgent()
        self.summary_agent = SummaryAgent(client=self.llm)
        self.question_agent = QuestionAgent(
            vector_store=self.vector_store, client=self.llm
        )
        self.report_agent = ReportAgent(database=self.db, client=self.llm)
        self.live_agent = LiveScienceAssistant(
            search_agent=self.search_agent, client=self.llm
        )
        self.biorxiv_agent = BioRxivAssistant(client=self.llm)
        self.graph_builder = KnowledgeGraphBuilder(database=self.db)

    # ------------------------------------------------------------- diagnostics
    def health_check(self) -> dict[str, object]:
        """Report connectivity and library status for startup diagnostics."""
        llm_up = self.llm.is_available()
        return {
            "provider": self.settings.llm_provider,
            "model": self.llm.model,
            "llm_reachable": llm_up,
            "model_available": self.llm.model_available() if llm_up else False,
            "papers_in_db": self.db.count(),
            "chunks_in_vector_store": self.vector_store.count(),
        }

    # ----------------------------------------------------------------- search
    def search_and_store(
        self,
        fields: list[str] | None = None,
        query: str | None = None,
        max_results: int | None = None,
    ) -> list[Paper]:
        """Search arXiv (by field or free text) and store new metadata.

        Returns the list of newly stored papers (already-known papers are
        refreshed but not returned as new).
        """
        if query:
            found = self.search_agent.search_query(
                query, max_results=max_results or 20
            )
        else:
            found = self.search_agent.search_fields(
                fields=fields, max_results=max_results
            )

        new_papers: list[Paper] = []
        for paper in found:
            is_new = not self.db.exists(paper.arxiv_id)
            self.db.upsert_paper(paper)
            if is_new:
                new_papers.append(paper)
        logger.info(
            "Search stored %d papers (%d new)", len(found), len(new_papers)
        )
        return new_papers

    # ---------------------------------------------------------------- process
    def process_papers(
        self,
        limit: int | None = None,
        field: str | None = None,
        arxiv_ids: list[str] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> list[ProcessResult]:
        """Download, extract, summarise, and embed unprocessed papers.

        Synchronous wrapper around the async pipeline so it is easy to call from
        scripts and Streamlit. Pass ``progress_callback`` to receive live
        :class:`ProgressEvent` updates as each paper moves through the stages.
        """
        return asyncio.run(
            self.aprocess_papers(
                limit=limit,
                field=field,
                arxiv_ids=arxiv_ids,
                progress_callback=progress_callback,
            )
        )

    async def aprocess_papers(
        self,
        limit: int | None = None,
        field: str | None = None,
        arxiv_ids: list[str] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> list[ProcessResult]:
        """Async pipeline: process unsummarised papers with bounded concurrency.

        If ``arxiv_ids`` is given, only those papers (that aren't summarised yet)
        are processed — used when adding specific papers from an assistant.
        """
        if arxiv_ids is not None:
            wanted = set(arxiv_ids)
            pending = [
                p for p in self.db.list_papers(summarized=False) if p.arxiv_id in wanted
            ]
        else:
            pending = self.db.list_papers(field=field, summarized=False, limit=limit)
        if not pending:
            logger.info("No unprocessed papers found.")
            return []

        total = len(pending)
        logger.info("Processing %d papers...", total)
        semaphore = asyncio.Semaphore(self.settings.max_concurrent_jobs)
        completed = 0  # closure counter shared across tasks (single event loop)

        def emit(paper: Paper, stage: str, elapsed: float = 0.0, error: str | None = None) -> None:
            if progress_callback is None:
                return
            try:
                progress_callback(
                    ProgressEvent(
                        done=completed,
                        total=total,
                        arxiv_id=paper.arxiv_id,
                        title=paper.title,
                        stage=stage,
                        elapsed=elapsed,
                        error=error,
                    )
                )
            except Exception:  # noqa: BLE001 - a bad UI callback must not break processing
                logger.debug("progress_callback raised", exc_info=True)

        async def _run(paper: Paper) -> ProcessResult:
            nonlocal completed
            result = await self._process_one(paper, semaphore, emit)
            completed += 1
            stage = "error" if (result.error and not result.summarized) else "done"
            emit(paper, stage, error=result.error)
            return result

        return await asyncio.gather(*(_run(p) for p in pending))

    async def _process_one(
        self,
        paper: Paper,
        semaphore: asyncio.Semaphore,
        emit: Callable[..., None] = lambda *a, **k: None,
    ) -> ProcessResult:
        """Run the full per-paper pipeline: download -> extract -> summarise -> embed."""
        result = ProcessResult(arxiv_id=paper.arxiv_id)
        started = time.time()
        async with semaphore:
            try:
                # 1. Download PDF.
                emit(paper, "downloading", time.time() - started)
                pdf_path = await pdf_loader.download_pdf(paper)
                if pdf_path is None:
                    result.error = "download_failed"
                    return result
                result.downloaded = True
                self.db.set_paths(paper.arxiv_id, local_pdf_path=str(pdf_path))

                # 2. Extract text — reuse a previous extraction if we have one,
                #    so retrying a paper (e.g. one whose summary failed) doesn't
                #    needlessly re-parse the PDF.
                emit(paper, "extracting", time.time() - started)
                text = ""
                if paper.text_path and Path(paper.text_path).exists():
                    text = await asyncio.to_thread(
                        Path(paper.text_path).read_text, encoding="utf-8"
                    )
                    result.extracted = True
                if not text:
                    text = await asyncio.to_thread(extract_text, pdf_path)
                    if text:
                        text_path = self.settings.papers_dir / f"{_safe(paper.arxiv_id)}.txt"
                        await asyncio.to_thread(save_text, text, text_path)
                        self.db.set_paths(paper.arxiv_id, text_path=str(text_path))
                        result.extracted = True
                if not text:
                    text = paper.abstract  # degrade gracefully to the abstract

                # 3. Summarise (network-bound -> async). A blank analysis means
                #    the model call failed; leave the paper un-summarised so the
                #    next `process` run retries it.
                emit(paper, "summarizing", time.time() - started)
                analysis = await self.summary_agent.aanalyze(paper, text)
                if analysis.is_empty:
                    result.error = "empty_summary"
                    logger.warning("Empty analysis for %s; will retry later.", paper.arxiv_id)
                else:
                    self.db.save_analysis(paper.arxiv_id, analysis)
                    result.summarized = True

                # 4. Embed into the vector store, unless it's already embedded
                #    (retries of an embedded-but-unsummarised paper skip this).
                if paper.embedded:
                    result.embedded = True
                else:
                    emit(paper, "embedding", time.time() - started)
                    added = await asyncio.to_thread(
                        self.vector_store.add_paper, paper, text
                    )
                    if added:
                        self.db.mark_embedded(paper.arxiv_id, True)
                        result.embedded = True

                logger.info("Processed %s in %.0fs", paper.arxiv_id, time.time() - started)
            except Exception as exc:  # noqa: BLE001 - never kill the whole batch
                logger.exception("Processing failed for %s", paper.arxiv_id)
                result.error = str(exc)
        return result

    # -------------------------------------------------------------------- ask
    def ask(
        self, question: str, top_k: int | None = None, field: str | None = None
    ) -> Answer:
        """Answer a natural-language question via RAG over the library."""
        return self.question_agent.ask(question, top_k=top_k, field=field)

    def live_assistant(self, question: str, max_results: int = 6) -> LiveAnswer:
        """Answer a question with a fresh live arXiv search (not the library)."""
        return self.live_agent.answer(question, max_results=max_results)

    def biorxiv_assistant(self, question: str, max_results: int = 6) -> BioAnswer:
        """Answer a question with a fresh live bioRxiv preprint search."""
        return self.biorxiv_agent.answer(question, max_results=max_results)

    def save_papers(self, papers: list[Paper]) -> int:
        """Store a list of papers' metadata in the library; return new count."""
        new = 0
        for paper in papers:
            if not self.db.exists(paper.arxiv_id):
                new += 1
            self.db.upsert_paper(paper)
        logger.info("Saved %d papers to library (%d new)", len(papers), new)
        return new

    def add_and_process(
        self,
        papers: list[Paper],
        progress_callback: ProgressCallback | None = None,
    ) -> list[ProcessResult]:
        """Save arXiv papers and fully process them (download, summarise, embed).

        This is the primary way the library is built: pick papers from the Live
        Science Assistant results and add them, after which they appear in Browse
        Library and are searchable in Ask Questions.
        """
        self.save_papers(papers)
        ids = [p.arxiv_id for p in papers]
        return self.process_papers(arxiv_ids=ids, progress_callback=progress_callback)

    def add_biorxiv_papers(self, bio_papers: list[BioPaper]) -> int:
        """Add bioRxiv preprints to the library (summarise + embed from abstract).

        bioRxiv papers have no arXiv PDF pipeline, so we summarise and embed from
        the abstract — enough to browse them and answer questions via RAG.
        """
        added = 0
        for bp in bio_papers:
            if not bp.doi or self.db.exists(bp.doi):
                continue
            paper = Paper(
                arxiv_id=bp.doi,
                title=bp.title,
                authors=bp.authors,
                abstract=bp.abstract,
                primary_category="bioRxiv",
                field="bioRxiv",
                published=_parse_date(bp.published),
                entry_url=bp.url,
            )
            self.db.upsert_paper(paper)
            if bp.abstract:
                analysis = self.summary_agent.analyze(paper, bp.abstract)
                if not analysis.is_empty:
                    self.db.save_analysis(bp.doi, analysis)
                if self.vector_store.add_paper(paper, bp.abstract):
                    self.db.mark_embedded(bp.doi, True)
            added += 1
        logger.info("Added %d bioRxiv papers to library", added)
        return added

    def remove_papers_by_field(self, field: str) -> int:
        """Delete every paper in a field from the DB, vector store, and disk."""
        ids = self.db.ids_by_field(field)
        for arxiv_id in ids:
            paper = self.db.get_paper(arxiv_id)
            self.vector_store.remove_paper(arxiv_id)
            self.db.delete_paper(arxiv_id)
            if paper:
                for path_str in (paper.local_pdf_path, paper.text_path):
                    if path_str:
                        try:
                            Path(path_str).unlink(missing_ok=True)
                        except OSError:
                            pass
        logger.info("Removed %d papers in field '%s'", len(ids), field)
        return len(ids)

    # ----------------------------------------------------------------- report
    def generate_report(self, days: int | None = None) -> str:
        """Generate and persist a weekly markdown report."""
        return self.report_agent.generate(days=days)

    # ------------------------------------------------------------------ graph
    def build_graph(
        self, field: str | None = None, export_html: bool = True
    ) -> KnowledgeGraphBuilder:
        """Rebuild the knowledge graph and optionally export visualisations."""
        self.graph_builder.build(field=field)
        self.graph_builder.save_graphml()
        if export_html:
            self.graph_builder.to_pyvis_html()
        return self.graph_builder

    # ---------------------------------------------------------------- one-shot
    def run_full_cycle(
        self,
        fields: list[str] | None = None,
        max_results: int | None = None,
        process_limit: int | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, object]:
        """Search -> process -> graph -> report, returning a summary dict.

        ``process_limit`` caps how many papers are summarised in this cycle so a
        big search can't queue dozens of slow CPU jobs at once; it defaults to
        ``settings.process_batch_size``. Pass ``0`` to process everything.
        """
        if process_limit is None:
            process_limit = self.settings.process_batch_size
        limit = None if process_limit == 0 else process_limit

        new_papers = self.search_and_store(fields=fields, max_results=max_results)
        pending = self.db.list_papers(summarized=False)
        results = self.process_papers(
            limit=limit, progress_callback=progress_callback
        )
        self.build_graph()
        report = self.generate_report()
        return {
            "new_papers": len(new_papers),
            "pending_total": len(pending),
            "processed": len(results),
            "succeeded": sum(1 for r in results if r.summarized),
            "remaining_unprocessed": max(len(pending) - len(results), 0),
            "report_chars": len(report),
        }


def _safe(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_")


def _parse_date(value: str) -> datetime:
    """Parse a bioRxiv date string ('YYYY-MM-DD'); default to now on failure."""
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
    return datetime.now(timezone.utc)
