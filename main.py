"""Local AI Scientist — command-line entry point.

Examples
--------
Search recent papers in two fields and store metadata::

    python main.py search --fields Paleogenetics "Artificial Intelligence" --max 5

Download, extract, summarise, and embed everything not yet processed::

    python main.py process

Do the whole cycle (search -> process -> graph -> report)::

    python main.py run --fields Genetics --max 5

Ask a question against your library::

    python main.py ask "What new papers discuss Denisovan DNA?"

Generate a weekly report / rebuild the graph / launch the UI::

    python main.py report
    python main.py graph
    python main.py ui
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from config.settings import ARXIV_FIELDS, settings
from utils.logging_config import configure_logging, get_logger

configure_logging(settings.logs_dir, level=settings.log_level)
logger = get_logger(__name__)


def _print_health(assistant) -> bool:
    """Print a health summary; return False if the LLM backend is unavailable."""
    health = assistant.health_check()
    provider = str(health["provider"]).lower()
    print("System status:")
    for key, value in health.items():
        print(f"  - {key}: {value}")
    if not health["llm_reachable"]:
        if provider == "ollama":
            print(
                "\n[!] Ollama is not reachable. Start it with `ollama serve` and pull "
                f"the model: `ollama pull {settings.ollama_model}`."
            )
        else:
            print(
                f"\n[!] Provider '{provider}' not reachable. Check OPENAI_API_KEY and "
                f"OPENAI_BASE_URL ({settings.openai_base_url}) in your .env."
            )
        return False
    if not health["model_available"]:
        print(f"\n[!] Model '{health['model']}' not available on provider '{provider}'.")
    return True


def cmd_search(args: argparse.Namespace) -> None:
    from core.pipeline import ResearchAssistant

    assistant = ResearchAssistant()
    new_papers = assistant.search_and_store(
        fields=args.fields, query=args.query, max_results=args.max
    )
    print(f"Stored search results. {len(new_papers)} new papers added.")
    for paper in new_papers[:20]:
        print(f"  [{paper.arxiv_id}] {paper.title}")


def _cli_progress(event) -> None:
    """Render a live, single-line progress update for the terminal (ASCII-only)."""
    if event.stage in ("done", "error"):
        mark = "OK" if event.stage == "done" else "--"
        note = f"({event.error})" if event.error else "summarised"
        # Overwrite the in-progress line, then commit it with a newline.
        sys.stdout.write(
            f"\r  [{event.done}/{event.total}] {mark} {event.arxiv_id}  {note}".ljust(72) + "\n"
        )
    else:
        label = f"{event.stage}...".ljust(14)
        sys.stdout.write(
            f"\r  [{event.done}/{event.total}] {event.arxiv_id}  {label} {event.elapsed:>4.0f}s".ljust(72)
        )
    sys.stdout.flush()


def cmd_process(args: argparse.Namespace) -> None:
    from core.pipeline import ResearchAssistant

    assistant = ResearchAssistant()
    if not _print_health(assistant):
        sys.exit(1)
    print("Processing (live progress below; first paper includes model load)...\n")
    results = assistant.process_papers(
        limit=args.limit, field=args.field, progress_callback=_cli_progress
    )
    ok = sum(1 for r in results if r.summarized)
    print(f"\nDone: {len(results)} papers processed, {ok} summarised successfully.")


def cmd_run(args: argparse.Namespace) -> None:
    from core.pipeline import ResearchAssistant

    assistant = ResearchAssistant()
    if not _print_health(assistant):
        sys.exit(1)
    print("Running full cycle (search -> process -> graph -> report)...\n")
    summary = assistant.run_full_cycle(
        fields=args.fields,
        max_results=args.max,
        process_limit=args.process_limit,
        progress_callback=_cli_progress,
    )
    print("\nFull cycle complete:")
    for key, value in summary.items():
        print(f"  - {key}: {value}")
    remaining = summary.get("remaining_unprocessed", 0)
    if remaining:
        print(
            f"\n{remaining} papers still un-summarised. Run `python main.py process "
            f"--limit {min(remaining, 10)}` again to continue in batches."
        )


def cmd_ask(args: argparse.Namespace) -> None:
    from core.pipeline import ResearchAssistant

    assistant = ResearchAssistant()
    answer = assistant.ask(args.question, top_k=args.top_k, field=args.field)
    print(f"\nQ: {answer.question}\n")
    print(answer.answer)
    if answer.cited_papers():
        print("\nSources:")
        for aid in answer.cited_papers():
            print(f"  - {aid}")


def cmd_live(args: argparse.Namespace) -> None:
    from core.pipeline import ResearchAssistant

    assistant = ResearchAssistant()
    result = assistant.live_assistant(args.question, max_results=args.max)
    print(f"\nQ: {result.question}")
    print(f"arXiv query: {result.arxiv_query}\n")
    print(result.answer)
    if result.papers:
        print("\narXiv results:")
        for paper in result.papers:
            print(f"  [{paper.arxiv_id}] {paper.title}")


def cmd_biorxiv(args: argparse.Namespace) -> None:
    from core.pipeline import ResearchAssistant

    assistant = ResearchAssistant()
    result = assistant.biorxiv_assistant(args.question, max_results=args.max)
    print(f"\nQ: {result.question}")
    print(f"bioRxiv query: {result.query}\n")
    print(result.answer)
    if result.papers:
        print("\nbioRxiv preprints:")
        for paper in result.papers:
            print(f"  [{paper.doi}] {paper.title}")


def cmd_report(args: argparse.Namespace) -> None:
    from core.pipeline import ResearchAssistant

    assistant = ResearchAssistant()
    report = assistant.generate_report(days=args.days)
    print(report)


def cmd_graph(args: argparse.Namespace) -> None:
    from core.pipeline import ResearchAssistant

    assistant = ResearchAssistant()
    builder = assistant.build_graph(field=args.field)
    print("Knowledge graph rebuilt:")
    for key, value in builder.stats().items():
        print(f"  - {key}: {value}")
    print("\nTop topics:")
    for label, degree in builder.central_topics():
        print(f"  - {label} ({degree})")


def cmd_status(args: argparse.Namespace) -> None:
    from core.pipeline import ResearchAssistant

    _print_health(ResearchAssistant())


def cmd_ui(args: argparse.Namespace) -> None:
    """Launch the Streamlit interface."""
    app_path = Path(__file__).parent / "ui" / "streamlit_app.py"
    print(f"Launching Streamlit UI: {app_path}")
    subprocess.run(["streamlit", "run", str(app_path)], check=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local-ai-scientist",
        description="A local AI research assistant powered by Granite via Ollama.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    field_choices = list(ARXIV_FIELDS.keys())

    p_search = sub.add_parser("search", help="Search arXiv and store metadata.")
    p_search.add_argument(
        "--fields", nargs="+", choices=field_choices, help="Research fields to search."
    )
    p_search.add_argument("--query", help="Free-text arXiv query (overrides --fields).")
    p_search.add_argument("--max", type=int, default=None, help="Max results per field.")
    p_search.set_defaults(func=cmd_search)

    p_process = sub.add_parser(
        "process", help="Download, extract, summarise and embed unprocessed papers."
    )
    p_process.add_argument("--limit", type=int, default=None, help="Max papers to process.")
    p_process.add_argument("--field", default=None, help="Restrict to one field.")
    p_process.set_defaults(func=cmd_process)

    p_run = sub.add_parser("run", help="Full cycle: search, process, graph, report.")
    p_run.add_argument("--fields", nargs="+", choices=field_choices)
    p_run.add_argument("--max", type=int, default=None, help="Max search results per field.")
    p_run.add_argument(
        "--process-limit",
        type=int,
        default=None,
        dest="process_limit",
        help="Max papers to summarise this cycle (default from config; 0 = all).",
    )
    p_run.set_defaults(func=cmd_run)

    p_ask = sub.add_parser("ask", help="Ask a question about your library (RAG).")
    p_ask.add_argument("question", help="The question to answer.")
    p_ask.add_argument("--top-k", type=int, default=None, dest="top_k")
    p_ask.add_argument("--field", default=None)
    p_ask.set_defaults(func=cmd_ask)

    p_live = sub.add_parser(
        "live", help="Live arXiv-backed Q&A (searches arXiv fresh per query)."
    )
    p_live.add_argument("question", help="The question to answer.")
    p_live.add_argument("--max", type=int, default=6, help="arXiv results to consider.")
    p_live.set_defaults(func=cmd_live)

    p_bio = sub.add_parser(
        "biorxiv", help="Live bioRxiv preprint Q&A (searches bioRxiv fresh per query)."
    )
    p_bio.add_argument("question", help="The question to answer.")
    p_bio.add_argument("--max", type=int, default=6, help="bioRxiv results to consider.")
    p_bio.set_defaults(func=cmd_biorxiv)

    p_report = sub.add_parser("report", help="Generate a weekly markdown report.")
    p_report.add_argument("--days", type=int, default=None, help="Look-back window.")
    p_report.set_defaults(func=cmd_report)

    p_graph = sub.add_parser("graph", help="Rebuild and export the knowledge graph.")
    p_graph.add_argument("--field", default=None)
    p_graph.set_defaults(func=cmd_graph)

    sub.add_parser("status", help="Show system / connectivity status.").set_defaults(
        func=cmd_status
    )
    sub.add_parser("ui", help="Launch the Streamlit web interface.").set_defaults(
        func=cmd_ui
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
