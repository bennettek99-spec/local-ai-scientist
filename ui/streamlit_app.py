"""Streamlit interface for Local AI Scientist.

Run with::

    streamlit run ui/streamlit_app.py

or::

    python main.py ui

Pages: Search Papers, Browse Library, Ask Questions, Weekly Reports,
Knowledge Graph.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# set_page_config must be the FIRST Streamlit call — do it before touching
# st.secrets so a secrets read can't trip the "first command" rule.
st.set_page_config(page_title="Local AI Scientist", page_icon="🔬", layout="wide")

# Make the project root importable when Streamlit runs this file directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# When deployed (e.g. Streamlit Community Cloud) there is no .env file; config
# comes from the Streamlit "secrets" manager instead. Copy those secrets into
# the environment BEFORE importing settings so they're picked up. No-op locally.
# (Overwrite, so secrets always win.) SECRETS_STATUS is shown in the sidebar so
# a deployment can self-report whether secrets actually loaded.
SECRETS_STATUS = "no secrets (local .env)"
try:
    _items = dict(st.secrets.items())
    for _key, _value in _items.items():
        if isinstance(_value, (str, int, float, bool)):
            os.environ[_key] = str(_value)
    if _items:
        SECRETS_STATUS = f"loaded {len(_items)} secret(s)"
except Exception as _exc:  # noqa: BLE001 - no secrets file locally is fine
    SECRETS_STATUS = f"not loaded ({type(_exc).__name__})"

from config.settings import ARXIV_FIELDS, settings  # noqa: E402
from core.pipeline import ResearchAssistant  # noqa: E402
from utils.logging_config import configure_logging  # noqa: E402

configure_logging(settings.logs_dir, level=settings.log_level)


def check_password() -> bool:
    """Gate the app behind APP_PASSWORD when one is configured.

    Set APP_PASSWORD in Streamlit secrets to protect a public deployment. If it
    is unset (e.g. running locally), the app is open.
    """
    password = os.environ.get("APP_PASSWORD")
    if not password:
        return True
    if st.session_state.get("authenticated"):
        return True

    st.title("🔬 Local AI Scientist")
    st.caption("This deployment is private. Enter the password to continue.")
    entered = st.text_input("Password", type="password")
    if entered:
        if entered == password:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


@st.cache_resource(show_spinner="Initialising assistant…")
def get_assistant() -> ResearchAssistant:
    """Build the assistant once and reuse it across reruns."""
    return ResearchAssistant()


# Colour accents per research field, used for badges/cards.
FIELD_COLORS = {
    "Physics": "#3B82F6",
    "Astrophysics": "#6366F1",
    "Materials Science": "#0EA5E9",
    "Genetics": "#10B981",
    "Paleogenetics": "#14B8A6",
    "Paleoanthropology": "#F59E0B",
    "Artificial Intelligence": "#EC4899",
    "bioRxiv": "#8B5CF6",
    "NTRS": "#E03C31",
}


def _field_color(field: str) -> str:
    return FIELD_COLORS.get(field, "#64748B")


def chip(text: str, color: str = "#64748B", solid: bool = False) -> str:
    """Return HTML for a small rounded pill/badge."""
    if solid:
        style = f"background:{color};color:#fff;"
    else:
        style = f"background:{color}1A;color:{color};border:1px solid {color}33;"
    return (
        f"<span style='display:inline-block;padding:2px 10px;border-radius:999px;"
        f"font-size:0.72rem;font-weight:600;line-height:1.4;{style}'>{text}</span>"
    )


def page_header(icon: str, title: str, subtitle: str = "") -> None:
    """Render a consistent, modern page header."""
    sub = f"<div class='page-sub'>{subtitle}</div>" if subtitle else ""
    st.markdown(
        f"<div class='page-head'><div class='page-icon'>{icon}</div>"
        f"<div><div class='page-title'>{title}</div>{sub}</div></div>",
        unsafe_allow_html=True,
    )


def inject_theme() -> None:
    """Inject the custom stylesheet (fonts, colours, cards, buttons)."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Sora:wght@600;700;800&display=swap');

        :root {
            --primary:#6366F1; --primary2:#8B5CF6; --accent:#0EA5E9;
            --bg:#F5F7FB; --card:#FFFFFF; --border:#E7EAF3;
            --ink:#0F172A; --muted:#64748B;
        }

        .stApp { background:
            radial-gradient(1200px 500px at 100% -10%, #EAE9FE 0%, rgba(234,233,254,0) 55%),
            linear-gradient(180deg, #F7F8FC 0%, #F2F4FA 100%); }
        html, body, [class*="css"], .stMarkdown, p, li, label, input, textarea {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: var(--ink);
        }
        h1,h2,h3,h4 { font-family:'Sora','Inter',sans-serif; letter-spacing:-0.02em; }
        .block-container { padding-top: 2.2rem; max-width: 1120px; }
        #MainMenu, footer, [data-testid="stDecoration"] { visibility: hidden; }

        /* ---- Page header ---- */
        .page-head { display:flex; align-items:center; gap:14px; margin:0 0 1.1rem; }
        .page-icon { font-size:1.6rem; width:50px; height:50px; display:flex;
            align-items:center; justify-content:center; border-radius:14px;
            background:linear-gradient(135deg, var(--primary), var(--primary2));
            box-shadow:0 8px 20px -8px var(--primary); }
        .page-title { font-family:'Sora',sans-serif; font-size:1.7rem; font-weight:700;
            color:var(--ink); line-height:1.1; }
        .page-sub { color:var(--muted); font-size:0.92rem; margin-top:2px; }

        /* ---- Sidebar ---- */
        [data-testid="stSidebar"] {
            background:linear-gradient(185deg,#141B2E 0%, #0C1120 100%);
            border-right:1px solid rgba(255,255,255,.06); }
        [data-testid="stSidebar"] * { color:#C7D0E0; }
        [data-testid="stSidebar"] h1 { color:#fff; font-size:1.25rem; }
        /* nav radio as menu */
        [data-testid="stSidebar"] [role="radiogroup"] label {
            padding:9px 12px; border-radius:10px; margin:2px 0; transition:all .12s ease;
            border:1px solid transparent; }
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background:rgba(255,255,255,.06); }
        [data-testid="stSidebar"] [role="radiogroup"] label p {
            font-weight:600; font-size:0.95rem; }

        /* ---- Buttons ---- */
        .stButton > button {
            border-radius:11px; border:1px solid var(--border); font-weight:600;
            padding:.5rem 1rem; transition:all .15s ease; }
        .stButton > button:hover { border-color:var(--primary); color:var(--primary); }
        .stButton > button[kind="primary"] {
            background:linear-gradient(135deg,var(--primary),var(--primary2));
            color:#fff; border:none; box-shadow:0 10px 22px -10px var(--primary); }
        .stButton > button[kind="primary"]:hover {
            filter:brightness(1.07); transform:translateY(-1px); color:#fff; }

        /* ---- Inputs ---- */
        .stTextInput input, .stNumberInput input, textarea,
        [data-baseweb="select"] > div { border-radius:11px !important; }

        /* ---- Cards (expanders + bordered containers) ---- */
        [data-testid="stExpander"] { border:1px solid var(--border); border-radius:16px;
            background:var(--card); box-shadow:0 1px 3px rgba(16,24,40,.05);
            margin-bottom:.55rem; overflow:hidden; }
        [data-testid="stExpander"] summary:hover { color:var(--primary); }
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius:18px !important; border-color:var(--border) !important;
            box-shadow:0 2px 10px rgba(16,24,40,.05); }

        /* ---- Metrics ---- */
        [data-testid="stMetric"] { background:var(--card); border:1px solid var(--border);
            border-radius:16px; padding:1rem 1.1rem; box-shadow:0 1px 3px rgba(16,24,40,.05); }

        /* ---- Cards used in the library grid ---- */
        .lib-card { background:var(--card); border:1px solid var(--border);
            border-radius:16px; padding:14px 16px; box-shadow:0 1px 3px rgba(16,24,40,.05); }
        .lib-title { font-weight:700; font-size:0.98rem; line-height:1.3; color:var(--ink);
            margin-bottom:6px; }
        .lib-meta { color:var(--muted); font-size:0.8rem; margin-bottom:8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar(assistant: ResearchAssistant) -> str:
    """Render the sidebar navigation and status panel; return the page name."""
    st.sidebar.markdown(
        "<div style='display:flex;align-items:center;gap:10px;margin:.2rem 0 .2rem;'>"
        "<div style='font-size:1.5rem;width:42px;height:42px;display:flex;align-items:center;"
        "justify-content:center;border-radius:12px;"
        "background:linear-gradient(135deg,#6366F1,#8B5CF6);'>🔬</div>"
        "<div><div style='font-family:Sora,sans-serif;font-weight:700;font-size:1.05rem;"
        "color:#fff;line-height:1.1;'>Local AI Scientist</div>"
        "<div style='font-size:.72rem;color:#8B96AE;'>your research, on tap</div></div></div>",
        unsafe_allow_html=True,
    )
    st.sidebar.divider()
    page = st.sidebar.radio(
        "Navigate",
        list(PAGES.keys()),
        label_visibility="collapsed",
    )
    st.sidebar.divider()

    health = assistant.health_check()
    ok = health["llm_reachable"] and health["model_available"]
    dot = "#22C55E" if ok else "#EF4444"
    st.sidebar.markdown(
        f"<div style='display:flex;align-items:center;gap:8px;font-size:.85rem;'>"
        f"<span style='width:9px;height:9px;border-radius:50%;background:{dot};"
        f"box-shadow:0 0 8px {dot};'></span>"
        f"<span>{'LLM ready' if ok else 'LLM unavailable'}</span></div>",
        unsafe_allow_html=True,
    )
    st.sidebar.caption(f"`{health['provider']}` · `{health['model']}`")
    c1, c2 = st.sidebar.columns(2)
    c1.metric("Papers", health["papers_in_db"])
    c2.metric("Chunks", health["chunks_in_vector_store"])
    with st.sidebar.expander("Diagnostics"):
        st.caption(f"Secrets: {SECRETS_STATUS}")
        st.caption(f"API key detected: {'yes' if os.environ.get('OPENAI_API_KEY') else 'no'}")
    return page


# ----------------------------------------------------------------------- pages
def _processing_progress():
    """Build a live Streamlit progress UI for paper processing.

    Returns ``(on_progress, finalize)``: pass ``on_progress`` to
    ``process_papers``/``add_and_process`` and call ``finalize`` when done.
    """
    progress_bar = st.progress(0.0, text="Starting…")
    status = st.empty()            # live, single-line current-stage indicator
    log_area = st.container()      # one persistent line per finished paper

    # Fraction of a paper complete at the start of each stage, so the bar moves
    # smoothly *within* a paper, not just between papers.
    stage_weight = {"downloading": 0.1, "extracting": 0.3, "summarizing": 0.55, "embedding": 0.9}

    def on_progress(ev) -> None:
        if ev.stage in ("done", "error"):
            frac = ev.done / ev.total if ev.total else 1.0
            icon = "✅" if ev.stage == "done" else "⚠️"
            detail = ev.error or "summarised & embedded"
            log_area.write(f"{icon} [{ev.done}/{ev.total}] `{ev.arxiv_id}` — {detail}")
        else:
            frac = (ev.done + stage_weight.get(ev.stage, 0.0)) / (ev.total or 1)
            status.markdown(
                f"⏳ **[{ev.done}/{ev.total}]** `{ev.arxiv_id}` — "
                f"**{ev.stage}…** ({ev.elapsed:.0f}s) — *{ev.title[:60]}*"
            )
        progress_bar.progress(min(frac, 1.0), text=f"{ev.done}/{ev.total} papers done")

    def finalize() -> None:
        progress_bar.progress(1.0, text="Complete")
        status.empty()

    return on_progress, finalize


def page_library(assistant: ResearchAssistant) -> None:
    counts = assistant.db.field_counts()
    total = sum(counts.values())
    page_header("📚", "Library", f"{total} papers in your collection")

    present = [f for f in counts if f != "Uncategorised"]
    field_options = ["All"] + sorted(set(list(ARXIV_FIELDS.keys()) + present))
    col_a, col_b = st.columns([2, 3])
    field = col_a.selectbox("Filter by field", field_options, label_visibility="collapsed")
    query = col_b.text_input(
        "search", placeholder="🔍  Filter by title or author…", label_visibility="collapsed"
    )

    papers = assistant.db.list_papers(field=None if field == "All" else field)
    if query:
        q = query.lower()
        papers = [
            p for p in papers
            if q in p.title.lower() or any(q in a.lower() for a in p.authors)
        ]

    if not papers:
        st.info(
            "No papers match. Use the **🔭 Live arXiv** or **🧬 bioRxiv** tabs to "
            "find papers and add them to your library."
        )
        return

    # Field-count chips
    chips_html = " ".join(
        chip(f"{k} · {v}", _field_color(k)) for k, v in sorted(counts.items(), key=lambda x: -x[1])
    )
    st.markdown(chips_html, unsafe_allow_html=True)
    st.write("")

    # Two-column card grid.
    cols = st.columns(2)
    for i, paper in enumerate(papers):
        with cols[i % 2]:
            with st.container(border=True):
                status = (
                    chip("✓ summarised", "#22C55E") if paper.summarized
                    else chip("⏳ pending", "#F59E0B")
                )
                badge = chip(paper.field or "Uncategorised", _field_color(paper.field))
                authors = ", ".join(paper.authors[:3]) + (" et al." if len(paper.authors) > 3 else "")
                year = paper.published.year if paper.published else ""
                st.markdown(
                    f"<div class='lib-title'>{paper.title}</div>"
                    f"<div class='lib-meta'>{authors} · {year}</div>"
                    f"{badge}&nbsp;{status}",
                    unsafe_allow_html=True,
                )
                with st.expander("Details"):
                    if paper.entry_url:
                        st.markdown(f"[🔗 View source]({paper.entry_url})")
                    analysis = assistant.db.get_analysis(paper.arxiv_id)
                    if analysis:
                        st.markdown(analysis.to_markdown())
                    else:
                        st.write(paper.abstract)
                        st.caption("Not yet summarised.")


def page_questions(assistant: ResearchAssistant) -> None:
    page_header("💬", "Ask Your Library", "Retrieval-augmented answers from your saved papers")

    field = st.selectbox(
        "Limit to a field (optional)", ["All"] + list(ARXIV_FIELDS.keys())
    )
    examples = [
        "What new papers discuss Denisovan DNA?",
        "Show recent work on high-entropy alloys.",
        "Find connections between paleogenetics and machine learning.",
    ]
    st.caption("Try: " + " · ".join(f"_{e}_" for e in examples))

    # A form so pressing Enter (not just clicking) submits the question.
    with st.form("ask_form"):
        question = st.text_input("Your question", "")
        submitted = st.form_submit_button("Ask", type="primary")

    if submitted and question.strip():
        with st.spinner("Searching your library and reasoning over the papers…"):
            try:
                answer = assistant.ask(question, field=None if field == "All" else field)
            except Exception as exc:  # noqa: BLE001 - surface any backend error
                st.error(f"Question failed: {exc}")
                return
        if answer.answer.strip():
            st.markdown(answer.answer)
        else:
            st.warning(
                "The model returned an empty answer — it may be rate-limited. "
                "Check the provider in the sidebar status, or try again shortly."
            )
        if answer.sources:
            with st.expander("Sources / retrieved passages"):
                for src in answer.sources:
                    meta = src["metadata"]
                    st.markdown(f"**[{meta.get('arxiv_id')}]** {meta.get('title', '')}")
                    st.caption(src["text"][:400] + "…")
    elif submitted:
        st.info("Type a question first.")


def page_live(assistant: ResearchAssistant) -> None:
    page_header("🔭", "Live arXiv Assistant", "Searches arXiv live for every question")
    st.caption(
        "Answers from fresh results, even papers not in your library. "
        "Try: _ancient DNA contamination removal methods_ · _high-entropy alloy design_"
    )

    with st.form("live_form"):
        question = st.text_input("Your question", "")
        max_results = st.slider("arXiv results to consider", 3, 12, 6)
        submitted = st.form_submit_button("Search arXiv & Answer", type="primary")

    if submitted and question.strip():
        with st.spinner("Searching arXiv live and reasoning over fresh results…"):
            try:
                result = assistant.live_assistant(
                    question, max_results=int(max_results)
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Live query failed: {exc}")
                return
        # Persist across reruns so the "Add to library" button keeps the results.
        st.session_state["live_result"] = result
    elif submitted:
        st.info("Type a question first.")

    result = st.session_state.get("live_result")
    if result is None:
        return

    st.caption(f"arXiv query used: `{result.arxiv_query}`")
    if result.answer.strip():
        st.markdown(result.answer)
    else:
        st.warning("The model returned an empty answer — try again or rephrase.")

    if result.papers:
        st.subheader(f"{len(result.papers)} papers from arXiv")
        st.caption("Ranked by **relevance** to your question (not by date — note the mixed years).")
        st.caption(
            "Adding downloads each PDF and summarises it (~10–20s/paper on cloud), "
            "then it's browsable and searchable in Ask Questions."
        )
        if st.button("➕ Add & process these papers", type="primary"):
            on_progress, finalize = _processing_progress()
            with st.spinner("Adding & processing papers…"):
                results = assistant.add_and_process(
                    result.papers, progress_callback=on_progress
                )
            finalize()
            ok = sum(1 for r in results if r.summarized)
            st.success(
                f"Added & processed {len(results)} papers ({ok} summarised). "
                "Find them in Browse Library and Ask Questions."
            )
        for paper in result.papers:
            with st.expander(f"[{paper.arxiv_id}] {paper.title}"):
                st.caption(
                    f"{', '.join(paper.authors[:6])} · {paper.primary_category} "
                    f"· {paper.published.date()}"
                )
                if paper.entry_url:
                    st.markdown(f"[View on arXiv]({paper.entry_url})")
                st.write(paper.abstract)


def page_biorxiv(assistant: ResearchAssistant) -> None:
    page_header("🧬", "bioRxiv Assistant", "Live preprint search for biology & genetics")
    st.caption(
        "Searches **bioRxiv preprints** live (via Europe PMC) — much richer than "
        "arXiv for biology, genetics, and paleogenetics."
    )
    st.caption(
        "Try: _What does recent work say about Neanderthal introgression and immunity?_ · "
        "_Methods for authenticating ancient DNA_"
    )
    st.info("bioRxiv papers are **preprints** — not yet peer-reviewed.", icon="🧬")

    with st.form("biorxiv_form"):
        question = st.text_input("Your question", "")
        max_results = st.slider("bioRxiv results to consider", 3, 12, 6)
        submitted = st.form_submit_button("Search bioRxiv & Answer", type="primary")

    if submitted and question.strip():
        with st.spinner("Searching bioRxiv live and reasoning over fresh preprints…"):
            try:
                result = assistant.biorxiv_assistant(
                    question, max_results=int(max_results)
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"bioRxiv query failed: {exc}")
                return
        st.session_state["biorxiv_result"] = result
    elif submitted:
        st.info("Type a question first.")

    result = st.session_state.get("biorxiv_result")
    if result is None:
        return

    st.caption(f"bioRxiv query used: `{result.query}`")
    if result.answer.strip():
        st.markdown(result.answer)
    else:
        st.warning("The model returned an empty answer — try again or rephrase.")

    if result.papers:
        st.subheader(f"{len(result.papers)} preprints from bioRxiv")
        st.caption("Ranked by **relevance** to your question.")
        st.caption("Adding summarises & indexes each preprint from its abstract.")
        if st.button("➕ Add & summarise these preprints", type="primary"):
            with st.spinner("Summarising & indexing preprints…"):
                added = assistant.add_biorxiv_papers(result.papers)
            st.success(
                f"Added {added} preprints to your library (field: bioRxiv). "
                "Find them in Browse Library and Ask Questions."
            )
        for paper in result.papers:
            with st.expander(f"{paper.title}"):
                st.caption(
                    f"{', '.join(paper.authors[:6])} · {paper.published}"
                )
                if paper.url:
                    st.markdown(f"[View on bioRxiv / DOI]({paper.url})  ·  `{paper.doi}`")
                st.write(paper.abstract[:1500] + ("…" if len(paper.abstract) > 1500 else ""))


def page_ntrs(assistant: ResearchAssistant) -> None:
    page_header("🚀", "NASA NTRS", "Live search of NASA technical reports")
    st.caption(
        "Searches the **NASA Technical Reports Server** live — strong on aerospace, "
        "planetary science, astrophysics, propulsion, and materials."
    )
    st.caption(
        "Try: _thermal performance of the Mars rover_ · "
        "_high-entropy alloys for turbine applications_"
    )

    with st.form("ntrs_form"):
        question = st.text_input("Your question", "")
        max_results = st.slider("NTRS results to consider", 3, 12, 6)
        submitted = st.form_submit_button("Search NTRS & Answer", type="primary")

    if submitted and question.strip():
        with st.spinner("Searching NASA NTRS live and reasoning over reports…"):
            try:
                result = assistant.ntrs_assistant(question, max_results=int(max_results))
            except Exception as exc:  # noqa: BLE001
                st.error(f"NTRS query failed: {exc}")
                return
        st.session_state["ntrs_result"] = result
    elif submitted:
        st.info("Type a question first.")

    result = st.session_state.get("ntrs_result")
    if result is None:
        return

    st.caption(f"NTRS query used: `{result.query}`")
    if result.answer.strip():
        st.markdown(result.answer)
    else:
        st.warning("The model returned an empty answer — try again or rephrase.")

    if result.papers:
        st.subheader(f"{len(result.papers)} reports from NASA NTRS")
        st.caption("Ranked by **relevance** to your question.")
        st.caption("Adding summarises & indexes each report from its abstract.")
        if st.button("➕ Add & summarise these reports", type="primary"):
            with st.spinner("Summarising & indexing reports…"):
                added = assistant.add_ntrs_papers(result.papers)
            st.success(
                f"Added {added} reports to your library (field: NTRS). "
                "Find them in Browse Library and Ask Questions."
            )
        for paper in result.papers:
            with st.expander(f"{paper.title}"):
                meta = f"{', '.join(paper.authors[:6])} · {paper.published}"
                if paper.center:
                    meta += f" · {paper.center}"
                st.caption(meta)
                if paper.url:
                    st.markdown(f"[View on NTRS]({paper.url})  ·  `{paper.id}`")
                st.write(paper.abstract[:1500] + ("…" if len(paper.abstract) > 1500 else ""))


def page_reports(assistant: ResearchAssistant) -> None:
    page_header("📰", "Weekly Reports", "AI-synthesised digests of your recent papers")
    days = st.number_input("Look-back window (days)", 1, 60, settings.search_lookback_days)
    st.caption(
        "Synthesises your *summarised* papers (capped to keep the request under "
        "provider rate limits)."
    )
    if st.button("Generate new report", type="primary"):
        with st.spinner("Synthesising weekly report…"):
            try:
                report = assistant.generate_report(days=int(days))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Report generation failed: {exc}")
                return
        st.markdown(report)

    st.divider()
    st.subheader("Saved reports")
    from agents.report_agent import ReportAgent

    saved = ReportAgent.list_reports()
    if not saved:
        st.info("No saved reports yet.")
        return
    chosen = st.selectbox("Open a saved report", saved)
    if chosen:
        st.markdown((settings.reports_dir / chosen).read_text(encoding="utf-8"))


def page_graph(assistant: ResearchAssistant) -> None:
    page_header("🕸️", "Knowledge Graph", "How your papers, authors, and topics connect")

    field = st.selectbox(
        "Limit to a field (optional)", ["All"] + list(ARXIV_FIELDS.keys())
    )
    if st.button("Build / refresh graph", type="primary"):
        with st.spinner("Building knowledge graph…"):
            builder = assistant.build_graph(
                field=None if field == "All" else field, export_html=True
            )
        st.session_state["graph_built"] = True

        stats = builder.stats()
        cols = st.columns(len(stats))
        for col, (key, value) in zip(cols, stats.items()):
            col.metric(key, value)

        st.subheader("Most central topics")
        for label, degree in builder.central_topics():
            st.write(f"- **{label}** — {degree} connections")

    # Show the interactive pyvis HTML if it was generated.
    html_path = settings.data_dir / "knowledge_graph.html"
    if html_path.exists():
        st.subheader("Interactive view")
        st.components.v1.html(html_path.read_text(encoding="utf-8"), height=750)
    else:
        st.info("Build the graph to generate an interactive visualisation.")


PAGES = {
    "📚  Library": page_library,
    "💬  Ask": page_questions,
    "🔭  Live arXiv": page_live,
    "🧬  bioRxiv": page_biorxiv,
    "🚀  NASA NTRS": page_ntrs,
    "📰  Reports": page_reports,
    "🕸️  Graph": page_graph,
}


def main() -> None:
    inject_theme()
    if not check_password():
        return
    assistant = get_assistant()
    page = sidebar(assistant)
    PAGES[page](assistant)


main()
