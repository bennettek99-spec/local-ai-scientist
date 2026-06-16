# 🔬 Local AI Scientist

A fully local, modular AI research assistant. It searches arXiv across the
fields you care about, downloads and reads papers, summarises them with **IBM
Granite 4.1 running through Ollama**, builds a semantic search index and a
knowledge graph, and answers natural-language questions about your collection —
all on your own machine.

> Nothing leaves your computer except the arXiv queries and PDF downloads. The
> LLM, embeddings, vector database, and knowledge graph are all local.

---

## Features

| Capability | Module |
|---|---|
| Search arXiv by field (Physics, Astrophysics, Materials Science, Genetics, Paleogenetics, Paleoanthropology, AI) | `agents/search_agent.py` |
| Download PDFs + extract text | `pdf_processing/` |
| Summaries, key findings, equations, assumptions/limitations, simplified explanation, future work, related topics | `agents/summary_agent.py` |
| Semantic search + RAG Q&A | `database/vector_store.py`, `agents/question_agent.py` |
| Live Science Assistant (fresh arXiv search per query) | `agents/live_agent.py` |
| bioRxiv Assistant (live preprint search via Europe PMC) | `agents/biorxiv_agent.py` |
| Weekly markdown reports (discoveries, trends, cross-disciplinary links) | `agents/report_agent.py` |
| Knowledge graph (papers ↔ authors ↔ topics ↔ fields) | `knowledge_graph/graph_builder.py` |
| Web UI with 5 pages | `ui/streamlit_app.py` |
| CLI orchestration | `main.py` + `core/pipeline.py` |

---

## Project structure

```
local-ai-scientist/
├── main.py                     # CLI entry point
├── requirements.txt
├── .env.example                # copy to .env to override defaults
├── config/
│   └── settings.py             # all configuration (env-overridable, no hardcoded paths)
├── core/
│   ├── models.py               # Paper / PaperAnalysis data models
│   ├── ollama_client.py        # Granite 4.1 via Ollama (sync + async)
│   └── pipeline.py             # ResearchAssistant orchestrator
├── agents/
│   ├── search_agent.py
│   ├── summary_agent.py
│   ├── question_agent.py
│   └── report_agent.py
├── database/
│   ├── paper_database.py       # SQLite metadata store
│   └── vector_store.py         # ChromaDB + sentence-transformers
├── knowledge_graph/
│   └── graph_builder.py        # NetworkX + pyvis
├── pdf_processing/
│   ├── pdf_loader.py           # async PDF downloads
│   └── text_extractor.py       # PyMuPDF extraction + chunking
├── ui/
│   └── streamlit_app.py
├── utils/
│   └── logging_config.py
├── data/                       # papers/, embeddings/, reports/ (generated)
└── logs/
```

---

## Setup

### 1. Install Ollama and pull Granite

Install Ollama from <https://ollama.com>, then start it and pull the model:

```bash
ollama serve              # if it isn't already running as a service
ollama pull granite4.1    # use the exact tag you want; see `ollama list`
```

> **Model tag matters.** Set `OLLAMA_MODEL` in `.env` to exactly what
> `ollama list` shows (for example `granite4.1`, or `granite4:small-h`). The app
> prints a clear warning at startup if the configured model isn't present.

### 2. Create a virtual environment and install dependencies

```powershell
# from the local-ai-scientist/ folder
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate         # macOS / Linux

pip install -r requirements.txt
```

The first run downloads the sentence-transformers embedding model
(`all-MiniLM-L6-v2`, ~90 MB) automatically.

### 3. (Optional) configure

```powershell
copy .env.example .env              # then edit values as desired
```

---

## Usage

### Command line

```bash
# Check connectivity and library status
python main.py status

# Search recent papers in selected fields
python main.py search --fields Paleogenetics "Artificial Intelligence" --max 5

# Download, extract, summarise, and embed everything not yet processed
python main.py process

# Or do the whole cycle at once (search -> process -> graph -> report)
python main.py run --fields Genetics "Materials Science" --max 5

# Ask questions about your library (RAG)
python main.py ask "What new papers discuss Denisovan DNA?"
python main.py ask "Find connections between paleogenetics and machine learning."

# Generate a weekly report / rebuild the knowledge graph
python main.py report
python main.py graph
```

### Web interface

```bash
python main.py ui          # or: streamlit run ui/streamlit_app.py
```

Pages: **Search Papers**, **Browse Library**, **Ask Questions**,
**Weekly Reports**, **Knowledge Graph**.

---

## How it works

1. **Search** — `SearchAgent` maps your fields to arXiv category codes and pulls
   the most recent submissions.
2. **Process** — for each new paper the pipeline downloads the PDF
   (async, concurrent), extracts text with PyMuPDF, asks Granite for a
   structured analysis, and embeds the chunks into ChromaDB.
3. **Ask** — `QuestionAgent` retrieves the most relevant passages and has
   Granite answer with inline `[arXiv-id]` citations.
4. **Report** — `ReportAgent` feeds recent summaries to Granite to produce a
   skimmable weekly digest in `data/reports/`.
5. **Graph** — `KnowledgeGraphBuilder` links papers, authors, topics, and
   fields with NetworkX and renders an interactive pyvis view.

---

## Choosing an LLM backend

The LLM is pluggable via `LLM_PROVIDER` in `.env`. Embeddings always stay local;
only text generation uses the chosen backend. Switching is config-only — no code
changes — because every backend implements the same interface (`core/llm.py`).

| Mode | `.env` settings | Speed (CPU laptop) | Notes |
|---|---|---|---|
| **Local Ollama** | `LLM_PROVIDER=ollama`, `OLLAMA_MODEL=granite4.1:3b` | ~60–90s/paper | 100% offline & private; unlimited |
| **Ollama Cloud** | `LLM_PROVIDER=ollama`, `OLLAMA_MODEL=gpt-oss:120b-cloud` | ~10–20s/paper | Runs on Ollama's GPUs; free tier w/ rate limits; needs `ollama signin` |
| **Groq** | `LLM_PROVIDER=groq`, `OPENAI_API_KEY=…` | very fast | Free tier; key from <https://console.groq.com/keys> |
| **Gemini** | `LLM_PROVIDER=gemini`, set `OPENAI_BASE_URL`+key | fast | Free tier; key from <https://aistudio.google.com/apikey> |

Groq/Gemini/OpenRouter all use the OpenAI-compatible client (`core/openai_client.py`);
set `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `OPENAI_MODEL` accordingly. On a CPU-only
machine, drop `MAX_CHARS_FOR_SUMMARY` to ~6000 and `MAX_CONCURRENT_JOBS=1` for local
Ollama; raise them again for cloud providers.

---

## Configuration reference

Everything is overridable via environment variables / `.env`. Key settings:

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `granite4.1` | Granite model tag |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `RETRIEVAL_TOP_K` | `6` | passages retrieved per question |
| `MAX_RESULTS_PER_FIELD` | `10` | papers fetched per field per search |
| `MAX_CONCURRENT_JOBS` | `2` | concurrent download/summarise jobs |
| `SEARCH_LOOKBACK_DAYS` | `7` | weekly-report window |

See `config/settings.py` and `.env.example` for the full list.

---

## Notes & tips

- **First processing run is the slow one** — Granite analyses each paper
  sequentially-ish (bounded concurrency). Start with `--max 3` to try it out.
- **Be polite to arXiv** — the search agent rate-limits requests; avoid huge
  `--max` values in rapid succession.
- **Storage** — PDFs, extracted text, the SQLite DB, the Chroma index, reports,
  and the graph all live under `data/`. Delete `data/` to start fresh.
- **Errors degrade gracefully** — if a PDF fails to download or extract, the
  paper falls back to its abstract; if Ollama is down, you get a clear message
  rather than a crash.

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) with a Granite 4.x model pulled
- ~2 GB free disk for models + a growing `data/` directory
