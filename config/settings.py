"""Central configuration for Local AI Scientist.

All tunable values live here and can be overridden via environment variables
(or a ``.env`` file in the project root). Nothing is hardcoded to an absolute
path: every directory is derived from the project root at import time, but can
still be overridden through the environment.

Example::

    from config.settings import settings
    print(settings.ollama_model)
    print(settings.papers_dir)
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = two levels up from this file (config/settings.py -> project_root/)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


# arXiv category codes grouped by the human-friendly research fields the user
# cares about. The search agent expands these into arXiv query strings.
ARXIV_FIELDS: dict[str, list[str]] = {
    "Physics": ["physics.gen-ph", "cond-mat.mtrl-sci", "quant-ph"],
    "Astrophysics": ["astro-ph.GA", "astro-ph.CO", "astro-ph.SR", "astro-ph.HE"],
    "Materials Science": ["cond-mat.mtrl-sci", "cond-mat.soft"],
    "Genetics": ["q-bio.GN", "q-bio.PE"],
    "Paleogenetics": ["q-bio.PE", "q-bio.GN"],
    "Paleoanthropology": ["q-bio.PE"],
    "Artificial Intelligence": ["cs.AI", "cs.LG", "cs.CL"],
}


class Settings(BaseSettings):
    """Strongly-typed application settings, sourced from env / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),  # allow field names that start with "model_"
    )

    # ----- LLM provider selection -------------------------------------------
    llm_provider: str = Field(
        default="ollama",
        description=(
            "Which LLM backend to use: 'ollama' (local or Ollama Cloud) or an "
            "OpenAI-compatible provider ('openai'/'groq'/'gemini'/'openrouter')."
        ),
    )

    # ----- Ollama / Granite -------------------------------------------------
    ollama_host: str = Field(
        default="http://localhost:11434",
        description="Base URL of the local Ollama server.",
    )
    ollama_model: str = Field(
        default="granite4.1",
        description=(
            "Tag of the Granite model to use, as shown by `ollama list`. "
            "Override with OLLAMA_MODEL if you pulled a different tag."
        ),
    )
    ollama_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    ollama_timeout: int = Field(default=600, description="Per-request timeout (s).")
    ollama_num_ctx: int = Field(
        default=8192, description="Context window passed to the model."
    )

    # ----- OpenAI-compatible provider (Groq / Gemini / OpenRouter) ----------
    openai_base_url: str = Field(
        default="https://api.groq.com/openai/v1",
        description="Base URL of an OpenAI-compatible chat-completions API.",
    )
    openai_api_key: str = Field(
        default="", description="API key for the OpenAI-compatible provider."
    )
    openai_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Model id on the OpenAI-compatible provider.",
    )

    # ----- Embeddings / vector search --------------------------------------
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="sentence-transformers model used for embeddings.",
    )
    chunk_size: int = Field(default=1200, description="Chars per text chunk.")
    chunk_overlap: int = Field(default=200, description="Char overlap between chunks.")
    retrieval_top_k: int = Field(default=6, description="Chunks retrieved per query.")

    # ----- Search defaults --------------------------------------------------
    default_fields: list[str] = Field(
        default_factory=lambda: list(ARXIV_FIELDS.keys()),
        description="Research fields searched when none are specified.",
    )
    max_results_per_field: int = Field(default=10)
    search_lookback_days: int = Field(default=7)
    report_max_papers: int = Field(
        default=30,
        description="Cap on papers included in one weekly report (keeps the "
        "single LLM call under provider token-per-minute limits).",
    )

    # ----- Processing -------------------------------------------------------
    max_concurrent_jobs: int = Field(
        default=2, description="Concurrent download/summarise jobs."
    )
    process_batch_size: int = Field(
        default=10,
        description=(
            "Default cap on papers processed by a single `run`/full-cycle, so a "
            "large search can't queue dozens of slow CPU summarisations at once."
        ),
    )
    max_chars_for_summary: int = Field(
        default=24000,
        description="Truncate extracted text to this many chars before summarising.",
    )

    # ----- Paths (override via env if desired) ------------------------------
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    logs_dir: Path = Field(default=PROJECT_ROOT / "logs")
    log_level: str = Field(default="INFO")

    @field_validator("data_dir", "logs_dir", mode="before")
    @classmethod
    def _expand_path(cls, value: str | Path) -> Path:
        """Expand ``~`` and resolve relative paths against the project root."""
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    # ----- Derived directories ---------------------------------------------
    @property
    def papers_dir(self) -> Path:
        return self.data_dir / "papers"

    @property
    def embeddings_dir(self) -> Path:
        return self.data_dir / "embeddings"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "papers.sqlite3"

    @property
    def graph_path(self) -> Path:
        return self.data_dir / "knowledge_graph.graphml"

    def ensure_dirs(self) -> None:
        """Create every directory the application writes to."""
        for path in (
            self.data_dir,
            self.papers_dir,
            self.embeddings_dir,
            self.reports_dir,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def categories_for_fields(self, fields: list[str] | None) -> dict[str, list[str]]:
        """Map requested field names to their arXiv category codes."""
        selected = fields or self.default_fields
        return {f: ARXIV_FIELDS[f] for f in selected if f in ARXIV_FIELDS}


# A single shared instance imported across the codebase.
settings = Settings()
settings.ensure_dirs()
