"""Shared helpers for coaxing structured JSON out of LLM responses.

Used by both the Ollama and OpenAI-compatible clients. Models usually honour a
JSON-mode request but occasionally wrap output in prose or code fences; we strip
those and retry before giving up.
"""

from __future__ import annotations

import json
import re
from typing import Any

from utils.logging_config import get_logger

logger = get_logger(__name__)


def parse_json_blob(raw: str) -> dict[str, Any]:
    """Best-effort parse of model output into a dict (empty dict on failure)."""
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strip ```json ... ``` fences if present.
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    # Fall back to the first balanced-looking { ... } block.
    brace = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group(0))
        except json.JSONDecodeError:
            logger.warning("Could not parse JSON from model output.")
    return {}


def describe_exception(exc: Exception) -> str:
    """Render an exception with its type, since some carry an empty message."""
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__
