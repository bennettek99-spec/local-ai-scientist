"""LLM backend factory.

Returns the right client for ``settings.llm_provider`` so the rest of the app
never has to care which provider is in use. Both clients expose the same method
surface (``generate``, ``agenerate``, ``generate_json``, ``agenerate_json``,
``is_available``, ``model_available``, ``.model``).

Providers:
- ``ollama`` (default): local or Ollama-Cloud models via the local Ollama API.
- ``openai`` / ``groq`` / ``gemini`` / ``openrouter``: any OpenAI-compatible API.
"""

from __future__ import annotations

from typing import Union

from config.settings import settings
from core.ollama_client import OllamaClient
from core.openai_client import OpenAIClient
from utils.logging_config import get_logger

logger = get_logger(__name__)

LLMClient = Union[OllamaClient, OpenAIClient]

_OPENAI_ALIASES = {"openai", "groq", "gemini", "openrouter", "compat"}


def make_llm_client() -> LLMClient:
    """Construct the LLM client selected by ``settings.llm_provider``."""
    provider = (settings.llm_provider or "ollama").lower()
    if provider in _OPENAI_ALIASES:
        logger.info(
            "LLM provider: %s - model '%s' at %s",
            provider, settings.openai_model, settings.openai_base_url,
        )
        return OpenAIClient()
    logger.info("LLM provider: ollama - model '%s'", settings.ollama_model)
    return OllamaClient()


def active_model_name() -> str:
    """Name of the model the active provider will use (for status displays)."""
    provider = (settings.llm_provider or "ollama").lower()
    return settings.openai_model if provider in _OPENAI_ALIASES else settings.ollama_model
