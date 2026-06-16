"""Thin wrapper around the local Ollama server for Granite 4.1.

Provides synchronous and asynchronous text generation plus a couple of helpers
for coaxing structured JSON out of the model. All network errors are caught and
re-raised as :class:`OllamaError` so callers can degrade gracefully.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import ollama

from config.settings import settings
from core.json_utils import parse_json_blob
from core.llm_base import LLMError
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Generation calls are retried this many times. The first call to a model often
# triggers a multi-GB load into memory, which can transiently fail/time out.
_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 4.0


def _describe(exc: Exception) -> str:
    """Render an exception with its type, since some carry an empty message."""
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


class OllamaError(LLMError):
    """Raised when the Ollama server cannot satisfy a request."""


class OllamaClient:
    """Wrapper exposing the small slice of Ollama functionality we need."""

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        timeout: int | None = None,
        num_ctx: int | None = None,
    ) -> None:
        self.host = host or settings.ollama_host
        self.model = model or settings.ollama_model
        self.temperature = (
            settings.ollama_temperature if temperature is None else temperature
        )
        self.timeout = timeout or settings.ollama_timeout
        self.num_ctx = num_ctx or settings.ollama_num_ctx

        self._client = ollama.Client(host=self.host, timeout=self.timeout)
        self._aclient = ollama.AsyncClient(host=self.host, timeout=self.timeout)

    # ------------------------------------------------------------------ checks
    def is_available(self) -> bool:
        """Return True if the Ollama server responds to a list request."""
        try:
            self._client.list()
            return True
        except Exception as exc:  # noqa: BLE001 - report any connectivity issue
            logger.warning("Ollama server not reachable at %s: %s", self.host, exc)
            return False

    def model_available(self) -> bool:
        """Return True if the configured model is present locally."""
        try:
            response = self._client.list()
            names = {m.get("model", "") for m in response.get("models", [])}
            # Match either the exact tag or the base name before the ':' tag.
            base = self.model.split(":")[0]
            return any(self.model == n or n.split(":")[0] == base for n in names)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not list Ollama models: %s", exc)
            return False

    def _options(self, **overrides: Any) -> dict[str, Any]:
        options = {"temperature": self.temperature, "num_ctx": self.num_ctx}
        options.update({k: v for k, v in overrides.items() if v is not None})
        return options

    # --------------------------------------------------------------- sync APIs
    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        as_json: bool = False,
    ) -> str:
        """Generate a completion for ``prompt`` and return the raw text.

        Retries transient failures (e.g. cold model loads) before giving up.
        """
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = self._client.generate(
                    model=self.model,
                    prompt=prompt,
                    system=system,
                    format="json" if as_json else "",
                    options=self._options(temperature=temperature),
                )
                return response.get("response", "").strip()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "Generation attempt %d/%d failed: %s",
                    attempt, _MAX_ATTEMPTS, _describe(exc),
                )
                if attempt < _MAX_ATTEMPTS:
                    time.sleep(_RETRY_BACKOFF_SECONDS * attempt)
        raise OllamaError(f"Generation failed: {_describe(last_exc)}") from last_exc

    # -------------------------------------------------------------- async APIs
    async def agenerate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        as_json: bool = False,
    ) -> str:
        """Async counterpart of :meth:`generate` (with the same retry policy)."""
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = await self._aclient.generate(
                    model=self.model,
                    prompt=prompt,
                    system=system,
                    format="json" if as_json else "",
                    options=self._options(temperature=temperature),
                )
                return response.get("response", "").strip()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "Async generation attempt %d/%d failed: %s",
                    attempt, _MAX_ATTEMPTS, _describe(exc),
                )
                if attempt < _MAX_ATTEMPTS:
                    await asyncio.sleep(_RETRY_BACKOFF_SECONDS * attempt)
        raise OllamaError(
            f"Async generation failed: {_describe(last_exc)}"
        ) from last_exc

    async def agenerate_json(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Async generate and parse the result as JSON (with a salvage pass)."""
        raw = await self.agenerate(
            prompt, system=system, temperature=temperature, as_json=True
        )
        return parse_json_blob(raw)

    def generate_json(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Generate and parse the result as JSON (with a salvage pass)."""
        raw = self.generate(prompt, system=system, temperature=temperature, as_json=True)
        return parse_json_blob(raw)
