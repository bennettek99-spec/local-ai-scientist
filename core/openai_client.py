"""OpenAI-compatible LLM client (Groq, Google Gemini, OpenRouter, etc.).

Talks to any endpoint that implements the OpenAI ``/chat/completions`` shape.
Exposes the exact same method surface as :class:`OllamaClient` so the rest of
the app (agents, pipeline) is backend-agnostic — the factory in ``core.llm``
picks one based on ``settings.llm_provider``.

Configure via ``.env``::

    LLM_PROVIDER=openai
    OPENAI_BASE_URL=https://api.groq.com/openai/v1
    OPENAI_API_KEY=gsk_...
    OPENAI_MODEL=llama-3.3-70b-versatile
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from config.settings import settings
from core.json_utils import describe_exception, parse_json_blob
from core.llm_base import LLMError
from utils.logging_config import get_logger

logger = get_logger(__name__)

_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 3.0


class OpenAIError(LLMError):
    """Raised when the OpenAI-compatible endpoint cannot satisfy a request."""


def _describe_http(exc: Exception) -> str:
    """Like describe_exception, but surfaces HTTP status + body for API errors."""
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            detail = exc.response.text[:200]
        except Exception:  # noqa: BLE001
            detail = ""
        return f"HTTP {exc.response.status_code}: {detail}".strip()
    return describe_exception(exc)


def _retry_delay(exc: Exception, attempt: int) -> float:
    """How long to wait before the next attempt.

    On HTTP 429 (rate limit) honour the provider's ``Retry-After`` header so we
    wait out the limit window instead of burning retries in the same minute.
    """
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
        retry_after = exc.response.headers.get("retry-after")
        if retry_after:
            try:
                return min(float(retry_after) + 1.0, 65.0)
            except ValueError:
                pass
        return 30.0
    return _RETRY_BACKOFF_SECONDS * attempt


class OpenAIClient:
    """Minimal client for an OpenAI-compatible chat-completions API."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        timeout: int | None = None,
    ) -> None:
        self.base_url = (base_url or settings.openai_base_url).rstrip("/")
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_model
        self.temperature = (
            settings.ollama_temperature if temperature is None else temperature
        )
        self.timeout = timeout or settings.ollama_timeout

    # ------------------------------------------------------------------ checks
    def is_available(self) -> bool:
        """Return True if the endpoint authenticates and responds."""
        if not self.api_key:
            logger.warning(
                "No OPENAI_API_KEY set for provider at %s.", self.base_url
            )
            return False
        try:
            response = httpx.get(
                f"{self.base_url}/models", headers=self._headers(), timeout=15
            )
            return response.status_code == 200
        except Exception as exc:  # noqa: BLE001
            logger.warning("Provider endpoint unreachable at %s: %s", self.base_url, exc)
            return False

    def model_available(self) -> bool:
        """If the endpoint authenticates, assume the configured model id is valid.

        Listing exact ids varies by provider; an invalid id surfaces a clear
        error at call time instead.
        """
        return self.is_available()

    # ----------------------------------------------------------------- helpers
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _payload(
        self, prompt: str, system: str | None, temperature: float | None, as_json: bool
    ) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if as_json:
            body["response_format"] = {"type": "json_object"}
        return body

    @staticmethod
    def _extract(data: dict[str, Any]) -> str:
        try:
            return (data["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, TypeError):
            logger.warning("Unexpected response shape from provider.")
            return ""

    # --------------------------------------------------------------- sync APIs
    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        as_json: bool = False,
    ) -> str:
        """Generate a completion and return the raw text (with retries)."""
        body = self._payload(prompt, system, temperature, as_json)
        url = f"{self.base_url}/chat/completions"
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = httpx.post(
                    url, headers=self._headers(), json=body, timeout=self.timeout
                )
                response.raise_for_status()
                return self._extract(response.json())
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "Provider generate attempt %d/%d failed: %s",
                    attempt, _MAX_ATTEMPTS, _describe_http(exc),
                )
                if attempt < _MAX_ATTEMPTS:
                    time.sleep(_retry_delay(exc, attempt))
        raise OpenAIError(f"Generation failed: {_describe_http(last_exc)}") from last_exc

    def generate_json(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        raw = self.generate(prompt, system=system, temperature=temperature, as_json=True)
        return parse_json_blob(raw)

    # -------------------------------------------------------------- async APIs
    async def agenerate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        as_json: bool = False,
    ) -> str:
        """Async counterpart of :meth:`generate`."""
        body = self._payload(prompt, system, temperature, as_json)
        url = f"{self.base_url}/chat/completions"
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        url, headers=self._headers(), json=body
                    )
                    response.raise_for_status()
                    return self._extract(response.json())
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "Provider async generate attempt %d/%d failed: %s",
                    attempt, _MAX_ATTEMPTS, _describe_http(exc),
                )
                if attempt < _MAX_ATTEMPTS:
                    await asyncio.sleep(_retry_delay(exc, attempt))
        raise OpenAIError(
            f"Async generation failed: {_describe_http(last_exc)}"
        ) from last_exc

    async def agenerate_json(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        raw = await self.agenerate(
            prompt, system=system, temperature=temperature, as_json=True
        )
        return parse_json_blob(raw)
