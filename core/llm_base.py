"""Shared base type for LLM backends.

Both :class:`OllamaError` and :class:`OpenAIError` subclass :class:`LLMError`,
so agents can catch a single error type regardless of which provider is active.
"""

from __future__ import annotations


class LLMError(RuntimeError):
    """Base class for errors raised by any LLM backend."""
