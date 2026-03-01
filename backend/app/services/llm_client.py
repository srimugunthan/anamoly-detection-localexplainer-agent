"""LLM client: thin wrapper around Anthropic (Claude) and Google (Gemini).

Selected via the LLM_PROVIDER environment variable ("anthropic" | "gemini").
Supports optional base64-encoded image attachments.
Retries up to 3 times with exponential backoff on transient errors.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.config import ANTHROPIC_API_KEY, GEMINI_API_KEY, LLM_PROVIDER

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """Raised when the LLM call fails after all retries."""


# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3
_BASE_BACKOFF = 1.0  # seconds; doubled each retry


# ---------------------------------------------------------------------------
# Message construction helpers
# ---------------------------------------------------------------------------


def _build_anthropic_message(prompt: str, images: list[dict] | None):
    """Build a LangChain HumanMessage with Anthropic-style content blocks."""
    from langchain_core.messages import HumanMessage

    content: list[dict[str, Any]] = []

    # Image blocks first (before text, following Anthropic's recommended order)
    for img in images or []:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img.get("media_type", "image/png"),
                    "data": img["data"],
                },
            }
        )

    # Text block
    content.append({"type": "text", "text": prompt})

    return HumanMessage(content=content)


def _build_gemini_message(prompt: str, images: list[dict] | None):
    """Build a LangChain HumanMessage with Gemini-compatible content blocks."""
    from langchain_core.messages import HumanMessage

    content: list[dict[str, Any]] = []

    # Gemini uses "image_url" blocks with base64 data URIs
    for img in images or []:
        media_type = img.get("media_type", "image/png")
        data_uri = f"data:{media_type};base64,{img['data']}"
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": data_uri},
            }
        )

    # Text block
    content.append({"type": "text", "text": prompt})

    return HumanMessage(content=content)


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------


def _make_anthropic_client():
    """Instantiate ChatAnthropic."""
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model="claude-sonnet-4-6",
        api_key=ANTHROPIC_API_KEY,  # type: ignore[arg-type]
        max_tokens=2048,
    )


def _make_gemini_client():
    """Instantiate ChatGoogleGenerativeAI."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GEMINI_API_KEY,  # type: ignore[arg-type]
        max_output_tokens=4096,
    )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


async def call_llm(prompt: str, images: list[dict] | None = None) -> str:
    """Call the configured LLM with optional base64-encoded images.

    Parameters
    ----------
    prompt:
        Plain-text prompt string.
    images:
        Optional list of dicts with keys ``"data"`` (base64 string) and
        ``"media_type"`` (e.g. ``"image/png"``).

    Returns
    -------
    str
        The text content of the LLM response.

    Raises
    ------
    LLMError
        After all retries are exhausted.
    """
    provider = LLM_PROVIDER.lower()

    if provider == "anthropic":
        model = _make_anthropic_client()
        message = _build_anthropic_message(prompt, images)
    elif provider == "gemini":
        model = _make_gemini_client()
        message = _build_gemini_message(prompt, images)
    else:
        raise LLMError(
            f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Set to 'anthropic' or 'gemini'."
        )

    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = await model.ainvoke([message])
            # .content may be a string or a list of content blocks
            content = response.content
            if isinstance(content, str):
                return content
            # If content is a list, join text parts
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "".join(parts)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning(
                "LLM call attempt %d/%d failed: %s", attempt, _MAX_RETRIES, exc
            )
            if attempt < _MAX_RETRIES:
                backoff = _BASE_BACKOFF * (2 ** (attempt - 1))
                await asyncio.sleep(backoff)

    raise LLMError(
        f"LLM call failed after {_MAX_RETRIES} attempts. Last error: {last_exc}"
    ) from last_exc
