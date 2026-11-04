"""Vision providers. To add another (OpenAI, Gemini, a local detector), subclass
CountProvider, implement `count`, and register it in `make_provider`."""

from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import dataclass, field

import httpx

from .config import Settings
from .prompt import REPORT_TOOL, SYSTEM_PROMPT, user_text

log = logging.getLogger("snapcount")


class ProviderError(Exception):
    """The AI provider failed or returned something unusable."""


class ProviderTimeout(ProviderError):
    pass


@dataclass
class Section:
    label: str
    count: int


@dataclass
class CountResult:
    item: str
    count: int | None
    confidence: str  # high | medium | low
    notes: str
    retake_tip: str
    sections: list[Section] = field(default_factory=list)
    # Centre of each counted item as [x, y] fractions of the image (0..1). Detectors fill
    # this; the Claude provider doesn't, because it returns no reliable positions.
    points: list[list[float]] = field(default_factory=list)
    # Usage info for logging only.
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


def normalize(raw: dict, fallback_item: str | None) -> CountResult:
    """Turn the model's tool input into a safe, consistent result."""
    if not isinstance(raw, dict):
        raise ProviderError("Model returned no structured result")

    confidence = str(raw.get("confidence", "low")).lower()
    if confidence not in ("high", "medium", "low"):
        confidence = "low"

    sections: list[Section] = []
    for s in raw.get("sections") or []:
        try:
            n = int(s.get("count"))
        except (TypeError, ValueError, AttributeError):
            continue
        if n >= 0:
            sections.append(Section(label=str(s.get("label", ""))[:60], count=n))

    countable = bool(raw.get("countable", True))
    count = raw.get("count")
    try:
        count = int(count) if count is not None else None
    except (TypeError, ValueError):
        count = None
    if count is not None and count < 0:
        count = None

    notes = str(raw.get("notes", "")).strip()
    tip = str(raw.get("retake_tip", "")).strip()

    if countable and sections:
        total = sum(s.count for s in sections)
        if count is None or total != count:
            # Trust the methodical per-section counts over the model's own addition.
            log.info("section sum %s differs from reported total %s; using sum", total, count)
            count = total

    if not countable or count is None:
        count = None
        confidence = "low"
        if not notes:
            notes = "The photo doesn't show enough to count these items."

    if confidence == "high":
        tip = ""
    elif not tip:
        tip = "Take a sharper photo, closer and straight on, with all the items in the frame."

    return CountResult(
        item=str(raw.get("item") or fallback_item or "items").strip()[:80],
        count=count,
        confidence=confidence,
        notes=notes[:400],
        retake_tip=tip[:200],
        sections=sections[:50],
    )


class CountProvider:
    name = "base"

    async def count(self, image: bytes, media_type: str, item: str | None) -> CountResult:
        raise NotImplementedError


class AnthropicProvider(CountProvider):
    name = "anthropic"
    url = "https://api.anthropic.com/v1/messages"

    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.s = settings
        self.transport = transport

    async def count(self, image: bytes, media_type: str, item: str | None) -> CountResult:
        body = {
            "model": self.s.anthropic_model,
            "max_tokens": 1500,
            "system": SYSTEM_PROMPT,
            "tools": [REPORT_TOOL],
            "tool_choice": {"type": "tool", "name": "report_count"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64.b64encode(image).decode(),
                            },
                        },
                        {"type": "text", "text": user_text(item)},
                    ],
                }
            ],
        }
        headers = {
            "x-api-key": self.s.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        data = await self._post_with_retry(body, headers)

        tool_input = None
        for block in data.get("content", []):
            if block.get("type") == "tool_use" and block.get("name") == "report_count":
                tool_input = block.get("input")
                break
        if tool_input is None:
            raise ProviderError("Model did not return a count")

        result = normalize(tool_input, item)
        usage = data.get("usage", {})
        result.model = data.get("model", self.s.anthropic_model)
        result.input_tokens = int(usage.get("input_tokens", 0))
        result.output_tokens = int(usage.get("output_tokens", 0))
        return result

    async def _post_with_retry(self, body: dict, headers: dict) -> dict:
        timeout = httpx.Timeout(self.s.ai_timeout_seconds, connect=10)
        attempts = 2
        async with httpx.AsyncClient(timeout=timeout, transport=self.transport) as client:
            for attempt in range(1, attempts + 1):
                try:
                    r = await client.post(self.url, json=body, headers=headers)
                except httpx.TimeoutException as e:
                    raise ProviderTimeout("The AI service took too long") from e
                except httpx.HTTPError as e:
                    if attempt < attempts:
                        await asyncio.sleep(1)
                        continue
                    raise ProviderError("Couldn't reach the AI service") from e

                if r.status_code == 200:
                    return r.json()
                # Retry once on rate limits and server-side errors.
                if r.status_code in (429, 500, 502, 503, 529) and attempt < attempts:
                    await asyncio.sleep(2)
                    continue
                log.warning("anthropic error %s: %s", r.status_code, r.text[:300])
                raise ProviderError(f"AI service error ({r.status_code})")
        raise ProviderError("AI service unavailable")


class MockProvider(CountProvider):
    """Returns a fixed answer. For local testing without an API key."""

    name = "mock"

    async def count(self, image: bytes, media_type: str, item: str | None) -> CountResult:
        result = normalize(
            {
                "item": item or "items",
                "sections": [{"label": "left", "count": 7}, {"label": "right", "count": 5}],
                "countable": True,
                "count": 12,
                "confidence": "medium",
                "notes": "Mock result - no AI was called.",
                "retake_tip": "",
            },
            item,
        )
        result.model = "mock"
        return result


def make_provider(settings: Settings) -> CountProvider:
    if settings.provider == "mock":
        return MockProvider()
    if settings.provider == "anthropic":
        return AnthropicProvider(settings)
    raise RuntimeError(f"Unknown SNAPCOUNT_PROVIDER: {settings.provider}")
