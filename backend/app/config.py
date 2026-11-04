"""Settings, all read from environment variables."""

import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _list(name: str, default: str) -> tuple[str, ...]:
    return tuple(x.strip().lower() for x in os.getenv(name, default).split(",") if x.strip())


@dataclass(frozen=True)
class Settings:
    # Models the server offers, by id. Free, self-hosted: owlv2, grounding_dino, omdet_turbo, stack.
    # Paid API: claude (only offered if ANTHROPIC_API_KEY is set). Testing only: mock.
    models: tuple[str, ...] = _list("SNAPCOUNT_MODELS", "owlv2,grounding_dino,omdet_turbo,stack,claude")
    # Models that Auto runs and combines. Defaults to the free models above.
    auto_models: tuple[str, ...] = _list("AUTO_MODELS", "owlv2,grounding_dino,omdet_turbo")
    # Load the free models when the server starts, so the first count isn't slow.
    preload: bool = os.getenv("PRELOAD_MODELS", "1") not in ("0", "false", "no")
    # Legacy switch: SNAPCOUNT_PROVIDER=mock behaves like SNAPCOUNT_MODELS=mock.
    provider: str = os.getenv("SNAPCOUNT_PROVIDER", "").lower()
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
    # USD per million tokens, used only for the cost estimate in logs.
    price_in_per_mtok: float = float(os.getenv("PRICE_IN_PER_MTOK", "2"))
    price_out_per_mtok: float = float(os.getenv("PRICE_OUT_PER_MTOK", "10"))
    # Optional shared secret. If set, the app must send it in the X-App-Token header.
    app_token: str = os.getenv("APP_TOKEN", "")
    max_upload_mb: int = _int("MAX_UPLOAD_MB", 10)
    max_image_side: int = _int("MAX_IMAGE_SIDE", 1600)
    ai_timeout_seconds: int = _int("AI_TIMEOUT_SECONDS", 60)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


def get_settings() -> Settings:
    return Settings()
