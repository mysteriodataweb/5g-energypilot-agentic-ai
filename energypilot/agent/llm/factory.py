"""Build the configured LLM client from environment variables (.env supported).

``LLM_PROVIDER=none`` (the default) returns ``None``: the agent then runs its
deterministic fallback workflow and says so in every report.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from energypilot.agent.llm.base import LLMClient, LLMConfigurationError, Pricing
from energypilot.config.settings import AppConfig

logger = logging.getLogger(__name__)


def _float_env(name: str) -> float | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise LLMConfigurationError(f"{name} must be a number") from exc


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    timeout_s: float
    max_tokens: int
    max_retries: int
    effort: str | None
    pricing: Pricing

    @classmethod
    def from_env(cls, config: AppConfig, *, load_dotenv_file: bool = True) -> "LLMSettings":
        if load_dotenv_file:
            try:
                from dotenv import load_dotenv

                load_dotenv(override=False)
            except ImportError:  # pragma: no cover
                pass
        timeout = _float_env("LLM_TIMEOUT_S")
        max_tokens = _float_env("LLM_MAX_TOKENS")
        return cls(
            provider=os.getenv("LLM_PROVIDER", "none").strip().lower() or "none",
            model=os.getenv("LLM_MODEL", "").strip(),
            timeout_s=timeout if timeout is not None else config.agent.llm_timeout_s,
            max_tokens=int(max_tokens) if max_tokens is not None else config.agent.max_tokens,
            max_retries=config.agent.llm_max_retries,
            effort=os.getenv("LLM_EFFORT", "").strip() or None,
            pricing=Pricing(_float_env("LLM_PRICE_INPUT_PER_MTOK"), _float_env("LLM_PRICE_OUTPUT_PER_MTOK")),
        )


def create_llm_client(settings: LLMSettings) -> LLMClient | None:
    """Return a client, or ``None`` when no LLM is configured. Raises on invalid configuration."""
    if settings.provider in ("none", "", "off", "fallback"):
        return None
    if settings.provider == "anthropic":
        from energypilot.agent.llm.anthropic_client import AnthropicClient

        return AnthropicClient(
            settings.model,
            api_key=os.getenv("ANTHROPIC_API_KEY") or None,
            max_retries=settings.max_retries,
            effort=settings.effort,
        )
    if settings.provider in ("openai", "openai_compatible"):
        from energypilot.agent.llm.openai_compat import OpenAICompatibleClient

        return OpenAICompatibleClient(
            settings.model,
            api_key=os.getenv("OPENAI_API_KEY") or None,
            base_url=os.getenv("OPENAI_BASE_URL") or None,
            max_retries=settings.max_retries,
            effort=settings.effort,
        )
    raise LLMConfigurationError(f"unknown LLM_PROVIDER '{settings.provider}' (expected none, anthropic or openai)")


def try_create_llm_client(config: AppConfig) -> tuple[LLMClient | None, LLMSettings, str | None]:
    """Like :func:`create_llm_client` but never raises: returns (client, settings, error)."""
    settings = LLMSettings.from_env(config)
    try:
        return create_llm_client(settings), settings, None
    except LLMConfigurationError as exc:
        logger.warning("LLM disabled: %s", exc)
        return None, settings, str(exc)
