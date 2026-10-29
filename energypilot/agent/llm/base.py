"""Provider-neutral LLM interface used by the orchestration agent.

Only this small surface is visible to the agent: a ``chat`` call with a system
prompt, a neutral message history and JSON-Schema tool definitions, returning
text, tool calls, token usage and latency. Provider SDKs are imported lazily by
the adapters, so the prototype runs without any of them installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


class LLMError(Exception):
    """Base class: any failure of the language-model layer (always recoverable by fallback)."""


class LLMConfigurationError(LLMError):
    """Missing SDK, key or model identifier."""


class LLMTimeoutError(LLMError):
    """The provider did not answer within the configured timeout."""


class LLMProviderError(LLMError):
    """Transport or API error (rate limit, 5xx, connection, authentication...)."""


class LLMResponseError(LLMError):
    """The model answered but the answer is unusable (refusal, truncation, malformed tool call)."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema (object, additionalProperties: false)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    parse_error: str | None = None


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    name: str
    content: str  # JSON text
    is_error: bool = False


@dataclass
class Message:
    """Neutral conversation message.

    ``provider_payload`` holds the provider's own representation of an
    assistant turn; adapters replay it verbatim (some providers require that,
    e.g. to keep reasoning blocks attached to tool calls).
    """

    role: Literal["user", "assistant", "tool_results"]
    text: str | None = None
    tool_results: list[ToolResult] = field(default_factory=list)
    provider_payload: Any = None


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    stop_reason: str
    usage: Usage
    latency_s: float
    provider: str
    model: str
    assistant_message: Message


class LLMClient(Protocol):
    provider: str
    model: str
    is_real_llm: bool  # False for test doubles: their output must never be reported as LLM results

    def chat(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        max_tokens: int,
        timeout_s: float,
    ) -> LLMResponse: ...


@dataclass(frozen=True)
class Pricing:
    """Optional, user-supplied prices (per million tokens). ``None`` = unknown."""

    input_per_mtok: float | None = None
    output_per_mtok: float | None = None

    def estimate(self, usage: Usage) -> float | None:
        if self.input_per_mtok is None or self.output_per_mtok is None:
            return None
        return (usage.input_tokens * self.input_per_mtok + usage.output_tokens * self.output_per_mtok) / 1e6
