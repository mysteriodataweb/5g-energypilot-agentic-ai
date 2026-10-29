"""Anthropic Messages API adapter (official ``anthropic`` SDK).

* Tools are declared with ``strict: true`` so arguments match their schema.
  Forced ``tool_choice`` is not used (recent models reject it); the agent
  states which tool to call in its prompt and checks that a call was made.
* Assistant turns are replayed with their original content blocks, which keeps
  any reasoning blocks attached to the tool calls that follow them.
* ``refusal`` and ``max_tokens`` stop reasons are surfaced as errors so the
  agent falls back to its deterministic workflow instead of acting on them.
* No sampling parameters are sent.
"""

from __future__ import annotations

import time
from typing import Any

from energypilot.agent.llm.base import (
    LLMConfigurationError,
    LLMProviderError,
    LLMResponse,
    LLMResponseError,
    LLMTimeoutError,
    Message,
    ToolCall,
    ToolSpec,
    Usage,
)


class AnthropicClient:
    provider = "anthropic"
    is_real_llm = True

    def __init__(self, model: str, *, api_key: str | None = None, max_retries: int = 1, effort: str | None = None,
                 base_url: str | None = None) -> None:  # fmt: skip
        if not model:
            raise LLMConfigurationError("LLM_MODEL must be set for the anthropic provider")
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise LLMConfigurationError("the 'anthropic' package is not installed (pip install anthropic)") from exc
        self._sdk = anthropic
        kwargs: dict[str, Any] = {"max_retries": max_retries}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        try:
            self._client = anthropic.Anthropic(**kwargs)
        except anthropic.AnthropicError as exc:
            raise LLMConfigurationError(f"cannot create Anthropic client: {exc}") from exc
        self.model = model
        self.effort = effort

    @staticmethod
    def _to_api_messages(messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "user":
                out.append({"role": "user", "content": m.text or ""})
            elif m.role == "assistant":
                out.append({"role": "assistant", "content": m.provider_payload if m.provider_payload is not None else (m.text or "")})
            else:  # all tool results of one turn go back in a single user message
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": r.call_id, "content": r.content, "is_error": r.is_error}
                            for r in m.tool_results
                        ],
                    }
                )
        return out

    @staticmethod
    def _to_api_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
        return [{"name": t.name, "description": t.description, "input_schema": t.parameters, "strict": True} for t in tools]

    def chat(self, system: str, messages: list[Message], tools: list[ToolSpec], *, max_tokens: int,
             timeout_s: float) -> LLMResponse:  # fmt: skip
        sdk = self._sdk
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": self._to_api_messages(messages),
        }
        if tools:
            params["tools"] = self._to_api_tools(tools)
        if self.effort:
            params["output_config"] = {"effort": self.effort}
        started = time.perf_counter()
        try:
            response = self._client.with_options(timeout=timeout_s).messages.create(**params)
        except sdk.APITimeoutError as exc:
            raise LLMTimeoutError(f"no answer within {timeout_s:g} s") from exc
        except sdk.RateLimitError as exc:
            raise LLMProviderError("rate limited by the provider") from exc
        except sdk.APIStatusError as exc:
            raise LLMProviderError(f"API error {exc.status_code}: {exc.message}") from exc
        except sdk.APIConnectionError as exc:
            raise LLMProviderError(f"connection error: {exc}") from exc
        latency = time.perf_counter() - started

        if response.stop_reason == "refusal":
            raise LLMResponseError("the model declined the request")
        if response.stop_reason == "max_tokens":
            raise LLMResponseError("the answer was truncated (max_tokens reached)")
        text = "".join(b.text for b in response.content if b.type == "text")
        calls = [
            ToolCall(id=b.id, name=b.name, arguments=b.input if isinstance(b.input, dict) else {},
                     parse_error=None if isinstance(b.input, dict) else "tool input is not an object")  # fmt: skip
            for b in response.content
            if b.type == "tool_use"
        ]
        return LLMResponse(
            text=text,
            tool_calls=calls,
            stop_reason=str(response.stop_reason),
            usage=Usage(response.usage.input_tokens, response.usage.output_tokens),
            latency_s=latency,
            provider=self.provider,
            model=getattr(response, "model", self.model) or self.model,
            assistant_message=Message(role="assistant", text=text, provider_payload=response.content),
        )
