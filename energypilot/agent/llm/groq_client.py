"""Groq Chat Completions adapter (official ``groq`` SDK).

* Tools are sent as standard function definitions. ``strict`` is not sent
  (not every Groq model accepts it), so arguments are always validated again
  by the agent.
* Assistant turns are replayed with only ``role``, ``content`` and
  ``tool_calls``: Groq rejects output-only fields (e.g. ``reasoning``) in input
  messages.
* ``length`` and ``content_filter`` finish reasons are surfaced as errors so the
  agent falls back to its deterministic workflow instead of acting on them.
* No sampling parameters are sent.
* TLS uses the operating-system trust store when ``truststore`` is installed, so
  HTTPS-inspecting antivirus or corporate proxies (whose root certificate is in
  the OS store but not in ``certifi``) do not break the connection.
"""

from __future__ import annotations

import json
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


class GroqClient:
    provider = "groq"
    is_real_llm = True

    def __init__(self, model: str, *, api_key: str | None = None, base_url: str | None = None,
                 max_retries: int = 1, effort: str | None = None) -> None:  # fmt: skip
        if not model:
            raise LLMConfigurationError("LLM_MODEL must be set for the groq provider")
        try:
            import groq
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise LLMConfigurationError("the 'groq' package is not installed (pip install groq)") from exc
        self._sdk = groq
        kwargs: dict[str, Any] = {"max_retries": max_retries}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        try:
            import ssl

            import truststore

            kwargs["http_client"] = groq.DefaultHttpxClient(verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT))
        except ImportError:  # fall back to the SDK's default certificate bundle
            pass
        try:
            self._client = groq.Groq(**kwargs)
        except groq.GroqError as exc:
            raise LLMConfigurationError(f"cannot create Groq client: {exc}") from exc
        self.model = model
        self.effort = effort

    @staticmethod
    def _to_api_messages(system: str, messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for m in messages:
            if m.role == "user":
                out.append({"role": "user", "content": m.text or ""})
            elif m.role == "assistant":
                out.append(m.provider_payload if m.provider_payload is not None else {"role": "assistant", "content": m.text or ""})
            else:
                out.extend({"role": "tool", "tool_call_id": r.call_id, "content": r.content} for r in m.tool_results)
        return out

    @staticmethod
    def _to_api_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
        return [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in tools
        ]

    @staticmethod
    def _replay_payload(msg: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            payload["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"}}  # fmt: skip
                for tc in msg.tool_calls
            ]
        return payload

    def chat(self, system: str, messages: list[Message], tools: list[ToolSpec], *, max_tokens: int,
             timeout_s: float) -> LLMResponse:  # fmt: skip
        sdk = self._sdk
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": self._to_api_messages(system, messages),
        }
        if tools:
            params["tools"] = self._to_api_tools(tools)
        if self.effort:
            params["reasoning_effort"] = self.effort
        started = time.perf_counter()
        try:
            response = self._client.with_options(timeout=timeout_s).chat.completions.create(**params)
        except sdk.APITimeoutError as exc:
            raise LLMTimeoutError(f"no answer within {timeout_s:g} s") from exc
        except sdk.RateLimitError as exc:
            raise LLMProviderError("rate limited by the provider") from exc
        except sdk.APIStatusError as exc:
            raise LLMProviderError(f"API error {exc.status_code}: {exc.message}") from exc
        except sdk.APIConnectionError as exc:
            raise LLMProviderError(f"connection error: {exc}") from exc
        latency = time.perf_counter() - started
        if not response.choices:
            raise LLMResponseError("empty response")
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise LLMResponseError("the answer was truncated (max_tokens reached)")
        if choice.finish_reason == "content_filter":
            raise LLMResponseError("the provider filtered the answer")
        msg = choice.message
        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
                error = None if isinstance(args, dict) else "tool arguments are not a JSON object"
            except json.JSONDecodeError as exc:
                args, error = {}, f"invalid JSON arguments: {exc.msg}"
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args if isinstance(args, dict) else {}, parse_error=error))
        usage = Usage(
            getattr(response.usage, "prompt_tokens", 0) or 0, getattr(response.usage, "completion_tokens", 0) or 0
        )
        return LLMResponse(
            text=msg.content or "",
            tool_calls=calls,
            stop_reason=str(choice.finish_reason),
            usage=usage,
            latency_s=latency,
            provider=self.provider,
            model=getattr(response, "model", self.model) or self.model,
            assistant_message=Message(role="assistant", text=msg.content or "", provider_payload=self._replay_payload(msg)),
        )
