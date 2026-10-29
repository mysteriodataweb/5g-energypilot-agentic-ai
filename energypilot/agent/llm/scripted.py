"""Scripted LLM test double.

It replays pre-written responses (or raises pre-defined errors) so that the
agent's LLM code path - tool loop, timeouts, malformed tool calls, refusals -
can be tested deterministically without any external API.

It is NOT a language model: ``is_real_llm`` is False and the agent labels every
report produced with it accordingly. It must never be used to produce results
presented as LLM results.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Union

from energypilot.agent.llm.base import LLMResponse, Message, ToolCall, ToolSpec, Usage


@dataclass
class ScriptedTurn:
    """One scripted answer: optional text and tool calls (name, arguments)."""

    tool_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    text: str = ""
    usage: Usage = Usage(100, 50)
    latency_s: float = 0.0


ScriptItem = Union[ScriptedTurn, Exception, Callable[[str, list[Message], list[ToolSpec]], ScriptedTurn]]


class ScriptedLLMClient:
    provider = "scripted-test-double"
    is_real_llm = False

    def __init__(self, script: list[ScriptItem], model: str = "scripted") -> None:
        self.script = list(script)
        self.model = model
        self.calls: list[dict[str, Any]] = []
        self._ids = itertools.count()

    def chat(self, system: str, messages: list[Message], tools: list[ToolSpec], *, max_tokens: int,
             timeout_s: float) -> LLMResponse:  # fmt: skip
        self.calls.append({"system": system, "messages": list(messages), "tools": [t.name for t in tools]})
        if not self.script:
            raise AssertionError("scripted LLM exhausted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        turn = item(system, messages, tools) if callable(item) else item
        if turn.latency_s:
            time.sleep(turn.latency_s)
        calls = [ToolCall(id=f"call_{next(self._ids)}", name=name, arguments=args) for name, args in turn.tool_calls]
        return LLMResponse(
            text=turn.text,
            tool_calls=calls,
            stop_reason="tool_use" if calls else "end_turn",
            usage=turn.usage,
            latency_s=turn.latency_s,
            provider=self.provider,
            model=self.model,
            assistant_message=Message(role="assistant", text=turn.text, provider_payload={"tool_calls": turn.tool_calls}),
        )
