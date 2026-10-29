"""Structured decision report produced for every instruction."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class DecisionStatus(StrEnum):
    ACCEPTED = "accepted"  # valid instruction, feasible strategy, validation passed
    NEEDS_CLARIFICATION = "needs_clarification"  # missing or ambiguous information
    REJECTED_INVALID = "rejected_invalid_instruction"  # out of range, unsupported, contradictory, unsafe
    INFEASIBLE = "infeasible"  # proven infeasible by the optimiser
    REJECTED_BY_VALIDATOR = "rejected_by_validator"  # every candidate strategy failed safety validation
    REJECTED_BY_AGENT = "rejected_by_agent"  # the LLM declined without proof of infeasibility
    TOOL_FAILURE = "tool_failure"  # decision tools failed; no valid strategy could be produced


EXPECTED_OUTCOME_STATUSES = {
    "accept": {DecisionStatus.ACCEPTED},
    "clarify": {DecisionStatus.NEEDS_CLARIFICATION},
    "reject": {
        DecisionStatus.REJECTED_INVALID,
        DecisionStatus.INFEASIBLE,
        DecisionStatus.REJECTED_BY_VALIDATOR,
        DecisionStatus.REJECTED_BY_AGENT,
    },
}


class LLMUsageSummary(BaseModel):
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    estimated_cost_eur: float | None = None  # None = unknown (no price configured), never guessed
    failures: list[str] = Field(default_factory=list)


class DecisionReport(BaseModel):
    approach: str
    run_id: str
    status: DecisionStatus
    instruction_text: str
    issued_at_minute: int
    interpretation_method: str
    interpretation_status: str
    interpretation_issues: list[dict[str, Any]] = Field(default_factory=list)
    parsed_instruction: dict[str, Any] | None = None
    orchestration_mode: str  # "llm" | "deterministic" | "llm->deterministic_fallback"
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_is_real: bool = False
    selected_controller: str | None = None  # "optimizer_mpc" | "rule_based" | None
    strategy_id: str | None = None
    solver_status: str | None = None
    validation: dict[str, Any] | None = None
    conflict_report: dict[str, Any] | None = None
    clarification_question: str | None = None
    explanation: str = ""
    explanation_source: str = "template"  # "template" (deterministic facts) or "llm"
    llm_explanation: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    fallback_reasons: list[str] = Field(default_factory=list)
    llm_decision_overridden: bool = False
    llm_usage: LLMUsageSummary = Field(default_factory=LLMUsageSummary)
    latency_s: float = 0.0
    active_instruction_fingerprint: str = ""

    @property
    def accepted(self) -> bool:
        return self.status == DecisionStatus.ACCEPTED

    def matches_expected(self, expected: str) -> bool:
        return self.status in EXPECTED_OUTCOME_STATUSES[expected]
