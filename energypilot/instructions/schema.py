"""Strict schema for operational instructions and parse results.

An :class:`OperationalInstruction` is the only form in which instructions reach
the controllers, the optimiser and the safety validator, whatever produced it
(deterministic parser, structured JSON, or an LLM). Numeric and boolean fields
are *strict*: ``"70"`` or ``"yes"`` are rejected rather than coerced, and
unknown keys are errors.

Times are minutes since 00:00 of the simulated day (step 0 starts at 00:00).
Windows are half-open ``[start_minute, end_minute)``.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Objective(StrEnum):
    MINIMIZE_COST = "minimize_cost"
    MINIMIZE_GRID_ENERGY = "minimize_grid_energy"
    MINIMIZE_PEAK_IMPORT = "minimize_peak_import"


class QoSLevel(StrEnum):
    STRICT = "strict"
    STANDARD = "standard"
    RELAXED = "relaxed"


_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class _Windowed(BaseModel):
    model_config = _MODEL_CONFIG

    start_minute: int = Field(strict=True, ge=0, description="minutes since 00:00, inclusive")
    end_minute: int = Field(strict=True, gt=0, description="minutes since 00:00, exclusive")

    @model_validator(mode="after")
    def _ordered(self):
        if self.end_minute <= self.start_minute:
            raise ValueError("end_minute must be after start_minute")
        return self


class ReserveConstraint(_Windowed):
    """Battery SoC must stay >= ``min_soc_pct`` throughout the window."""

    min_soc_pct: float = Field(strict=True, gt=0, le=100)


class GridImportLimit(_Windowed):
    """Grid import must stay <= ``max_import_kw`` throughout the window (e.g. demand response)."""

    max_import_kw: float = Field(strict=True, ge=0)


class OperationalInstruction(BaseModel):
    """Validated, structured operational instruction."""

    model_config = _MODEL_CONFIG

    objective: Objective = Objective.MINIMIZE_COST
    reserves: tuple[ReserveConstraint, ...] = ()
    grid_limits: tuple[GridImportLimit, ...] = ()
    qos_level: QoSLevel = QoSLevel.STANDARD
    max_utilization: float | None = Field(
        default=None, strict=True, gt=0, le=1, description="explicit override of the QoS level"
    )
    allow_capacity_reduction: bool = Field(default=True, strict=True)
    issued_at_minute: int = Field(default=0, strict=True, ge=0)
    defaults_applied: tuple[str, ...] = Field(
        default=(), description="fields set from documented defaults, not from the instruction text"
    )

    def requirement_dict(self) -> dict:
        """Operational content only (no provenance fields): used for comparisons."""
        data = self.model_dump(mode="json", exclude={"defaults_applied", "issued_at_minute"})
        data["reserves"] = sorted(data["reserves"], key=lambda r: (r["start_minute"], r["end_minute"], r["min_soc_pct"]))
        data["grid_limits"] = sorted(
            data["grid_limits"], key=lambda g: (g["start_minute"], g["end_minute"], g["max_import_kw"])
        )
        return data

    def fingerprint(self) -> str:
        """Stable hash of the operational content plus issue time.

        Strategies carry the fingerprint of the instruction they were computed
        for, so the validator can detect a plan built for a different (e.g.
        silently relaxed) instruction.
        """
        payload = {**self.requirement_dict(), "issued_at_minute": self.issued_at_minute}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def effective_max_utilization(self, levels: dict[str, float]) -> float:
        if self.max_utilization is not None:
            return self.max_utilization
        return levels[self.qos_level.value]


def default_instruction(issued_at_minute: int = 0) -> OperationalInstruction:
    """Operating policy in force before (or instead of) any accepted instruction."""
    return OperationalInstruction(
        issued_at_minute=issued_at_minute,
        defaults_applied=("objective", "qos_level", "allow_capacity_reduction"),
    )


class Issue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    message: str
    severity: Literal["error", "clarification", "warning"]
    field: str | None = None


class ParseStatus(StrEnum):
    OK = "ok"
    NEEDS_CLARIFICATION = "needs_clarification"
    REJECTED = "rejected"


class ParseResult(BaseModel):
    """Outcome of interpreting an instruction (any method)."""

    model_config = ConfigDict(extra="forbid")

    status: ParseStatus
    instruction: OperationalInstruction | None = None
    issues: list[Issue] = Field(default_factory=list)
    method: str
    raw_input: str
    clarification_question: str | None = None
    latency_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == ParseStatus.OK and self.instruction is not None

    def messages(self, severity: str | None = None) -> list[str]:
        return [i.message for i in self.issues if severity is None or i.severity == severity]
