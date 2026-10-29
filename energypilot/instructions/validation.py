"""Semantic validation of instructions and the structured-JSON interface.

Schema validation (types, ranges, unknown keys) is done by pydantic. This
module adds checks that depend on the site configuration and the issue time
(permitted operating range, horizon, elapsed windows) and turns the result
into a :class:`ParseResult` with an explicit status:

* ``rejected``: invalid, out of the permitted range, or unsupported;
* ``needs_clarification``: information missing or ambiguous;
* ``ok``: valid (warnings may still be attached and are logged).
"""

from __future__ import annotations

import json
import time
from typing import Any

from pydantic import ValidationError

from energypilot.config.settings import AppConfig
from energypilot.instructions.schema import (
    Issue,
    OperationalInstruction,
    ParseResult,
    ParseStatus,
)

DEFAULTABLE_FIELDS = ("objective", "qos_level", "allow_capacity_reduction")


def semantic_issues(instr: OperationalInstruction, config: AppConfig) -> list[Issue]:
    """Checks against the site's permitted operating range and the horizon."""
    issues: list[Issue] = []
    bat = config.battery
    horizon_min = config.simulation.horizon_hours * 60
    issued = instr.issued_at_minute
    if issued >= horizon_min:
        issues.append(Issue(code="issued_after_horizon", message="instruction issued after the horizon", severity="error"))

    def _window_checks(kind: str, idx: int, start: int, end: int) -> None:
        name = f"{kind}[{idx}]"
        if end > horizon_min:
            issues.append(
                Issue(
                    code="window_beyond_horizon",
                    field=name,
                    severity="error",
                    message=f"{name} ends at {_hhmm(end)}, after the {config.simulation.horizon_hours} h horizon",
                )
            )
        if end <= issued:
            issues.append(
                Issue(
                    code="window_elapsed",
                    field=name,
                    severity="error",
                    message=f"{name} ({_hhmm(start)}-{_hhmm(end)}) has already elapsed at {_hhmm(issued)}",
                )
            )
        elif start < issued:
            issues.append(
                Issue(
                    code="window_partially_elapsed",
                    field=name,
                    severity="warning",
                    message=f"{name} started before the issue time; it applies from {_hhmm(issued)}",
                )
            )

    for i, r in enumerate(instr.reserves):
        _window_checks("reserves", i, r.start_minute, r.end_minute)
        if r.min_soc_pct > bat.soc_max_pct:
            issues.append(
                Issue(
                    code="reserve_above_soc_max",
                    field=f"reserves[{i}].min_soc_pct",
                    severity="error",
                    message=(
                        f"requested reserve {r.min_soc_pct:g}% exceeds the battery's maximum "
                        f"permitted SoC ({bat.soc_max_pct:g}%)"
                    ),
                )
            )
        elif r.min_soc_pct <= bat.soc_min_pct:
            issues.append(
                Issue(
                    code="reserve_redundant",
                    field=f"reserves[{i}].min_soc_pct",
                    severity="warning",
                    message=f"reserve {r.min_soc_pct:g}% is not above the BMS minimum ({bat.soc_min_pct:g}%)",
                )
            )
    for i, g in enumerate(instr.grid_limits):
        _window_checks("grid_limits", i, g.start_minute, g.end_minute)
        if g.max_import_kw >= config.grid.max_import_kw:
            issues.append(
                Issue(
                    code="grid_limit_redundant",
                    field=f"grid_limits[{i}].max_import_kw",
                    severity="warning",
                    message="grid limit is not below the physical connection limit",
                )
            )
    return issues


def build_result(
    instruction: OperationalInstruction | None,
    issues: list[Issue],
    *,
    method: str,
    raw_input: str,
    config: AppConfig,
    started: float | None = None,
    clarification_question: str | None = None,
) -> ParseResult:
    """Combine schema/semantic issues into a ParseResult with an explicit status."""
    issues = list(issues)
    if instruction is not None:
        issues.extend(semantic_issues(instruction, config))
    if any(i.severity == "error" for i in issues):
        status = ParseStatus.REJECTED
    elif instruction is None or any(i.severity == "clarification" for i in issues):
        status = ParseStatus.NEEDS_CLARIFICATION
    else:
        status = ParseStatus.OK
    if status == ParseStatus.NEEDS_CLARIFICATION and clarification_question is None:
        clarification_question = " ".join(i.message for i in issues if i.severity == "clarification") or None
    # A rejected/unclear instruction is kept for transparency; callers must check ``ok``.
    return ParseResult(
        status=status,
        instruction=instruction,
        issues=issues,
        method=method,
        raw_input=raw_input,
        clarification_question=clarification_question,
        latency_s=(time.perf_counter() - started) if started is not None else 0.0,
    )


def issues_from_validation_error(exc: ValidationError) -> list[Issue]:
    """Missing fields need clarification; any other schema violation is an error."""
    issues = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or None
        missing = err["type"] == "missing"
        issues.append(
            Issue(
                code="missing_field" if missing else f"schema_{err['type']}",
                field=loc,
                severity="clarification" if missing else "error",
                message=f"{loc}: {err['msg']}" if loc else err["msg"],
            )
        )
    return issues


def instruction_from_mapping(
    payload: dict[str, Any], issued_at_minute: int
) -> tuple[OperationalInstruction | None, list[Issue]]:
    """Validate a mapping against the schema, recording which defaults were applied."""
    data = dict(payload)
    data.pop("defaults_applied", None)
    data.setdefault("issued_at_minute", issued_at_minute)
    data["defaults_applied"] = tuple(f for f in DEFAULTABLE_FIELDS if f not in data)
    try:
        return OperationalInstruction.model_validate(data), []
    except ValidationError as exc:
        return None, issues_from_validation_error(exc)


def parse_structured(payload: str | dict[str, Any], issued_at_minute: int, config: AppConfig) -> ParseResult:
    """Structured JSON instruction interface (reliable fallback, no LLM involved)."""
    started = time.perf_counter()
    raw = payload if isinstance(payload, str) else json.dumps(payload, sort_keys=True)
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            issue = Issue(code="malformed_json", severity="error", message=f"invalid JSON: {exc.msg}")
            return build_result(None, [issue], method="structured_json", raw_input=raw, config=config, started=started)
    if not isinstance(payload, dict):
        issue = Issue(code="malformed_json", severity="error", message="instruction JSON must be an object")
        return build_result(None, [issue], method="structured_json", raw_input=raw, config=config, started=started)
    instruction, issues = instruction_from_mapping(payload, issued_at_minute)
    return build_result(instruction, issues, method="structured_json", raw_input=raw, config=config, started=started)


def _hhmm(minute: int) -> str:
    return f"{minute // 60:02d}:{minute % 60:02d}"
