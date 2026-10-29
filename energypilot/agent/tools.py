"""Typed decision tools exposed to the orchestrator (LLM or deterministic).

The tool surface is deliberately small and read-mostly:

    get_current_state, get_active_constraints, run_optimizer,
    run_rule_controller, validate_strategy, diagnose_infeasibility

No tool can modify the battery state, edit a strategy, change the validated
instruction or skip validation. Strategies are created only by the numerical
tools and referenced by ID. Every call - successful or not - is recorded in
the audit log with its arguments, duration and result summary.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from energypilot.agent.llm.base import ToolSpec
from energypilot.config.settings import AppConfig
from energypilot.controllers.rule_based import RuleBasedController
from energypilot.core.types import Forecast, Observation, Strategy
from energypilot.data.profiles import tariff_label
from energypilot.instructions.constraints import compile_constraints
from energypilot.instructions.schema import OperationalInstruction
from energypilot.optimizer.diagnosis import ConflictReport, diagnose
from energypilot.optimizer.milp import OptimizationResult, OptimizerError, optimize
from energypilot.safety.audit import AuditLogger
from energypilot.safety.validator import SafetyValidator, ValidationReport


class ToolFailure(RuntimeError):
    """A decision tool could not complete (e.g. solver crash, injected fault)."""


@dataclass(frozen=True)
class DecisionContext:
    """Everything known when an instruction arrives (no future measurements)."""

    config: AppConfig
    observation: Observation
    forecast: Forecast
    active_instruction: OperationalInstruction
    terminal_target_kwh: float | None

    @property
    def issued_at_minute(self) -> int:
        return self.observation.step * self.config.simulation.time_step_minutes


@dataclass
class FaultInjection:
    """Deterministic fault injection used by robustness experiments and tests."""

    optimizer_fails: bool = False
    rule_controller_fails: bool = False


class _NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _StrategyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy_id: str = Field(strict=True, min_length=1, max_length=64)


class ToolCallRecord(BaseModel):
    seq: int
    tool: str
    caller: str
    arguments: dict[str, Any]
    ok: bool
    error: str | None = None
    duration_s: float
    result: dict[str, Any] = Field(default_factory=dict)


_EMPTY = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
_STRATEGY = {
    "type": "object",
    "properties": {"strategy_id": {"type": "string", "description": "ID returned by run_optimizer or run_rule_controller"}},
    "required": ["strategy_id"],
    "additionalProperties": False,
}

TOOL_SPECS: dict[str, ToolSpec] = {
    "get_current_state": ToolSpec(
        "get_current_state",
        "Read the simulated site state now: time, battery SoC, PV, traffic, price, active instruction and limits.",
        _EMPTY,
    ),
    "get_active_constraints": ToolSpec(
        "get_active_constraints",
        "Read the validated candidate instruction (objective, reserves, grid limits, QoS) and the site's hard limits.",
        _EMPTY,
    ),
    "run_optimizer": ToolSpec(
        "run_optimizer",
        "Solve the MILP schedule for the rest of the day under the validated instruction. Returns solver status, "
        "objective value, planned KPIs and a strategy_id when feasible. It cannot relax any constraint.",
        _EMPTY,
    ),
    "run_rule_controller": ToolSpec(
        "run_rule_controller",
        "Roll out the interpretable rule-based controller on the forecast. Returns planned KPIs and a strategy_id. "
        "It does not guarantee feasibility: validate the result.",
        _EMPTY,
    ),
    "validate_strategy": ToolSpec(
        "validate_strategy",
        "Run the independent safety validator on a strategy. Returns accepted/rejected and the failed checks.",
        _STRATEGY,
    ),
    "diagnose_infeasibility": ToolSpec(
        "diagnose_infeasibility",
        "Explain why the validated instruction cannot be satisfied (conflicting requirements, physical limits).",
        _EMPTY,
    ),
}


def _hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


class AgentToolbox:
    """Tool implementations bound to one decision (one instruction, one context)."""

    def __init__(
        self,
        ctx: DecisionContext,
        instruction: OperationalInstruction,
        *,
        validator: SafetyValidator,
        audit: AuditLogger,
        faults: FaultInjection | None = None,
    ) -> None:
        self.ctx = ctx
        self.instruction = instruction
        self.validator = validator
        self.audit = audit
        self.faults = faults or FaultInjection()
        self.constraints = compile_constraints(instruction, ctx.config)
        self.strategies: dict[str, Strategy] = {}
        self.validations: dict[str, ValidationReport] = {}
        self.optimization: OptimizationResult | None = None
        self.conflict: ConflictReport | None = None
        self.records: list[ToolCallRecord] = []
        self._impl: dict[str, tuple[type[BaseModel], Callable[..., dict[str, Any]]]] = {
            "get_current_state": (_NoArgs, self.get_current_state),
            "get_active_constraints": (_NoArgs, self.get_active_constraints),
            "run_optimizer": (_NoArgs, self.run_optimizer),
            "run_rule_controller": (_NoArgs, self.run_rule_controller),
            "validate_strategy": (_StrategyArgs, self.validate_strategy),
            "diagnose_infeasibility": (_NoArgs, self.diagnose_infeasibility),
        }

    # ------------------------------------------------------------- dispatch
    def specs(self) -> list[ToolSpec]:
        return list(TOOL_SPECS.values())

    def call(self, name: str, arguments: dict[str, Any] | None, *, caller: str) -> ToolCallRecord:
        arguments = arguments or {}
        started = time.perf_counter()
        ok, error, result = True, None, {}
        if name not in self._impl:
            ok, error = False, f"unknown tool '{name}'"
        else:
            model, fn = self._impl[name]
            try:
                args = model.model_validate(arguments)
                result = fn(**args.model_dump())
            except ValidationError as exc:
                ok, error = False, f"invalid arguments: {exc.errors()[0]['msg']}"
            except (ToolFailure, OptimizerError, KeyError) as exc:
                ok, error = False, f"{type(exc).__name__}: {exc}"
        record = ToolCallRecord(
            seq=len(self.records), tool=name, caller=caller, arguments=arguments, ok=ok, error=error,
            duration_s=time.perf_counter() - started, result=result,
        )  # fmt: skip
        self.records.append(record)
        self.audit.log("tool_call", **record.model_dump())
        return record

    # ---------------------------------------------------------------- tools
    def get_current_state(self) -> dict[str, Any]:
        cfg, obs = self.ctx.config, self.ctx.observation
        bat = cfg.battery
        return {
            "time": _hhmm(self.ctx.issued_at_minute),
            "soc_pct": round(obs.soc_pct, 2),
            "soc_kwh": round(obs.soc_kwh, 3),
            "pv_kw": round(obs.pv_kw, 3),
            "traffic_normalised": round(obs.traffic, 3),
            "price_eur_per_kwh": obs.price_eur_per_kwh,
            "tariff_period": tariff_label(obs.time_h + cfg.simulation.dt_h / 2, cfg.tariff),
            "battery_limits": {
                "capacity_kwh": bat.capacity_kwh,
                "soc_min_pct": bat.soc_min_pct,
                "soc_max_pct": bat.soc_max_pct,
                "max_charge_kw": bat.max_charge_kw,
                "max_discharge_kw": bat.max_discharge_kw,
            },
            "active_instruction": self.ctx.active_instruction.requirement_dict(),
            "data_note": "synthetic simulation state, not a real site",
        }

    def get_active_constraints(self) -> dict[str, Any]:
        instr, cfg = self.instruction, self.ctx.config
        return {
            "objective": instr.objective.value,
            "reserves": [
                {"min_soc_pct": r.min_soc_pct, "start": _hhmm(r.start_minute), "end": _hhmm(r.end_minute)}
                for r in instr.reserves
            ],
            "grid_limits": [
                {"max_import_kw": g.max_import_kw, "start": _hhmm(g.start_minute), "end": _hhmm(g.end_minute)}
                for g in instr.grid_limits
            ],
            "qos_level": instr.qos_level.value,
            "max_utilization": self.constraints.max_utilization,
            "allow_capacity_reduction": instr.allow_capacity_reduction,
            "defaults_applied": list(instr.defaults_applied),
            "grid_connection_limit_kw": cfg.grid.max_import_kw,
            "end_of_day_soc_target_pct": (
                round(cfg.battery.kwh_to_pct(self.ctx.terminal_target_kwh), 2)
                if self.ctx.terminal_target_kwh is not None
                else None
            ),
        }

    def _register(self, strategy: Strategy) -> None:
        self.strategies[strategy.strategy_id] = strategy

    def _plan_kpis(self, s: Strategy) -> dict[str, float]:
        cfg = self.ctx.config
        dt = cfg.simulation.dt_h
        return {
            "planned_cost_eur": round(float(np.sum(s.grid_import_kw * s.forecast.price_eur_per_kwh) * dt), 4),
            "planned_grid_kwh": round(float(np.sum(s.grid_import_kw) * dt), 3),
            "planned_min_soc_pct": round(cfg.battery.kwh_to_pct(float(np.min(s.soc_kwh))), 2),
            "planned_final_soc_pct": round(cfg.battery.kwh_to_pct(float(s.soc_kwh[-1])), 2),
            "planned_peak_import_kw": round(float(np.max(s.grid_import_kw)), 3),
        }

    def run_optimizer(self) -> dict[str, Any]:
        if self.faults.optimizer_fails:
            raise ToolFailure("injected optimizer fault")
        result = optimize(
            self.ctx.config,
            self.constraints,
            self.ctx.forecast,
            self.ctx.observation.soc_kwh,
            terminal_target_kwh=self.ctx.terminal_target_kwh,
        )
        self.optimization = result
        self.audit.log("solver_result", **result.summary())
        if result.feasible:
            self._register(result.strategy)
            return {**result.summary(), **self._plan_kpis(result.strategy)}
        return result.summary()

    def run_rule_controller(self) -> dict[str, Any]:
        if self.faults.rule_controller_fails:
            raise ToolFailure("injected rule-controller fault")
        ctrl = RuleBasedController(self.ctx.config, self.constraints)
        plan = ctrl.plan(self.ctx.observation, self.ctx.forecast)
        self._register(plan)
        return {"strategy_id": plan.strategy_id, "source": plan.source, **self._plan_kpis(plan)}

    def validate(self, strategy_id: str) -> ValidationReport:
        if strategy_id not in self.strategies:
            raise KeyError(f"unknown strategy_id '{strategy_id}'")
        report = self.validator.validate(
            self.strategies[strategy_id],
            self.instruction,
            current_energy_kwh=self.ctx.observation.soc_kwh,
            reference_forecast=self.ctx.forecast,
            terminal_target_kwh=self.ctx.terminal_target_kwh,
        )
        self.validations[strategy_id] = report
        self.audit.log("validation_result", **report.summary())
        return report

    def validate_strategy(self, strategy_id: str) -> dict[str, Any]:
        return self.validate(strategy_id).summary()

    def diagnose_infeasibility(self) -> dict[str, Any]:
        self.conflict = diagnose(
            self.ctx.config,
            self.constraints,
            self.ctx.forecast,
            self.ctx.observation.soc_kwh,
            terminal_target_kwh=self.ctx.terminal_target_kwh,
        )
        self.audit.log("infeasibility_diagnosis", **self.conflict.to_dict())
        return self.conflict.to_dict()
