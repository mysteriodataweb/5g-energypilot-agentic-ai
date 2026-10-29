"""Closed-loop simulation of one (scenario, seed, approach) episode.

Timeline of an episode (identical for every approach):

1. 00:00 - the default operating instruction is active; the approach's primary
   controller (rule-based or MPC) runs.
2. At the scenario's issue time the instruction is submitted to the approach's
   decision pipeline. If accepted, the selected controller takes over with the
   new instruction; otherwise the previous instruction stays in force.
3. Every action passes through the same runtime shield before the simulator.

The simulator runs on *realised* profiles; controllers only see forecasts.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from energypilot.agent.orchestrator import DecisionPipeline
from energypilot.agent.report import DecisionReport
from energypilot.agent.tools import DecisionContext
from energypilot.config.settings import AppConfig
from energypilot.controllers.mpc import MPCController
from energypilot.controllers.rule_based import RuleBasedController
from energypilot.core.types import Strategy
from energypilot.data.forecast import ForecastProvider
from energypilot.data.scenarios import ScenarioData
from energypilot.evaluation.metrics import compute_metrics
from energypilot.instructions.constraints import CompiledConstraints, compile_constraints
from energypilot.instructions.schema import OperationalInstruction, default_instruction
from energypilot.safety.audit import AuditLogger
from energypilot.safety.shield import RuntimeShield
from energypilot.safety.validator import SafetyValidator
from energypilot.simulator.environment import BaseStationEnv

InputMode = str  # "natural_language" | "structured" | "paraphrase_<k>"


@dataclass
class EpisodeResult:
    scenario: str
    seed: int
    approach: str
    input_mode: str
    history: pd.DataFrame
    decision: DecisionReport
    metrics: dict[str, Any]
    audit: AuditLogger
    scenario_data: ScenarioData


def _controller(kind: str, config: AppConfig, instruction: OperationalInstruction, terminal_kwh: float,
                validator: SafetyValidator, audit: AuditLogger, initial_plan: Strategy | None = None):  # fmt: skip
    constraints = compile_constraints(instruction, config)
    if kind == "rule_based":
        return RuleBasedController(config, constraints), constraints
    return (
        MPCController(
            config,
            constraints,
            terminal_target_kwh=terminal_kwh,
            plan_validator=lambda s: validator.validate(s, instruction, terminal_target_kwh=terminal_kwh),
            event_sink=lambda event, payload: audit.log(event, **payload),
            initial_plan=initial_plan,
        ),
        constraints,
    )


def instruction_input(scenario: ScenarioData, input_mode: InputMode) -> str | dict:
    """Instruction submitted in a given input mode.

    * ``natural_language``: the scenario's canonical text;
    * ``paraphrase_<k>``: the k-th held-out rephrasing (same ground truth);
    * ``structured``: the ground-truth JSON. Scenarios whose correct outcome is a
      clarification have no ground truth and are submitted as text (documented).
    """
    if input_mode == "structured" and scenario.spec.ground_truth is not None:
        return dict(scenario.spec.ground_truth)
    if input_mode.startswith("paraphrase_"):
        return scenario.spec.paraphrases[int(input_mode.split("_", 1)[1])]
    if input_mode not in ("natural_language", "structured"):
        raise ValueError(f"unknown input mode '{input_mode}'")
    return scenario.spec.instruction_text


def run_episode(
    config: AppConfig,
    scenario: ScenarioData,
    pipeline: DecisionPipeline,
    *,
    input_mode: InputMode = "natural_language",
    forecast_mode: str | None = None,
    audit_dir: str | Path | None = None,
) -> EpisodeResult:
    run_id = f"{scenario.spec.name}-s{scenario.seed}-{pipeline.name}-{input_mode}"
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", run_id)
    audit_path = Path(audit_dir) / f"{safe_name}.jsonl" if audit_dir is not None else None
    if audit_path is not None and audit_path.exists():
        audit_path.unlink()
    audit = AuditLogger(run_id=run_id, path=audit_path)
    audit.log("run_started", scenario=scenario.spec.name, seed=scenario.seed, approach=pipeline.name,
              input_mode=input_mode, data="synthetic", provenance=scenario.provenance)  # fmt: skip

    env = BaseStationEnv(config, scenario.pv_actual_kw, scenario.traffic_actual, scenario.price_eur_per_kwh,
                         scenario.initial_soc_pct)  # fmt: skip
    forecasts = ForecastProvider(scenario, config, mode=forecast_mode)
    validator = SafetyValidator(config)
    shield = RuntimeShield(config)
    terminal_kwh = env.initial_soc_kwh
    active = default_instruction(0)
    default_kind = "optimizer_mpc" if pipeline.policy.primary == "optimizer" else "rule_based"
    controller, constraints = _controller(default_kind, config, active, terminal_kwh, validator, audit)
    decision: DecisionReport | None = None
    rows: list[dict[str, Any]] = []
    decide_times: list[float] = []

    while not env.done:
        obs = env.observe()
        forecast = forecasts.forecast(obs.step)
        if obs.step == scenario.issue_step:
            ctx = DecisionContext(config, obs, forecast, active, terminal_kwh)
            outcome = pipeline.handle_instruction(instruction_input(scenario, input_mode), ctx, audit)
            decision = outcome.report
            if outcome.controller is not None:
                active = outcome.instruction
                controller, constraints = _controller(outcome.controller, config, active, terminal_kwh, validator,
                                                      audit, initial_plan=outcome.strategy)  # fmt: skip
                audit.log("instruction_activated", step=obs.step, controller=outcome.controller,
                          fingerprint=active.fingerprint())  # fmt: skip
        started = time.perf_counter()
        action = controller.decide(obs, forecast)
        decide_times.append(time.perf_counter() - started)
        safe, notes = shield.filter(obs, action, constraints)
        if notes:
            audit.log("shield_intervention", step=obs.step, notes=notes, requested=action, applied=safe)
        env.step(safe)
        rows.append({"controller": controller.name, "active_fingerprint": constraints.fingerprint,
                     "shield_notes": ";".join(notes)})  # fmt: skip

    history = pd.concat([env.history(), pd.DataFrame(rows)], axis=1)
    if decision is None:  # pragma: no cover - issue step is always within the horizon
        raise RuntimeError("the instruction was never issued")
    active_by_fp: dict[str, CompiledConstraints] = {
        compile_constraints(default_instruction(0), config).fingerprint: compile_constraints(default_instruction(0), config)
    }
    if decision.selected_controller is not None:
        active_by_fp[constraints.fingerprint] = constraints
    metrics = compute_metrics(
        config, scenario, history, decision, active_by_fp,
        decide_times=np.asarray(decide_times), mpc_stats=getattr(controller, "stats", None),
    )  # fmt: skip
    metrics.update({"approach": pipeline.name, "input_mode": input_mode, "run_id": run_id})
    audit.log("run_completed", **{k: v for k, v in metrics.items() if not isinstance(v, (list, dict))})
    return EpisodeResult(scenario.spec.name, scenario.seed, pipeline.name, input_mode, history, decision, metrics, audit, scenario)
