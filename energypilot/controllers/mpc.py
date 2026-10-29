"""Receding-horizon (MPC) execution of the MILP optimiser.

Every ``replan_interval_steps`` the optimiser re-plans the remaining horizon
from the measured SoC and an updated forecast. Each new plan must pass the
injected validator before it replaces the current one. Between re-plans the
current plan is executed with deterministic real-time corrections for forecast
errors (never relaxing a constraint):

* capacity is raised if the measured traffic needs more than planned (QoS);
* PV surplus not anticipated by the plan is stored if the battery has room;
* if the measured grid import would exceed an active cap, charging is reduced
  and the battery discharges down to its floor.

If the optimiser fails or no valid plan covers the current step, the
rule-based controller takes over for that step (logged as a fallback).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from energypilot.config.settings import AppConfig
from energypilot.controllers.rule_based import RuleBasedController
from energypilot.core.types import ControlAction, Forecast, Observation, Strategy
from energypilot.instructions.constraints import CompiledConstraints
from energypilot.optimizer.milp import OptimizationResult, OptimizerError, optimize
from energypilot.simulator import physics

logger = logging.getLogger(__name__)

EventSink = Callable[[str, dict[str, Any]], None]
PlanValidator = Callable[[Strategy], Any]  # returns an object with ``accepted`` and ``summary()``


def realtime_adjust(
    obs: Observation, planned: ControlAction, constraints: CompiledConstraints, config: AppConfig
) -> ControlAction:
    """Correct planned set-points for the measured PV / traffic of the current step."""
    bat, dt = config.battery, config.simulation.dt_h
    capacity = max(planned.capacity_fraction, float(constraints.required_capacity(obs.traffic, config)))
    capacity = min(capacity, 1.0)
    net = float(physics.bs_power_kw(capacity, obs.traffic, config.base_station)) - obs.pv_kw
    floor_next = constraints.floor_kwh(obs.step + 1, bat.e_min_kwh)
    ch_max = physics.max_charge_kw(obs.soc_kwh, dt, bat)
    dis_max = physics.max_discharge_kw(obs.soc_kwh, dt, bat, floor_kwh=floor_next)
    p_ch = min(planned.p_charge_kw, ch_max)
    p_dis = min(planned.p_discharge_kw, dis_max, max(net, 0.0))
    grid = net + p_ch - p_dis
    if grid < 0:  # unexpected surplus: store it rather than curtail
        p_dis = max(p_dis + grid, 0.0)
        grid = net + p_ch - p_dis
        if grid < 0:
            p_ch = min(p_ch - grid, ch_max)
    cap = float(constraints.grid_cap_kw[obs.step])
    excess = net + p_ch - p_dis - cap
    if excess > 1e-9:
        cut = min(p_ch, excess)
        p_ch -= cut
        excess -= cut
        if excess > 1e-9 and net > 0:
            p_dis = min(p_dis + excess, dis_max, net)
    if p_ch > 1e-9 and p_dis > 1e-9:
        p_ch, p_dis = max(p_ch - p_dis, 0.0), max(p_dis - p_ch, 0.0)
    return ControlAction(p_charge_kw=max(p_ch, 0.0), p_discharge_kw=max(p_dis, 0.0), capacity_fraction=capacity)


class MPCController:
    name = "optimizer_mpc"

    def __init__(
        self,
        config: AppConfig,
        constraints: CompiledConstraints,
        *,
        terminal_target_kwh: float | None,
        plan_validator: PlanValidator | None = None,
        event_sink: EventSink | None = None,
        initial_plan: Strategy | None = None,
        optimize_fn: Callable[..., OptimizationResult] = optimize,
    ) -> None:
        self.config = config
        self.constraints = constraints
        self.terminal_target_kwh = terminal_target_kwh
        self.plan_validator = plan_validator
        self.event_sink = event_sink or (lambda event, payload: None)
        self.optimize_fn = optimize_fn
        self.fallback = RuleBasedController(config, constraints)
        self.plan = initial_plan
        self.plan_step = initial_plan.start_step if initial_plan is not None else None
        self.stats = {"replans": 0, "replan_failures": 0, "validation_rejections": 0, "fallback_steps": 0}

    def _replan(self, obs: Observation, forecast: Forecast) -> None:
        self.stats["replans"] += 1
        try:
            result = self.optimize_fn(
                self.config, self.constraints, forecast, obs.soc_kwh, terminal_target_kwh=self.terminal_target_kwh
            )
        except OptimizerError as exc:
            self.stats["replan_failures"] += 1
            self.event_sink("mpc_replan_failed", {"step": obs.step, "error": str(exc)})
            return
        self.event_sink("mpc_replan", {"step": obs.step, **result.summary()})
        if not result.feasible:
            self.stats["replan_failures"] += 1
            return
        if self.plan_validator is not None:
            report = self.plan_validator(result.strategy)
            self.event_sink("mpc_plan_validation", {"step": obs.step, **report.summary()})
            if not report.accepted:
                self.stats["validation_rejections"] += 1
                return
        self.plan, self.plan_step = result.strategy, obs.step

    def _plan_covers(self, step: int) -> bool:
        return self.plan is not None and self.plan.start_step <= step < self.plan.start_step + len(self.plan)

    def decide(self, obs: Observation, forecast: Forecast) -> ControlAction:
        if self.plan_step is None or obs.step - self.plan_step >= self.config.optimizer.replan_interval_steps:
            self._replan(obs, forecast)
        if not self._plan_covers(obs.step):
            self.stats["fallback_steps"] += 1
            self.event_sink("mpc_fallback_rule_based", {"step": obs.step})
            return self.fallback.decide(obs, forecast)
        return realtime_adjust(obs, self.plan.action_at(obs.step), self.constraints, self.config)
