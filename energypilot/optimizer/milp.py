"""Mixed-integer linear programme for battery and capacity scheduling (PuLP).

Model class: MILP. All relations are linear; one binary per step forbids
simultaneous charging and discharging (``battery_mode_binaries = false`` turns
it into a pure LP, where the degradation cost normally prevents simultaneity
but does not guarantee it). See docs/mathematical_model.md for the full model.

Decision variables (per step k of the remaining horizon):
    ch[k], dis[k] >= 0      battery charge / discharge power (kW, AC side)
    grid[k] >= 0            grid import (kW)
    curt[k] in [0, pv[k]]   PV curtailment (kW)
    cap[k] in [c_lb[k], 1]  active radio capacity fraction
    E[k+1]                  battery energy at the end of step k (kWh)

Hard constraints: bus balance, battery dynamics, SoC bounds, power limits,
grid connection limit, operational reserves, grid-import caps, QoS capacity.
Soft: end-of-horizon SoC target (penalised shortfall). No hard constraint is
ever relaxed by the solver: if it cannot be met the status is ``infeasible``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pulp

from energypilot.config.settings import AppConfig
from energypilot.core.types import Forecast, Strategy
from energypilot.instructions.constraints import CompiledConstraints
from energypilot.instructions.schema import Objective

logger = logging.getLogger(__name__)

RELAXABLE_GROUPS = ("reserve", "grid_cap", "qos")
# Penalty on end-of-horizon SoC shortfall for non-cost objectives (primary units per kWh).
# It exceeds 1 / eta_dis so draining stored energy never pays off in kWh or kW terms.
_TERMINAL_PENALTY_NONCOST = 1.25


class OptimizerError(RuntimeError):
    """The optimisation tool failed (solver crash, invalid inputs, no solution returned)."""


@dataclass
class OptimizationResult:
    status: str  # optimal | feasible_time_limit | infeasible | unbounded | error
    solver: str
    solver_status: str
    objective_value: float | None
    strategy: Strategy | None
    model_class: str
    n_variables: int = 0
    n_constraints: int = 0
    n_binaries: int = 0
    build_time_s: float = 0.0
    solve_time_s: float = 0.0
    relaxed_groups: tuple[str, ...] = ()
    message: str = ""
    kpis: dict[str, float] = field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        return self.status in ("optimal", "feasible_time_limit") and self.strategy is not None

    def summary(self) -> dict:
        return {
            "status": self.status,
            "solver": self.solver,
            "solver_status": self.solver_status,
            "objective_value": self.objective_value,
            "model_class": self.model_class,
            "n_variables": self.n_variables,
            "n_constraints": self.n_constraints,
            "n_binaries": self.n_binaries,
            "build_time_s": round(self.build_time_s, 4),
            "solve_time_s": round(self.solve_time_s, 4),
            "relaxed_groups": list(self.relaxed_groups),
            "message": self.message,
            "strategy_id": self.strategy.strategy_id if self.strategy else None,
            **self.kpis,
        }


def _solver(config: AppConfig, time_limit: float):
    name = config.optimizer.solver
    if name == "CBC":
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit, threads=1)
    elif name == "HiGHS":
        solver = pulp.HiGHS(msg=False, timeLimit=time_limit)
    else:
        raise OptimizerError(f"unsupported solver '{name}'")
    if not solver.available():
        hint = "pip install highspy" if name == "HiGHS" else "reinstall pulp<3 (bundled CBC)"
        raise OptimizerError(f"solver {name} is not available ({hint}) - or select another solver in the config")
    return solver


def optimize(
    config: AppConfig,
    constraints: CompiledConstraints,
    forecast: Forecast,
    energy_kwh: float,
    *,
    terminal_target_kwh: float | None = None,
    relax: tuple[str, ...] = (),
    time_limit_s: float | None = None,
) -> OptimizationResult:
    """Optimise set-points from ``forecast.start_step`` to the end of the horizon.

    ``relax`` removes whole constraint groups and is used **only** by the
    infeasibility diagnosis; results computed with relaxations are labelled and
    must never be executed.
    """
    unknown = set(relax) - set(RELAXABLE_GROUPS)
    if unknown:
        raise OptimizerError(f"unknown constraint groups {sorted(unknown)}")
    bat, bs, sim, opt = config.battery, config.base_station, config.simulation, config.optimizer
    dt, t0, n = sim.dt_h, forecast.start_step, len(forecast)
    if t0 + n != sim.n_steps:
        raise OptimizerError("forecast must cover the remaining horizon")
    for arr in (forecast.pv_kw, forecast.traffic, forecast.price_eur_per_kwh):
        if not np.all(np.isfinite(arr)) or np.any(np.asarray(arr) < 0):
            raise OptimizerError("forecast contains invalid values")
    model_class = "MILP" if opt.battery_mode_binaries else "LP"
    base = dict(solver=opt.solver, model_class=model_class, relaxed_groups=tuple(relax))

    # The current boundary is fixed: an already violated reserve cannot be repaired by planning.
    if "reserve" not in relax and constraints.reserve_kwh[t0] > energy_kwh + config.safety.energy_tolerance_kwh:
        return OptimizationResult(
            status="infeasible",
            solver_status="pre-solve check",
            objective_value=None,
            strategy=None,
            message=(
                f"current SoC {bat.kwh_to_pct(energy_kwh):.1f}% is below the required reserve "
                f"{bat.kwh_to_pct(constraints.reserve_kwh[t0]):.1f}% at the current time"
            ),
            **base,
        )

    started = time.perf_counter()
    pv = np.asarray(forecast.pv_kw, dtype=float)
    lam = np.asarray(forecast.traffic, dtype=float)
    price = np.asarray(forecast.price_eur_per_kwh, dtype=float)
    served = np.minimum(lam, 1.0)
    if "qos" in relax:  # keep traffic carried (linearity), drop the utilisation headroom only
        c_lb = np.maximum(bs.min_capacity_fraction, served)
    else:
        c_lb = np.asarray(constraints.required_capacity(lam, config), dtype=float)
    retention = bat.retention_per_step(dt)

    prob = pulp.LpProblem("energypilot_schedule", pulp.LpMinimize)
    K = range(n)
    ch = [pulp.LpVariable(f"ch_{k}", 0, bat.max_charge_kw) for k in K]
    dis = [pulp.LpVariable(f"dis_{k}", 0, bat.max_discharge_kw) for k in K]
    grid = [pulp.LpVariable(f"grid_{k}", 0, config.grid.max_import_kw) for k in K]
    curt = [pulp.LpVariable(f"curt_{k}", 0, float(pv[k])) for k in K]
    cap = [pulp.LpVariable(f"cap_{k}", float(min(c_lb[k], 1.0)), 1.0) for k in K]
    energy = [pulp.LpVariable(f"E_{k + 1}", bat.e_min_kwh, bat.e_max_kwh) for k in K]
    n_bin = 0
    if opt.battery_mode_binaries:
        mode = [pulp.LpVariable(f"charging_{k}", cat=pulp.LpBinary) for k in K]
        n_bin = n
        for k in K:
            prob += ch[k] <= bat.max_charge_kw * mode[k], f"excl_ch_{k}"
            prob += dis[k] <= bat.max_discharge_kw * (1 - mode[k]), f"excl_dis_{k}"

    for k in K:
        load = bs.fixed_power_kw + bs.capacity_power_kw * cap[k] + bs.traffic_power_kw * float(served[k])
        prob += float(pv[k]) - curt[k] + grid[k] + dis[k] == load + ch[k], f"balance_{k}"
        previous = energy_kwh if k == 0 else energy[k - 1]
        prob += (
            energy[k] == retention * previous + bat.charge_efficiency * dt * ch[k] - (dt / bat.discharge_efficiency) * dis[k],
            f"soc_{k}",
        )
        boundary = t0 + k + 1
        if "reserve" not in relax and constraints.reserve_kwh[boundary] > bat.e_min_kwh:
            prob += energy[k] >= float(constraints.reserve_kwh[boundary]), f"reserve_{k}"
        if "grid_cap" not in relax and np.isfinite(constraints.grid_cap_kw[t0 + k]):
            prob += grid[k] <= float(constraints.grid_cap_kw[t0 + k]), f"grid_cap_{k}"

    cost = pulp.lpSum(float(price[k]) * dt * grid[k] for k in K)
    secondary = (
        bat.degradation_cost_eur_per_kwh * dt * pulp.lpSum(ch[k] + dis[k] for k in K)
        + opt.curtailment_penalty_eur_per_kwh * dt * pulp.lpSum(curt)
    )
    objective_kind = constraints.objective
    terminal_weight = (
        opt.terminal_soc_penalty_eur_per_kwh if objective_kind == Objective.MINIMIZE_COST else _TERMINAL_PENALTY_NONCOST
    )
    terminal_term = 0
    if terminal_target_kwh is not None:
        shortfall = pulp.LpVariable("terminal_shortfall", 0)
        prob += energy[-1] + shortfall >= float(min(terminal_target_kwh, bat.e_max_kwh)), "terminal_soc"
        terminal_term = terminal_weight * shortfall
    if objective_kind == Objective.MINIMIZE_COST:
        prob += cost + secondary + terminal_term
    elif objective_kind == Objective.MINIMIZE_GRID_ENERGY:
        prob += dt * pulp.lpSum(grid) + opt.secondary_cost_weight * (cost + secondary) + terminal_term
    else:
        peak = pulp.LpVariable("peak_import", 0)
        for k in K:
            prob += peak >= grid[k], f"peak_{k}"
        prob += peak + opt.secondary_cost_weight * (cost + secondary) + terminal_term
    build_time = time.perf_counter() - started

    solve_start = time.perf_counter()
    try:
        prob.solve(_solver(config, time_limit_s or opt.time_limit_s))
    except pulp.PulpSolverError as exc:
        raise OptimizerError(f"solver failure: {exc}") from exc
    solve_time = time.perf_counter() - solve_start
    raw_status = pulp.LpStatus[prob.status]
    sol_status = getattr(prob, "sol_status", None)
    common = dict(
        n_variables=prob.numVariables(),
        n_constraints=prob.numConstraints(),
        n_binaries=n_bin,
        build_time_s=build_time,
        solve_time_s=solve_time,
        **base,
    )
    if prob.status == pulp.LpStatusInfeasible:
        return OptimizationResult(status="infeasible", solver_status=raw_status, objective_value=None, strategy=None,
                                  message="the solver proved that no schedule satisfies all hard constraints", **common)  # fmt: skip
    if prob.status == pulp.LpStatusUnbounded:
        return OptimizationResult(status="unbounded", solver_status=raw_status, objective_value=None, strategy=None,
                                  message="unbounded model (formulation error)", **common)  # fmt: skip
    if prob.status != pulp.LpStatusOptimal:
        return OptimizationResult(status="error", solver_status=raw_status, objective_value=None, strategy=None,
                                  message="the solver returned no solution (time limit or solver error)", **common)  # fmt: skip

    def values(variables) -> np.ndarray:
        return np.array([max(v.value() or 0.0, 0.0) for v in variables])

    p_ch, p_dis = values(ch), values(dis)
    soc = np.concatenate([[energy_kwh], [v.value() for v in energy]])
    strategy = Strategy(
        source="optimizer" if not relax else "optimizer_relaxed_diagnostic",
        start_step=t0,
        p_charge_kw=p_ch,
        p_discharge_kw=p_dis,
        capacity_fraction=np.clip(values(cap), 0.0, 1.0),
        grid_import_kw=values(grid),
        pv_curtailed_kw=values(curt),
        soc_kwh=soc,
        forecast=forecast,
        instruction_fingerprint=constraints.fingerprint,
        metadata={"objective": objective_kind.value, "model_class": model_class, "solver": opt.solver},
    )
    status = "optimal" if sol_status in (None, pulp.LpSolutionOptimal) else "feasible_time_limit"
    kpis = {
        "planned_cost_eur": float(np.sum(price * strategy.grid_import_kw) * dt),
        "planned_grid_kwh": float(np.sum(strategy.grid_import_kw) * dt),
        "planned_final_soc_pct": float(bat.kwh_to_pct(soc[-1])),
    }
    return OptimizationResult(
        status=status,
        solver_status=raw_status,
        objective_value=float(pulp.value(prob.objective)),
        strategy=strategy,
        message="optimal schedule found" if status == "optimal" else "feasible schedule (time limit reached)",
        kpis=kpis,
        **common,
    )
