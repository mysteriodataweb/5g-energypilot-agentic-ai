"""Explain *why* an instruction is infeasible, without relaxing it.

Two complementary, deterministic methods:

1. Analytic pre-checks (necessary conditions): reserve reachability given the
   charging power limit, discharge power and energy needed to respect a grid
   cap, and reserve-versus-grid-cap energy budgets.
2. Group relaxation tests (deletion filter at group level): re-solve with one,
   then two, constraint groups removed ({reserve, grid_cap, qos}); the minimal
   sets whose removal restores feasibility identify the conflicting requirements.

The relaxed solutions are diagnostic only and are never executed.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from energypilot.config.settings import AppConfig
from energypilot.core.types import Forecast
from energypilot.instructions.constraints import CompiledConstraints
from energypilot.optimizer.milp import RELAXABLE_GROUPS, optimize

_GROUP_LABELS = {
    "reserve": "minimum battery reserve",
    "grid_cap": "grid-import limit",
    "qos": "QoS capacity requirement",
}


@dataclass
class ConflictReport:
    feasible: bool
    analytic_findings: list[str] = field(default_factory=list)
    relaxation_tests: dict[str, str] = field(default_factory=dict)
    conflicting_groups: list[list[str]] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "feasible": self.feasible,
            "analytic_findings": self.analytic_findings,
            "relaxation_tests": self.relaxation_tests,
            "conflicting_groups": self.conflicting_groups,
            "explanation": self.explanation,
        }


def _hhmm(step: int, dt_h: float) -> str:
    minutes = int(round(step * dt_h * 60))
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def analytic_checks(
    config: AppConfig, constraints: CompiledConstraints, forecast: Forecast, energy_kwh: float
) -> list[str]:
    bat, bs, dt = config.battery, config.base_station, config.simulation.dt_h
    t0, n = forecast.start_step, len(forecast)
    findings: list[str] = []

    reserve = constraints.reserve_kwh
    if reserve[t0] > energy_kwh + config.safety.energy_tolerance_kwh:
        findings.append(
            f"The battery is at {bat.kwh_to_pct(energy_kwh):.1f}% now, below the requested reserve of "
            f"{bat.kwh_to_pct(reserve[t0]):.0f}% that applies from {_hhmm(t0, dt)}."
        )
    for b in range(t0 + 1, t0 + n + 1):
        gap = reserve[b] - energy_kwh
        reachable = bat.charge_efficiency * bat.max_charge_kw * (b - t0) * dt
        if gap > reachable + 1e-9:
            needed_h = gap / (bat.charge_efficiency * bat.max_charge_kw)
            findings.append(
                f"Reaching {bat.kwh_to_pct(reserve[b]):.0f}% by {_hhmm(b, dt)} needs {gap:.1f} kWh but at most "
                f"{reachable:.1f} kWh can be charged in time; the earliest feasible time is about "
                f"{_hhmm(t0 + int(np.ceil(needed_h / dt)), dt)}."
            )
            break

    lam = np.asarray(forecast.traffic, dtype=float)
    min_capacity = np.asarray(constraints.required_capacity(lam, config), dtype=float)
    min_load = bs.fixed_power_kw + bs.capacity_power_kw * min_capacity + bs.traffic_power_kw * np.minimum(lam, 1.0)
    caps = constraints.grid_cap_kw[t0 : t0 + n]
    capped = np.isfinite(caps)
    if np.any(capped):
        deficit = np.where(capped, np.maximum(min_load - forecast.pv_kw - np.where(capped, caps, 0.0), 0.0), 0.0)
        too_fast = np.flatnonzero(deficit > bat.max_discharge_kw + 1e-9)
        if too_fast.size:
            k = int(too_fast[0])
            findings.append(
                f"At {_hhmm(t0 + k, dt)} the battery would have to discharge {deficit[k]:.1f} kW to respect the "
                f"grid cap, above its {bat.max_discharge_kw:g} kW limit."
            )
        energy_needed = float(np.sum(deficit) * dt / bat.discharge_efficiency)
        idx = np.flatnonzero(capped)
        floor_in_window = max(bat.e_min_kwh, float(np.max(reserve[t0 + idx[0] : t0 + idx[-1] + 2])))
        usable = bat.e_max_kwh - floor_in_window
        if energy_needed > usable + 1e-9:
            findings.append(
                f"Respecting the grid cap from {_hhmm(t0 + idx[0], dt)} to {_hhmm(t0 + idx[-1] + 1, dt)} requires about "
                f"{energy_needed:.1f} kWh from the battery (forecast load minus PV), but only {max(usable, 0):.1f} kWh lie "
                f"between the maximum SoC ({bat.soc_max_pct:g}%) and the applicable floor "
                f"({bat.kwh_to_pct(floor_in_window):.0f}%)."
            )
    return findings


def diagnose(
    config: AppConfig,
    constraints: CompiledConstraints,
    forecast: Forecast,
    energy_kwh: float,
    *,
    terminal_target_kwh: float | None = None,
) -> ConflictReport:
    findings = analytic_checks(config, constraints, forecast, energy_kwh)

    def feasible(relax: tuple[str, ...]) -> bool:
        return optimize(
            config, constraints, forecast, energy_kwh, terminal_target_kwh=terminal_target_kwh, relax=relax
        ).feasible

    if feasible(()):
        return ConflictReport(feasible=True, analytic_findings=findings, explanation="The instruction is feasible.")

    present = []
    t0 = forecast.start_step
    if np.any(constraints.reserve_kwh[t0:] > config.battery.e_min_kwh):
        present.append("reserve")
    if np.any(np.isfinite(constraints.grid_cap_kw[t0:])):
        present.append("grid_cap")
    present.append("qos")
    tests: dict[str, str] = {}
    minimal: list[list[str]] = []
    for size in (1, 2):
        for combo in itertools.combinations([g for g in RELAXABLE_GROUPS if g in present], size):
            if any(set(m) <= set(combo) for m in minimal):
                continue
            ok = feasible(combo)
            tests["without " + " + ".join(combo)] = "feasible" if ok else "infeasible"
            if ok:
                minimal.append(list(combo))
        if minimal:
            break

    if minimal:
        parts = [" and ".join(_GROUP_LABELS[g] for g in m) for m in minimal]
        explanation = (
            "No schedule satisfies all requirements. Feasibility is restored only by dropping: "
            + "; or ".join(parts)
            + ". These requirements conflict with the battery and site limits."
        )
    else:
        explanation = (
            "No schedule satisfies all requirements, and removing up to two requirement groups does not restore "
            "feasibility; the physical limits themselves (power, capacity, connection) are binding."
        )
    if findings:
        explanation += " " + " ".join(findings)
    return ConflictReport(
        feasible=False,
        analytic_findings=findings,
        relaxation_tests=tests,
        conflicting_groups=minimal,
        explanation=explanation,
    )
