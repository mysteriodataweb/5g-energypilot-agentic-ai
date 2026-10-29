"""Compile a validated instruction into per-step constraint arrays.

Conventions (``dt`` = step length in minutes, ``n`` = number of steps):

* Reserve windows constrain the battery energy at step *boundaries*
  ``k = floor(start/dt) .. ceil(end/dt)``. Energy varies linearly within a step,
  so constraining the enclosing boundaries is conservative.
* Grid-import limits constrain every step whose interval overlaps the window,
  ``k = floor(start/dt) .. ceil(end/dt) - 1``.
* Nothing is constrained before the issue time; a window that started earlier
  applies from the issue boundary (reported as a warning during validation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from energypilot.config.settings import AppConfig
from energypilot.instructions.schema import Objective, OperationalInstruction
from energypilot.simulator import physics


@dataclass(frozen=True)
class CompiledConstraints:
    n_steps: int
    dt_h: float
    issue_step: int
    reserve_kwh: np.ndarray  # (n + 1,) operational reserve per boundary, 0 where none
    grid_cap_kw: np.ndarray  # (n,) +inf where none
    max_utilization: float
    allow_capacity_reduction: bool
    objective: Objective
    fingerprint: str

    def floor_kwh(self, boundary: int, e_min_kwh: float) -> float:
        """Lowest admissible battery energy at a boundary (BMS minimum or reserve)."""
        return max(e_min_kwh, float(self.reserve_kwh[min(boundary, self.n_steps)]))

    def required_capacity(self, traffic, config: AppConfig):
        """Capacity fraction the QoS target requires for a given traffic."""
        if not self.allow_capacity_reduction:
            return np.ones_like(np.asarray(traffic, dtype=float)) if np.ndim(traffic) else 1.0
        return physics.required_capacity(traffic, self.max_utilization, config.base_station)

    def reserve_steps(self) -> np.ndarray:
        return np.flatnonzero(self.reserve_kwh > 0)

    def capped_steps(self) -> np.ndarray:
        return np.flatnonzero(np.isfinite(self.grid_cap_kw))


def compile_constraints(instruction: OperationalInstruction, config: AppConfig) -> CompiledConstraints:
    sim, bat = config.simulation, config.battery
    n, step_min = sim.n_steps, sim.time_step_minutes
    issue_step = min(instruction.issued_at_minute // step_min, n)
    reserve = np.zeros(n + 1)
    for r in instruction.reserves:
        lo = max(math.floor(r.start_minute / step_min), issue_step)
        hi = min(math.ceil(r.end_minute / step_min), n)
        if lo <= hi:
            reserve[lo : hi + 1] = np.maximum(reserve[lo : hi + 1], bat.pct_to_kwh(r.min_soc_pct))
    cap = np.full(n, np.inf)
    for g in instruction.grid_limits:
        lo = max(math.floor(g.start_minute / step_min), issue_step)
        hi = min(math.ceil(g.end_minute / step_min), n)
        if lo < hi:
            cap[lo:hi] = np.minimum(cap[lo:hi], g.max_import_kw)
    return CompiledConstraints(
        n_steps=n,
        dt_h=sim.dt_h,
        issue_step=issue_step,
        reserve_kwh=reserve,
        grid_cap_kw=cap,
        max_utilization=instruction.effective_max_utilization(config.qos.max_utilization_by_level),
        allow_capacity_reduction=instruction.allow_capacity_reduction,
        objective=instruction.objective,
        fingerprint=instruction.fingerprint(),
    )
