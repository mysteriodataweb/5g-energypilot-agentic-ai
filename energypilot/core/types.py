"""Domain types shared by the simulator, controllers, optimiser, validator and agent.

Sign convention (all power flows are non-negative, direction given by the name):

    pv_used + grid_import + p_discharge = load_served + p_charge

``p_charge`` is power drawn from the AC bus into the battery and ``p_discharge``
is power delivered by the battery to the AC bus. Grid export is not modelled:
PV surplus that cannot be stored is curtailed.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np


class InvalidActionError(ValueError):
    """A control action is malformed (NaN, negative power, ...)."""


@dataclass(frozen=True)
class ControlAction:
    """Set-points for one time step, as requested by a controller.

    The simulator enforces physical limits and records every adjustment it makes.
    """

    p_charge_kw: float = 0.0
    p_discharge_kw: float = 0.0
    capacity_fraction: float = 1.0

    def __post_init__(self) -> None:
        for name in ("p_charge_kw", "p_discharge_kw", "capacity_fraction"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise InvalidActionError(f"{name} must be a finite number, got {value!r}")
            if value < 0:
                raise InvalidActionError(f"{name} must be non-negative, got {value}")


@dataclass(frozen=True)
class Observation:
    """What a controller may observe at the beginning of a step.

    Current-step PV and traffic are measured values (nowcast assumption: the
    15-minute value measured at the start of the step is representative).
    """

    step: int
    time_h: float
    soc_kwh: float
    soc_pct: float
    pv_kw: float
    traffic: float
    price_eur_per_kwh: float


@dataclass(frozen=True)
class Forecast:
    """Forecast of exogenous inputs for steps ``start_step .. start_step + len - 1``.

    Index 0 refers to the current step and holds the measured values.
    """

    start_step: int
    pv_kw: np.ndarray
    traffic: np.ndarray
    price_eur_per_kwh: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.pv_kw)
        if len(self.traffic) != n or len(self.price_eur_per_kwh) != n:
            raise ValueError("forecast arrays must have equal length")

    def __len__(self) -> int:
        return len(self.pv_kw)


@dataclass
class Strategy:
    """A proposed plan of set-points over the remaining horizon.

    Strategies are produced only by the numerical tools (optimiser, rule-based
    roll-out). The agent references them by ``strategy_id``; it never edits them.
    ``soc_kwh`` has one more element than the per-step arrays (boundary values).
    """

    source: str
    start_step: int
    p_charge_kw: np.ndarray
    p_discharge_kw: np.ndarray
    capacity_fraction: np.ndarray
    grid_import_kw: np.ndarray
    pv_curtailed_kw: np.ndarray
    soc_kwh: np.ndarray
    forecast: Forecast
    instruction_fingerprint: str
    metadata: dict[str, Any] = field(default_factory=dict)
    strategy_id: str = field(default_factory=lambda: f"strat-{uuid.uuid4().hex[:10]}")

    def __len__(self) -> int:
        return len(self.p_charge_kw)

    def action_at(self, step: int) -> ControlAction:
        """Set-points planned for absolute simulation step ``step``."""
        k = step - self.start_step
        if not 0 <= k < len(self):
            raise IndexError(f"step {step} outside strategy range")
        return ControlAction(
            p_charge_kw=float(max(self.p_charge_kw[k], 0.0)),
            p_discharge_kw=float(max(self.p_discharge_kw[k], 0.0)),
            capacity_fraction=float(max(self.capacity_fraction[k], 0.0)),
        )

    def summary(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "source": self.source,
            "start_step": self.start_step,
            "n_steps": len(self),
            "soc_start_kwh": float(self.soc_kwh[0]),
            "soc_end_kwh": float(self.soc_kwh[-1]),
            "instruction_fingerprint": self.instruction_fingerprint,
            **{k: v for k, v in self.metadata.items() if isinstance(v, (str, int, float, bool))},
        }
