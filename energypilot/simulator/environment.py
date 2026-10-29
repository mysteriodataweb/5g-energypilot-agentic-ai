"""Discrete-time simulator of a 5G base station with PV, battery and grid.

The simulator is the "plant": it receives set-points, enforces physical limits
(power, SoC, grid connection) and returns what actually happened. It never
enforces *operational* instructions (reserves, grid caps, QoS targets): those
are the controllers' responsibility and are audited ex post. Every correction
it applies to a set-point is recorded in ``StepRecord.adjustments``.

The simulator is independent of any LLM provider: every controller uses it
through the same ``observe`` / ``step`` interface.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from energypilot.config.settings import AppConfig
from energypilot.core.types import ControlAction, InvalidActionError, Observation
from energypilot.simulator import physics

logger = logging.getLogger(__name__)

_TOL = 1e-9


@dataclass(frozen=True)
class StepRecord:
    """Realised flows for one step. Powers are step averages (kW); energies in kWh."""

    step: int
    time_h: float
    pv_available_kw: float
    pv_used_kw: float
    pv_curtailed_kw: float
    traffic: float
    capacity_requested: float
    capacity_fraction: float
    load_kw: float
    load_served_kw: float
    unserved_kw: float
    grid_import_kw: float
    p_charge_requested_kw: float
    p_discharge_requested_kw: float
    p_charge_kw: float
    p_discharge_kw: float
    soc_start_kwh: float
    soc_end_kwh: float
    soc_end_pct: float
    price_eur_per_kwh: float
    cost_eur: float
    utilization: float
    served_ratio: float
    delay_factor: float
    adjustments: str


class BaseStationEnv:
    """Closed-loop simulator driven by *actual* (realised) exogenous profiles.

    Args:
        config: validated application configuration.
        pv_kw: realised PV availability per step (kW, AC).
        traffic: realised normalised traffic per step.
        price_eur_per_kwh: grid price per step.
        initial_soc_pct: battery SoC at step 0.
    """

    def __init__(
        self,
        config: AppConfig,
        pv_kw: np.ndarray,
        traffic: np.ndarray,
        price_eur_per_kwh: np.ndarray,
        initial_soc_pct: float,
    ) -> None:
        n = config.simulation.n_steps
        for name, arr in (("pv_kw", pv_kw), ("traffic", traffic), ("price", price_eur_per_kwh)):
            if len(arr) != n:
                raise ValueError(f"{name} has {len(arr)} values, expected {n}")
            if not np.all(np.isfinite(arr)) or np.any(np.asarray(arr) < 0):
                raise ValueError(f"{name} must be finite and non-negative")
        bat = config.battery
        if not bat.soc_min_pct - 1e-9 <= initial_soc_pct <= bat.soc_max_pct + 1e-9:
            raise ValueError("initial SoC outside the battery's permitted range")
        self.config = config
        self.pv_kw = np.asarray(pv_kw, dtype=float)
        self.traffic = np.asarray(traffic, dtype=float)
        self.price = np.asarray(price_eur_per_kwh, dtype=float)
        self.initial_soc_kwh = bat.pct_to_kwh(initial_soc_pct)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> Observation:
        self.step_index = 0
        self.energy_kwh = self.initial_soc_kwh
        self.records: list[StepRecord] = []
        return self.observe()

    @property
    def n_steps(self) -> int:
        return self.config.simulation.n_steps

    @property
    def done(self) -> bool:
        return self.step_index >= self.n_steps

    def observe(self) -> Observation:
        t = min(self.step_index, self.n_steps - 1)
        return Observation(
            step=self.step_index,
            time_h=self.step_index * self.config.simulation.dt_h,
            soc_kwh=self.energy_kwh,
            soc_pct=self.config.battery.kwh_to_pct(self.energy_kwh),
            pv_kw=float(self.pv_kw[t]),
            traffic=float(self.traffic[t]),
            price_eur_per_kwh=float(self.price[t]),
        )

    # ------------------------------------------------------------------- step
    def step(self, action: ControlAction) -> StepRecord:
        """Apply set-points for the current step and advance time."""
        if self.done:
            raise RuntimeError("simulation horizon already reached")
        if not isinstance(action, ControlAction):
            raise InvalidActionError("action must be a ControlAction")
        cfg = self.config
        bat, bs = cfg.battery, cfg.base_station
        dt = cfg.simulation.dt_h
        t = self.step_index
        pv, lam, price = float(self.pv_kw[t]), float(self.traffic[t]), float(self.price[t])
        notes: list[str] = []

        # 1) Radio capacity.
        capacity = float(np.clip(action.capacity_fraction, bs.min_capacity_fraction, 1.0))
        if abs(capacity - action.capacity_fraction) > 1e-9:
            notes.append("capacity_clipped_to_hardware_range")
        if bs.ran_auto_wakeup and lam > capacity + _TOL:
            capacity = min(1.0, lam)
            notes.append("ran_auto_wakeup")
        load = float(physics.bs_power_kw(capacity, lam, bs))

        # 2) Battery set-points: no simultaneous charge/discharge, power and SoC limits.
        p_ch, p_dis = action.p_charge_kw, action.p_discharge_kw
        if p_ch > _TOL and p_dis > _TOL:
            net = p_ch - p_dis
            p_ch, p_dis = max(net, 0.0), max(-net, 0.0)
            notes.append("simultaneous_charge_discharge_netted")
        limit_ch = physics.max_charge_kw(self.energy_kwh, dt, bat)
        limit_dis = physics.max_discharge_kw(self.energy_kwh, dt, bat)
        if p_ch > limit_ch + _TOL:
            notes.append("charge_limited" + ("_power" if p_ch > bat.max_charge_kw else "_soc_max"))
            p_ch = limit_ch
        if p_dis > limit_dis + _TOL:
            notes.append("discharge_limited" + ("_power" if p_dis > bat.max_discharge_kw else "_soc_min"))
            p_dis = limit_dis

        # 3) Bus balance: pv_used + grid + p_dis = load_served + p_ch.
        net_demand = load + p_ch - p_dis - pv
        unserved = 0.0
        if net_demand >= 0.0:
            grid, curtailed = net_demand, 0.0
            if grid > cfg.grid.max_import_kw + _TOL:
                # Connection limit reached: curtail battery charging first, then shed load.
                excess = grid - cfg.grid.max_import_kw
                cut = min(p_ch, excess)
                if cut > 0:
                    p_ch -= cut
                    notes.append("charge_reduced_grid_limit")
                unserved = excess - cut
                if unserved > _TOL:
                    notes.append("load_shed_grid_limit")
                grid = cfg.grid.max_import_kw
        else:
            grid, curtailed = 0.0, -net_demand
        pv_used = pv - curtailed

        new_energy = float(physics.next_energy_kwh(self.energy_kwh, p_ch, p_dis, dt, bat))
        new_energy = float(np.clip(new_energy, bat.e_min_kwh, bat.e_max_kwh))  # numerical guard only

        qos = physics.qos_indicators(lam, capacity)
        served_ratio = float(qos["served_ratio"]) * (1.0 - unserved / load if load > 0 else 1.0)
        record = StepRecord(
            step=t,
            time_h=t * dt,
            pv_available_kw=pv,
            pv_used_kw=pv_used,
            pv_curtailed_kw=curtailed,
            traffic=lam,
            capacity_requested=action.capacity_fraction,
            capacity_fraction=capacity,
            load_kw=load,
            load_served_kw=load - unserved,
            unserved_kw=unserved,
            grid_import_kw=grid,
            p_charge_requested_kw=action.p_charge_kw,
            p_discharge_requested_kw=action.p_discharge_kw,
            p_charge_kw=p_ch,
            p_discharge_kw=p_dis,
            soc_start_kwh=self.energy_kwh,
            soc_end_kwh=new_energy,
            soc_end_pct=bat.kwh_to_pct(new_energy),
            price_eur_per_kwh=price,
            cost_eur=grid * dt * price,
            utilization=float(qos["utilization"]),
            served_ratio=served_ratio,
            delay_factor=float(qos["delay_factor"]),
            adjustments=";".join(notes),
        )
        if notes:
            logger.debug("step %d adjustments: %s", t, record.adjustments)
        self.records.append(record)
        self.energy_kwh = new_energy
        self.step_index += 1
        return record

    def history(self) -> pd.DataFrame:
        """All realised steps as a DataFrame (one row per step)."""
        return pd.DataFrame([asdict(r) for r in self.records])
