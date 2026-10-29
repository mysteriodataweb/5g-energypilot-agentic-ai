"""Interpretable rule-based controller (deterministic baseline).

The controller is reactive: it uses only the current measurements, the known
tariff signal and the schedule of operational constraints. It does not use
load or PV forecasts. Rules, evaluated every step:

R1  Capacity: smallest capacity meeting the QoS utilisation target
    (full capacity if capacity reduction is not allowed).
R2  Reserve preparation: charge at the minimal constant power that reaches
    every upcoming reserve level in time, multiplied by a safety margin.
R3  Solar surplus: store PV surplus in the battery (up to its limits).
R4  Off-peak charging (cost objective): charge from the grid up to a target
    SoC when the price is at or below the off-peak threshold.
R5  Discharge (only when not charging), never below the energy floor
    (BMS minimum or active reserve), and never more than the net deficit:
      cost objective: when the price is at or above the peak threshold;
      grid-energy objective: whenever there is a deficit;
      peak objective: the part of the deficit above a fixed threshold.
R6  Grid cap: if a grid-import limit would be exceeded, reduce charging first,
    then discharge (down to the floor) to stay below the cap.

Limitations (by design): myopic (no forecast), hand-tuned thresholds, no
guarantee of feasibility, and no end-of-day SoC management. When reserve and
grid-cap requirements conflict, the grid cap wins (R6 is applied last) and the
reserve may be missed: the ex-ante safety validation detects this.
"""

from __future__ import annotations

import numpy as np

from energypilot.config.settings import AppConfig
from energypilot.core.types import ControlAction, Forecast, Observation, Strategy
from energypilot.controllers.base import rollout_strategy
from energypilot.instructions.constraints import CompiledConstraints
from energypilot.instructions.schema import Objective
from energypilot.simulator import physics


class RuleBasedController:
    name = "rule_based"

    def __init__(self, config: AppConfig, constraints: CompiledConstraints) -> None:
        self.config = config
        self.constraints = constraints
        self.last_rules: list[str] = []

    def _precharge_power_kw(self, step: int, energy_kwh: float) -> float:
        """Minimal constant charging power reaching all upcoming reserves in time."""
        bat, dt = self.config.battery, self.config.simulation.dt_h
        reserve = self.constraints.reserve_kwh
        future = np.arange(step + 1, len(reserve))
        if future.size == 0:
            return 0.0
        gaps = reserve[future] - energy_kwh
        mask = gaps > 1e-9
        if not np.any(mask):
            return 0.0
        hours = (future[mask] - step) * dt
        return float(np.max(gaps[mask] / (bat.charge_efficiency * hours)))

    def decide(self, obs: Observation, forecast: Forecast | None = None) -> ControlAction:
        cfg, cons = self.config, self.constraints
        bat, rc, dt = cfg.battery, cfg.rule_controller, cfg.simulation.dt_h
        t, energy = obs.step, obs.soc_kwh
        rules: list[str] = []

        capacity = float(cons.required_capacity(obs.traffic, cfg))  # R1
        net = float(physics.bs_power_kw(capacity, obs.traffic, cfg.base_station)) - obs.pv_kw
        floor_next = cons.floor_kwh(t + 1, bat.e_min_kwh)
        ch_max = physics.max_charge_kw(energy, dt, bat)
        dis_max = physics.max_discharge_kw(energy, dt, bat, floor_kwh=floor_next)

        p_ch = p_dis = 0.0
        p_needed = self._precharge_power_kw(t, energy)  # R2
        if p_needed > 0:
            p_ch = min(ch_max, p_needed * rc.reserve_precharge_margin)
            rules.append("R2_reserve_precharge")
        if net < 0 and ch_max > 0:  # R3
            p_ch = max(p_ch, min(-net, ch_max))
            rules.append("R3_pv_surplus")
        if cons.objective == Objective.MINIMIZE_COST and obs.price_eur_per_kwh <= rc.offpeak_price_threshold_eur_per_kwh:
            target = bat.pct_to_kwh(rc.offpeak_charge_target_pct)
            if energy < target:  # R4
                room = (target - energy) / (bat.charge_efficiency * dt)
                p_ch = max(p_ch, min(rc.offpeak_charge_power_kw, ch_max, room))
                rules.append("R4_offpeak_charge")
        if p_ch <= 0 and net > 0 and dis_max > 0:  # R5
            if cons.objective == Objective.MINIMIZE_GRID_ENERGY:
                p_dis = min(net, dis_max)
            elif cons.objective == Objective.MINIMIZE_PEAK_IMPORT:
                p_dis = min(max(net - rc.peak_shaving_threshold_kw, 0.0), dis_max)
            elif obs.price_eur_per_kwh >= rc.peak_price_threshold_eur_per_kwh:
                p_dis = min(net, dis_max)
            if p_dis > 0:
                rules.append("R5_discharge")

        cap = float(cons.grid_cap_kw[t])  # R6
        excess = net + p_ch - p_dis - cap
        if excess > 1e-9:
            cut = min(p_ch, excess)
            p_ch -= cut
            excess -= cut
            if excess > 1e-9 and net > 0:
                p_dis = min(p_dis + excess, dis_max, net)
            rules.append("R6_grid_cap")

        self.last_rules = rules
        return ControlAction(p_charge_kw=max(p_ch, 0.0), p_discharge_kw=max(p_dis, 0.0), capacity_fraction=capacity)

    def plan(self, obs: Observation, forecast: Forecast) -> Strategy:
        """Ex-ante plan: roll the rules out on the forecast (for validation)."""
        return rollout_strategy(
            lambda o: self.decide(o, None),
            forecast,
            obs.soc_kwh,
            self.config,
            source=self.name,
            instruction_fingerprint=self.constraints.fingerprint,
        )
