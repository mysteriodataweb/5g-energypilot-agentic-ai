"""Runtime safety shield applied to every action before it reaches the simulator.

The shield only *reduces* or *raises to a safe level* the requested set-points
(projection onto a safe set); it never adds control intelligence. It is applied
identically to every approach, and each intervention is reported:

* no simultaneous charge and discharge (netting);
* battery power limits and SoC ceiling;
* discharge never below the energy floor (BMS minimum or active reserve);
* capacity within the hardware range and at least the QoS requirement for the
  *measured* traffic;
* charging reduced if it alone would push grid import above an active cap.

What the shield cannot guarantee (e.g. a grid cap that needs more discharge
than available) is measured ex post as a violation.
"""

from __future__ import annotations

from energypilot.config.settings import AppConfig
from energypilot.core.types import ControlAction, Observation
from energypilot.instructions.constraints import CompiledConstraints
from energypilot.simulator import physics

_TOL = 1e-6


class RuntimeShield:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def filter(
        self, obs: Observation, action: ControlAction, constraints: CompiledConstraints
    ) -> tuple[ControlAction, list[str]]:
        cfg = self.config
        bat, bs, dt = cfg.battery, cfg.base_station, cfg.simulation.dt_h
        notes: list[str] = []
        p_ch, p_dis, cap = action.p_charge_kw, action.p_discharge_kw, action.capacity_fraction

        if p_ch > _TOL and p_dis > _TOL:
            p_ch, p_dis = max(p_ch - p_dis, 0.0), max(p_dis - p_ch, 0.0)
            notes.append("netted_simultaneous_setpoints")
        ch_max = physics.max_charge_kw(obs.soc_kwh, dt, bat)
        if p_ch > ch_max + _TOL:
            p_ch = ch_max
            notes.append("charge_limited")
        floor = constraints.floor_kwh(obs.step + 1, bat.e_min_kwh)
        dis_max = physics.max_discharge_kw(obs.soc_kwh, dt, bat, floor_kwh=floor)
        if p_dis > dis_max + _TOL:
            p_dis = dis_max
            notes.append("discharge_limited_to_floor")

        required = float(constraints.required_capacity(obs.traffic, cfg))
        safe_cap = min(max(cap, bs.min_capacity_fraction, required), 1.0)
        if abs(safe_cap - cap) > _TOL:
            notes.append("capacity_raised_for_qos" if safe_cap > cap else "capacity_clipped")
        cap = safe_cap

        grid_cap = float(constraints.grid_cap_kw[obs.step])
        if p_ch > _TOL:
            net = float(physics.bs_power_kw(cap, obs.traffic, bs)) - obs.pv_kw
            excess = net + p_ch - p_dis - grid_cap
            if excess > _TOL:
                p_ch = max(p_ch - excess, 0.0)
                notes.append("charge_reduced_for_grid_cap")
        return ControlAction(p_charge_kw=p_ch, p_discharge_kw=p_dis, capacity_fraction=cap), notes
