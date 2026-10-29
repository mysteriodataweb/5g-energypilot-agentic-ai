"""Physical models: base-station power, battery energy dynamics, QoS indicators.

These pure functions are the single source of truth for physics. The simulator,
the rule-based roll-out and the safety validator all call them, so the validator
recomputes a plan's consequences independently of the optimiser's own model.

Battery dynamics (energy at step boundaries, kWh):

    E[k+1] = r * E[k] + eta_ch * P_ch[k] * dt - P_dis[k] * dt / eta_dis

where ``r`` is the per-step self-discharge retention and ``dt`` is in hours.
"""

from __future__ import annotations

import numpy as np

from energypilot.config.settings import BaseStationConfig, BatteryConfig

ArrayLike = float | np.ndarray


def bs_power_kw(capacity_fraction: ArrayLike, traffic: ArrayLike, bs: BaseStationConfig) -> ArrayLike:
    """Base-station electrical demand (kW) for an active capacity and a traffic load.

    Linear load-dependent model: fixed site overhead, idle power of the active
    radio capacity, and an amplifier term proportional to the *served* traffic.
    """
    capacity = np.asarray(capacity_fraction, dtype=float)
    served = np.minimum(np.asarray(traffic, dtype=float), capacity)
    power = bs.fixed_power_kw + bs.capacity_power_kw * capacity + bs.traffic_power_kw * served
    return float(power) if np.ndim(power) == 0 else power


def required_capacity(traffic: ArrayLike, max_utilization: float, bs: BaseStationConfig) -> ArrayLike:
    """Smallest capacity fraction keeping utilisation <= ``max_utilization``.

    Bounded to ``[min_capacity_fraction, 1]``. When the result is 1 and traffic
    still exceeds ``max_utilization``, the QoS target is unattainable with the
    site's hardware (an exogenous, capacity-limited shortfall).
    """
    if not 0 < max_utilization <= 1:
        raise ValueError("max_utilization must lie in (0, 1]")
    req = np.clip(np.asarray(traffic, dtype=float) / max_utilization, bs.min_capacity_fraction, 1.0)
    return float(req) if np.ndim(req) == 0 else req


def capacity_limited(traffic: ArrayLike, max_utilization: float) -> ArrayLike:
    """True where even full capacity cannot meet the utilisation target."""
    return np.asarray(traffic, dtype=float) > max_utilization + 1e-12


def next_energy_kwh(
    energy_kwh: ArrayLike, p_charge_kw: ArrayLike, p_discharge_kw: ArrayLike, dt_h: float, bat: BatteryConfig
) -> ArrayLike:
    """Battery energy at the end of a step."""
    return (
        bat.retention_per_step(dt_h) * energy_kwh
        + bat.charge_efficiency * p_charge_kw * dt_h
        - p_discharge_kw * dt_h / bat.discharge_efficiency
    )


def soc_trajectory_kwh(
    initial_kwh: float, p_charge_kw: np.ndarray, p_discharge_kw: np.ndarray, dt_h: float, bat: BatteryConfig
) -> np.ndarray:
    """Integrate the battery model; returns ``len(p_charge_kw) + 1`` boundary values."""
    p_charge_kw = np.asarray(p_charge_kw, dtype=float)
    p_discharge_kw = np.asarray(p_discharge_kw, dtype=float)
    energy = np.empty(len(p_charge_kw) + 1)
    energy[0] = initial_kwh
    for k in range(len(p_charge_kw)):
        energy[k + 1] = next_energy_kwh(energy[k], p_charge_kw[k], p_discharge_kw[k], dt_h, bat)
    return energy


def max_charge_kw(energy_kwh: float, dt_h: float, bat: BatteryConfig, ceiling_kwh: float | None = None) -> float:
    """Largest charging power that respects the power limit and the SoC ceiling."""
    ceiling = bat.e_max_kwh if ceiling_kwh is None else min(ceiling_kwh, bat.e_max_kwh)
    room = ceiling - bat.retention_per_step(dt_h) * energy_kwh
    return float(np.clip(room / (bat.charge_efficiency * dt_h), 0.0, bat.max_charge_kw))


def max_discharge_kw(energy_kwh: float, dt_h: float, bat: BatteryConfig, floor_kwh: float | None = None) -> float:
    """Largest discharging power that respects the power limit and an energy floor."""
    floor = bat.e_min_kwh if floor_kwh is None else max(floor_kwh, bat.e_min_kwh)
    available = bat.retention_per_step(dt_h) * energy_kwh - floor
    return float(np.clip(available * bat.discharge_efficiency / dt_h, 0.0, bat.max_discharge_kw))


def qos_indicators(traffic: ArrayLike, capacity_fraction: ArrayLike) -> dict[str, ArrayLike]:
    """QoS proxies derived from traffic and active capacity.

    * ``utilization`` = traffic / capacity (radio load of the active layers)
    * ``served_ratio`` = share of offered traffic that can be carried
    * ``delay_factor`` = 1 / (1 - min(u, 0.99)), an M/M/1-style congestion proxy
      (1.0 = no queueing). Indicative only; not used for optimisation.
    """
    traffic = np.asarray(traffic, dtype=float)
    capacity = np.maximum(np.asarray(capacity_fraction, dtype=float), 1e-9)
    utilization = traffic / capacity
    with np.errstate(divide="ignore", invalid="ignore"):
        served = np.where(traffic > 0, np.minimum(traffic, capacity) / np.where(traffic > 0, traffic, 1.0), 1.0)
    delay = 1.0 / (1.0 - np.minimum(utilization, 0.99))
    return {"utilization": utilization, "served_ratio": served, "delay_factor": delay}
