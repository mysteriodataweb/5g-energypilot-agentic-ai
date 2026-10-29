"""Synthetic exogenous profiles: PV, traffic, tariff.

All generators are deterministic given a ``numpy.random.Generator``. Values are
step averages sampled at the middle of each step.

* PV (synthetic measurement) = clear-sky shape x cloud attenuation x events.
* Traffic (synthetic measurement) = diurnal template x scale x AR(1) noise + events.
* Price (assumed tariff) = time-of-use lookup.
* Base-station demand is *not* generated here: it is modelled by the simulator
  from traffic and the controller's capacity decision (see physics.bs_power_kw).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from energypilot.config.settings import (
    PVConfig,
    SimulationConfig,
    TariffConfig,
    TrafficConfig,
    WeatherClass,
)


def step_mid_hours(sim: SimulationConfig) -> np.ndarray:
    """Clock hour at the middle of each step."""
    return (np.arange(sim.n_steps) + 0.5) * sim.dt_h


def step_start_hours(sim: SimulationConfig) -> np.ndarray:
    return np.arange(sim.n_steps) * sim.dt_h


def clear_sky_pv_kw(hours: np.ndarray, pv: PVConfig) -> np.ndarray:
    """Clear-sky AC output: ``peak * sin(pi * x) ** exponent`` between sunrise and sunset."""
    x = (hours - pv.sunrise_hour) / (pv.sunset_hour - pv.sunrise_hour)
    shape = np.where((x > 0) & (x < 1), np.sin(np.pi * np.clip(x, 0, 1)), 0.0)
    return pv.peak_power_kw * np.power(np.clip(shape, 0, 1), pv.shape_exponent)


def ar1_series(rng: np.random.Generator, n: int, std: float, phi: float) -> np.ndarray:
    """Stationary zero-mean AR(1) series with marginal standard deviation ``std``."""
    out = np.empty(n)
    innovation_std = std * np.sqrt(1.0 - phi**2)
    out[0] = rng.normal(0.0, std)
    for k in range(1, n):
        out[k] = phi * out[k - 1] + rng.normal(0.0, innovation_std)
    return out


def cloud_attenuation(rng: np.random.Generator, n: int, weather: WeatherClass, phi: float) -> np.ndarray:
    """Cloud multiplier in [0.05, 1] around the weather class mean."""
    return np.clip(weather.mean + ar1_series(rng, n, weather.std, phi), 0.05, 1.0)


def _circular_gaussian(hours: np.ndarray, center: float, width: float) -> np.ndarray:
    d = np.abs(hours - center)
    d = np.minimum(d, 24.0 - d)
    return np.exp(-0.5 * (d / width) ** 2)


def traffic_template(hours: np.ndarray, traffic: TrafficConfig) -> np.ndarray:
    """Expected normalised traffic (diurnal shape) before scaling and noise."""
    profile = np.full_like(hours, traffic.baseline, dtype=float)
    for peak in traffic.peaks:
        profile += peak.amplitude * _circular_gaussian(hours, peak.center_hour, peak.width_hours)
    return profile


def traffic_noise_factor(rng: np.random.Generator, n: int, traffic: TrafficConfig) -> np.ndarray:
    """Mean-one multiplicative log-normal AR(1) noise."""
    z = ar1_series(rng, n, traffic.noise_std, traffic.noise_ar)
    return np.exp(z - 0.5 * traffic.noise_std**2)


def tariff_prices(hours: np.ndarray, tariff: TariffConfig) -> np.ndarray:
    prices = np.empty_like(hours, dtype=float)
    h = np.mod(hours, 24.0)
    for period in tariff.periods:
        mask = (h >= period.start_hour) & (h < period.end_hour)
        prices[mask] = period.price_eur_per_kwh
    return prices


def tariff_label(hour: float, tariff: TariffConfig) -> str:
    h = hour % 24.0
    for period in tariff.periods:
        if period.start_hour <= h < period.end_hour:
            return period.label
    return "unknown"


@dataclass(frozen=True)
class ProfileEvent:
    """A localised modification of an exogenous profile.

    ``kind``:
      * ``pv_scale``: multiply PV by ``value`` (e.g. 0.2 for a sudden cloud bank)
      * ``traffic_add``: add ``value`` normalised traffic (e.g. a public event)
      * ``traffic_scale``: multiply traffic by ``value``
    ``in_forecast`` tells whether forecasts know about the event in advance.
    The modification ramps linearly over ``ramp_hours`` at both edges.
    """

    kind: Literal["pv_scale", "traffic_add", "traffic_scale"]
    start_hour: float
    end_hour: float
    value: float
    ramp_hours: float = 0.25
    in_forecast: bool = False

    def weight(self, hours: np.ndarray) -> np.ndarray:
        """Activation weight in [0, 1] (trapezoid with linear ramps)."""
        ramp = max(self.ramp_hours, 1e-6)
        up = np.clip((hours - self.start_hour) / ramp, 0.0, 1.0)
        down = np.clip((self.end_hour - hours) / ramp, 0.0, 1.0)
        return np.minimum(up, down)

    def apply(self, hours: np.ndarray, pv: np.ndarray, traffic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w = self.weight(hours)
        if self.kind == "pv_scale":
            pv = pv * (1.0 + w * (self.value - 1.0))
        elif self.kind == "traffic_scale":
            traffic = traffic * (1.0 + w * (self.value - 1.0))
        elif self.kind == "traffic_add":
            traffic = traffic + w * self.value
        else:  # pragma: no cover - guarded by Literal
            raise ValueError(f"unknown event kind {self.kind}")
        return pv, traffic
