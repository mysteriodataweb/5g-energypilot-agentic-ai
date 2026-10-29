"""Scenario catalogue and reproducible scenario generation.

A scenario couples (i) synthetic exogenous profiles, with a distinction between
what forecasts know and what actually happens, and (ii) an operational
instruction issued at a given time, with a labelled ground truth.

Randomness: ``numpy.random.SeedSequence([seed, crc32(name)])`` gives an
independent, platform-stable stream per (seed, scenario). Python's built-in
``hash`` is salted per process and is therefore never used.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from energypilot.config.settings import AppConfig
from energypilot.data import profiles
from energypilot.data.profiles import ProfileEvent

ExpectedOutcome = Literal["accept", "clarify", "reject"]

PROVENANCE: dict[str, str] = {
    "pv_actual_kw": "synthetic measurement (generated: clear-sky model x stochastic cloud attenuation x events)",
    "pv_forecast_kw": "synthetic forecast (clear-sky model x expected cloud attenuation; unforeseen events excluded)",
    "traffic_actual": "synthetic measurement (diurnal template x AR(1) noise + events)",
    "traffic_forecast": "synthetic forecast (diurnal template; unforeseen events excluded)",
    "price_eur_per_kwh": "assumption (illustrative time-of-use tariff from config)",
    "demand_kw": "modelled (linear base-station power model applied to traffic and capacity decisions)",
    "initial_soc_pct": "assumption (config value + seeded jitter, or scenario override)",
}


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    title: str
    description: str
    weather: str
    instruction_text: str
    instruction_issue_hour: float
    expected_outcome: ExpectedOutcome
    ground_truth: dict[str, Any] | None
    forecast_weather: str | None = None
    traffic_scale: float = 1.0
    forecast_traffic_scale: float = 1.0
    events: tuple[ProfileEvent, ...] = ()
    initial_soc_pct: float | None = None
    tags: tuple[str, ...] = ()


@dataclass
class ScenarioData:
    """Generated profiles for one (scenario, seed) pair."""

    spec: ScenarioSpec
    seed: int
    hours: np.ndarray
    pv_actual_kw: np.ndarray
    pv_forecast_kw: np.ndarray
    traffic_actual: np.ndarray
    traffic_forecast: np.ndarray
    price_eur_per_kwh: np.ndarray
    initial_soc_pct: float
    issue_step: int
    provenance: dict[str, str] = field(default_factory=lambda: dict(PROVENANCE))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "time_h": self.hours,
                "pv_actual_kw": self.pv_actual_kw,
                "pv_forecast_kw": self.pv_forecast_kw,
                "traffic_actual": self.traffic_actual,
                "traffic_forecast": self.traffic_forecast,
                "price_eur_per_kwh": self.price_eur_per_kwh,
            }
        )


def _window(start_h: float, end_h: float) -> dict[str, int]:
    return {"start_minute": int(round(start_h * 60)), "end_minute": int(round(end_h * 60))}


SCENARIOS: dict[str, ScenarioSpec] = {
    spec.name: spec
    for spec in (
        ScenarioSpec(
            name="normal",
            title="1. Normal operation",
            description="Sunny day, nominal traffic. Start-of-day instruction with a short reserve.",
            weather="sunny",
            initial_soc_pct=80.0,
            instruction_issue_hour=0.0,
            instruction_text=(
                "Maintain at least 70% battery charge during the next two hours, minimize "
                "electricity costs, and preserve the required quality of service."
            ),
            expected_outcome="accept",
            ground_truth={
                "objective": "minimize_cost",
                "reserves": [{"min_soc_pct": 70.0, **_window(0, 2)}],
                "grid_limits": [],
                "qos_level": "standard",
                "allow_capacity_reduction": True,
            },
            tags=("nominal",),
        ),
        ScenarioSpec(
            name="solar_drop",
            title="2. Sudden decrease in solar production",
            description="Forecast sunny; an unforeseen cloud bank cuts PV by 80% from 11:00 to 16:00.",
            weather="sunny",
            events=(ProfileEvent("pv_scale", 11.0, 16.0, 0.2, ramp_hours=0.5, in_forecast=False),),
            instruction_issue_hour=8.0,
            instruction_text=(
                "Minimize electricity costs and keep at least 60% battery charge between 17:00 and 21:00."
            ),
            expected_outcome="accept",
            ground_truth={
                "objective": "minimize_cost",
                "reserves": [{"min_soc_pct": 60.0, **_window(17, 21)}],
                "grid_limits": [],
                "qos_level": "standard",
                "allow_capacity_reduction": True,
            },
            tags=("forecast_error", "pv"),
        ),
        ScenarioSpec(
            name="traffic_peak",
            title="3. Unexpected traffic peak",
            description="An unforeseen public event adds +0.45 normalised traffic from 18:00 to 21:30.",
            weather="partly_cloudy",
            events=(ProfileEvent("traffic_add", 18.0, 21.5, 0.45, ramp_hours=0.5, in_forecast=False),),
            instruction_issue_hour=8.0,
            instruction_text=(
                "Strict quality of service is required today because of a public event; "
                "keep minimizing electricity costs."
            ),
            expected_outcome="accept",
            ground_truth={
                "objective": "minimize_cost",
                "reserves": [],
                "grid_limits": [],
                "qos_level": "strict",
                "allow_capacity_reduction": True,
            },
            tags=("forecast_error", "traffic", "qos"),
        ),
        ScenarioSpec(
            name="reserve_increase",
            title="4. Temporary increase in minimum battery reserve",
            description="Storm warning at 14:00: the reserve must be raised for the evening.",
            weather="partly_cloudy",
            instruction_issue_hour=14.0,
            instruction_text=(
                "Storm warning: temporarily raise the minimum battery reserve to 80% from 18:00 "
                "to 23:00. Continue to minimize costs."
            ),
            expected_outcome="accept",
            ground_truth={
                "objective": "minimize_cost",
                "reserves": [{"min_soc_pct": 80.0, **_window(18, 23)}],
                "grid_limits": [],
                "qos_level": "standard",
                "allow_capacity_reduction": True,
            },
            tags=("reserve",),
        ),
        ScenarioSpec(
            name="high_traffic_low_solar",
            title="5. High traffic and low solar production",
            description=(
                "Overcast day and traffic 25% above forecast, combined with a demand-response "
                "grid-import cap in the evening."
            ),
            weather="overcast",
            traffic_scale=1.25,
            instruction_issue_hour=10.0,
            instruction_text=(
                "Demand response event: limit grid import to 2 kW between 18:00 and 20:00 while "
                "preserving standard quality of service, and minimize costs otherwise."
            ),
            expected_outcome="accept",
            ground_truth={
                "objective": "minimize_cost",
                "reserves": [],
                "grid_limits": [{"max_import_kw": 2.0, **_window(18, 20)}],
                "qos_level": "standard",
                "allow_capacity_reduction": True,
            },
            tags=("forecast_error", "traffic", "pv", "demand_response"),
        ),
        ScenarioSpec(
            name="infeasible_instruction",
            title="6. Infeasible / contradictory instruction",
            description=(
                "A 90% reserve combined with zero grid import over the evening peak cannot be "
                "satisfied: the battery would have to supply the load without discharging."
            ),
            weather="partly_cloudy",
            instruction_issue_hour=16.0,
            instruction_text=(
                "Keep the battery above 90% from 17:00 to 21:00 and cap grid import at 0 kW during "
                "the same period, with strict quality of service."
            ),
            expected_outcome="reject",
            ground_truth={
                "objective": "minimize_cost",
                "reserves": [{"min_soc_pct": 90.0, **_window(17, 21)}],
                "grid_limits": [{"max_import_kw": 0.0, **_window(17, 21)}],
                "qos_level": "strict",
                "allow_capacity_reduction": True,
            },
            tags=("infeasible",),
        ),
        ScenarioSpec(
            name="ambiguous_instruction",
            title="7. Malformed / ambiguous instruction",
            description="Vague instruction without a reserve level or time window.",
            weather="partly_cloudy",
            instruction_issue_hour=10.0,
            instruction_text="Keep the battery reasonably high for a while and try to save some energy.",
            expected_outcome="clarify",
            ground_truth=None,
            tags=("ambiguous",),
        ),
    )
}


def _rng_streams(seed: int, name: str, n: int) -> list[np.random.Generator]:
    seq = np.random.SeedSequence([int(seed), zlib.crc32(name.encode("utf-8"))])
    return [np.random.default_rng(s) for s in seq.spawn(n)]


def generate_scenario(spec: ScenarioSpec | str, config: AppConfig, seed: int = 0) -> ScenarioData:
    """Generate the profiles of a scenario for a given seed (fully reproducible)."""
    if isinstance(spec, str):
        try:
            spec = SCENARIOS[spec]
        except KeyError as exc:
            raise KeyError(f"unknown scenario '{spec}'; available: {sorted(SCENARIOS)}") from exc
    sim = config.simulation
    n = sim.n_steps
    rng_cloud, rng_traffic, rng_soc, _spare = _rng_streams(seed, spec.name, 4)

    hours_mid = profiles.step_mid_hours(sim)
    clear = profiles.clear_sky_pv_kw(hours_mid, config.pv)
    weather = config.weather.classes[spec.weather]
    forecast_weather = config.weather.classes[spec.forecast_weather or spec.weather]

    pv_actual = clear * profiles.cloud_attenuation(rng_cloud, n, weather, config.weather.ar_coefficient)
    pv_forecast = clear * forecast_weather.mean

    template = profiles.traffic_template(hours_mid, config.traffic)
    traffic_actual = template * spec.traffic_scale * profiles.traffic_noise_factor(rng_traffic, n, config.traffic)
    traffic_forecast = template * spec.forecast_traffic_scale

    for event in spec.events:
        pv_actual, traffic_actual = event.apply(hours_mid, pv_actual, traffic_actual)
        if event.in_forecast:
            pv_forecast, traffic_forecast = event.apply(hours_mid, pv_forecast, traffic_forecast)

    cap = config.traffic.max_traffic
    traffic_actual = np.clip(traffic_actual, 0.0, cap)
    traffic_forecast = np.clip(traffic_forecast, 0.0, cap)
    pv_actual = np.clip(pv_actual, 0.0, None)
    pv_forecast = np.clip(pv_forecast, 0.0, None)

    bat = config.battery
    if spec.initial_soc_pct is not None:
        soc0 = spec.initial_soc_pct
    else:
        jitter = config.scenario.initial_soc_jitter_pct
        soc0 = bat.initial_soc_pct + rng_soc.uniform(-jitter, jitter)
    soc0 = float(np.clip(soc0, bat.soc_min_pct, bat.soc_max_pct))

    issue_step = int(round(spec.instruction_issue_hour / sim.dt_h))
    if not 0 <= issue_step < n:
        raise ValueError("instruction issue time outside the horizon")

    return ScenarioData(
        spec=spec,
        seed=int(seed),
        hours=profiles.step_start_hours(sim),
        pv_actual_kw=pv_actual,
        pv_forecast_kw=pv_forecast,
        traffic_actual=traffic_actual,
        traffic_forecast=traffic_forecast,
        price_eur_per_kwh=profiles.tariff_prices(hours_mid, config.tariff),
        initial_soc_pct=soc0,
        issue_step=issue_step,
    )
