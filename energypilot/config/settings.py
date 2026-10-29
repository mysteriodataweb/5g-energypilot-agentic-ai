"""Typed, validated configuration for the whole prototype.

Every physical or behavioural assumption used by the simulator, the controllers,
the optimiser, the safety validator and the agent is a field of :class:`AppConfig`.
Default values are mirrored in ``config/default.toml`` (a test checks that both
stay identical). All values are *assumptions* for a synthetic study, not
measurements of a real site.

Units are encoded in field names: ``_kw`` (power), ``_kwh`` (energy),
``_pct`` (percent of battery capacity), ``_hour`` (clock hour), ``_eur_per_kwh``.
"""

from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.toml"


class _Strict(BaseModel):
    """Base model: unknown keys are errors, so typos in config files are caught."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SimulationConfig(_Strict):
    """Discrete-time grid. Step 0 starts at 00:00 of the simulated day."""

    time_step_minutes: int = Field(15, ge=1, le=60)
    horizon_hours: int = Field(24, ge=1, le=48)

    @model_validator(mode="after")
    def _divides(self) -> "SimulationConfig":
        if (self.horizon_hours * 60) % self.time_step_minutes != 0:
            raise ValueError("time_step_minutes must divide the horizon exactly")
        return self

    @property
    def dt_h(self) -> float:
        """Time-step duration in hours (converts kW to kWh per step)."""
        return self.time_step_minutes / 60.0

    @property
    def n_steps(self) -> int:
        return self.horizon_hours * 60 // self.time_step_minutes


class BatteryConfig(_Strict):
    capacity_kwh: float = Field(40.0, gt=0)
    soc_min_pct: float = Field(10.0, ge=0, lt=100)
    soc_max_pct: float = Field(95.0, gt=0, le=100)
    initial_soc_pct: float = Field(60.0, ge=0, le=100)
    max_charge_kw: float = Field(10.0, gt=0)
    max_discharge_kw: float = Field(10.0, gt=0)
    charge_efficiency: float = Field(0.95, gt=0, le=1)
    discharge_efficiency: float = Field(0.95, gt=0, le=1)
    self_discharge_pct_per_day: float = Field(0.0, ge=0, lt=100)
    degradation_cost_eur_per_kwh: float = Field(0.02, ge=0)

    @model_validator(mode="after")
    def _bounds(self) -> "BatteryConfig":
        if self.soc_min_pct >= self.soc_max_pct:
            raise ValueError("soc_min_pct must be strictly below soc_max_pct")
        if not self.soc_min_pct <= self.initial_soc_pct <= self.soc_max_pct:
            raise ValueError("initial_soc_pct must lie within [soc_min_pct, soc_max_pct]")
        return self

    @property
    def e_min_kwh(self) -> float:
        return self.capacity_kwh * self.soc_min_pct / 100.0

    @property
    def e_max_kwh(self) -> float:
        return self.capacity_kwh * self.soc_max_pct / 100.0

    def pct_to_kwh(self, pct: float) -> float:
        return self.capacity_kwh * pct / 100.0

    def kwh_to_pct(self, kwh: float) -> float:
        return 100.0 * kwh / self.capacity_kwh

    def retention_per_step(self, dt_h: float) -> float:
        """Fraction of stored energy kept after one step (self-discharge)."""
        return 1.0 - (self.self_discharge_pct_per_day / 100.0) * dt_h / 24.0


class PVConfig(_Strict):
    """Photovoltaic array, expressed as AC output after the inverter."""

    peak_power_kw: float = Field(6.0, gt=0, description="AC output at solar noon on a clear day")
    sunrise_hour: float = Field(6.5, ge=0, lt=24)
    sunset_hour: float = Field(20.5, gt=0, le=24)
    shape_exponent: float = Field(1.5, gt=0)

    @model_validator(mode="after")
    def _order(self) -> "PVConfig":
        if self.sunrise_hour >= self.sunset_hour:
            raise ValueError("sunrise_hour must precede sunset_hour")
        return self


class WeatherClass(_Strict):
    """Cloud attenuation statistics (multiplier on clear-sky PV)."""

    mean: float = Field(ge=0, le=1)
    std: float = Field(ge=0)


class WeatherConfig(_Strict):
    classes: dict[str, WeatherClass] = Field(
        default_factory=lambda: {
            "sunny": WeatherClass(mean=0.95, std=0.03),
            "partly_cloudy": WeatherClass(mean=0.70, std=0.15),
            "overcast": WeatherClass(mean=0.30, std=0.08),
        }
    )
    ar_coefficient: float = Field(0.9, ge=0, lt=1)


class BaseStationConfig(_Strict):
    """Linear load-dependent power model (EARTH-style), see docs/mathematical_model.md.

    P_bs = fixed + capacity_power * c + traffic_power * min(traffic, c)
    where c is the active capacity fraction and traffic is the normalised load.
    """

    fixed_power_kw: float = Field(1.2, ge=0)
    capacity_power_kw: float = Field(1.0, ge=0)
    traffic_power_kw: float = Field(2.0, ge=0)
    min_capacity_fraction: float = Field(0.4, gt=0, le=1)
    ran_auto_wakeup: bool = Field(
        True, description="RAN re-activates capacity when traffic exceeds the active capacity"
    )


class GridConfig(_Strict):
    max_import_kw: float = Field(15.0, gt=0)


class TariffPeriod(_Strict):
    start_hour: float = Field(ge=0, lt=24)
    end_hour: float = Field(gt=0, le=24)
    price_eur_per_kwh: float = Field(ge=0)
    label: str


class TariffConfig(_Strict):
    """Illustrative time-of-use tariff. Not a real commercial offer."""

    periods: list[TariffPeriod] = Field(
        default_factory=lambda: [
            TariffPeriod(start_hour=0, end_hour=6, price_eur_per_kwh=0.15, label="off-peak"),
            TariffPeriod(start_hour=6, end_hour=17, price_eur_per_kwh=0.21, label="standard"),
            TariffPeriod(start_hour=17, end_hour=21, price_eur_per_kwh=0.32, label="peak"),
            TariffPeriod(start_hour=21, end_hour=22, price_eur_per_kwh=0.21, label="standard"),
            TariffPeriod(start_hour=22, end_hour=24, price_eur_per_kwh=0.15, label="off-peak"),
        ]
    )

    @model_validator(mode="after")
    def _coverage(self) -> "TariffConfig":
        spans = sorted((p.start_hour, p.end_hour) for p in self.periods)
        cursor = 0.0
        for start, end in spans:
            if abs(start - cursor) > 1e-9 or end <= start:
                raise ValueError("tariff periods must tile [0, 24) without gaps or overlaps")
            cursor = end
        if abs(cursor - 24.0) > 1e-9:
            raise ValueError("tariff periods must end at 24:00")
        return self


class TrafficPeak(_Strict):
    center_hour: float = Field(ge=0, lt=24)
    width_hours: float = Field(gt=0)
    amplitude: float = Field(ge=0)


class TrafficConfig(_Strict):
    """Normalised traffic: 1.0 = the site's full radio capacity (all layers active)."""

    baseline: float = Field(0.10, ge=0)
    peaks: list[TrafficPeak] = Field(
        default_factory=lambda: [
            TrafficPeak(center_hour=9.0, width_hours=1.6, amplitude=0.32),
            TrafficPeak(center_hour=13.5, width_hours=2.2, amplitude=0.38),
            TrafficPeak(center_hour=20.5, width_hours=2.0, amplitude=0.62),
        ]
    )
    noise_std: float = Field(0.06, ge=0)
    noise_ar: float = Field(0.8, ge=0, lt=1)
    max_traffic: float = Field(1.3, gt=0)


class QoSConfig(_Strict):
    """QoS proxy: maximum radio utilisation (traffic / active capacity) per level."""

    max_utilization_by_level: dict[str, float] = Field(
        default_factory=lambda: {"strict": 0.70, "standard": 0.85, "relaxed": 0.95}
    )

    @model_validator(mode="after")
    def _range(self) -> "QoSConfig":
        for level in ("strict", "standard", "relaxed"):
            if level not in self.max_utilization_by_level:
                raise ValueError(f"missing QoS level '{level}'")
        for value in self.max_utilization_by_level.values():
            if not 0 < value <= 1:
                raise ValueError("max utilisation must lie in (0, 1]")
        return self


class ScenarioGenConfig(_Strict):
    initial_soc_jitter_pct: float = Field(5.0, ge=0)


class ForecastConfig(_Strict):
    mode: Literal["persistence_corrected", "perfect"] = "persistence_corrected"
    pv_decay_per_step: float = Field(0.85, ge=0, le=1)
    traffic_decay_per_step: float = Field(0.90, ge=0, le=1)


class OptimizerConfig(_Strict):
    solver: Literal["CBC", "HiGHS"] = "CBC"
    time_limit_s: float = Field(10.0, gt=0)
    battery_mode_binaries: bool = True
    replan_interval_steps: int = Field(4, ge=1)
    terminal_soc_penalty_eur_per_kwh: float = Field(0.40, ge=0)
    curtailment_penalty_eur_per_kwh: float = Field(0.001, ge=0)
    secondary_cost_weight: float = Field(0.01, ge=0)


class RuleControllerConfig(_Strict):
    peak_price_threshold_eur_per_kwh: float = Field(0.30, ge=0)
    offpeak_price_threshold_eur_per_kwh: float = Field(0.16, ge=0)
    offpeak_charge_target_pct: float = Field(50.0, ge=0, le=100)
    offpeak_charge_power_kw: float = Field(5.0, ge=0)
    reserve_precharge_margin: float = Field(1.3, ge=1.0)


class AgentConfig(_Strict):
    max_orchestration_turns: int = Field(8, ge=1, le=30)
    llm_timeout_s: float = Field(30.0, gt=0)
    max_tokens: int = Field(1500, ge=64)
    max_instruction_chars: int = Field(1000, ge=10)


class SafetyConfig(_Strict):
    power_tolerance_kw: float = Field(1e-3, ge=0)
    energy_tolerance_kwh: float = Field(1e-3, ge=0)
    capacity_tolerance: float = Field(1e-4, ge=0)


class AppConfig(_Strict):
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    battery: BatteryConfig = Field(default_factory=BatteryConfig)
    pv: PVConfig = Field(default_factory=PVConfig)
    weather: WeatherConfig = Field(default_factory=WeatherConfig)
    base_station: BaseStationConfig = Field(default_factory=BaseStationConfig)
    grid: GridConfig = Field(default_factory=GridConfig)
    tariff: TariffConfig = Field(default_factory=TariffConfig)
    traffic: TrafficConfig = Field(default_factory=TrafficConfig)
    qos: QoSConfig = Field(default_factory=QoSConfig)
    scenario: ScenarioGenConfig = Field(default_factory=ScenarioGenConfig)
    forecast: ForecastConfig = Field(default_factory=ForecastConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    rule_controller: RuleControllerConfig = Field(default_factory=RuleControllerConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)

    def with_overrides(self, overrides: dict[str, Any]) -> "AppConfig":
        """Return a new validated config with nested ``overrides`` applied."""
        merged = _deep_merge(self.model_dump(), overrides)
        return AppConfig.model_validate(merged)


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    path: str | Path | None = DEFAULT_CONFIG_PATH, overrides: dict[str, Any] | None = None
) -> AppConfig:
    """Load a TOML configuration file and apply optional nested overrides.

    ``path=None`` returns the built-in defaults (identical to ``config/default.toml``).
    """
    data: dict[str, Any] = {}
    if path is not None:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    if overrides:
        data = _deep_merge(data, overrides)
    return AppConfig.model_validate(data)
