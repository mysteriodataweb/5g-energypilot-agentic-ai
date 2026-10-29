import pytest
from pydantic import ValidationError

from energypilot.config import AppConfig, load_config


def test_default_toml_matches_builtin_defaults():
    assert load_config() == AppConfig()


def test_derived_units(config):
    assert config.simulation.dt_h == pytest.approx(0.25)
    assert config.simulation.n_steps == 96
    assert config.battery.e_min_kwh == pytest.approx(4.0)
    assert config.battery.e_max_kwh == pytest.approx(38.0)


def test_overrides_are_validated(config):
    cfg = config.with_overrides({"battery": {"capacity_kwh": 80.0}})
    assert cfg.battery.capacity_kwh == 80.0
    assert cfg.battery.max_charge_kw == config.battery.max_charge_kw
    with pytest.raises(ValidationError):
        config.with_overrides({"battery": {"soc_min_pct": 96.0}})


@pytest.mark.parametrize(
    "overrides",
    [
        {"simulation": {"time_step_minutes": 7}},
        {"battery": {"charge_efficiency": 1.2}},
        {"battery": {"initial_soc_pct": 99.0}},
        {"pv": {"sunrise_hour": 21.0}},
        {"qos": {"max_utilization_by_level": {"strict": 1.5, "standard": 0.8, "relaxed": 0.9}}},
        {"tariff": {"periods": [{"start_hour": 0, "end_hour": 12, "price_eur_per_kwh": 0.2, "label": "x"}]}},
        {"unknown_section": {"a": 1}},
    ],
)
def test_invalid_configurations_are_rejected(overrides):
    with pytest.raises(ValidationError):
        AppConfig().with_overrides(overrides)


def test_other_time_steps_supported(config):
    cfg = config.with_overrides({"simulation": {"time_step_minutes": 30}})
    assert cfg.simulation.n_steps == 48
    assert cfg.simulation.dt_h == 0.5
