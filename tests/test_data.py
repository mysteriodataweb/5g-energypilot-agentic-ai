import numpy as np
import pytest

from energypilot.data import SCENARIOS, generate_scenario
from energypilot.data.forecast import ForecastProvider
from energypilot.data.profiles import tariff_prices


def test_seven_scenarios_with_expected_outcomes():
    assert len(SCENARIOS) == 7
    outcomes = {s.expected_outcome for s in SCENARIOS.values()}
    assert outcomes == {"accept", "clarify", "reject"}
    for spec in SCENARIOS.values():
        assert (spec.ground_truth is None) == (spec.expected_outcome == "clarify")


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_same_seed_reproduces_identical_data(config, name):
    a = generate_scenario(name, config, seed=7)
    b = generate_scenario(name, config, seed=7)
    for attr in ("pv_actual_kw", "pv_forecast_kw", "traffic_actual", "traffic_forecast", "price_eur_per_kwh"):
        np.testing.assert_array_equal(getattr(a, attr), getattr(b, attr))
    assert a.initial_soc_pct == b.initial_soc_pct


def test_different_seeds_differ(config):
    a = generate_scenario("traffic_peak", config, seed=1)
    b = generate_scenario("traffic_peak", config, seed=2)
    assert not np.allclose(a.traffic_actual, b.traffic_actual)
    assert not np.allclose(a.pv_actual_kw, b.pv_actual_kw)


def test_pv_is_physically_consistent(config):
    sc = generate_scenario("normal", config, seed=0)
    hours = sc.hours + config.simulation.dt_h / 2
    night = (hours < config.pv.sunrise_hour) | (hours > config.pv.sunset_hour)
    assert np.all(sc.pv_actual_kw[night] == 0)
    assert np.all(sc.pv_actual_kw >= 0)
    assert sc.pv_actual_kw.max() <= config.pv.peak_power_kw + 1e-9
    noon = np.argmin(np.abs(hours - 13.5))
    assert sc.pv_actual_kw[noon] > 0.8 * config.pv.peak_power_kw


def test_unforeseen_events_absent_from_forecast(config):
    sc = generate_scenario("solar_drop", config, seed=0)
    window = (sc.hours >= 12) & (sc.hours < 15)
    assert sc.pv_actual_kw[window].mean() < 0.35 * sc.pv_forecast_kw[window].mean()
    tp = generate_scenario("traffic_peak", config, seed=0)
    evening = (tp.hours >= 19) & (tp.hours < 21)
    assert tp.traffic_actual[evening].mean() > tp.traffic_forecast[evening].mean() + 0.3


def test_traffic_profile_shape(config):
    sc = generate_scenario("normal", config, seed=0)
    night = sc.traffic_forecast[(sc.hours >= 2) & (sc.hours < 5)].mean()
    evening = sc.traffic_forecast[(sc.hours >= 19.5) & (sc.hours < 21.5)].mean()
    assert evening > 4 * night
    assert np.all(sc.traffic_actual <= config.traffic.max_traffic)


def test_tariff_lookup(config):
    prices = tariff_prices(np.array([1.0, 12.0, 18.0, 21.5, 23.0]), config.tariff)
    np.testing.assert_allclose(prices, [0.15, 0.21, 0.32, 0.21, 0.15])


def test_initial_soc_jitter_within_bounds(config):
    socs = [generate_scenario("reserve_increase", config, seed=s).initial_soc_pct for s in range(20)]
    assert min(socs) >= 55 - 1e-9 and max(socs) <= 65 + 1e-9
    assert len(set(socs)) > 1
    assert generate_scenario("normal", config, seed=3).initial_soc_pct == 80.0


def test_unknown_scenario(config):
    with pytest.raises(KeyError):
        generate_scenario("nope", config)


def test_forecast_provider(config):
    sc = generate_scenario("solar_drop", config, seed=0)
    perfect = ForecastProvider(sc, config, mode="perfect").forecast(40)
    np.testing.assert_array_equal(perfect.pv_kw, sc.pv_actual_kw[40:])
    fp = ForecastProvider(sc, config)
    f = fp.forecast(50)
    assert len(f) == 96 - 50
    assert f.pv_kw[0] == sc.pv_actual_kw[50]  # measured now
    assert f.traffic[0] == sc.traffic_actual[50]
    # Persistence correction: f_adj[k] = f[k] * (1 + (r - 1) * decay**k), relaxing to the forecast.
    r = sc.pv_actual_kw[50] / sc.pv_forecast_kw[50]
    for k in (1, 5, 16):
        expected = sc.pv_forecast_kw[50 + k] * (1 + (r - 1) * config.forecast.pv_decay_per_step**k)
        assert f.pv_kw[k] == pytest.approx(expected)
    assert f.pv_kw[16] / sc.pv_forecast_kw[66] > f.pv_kw[1] / sc.pv_forecast_kw[51]
    with pytest.raises(IndexError):
        fp.forecast(96)
