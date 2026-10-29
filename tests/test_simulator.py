import numpy as np
import pytest

from energypilot.core import ControlAction, InvalidActionError
from energypilot.simulator import physics


def _balance_residual(r):
    return r.pv_used_kw + r.grid_import_kw + r.p_discharge_kw - r.load_served_kw - r.p_charge_kw


def test_energy_balance_holds_every_step(flat_env, config):
    rng = np.random.default_rng(1)
    env = flat_env(pv_kw=3.0, traffic=0.4, soc_pct=50)
    while not env.done:
        action = ControlAction(
            p_charge_kw=float(rng.uniform(0, 12)) if rng.random() < 0.5 else 0.0,
            p_discharge_kw=float(rng.uniform(0, 12)) if rng.random() < 0.5 else 0.0,
            capacity_fraction=float(rng.uniform(0, 1.2)),
        )
        r = env.step(action)
        assert abs(_balance_residual(r)) < 1e-9
        assert r.grid_import_kw >= 0 and r.pv_curtailed_kw >= -1e-12
        assert not (r.p_charge_kw > 1e-9 and r.p_discharge_kw > 1e-9)
        assert config.battery.e_min_kwh - 1e-9 <= r.soc_end_kwh <= config.battery.e_max_kwh + 1e-9


def test_soc_follows_battery_model(flat_env, config):
    env = flat_env(pv_kw=0.0, traffic=0.3, soc_pct=50)
    e0 = env.energy_kwh
    r = env.step(ControlAction(p_charge_kw=4.0, capacity_fraction=1.0))
    assert r.soc_end_kwh == pytest.approx(physics.next_energy_kwh(e0, 4.0, 0.0, 0.25, config.battery))
    assert r.soc_end_pct == pytest.approx(100 * r.soc_end_kwh / config.battery.capacity_kwh)


def test_charge_power_limit_is_enforced(flat_env, config):
    env = flat_env(soc_pct=30)
    r = env.step(ControlAction(p_charge_kw=25.0))
    assert r.p_charge_kw == pytest.approx(config.battery.max_charge_kw)
    assert "charge_limited_power" in r.adjustments


def test_discharge_stops_at_soc_min(flat_env, config):
    env = flat_env(soc_pct=11)
    r = env.step(ControlAction(p_discharge_kw=10.0))
    assert r.soc_end_kwh == pytest.approx(config.battery.e_min_kwh)
    assert "discharge_limited_soc_min" in r.adjustments
    r2 = env.step(ControlAction(p_discharge_kw=10.0))
    assert r2.p_discharge_kw == pytest.approx(0.0, abs=1e-9)


def test_charge_stops_at_soc_max(flat_env, config):
    env = flat_env(soc_pct=94)
    for _ in range(3):
        r = env.step(ControlAction(p_charge_kw=10.0))
    assert r.soc_end_kwh == pytest.approx(config.battery.e_max_kwh)


def test_pv_surplus_is_curtailed_without_export(flat_env, config):
    env = flat_env(pv_kw=8.0, traffic=0.2, soc_pct=50)
    r = env.step(ControlAction(capacity_fraction=0.4))
    expected_load = physics.bs_power_kw(0.4, 0.2, config.base_station)
    assert r.grid_import_kw == 0.0
    assert r.pv_curtailed_kw == pytest.approx(8.0 - expected_load)
    # Charging with the surplus reduces curtailment one-for-one.
    r2 = env.step(ControlAction(p_charge_kw=2.0, capacity_fraction=0.4))
    assert r2.pv_curtailed_kw == pytest.approx(8.0 - expected_load - 2.0)


def test_grid_covers_deficit_and_cost(flat_env, config):
    env = flat_env(pv_kw=1.0, traffic=0.5, price=0.3, soc_pct=50)
    r = env.step(ControlAction(capacity_fraction=1.0))
    load = physics.bs_power_kw(1.0, 0.5, config.base_station)
    assert r.grid_import_kw == pytest.approx(load - 1.0)
    assert r.cost_eur == pytest.approx((load - 1.0) * 0.25 * 0.3)


def test_simultaneous_setpoints_are_netted(flat_env):
    env = flat_env(soc_pct=50)
    r = env.step(ControlAction(p_charge_kw=5.0, p_discharge_kw=2.0))
    assert r.p_charge_kw == pytest.approx(3.0) and r.p_discharge_kw == 0.0
    assert "simultaneous_charge_discharge_netted" in r.adjustments


def test_grid_connection_limit_reduces_charge_then_sheds(flat_env, config):
    cfg = config.with_overrides({"grid": {"max_import_kw": 5.0}})
    env = flat_env(traffic=1.0, soc_pct=50, cfg=cfg)
    r = env.step(ControlAction(p_charge_kw=10.0, capacity_fraction=1.0))
    assert r.grid_import_kw == pytest.approx(5.0)
    assert r.p_charge_kw == pytest.approx(5.0 - 4.2)
    assert r.unserved_kw == 0.0
    cfg2 = config.with_overrides({"grid": {"max_import_kw": 3.0}})
    env2 = flat_env(traffic=1.0, soc_pct=50, cfg=cfg2)
    r2 = env2.step(ControlAction(capacity_fraction=1.0))
    assert r2.unserved_kw == pytest.approx(1.2)
    assert "load_shed_grid_limit" in r2.adjustments
    assert abs(_balance_residual(r2)) < 1e-9


def test_ran_auto_wakeup(flat_env):
    env = flat_env(traffic=0.9)
    r = env.step(ControlAction(capacity_fraction=0.5))
    assert r.capacity_fraction == pytest.approx(0.9)
    assert "ran_auto_wakeup" in r.adjustments


def test_invalid_actions_raise():
    with pytest.raises(InvalidActionError):
        ControlAction(p_charge_kw=float("nan"))
    with pytest.raises(InvalidActionError):
        ControlAction(p_discharge_kw=-1.0)


def test_horizon_end(flat_env):
    env = flat_env()
    while not env.done:
        env.step(ControlAction())
    assert len(env.history()) == 96
    with pytest.raises(RuntimeError):
        env.step(ControlAction())


def test_invalid_profiles_rejected(config):
    from energypilot.simulator import BaseStationEnv

    n = config.simulation.n_steps
    with pytest.raises(ValueError):
        BaseStationEnv(config, np.zeros(n - 1), np.zeros(n), np.zeros(n), 50)
    with pytest.raises(ValueError):
        BaseStationEnv(config, -np.ones(n), np.zeros(n), np.zeros(n), 50)
    with pytest.raises(ValueError):
        BaseStationEnv(config, np.zeros(n), np.zeros(n), np.zeros(n), 99)
