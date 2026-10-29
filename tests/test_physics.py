import numpy as np
import pytest

from energypilot.simulator import physics


def test_bs_power_model(config):
    bs = config.base_station
    # Full capacity, full traffic: all three terms.
    assert physics.bs_power_kw(1.0, 1.0, bs) == pytest.approx(1.2 + 1.0 + 2.0)
    # Only served traffic consumes amplifier power.
    assert physics.bs_power_kw(0.5, 0.8, bs) == pytest.approx(1.2 + 0.5 + 2.0 * 0.5)
    arr = physics.bs_power_kw(np.array([0.4, 1.0]), np.array([0.1, 0.1]), bs)
    assert arr.shape == (2,)


def test_required_capacity_bounds(config):
    bs = config.base_station
    assert physics.required_capacity(0.0, 0.85, bs) == pytest.approx(bs.min_capacity_fraction)
    assert physics.required_capacity(0.51, 0.85, bs) == pytest.approx(0.6)
    assert physics.required_capacity(2.0, 0.85, bs) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        physics.required_capacity(0.5, 0.0, bs)


def test_battery_energy_update_uses_efficiencies(config):
    bat, dt = config.battery, 0.25
    assert physics.next_energy_kwh(20.0, 4.0, 0.0, dt, bat) == pytest.approx(20.0 + 0.95 * 4.0 * 0.25)
    assert physics.next_energy_kwh(20.0, 0.0, 4.0, dt, bat) == pytest.approx(20.0 - 4.0 * 0.25 / 0.95)


def test_soc_trajectory_matches_step_updates(config):
    bat = config.battery
    ch = np.array([2.0, 0.0, 5.0])
    dis = np.array([0.0, 3.0, 0.0])
    traj = physics.soc_trajectory_kwh(20.0, ch, dis, 0.25, bat)
    assert len(traj) == 4
    e = 20.0
    for k in range(3):
        e = physics.next_energy_kwh(e, ch[k], dis[k], 0.25, bat)
        assert traj[k + 1] == pytest.approx(e)


def test_charge_and_discharge_headroom(config):
    bat, dt = config.battery, 0.25
    # Plenty of room: power limit binds.
    assert physics.max_charge_kw(20.0, dt, bat) == pytest.approx(bat.max_charge_kw)
    # Near full: SoC ceiling binds, and charging exactly that power reaches the ceiling.
    p = physics.max_charge_kw(37.5, dt, bat)
    assert p < bat.max_charge_kw
    assert physics.next_energy_kwh(37.5, p, 0, dt, bat) == pytest.approx(bat.e_max_kwh)
    # Near empty: floor binds.
    p = physics.max_discharge_kw(4.5, dt, bat)
    assert physics.next_energy_kwh(4.5, 0, p, dt, bat) == pytest.approx(bat.e_min_kwh)
    # A custom floor above the BMS limit (e.g. an operational reserve).
    assert physics.max_discharge_kw(20.0, dt, bat, floor_kwh=20.0) == 0.0


def test_self_discharge(config):
    bat = config.battery.model_copy(update={"self_discharge_pct_per_day": 2.4})
    r = bat.retention_per_step(0.25)
    assert r == pytest.approx(1 - 0.024 * 0.25 / 24)
    assert physics.next_energy_kwh(10.0, 0, 0, 0.25, bat) == pytest.approx(10.0 * r)


def test_qos_indicators():
    q = physics.qos_indicators(np.array([0.5, 0.0, 1.2]), np.array([1.0, 0.4, 1.0]))
    assert q["utilization"][0] == pytest.approx(0.5)
    assert q["served_ratio"][1] == pytest.approx(1.0)  # no traffic -> fully served
    assert q["served_ratio"][2] == pytest.approx(1 / 1.2)
    assert q["delay_factor"][0] == pytest.approx(2.0)
