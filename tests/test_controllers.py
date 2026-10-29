import numpy as np
import pytest

from energypilot.controllers import MPCController, RuleBasedController
from energypilot.core import Observation
from energypilot.data import generate_scenario
from energypilot.data.forecast import ForecastProvider
from energypilot.instructions import OperationalInstruction, compile_constraints, default_instruction
from energypilot.optimizer import OptimizerError
from energypilot.simulator import BaseStationEnv, physics


def _obs(step=40, soc_kwh=20.0, pv=0.0, traffic=0.5, price=0.21, config=None):
    return Observation(step, step * 0.25, soc_kwh, 100 * soc_kwh / 40, pv, traffic, price)


def _instr(**kw):
    return OperationalInstruction.model_validate(kw)


def _run(config, scenario, instruction, kind, seed=0):
    sc = generate_scenario(scenario, config, seed)
    env = BaseStationEnv(config, sc.pv_actual_kw, sc.traffic_actual, sc.price_eur_per_kwh, sc.initial_soc_pct)
    fp = ForecastProvider(sc, config)
    e0 = env.energy_kwh

    def make(instr):
        c = compile_constraints(instr, config)
        ctrl = RuleBasedController(config, c) if kind == "rule" else MPCController(config, c, terminal_target_kwh=e0)
        return c, ctrl

    cons, ctrl = make(default_instruction(0))
    while not env.done:
        obs = env.observe()
        if obs.step == sc.issue_step:
            cons, ctrl = make(instruction)
        env.step(ctrl.decide(obs, fp.forecast(obs.step)))
    return env.history(), cons, ctrl


def test_rule_capacity_follows_qos(config):
    c = compile_constraints(default_instruction(0), config)
    a = RuleBasedController(config, c).decide(_obs(traffic=0.51))
    assert a.capacity_fraction == pytest.approx(0.6)  # 0.51 / 0.85
    no_reduction = compile_constraints(_instr(allow_capacity_reduction=False), config)
    assert RuleBasedController(config, no_reduction).decide(_obs(traffic=0.1)).capacity_fraction == 1.0


def test_rule_stores_pv_surplus(config):
    c = compile_constraints(default_instruction(0), config)
    a = RuleBasedController(config, c).decide(_obs(pv=6.0, traffic=0.3))
    load = physics.bs_power_kw(a.capacity_fraction, 0.3, config.base_station)
    assert a.p_charge_kw == pytest.approx(6.0 - load) and a.p_discharge_kw == 0


def test_rule_discharges_at_peak_but_not_below_floor(config):
    instr = _instr(reserves=[{"min_soc_pct": 50.0, "start_minute": 1000, "end_minute": 1300}])
    ctrl = RuleBasedController(config, compile_constraints(instr, config))
    a = ctrl.decide(_obs(step=72, soc_kwh=30.0, price=0.32))
    assert 0 < a.p_discharge_kw
    nxt = physics.next_energy_kwh(30.0, 0, a.p_discharge_kw, 0.25, config.battery)
    assert nxt >= 20.0 - 1e-9
    at_floor = ctrl.decide(_obs(step=72, soc_kwh=20.0, price=0.32))
    assert at_floor.p_discharge_kw == pytest.approx(0.0, abs=1e-9)
    standard_price = ctrl.decide(_obs(step=60, soc_kwh=30.0, price=0.21))
    assert standard_price.p_discharge_kw == 0


def test_rule_respects_grid_cap_by_discharging(config):
    instr = _instr(grid_limits=[{"max_import_kw": 1.0, "start_minute": 1080, "end_minute": 1200}])
    a = RuleBasedController(config, compile_constraints(instr, config)).decide(_obs(step=73, soc_kwh=30.0))
    load = physics.bs_power_kw(a.capacity_fraction, 0.5, config.base_station)
    assert load - a.p_discharge_kw == pytest.approx(1.0)


@pytest.mark.parametrize("kind", ["rule", "mpc"])
def test_closed_loop_reserve_is_met(config, kind):
    instr = _instr(reserves=[{"min_soc_pct": 80.0, "start_minute": 1080, "end_minute": 1380}], issued_at_minute=840)
    hist, cons, _ = _run(config, "reserve_increase", instr, kind)
    soc = np.concatenate([[hist.soc_start_kwh.iloc[0]], hist.soc_end_kwh.to_numpy()])
    assert np.all(soc >= cons.reserve_kwh - 1e-6)


def test_rule_plan_rollout_is_consistent(config):
    sc = generate_scenario("reserve_increase", config, 0)
    fc = ForecastProvider(sc, config).forecast(56)
    instr = _instr(reserves=[{"min_soc_pct": 80.0, "start_minute": 1080, "end_minute": 1380}], issued_at_minute=840)
    ctrl = RuleBasedController(config, compile_constraints(instr, config))
    obs = Observation(56, 14.0, 24.0, 60.0, fc.pv_kw[0], fc.traffic[0], fc.price_eur_per_kwh[0])
    plan = ctrl.plan(obs, fc)
    assert len(plan) == 40 and len(plan.soc_kwh) == 41
    traj = physics.soc_trajectory_kwh(24.0, plan.p_charge_kw, plan.p_discharge_kw, 0.25, config.battery)
    np.testing.assert_allclose(traj, plan.soc_kwh, atol=1e-9)
    assert plan.instruction_fingerprint == instr.fingerprint()


def test_mpc_falls_back_when_optimizer_fails(config):
    events = []

    def broken(*args, **kwargs):
        raise OptimizerError("injected failure")

    c = compile_constraints(default_instruction(0), config)
    mpc = MPCController(config, c, terminal_target_kwh=None, optimize_fn=broken, event_sink=lambda e, p: events.append(e))
    sc = generate_scenario("normal", config, 0)
    fc = ForecastProvider(sc, config).forecast(0)
    a = mpc.decide(_obs(step=0, soc_kwh=30.0, pv=0.0, traffic=0.2, price=0.15), fc)
    assert a == RuleBasedController(config, c).decide(_obs(step=0, soc_kwh=30.0, pv=0.0, traffic=0.2, price=0.15))
    assert mpc.stats["replan_failures"] == 1 and mpc.stats["fallback_steps"] == 1
    assert "mpc_replan_failed" in events and "mpc_fallback_rule_based" in events


def test_mpc_does_not_adopt_rejected_plans(config):
    class Reject:
        accepted = False

        def summary(self):
            return {"accepted": False}

    c = compile_constraints(default_instruction(0), config)
    mpc = MPCController(config, c, terminal_target_kwh=None, plan_validator=lambda s: Reject())
    sc = generate_scenario("normal", config, 0)
    mpc.decide(_obs(step=0, soc_kwh=30.0, traffic=0.2, price=0.15), ForecastProvider(sc, config).forecast(0))
    assert mpc.plan is None and mpc.stats["validation_rejections"] == 1 and mpc.stats["fallback_steps"] == 1


def test_mpc_beats_rules_on_adjusted_cost_in_nominal_case(config):
    """Sanity check of the optimiser's value (not an experimental result)."""
    instr = _instr(reserves=[{"min_soc_pct": 70.0, "start_minute": 0, "end_minute": 120}])
    out = {}
    for kind in ("rule", "mpc"):
        hist, _, _ = _run(config, "normal", instr, kind)
        mean_price = hist.price_eur_per_kwh.mean()
        out[kind] = hist.cost_eur.sum() + (hist.soc_start_kwh.iloc[0] - hist.soc_end_kwh.iloc[-1]) * mean_price
    assert out["mpc"] < out["rule"]
