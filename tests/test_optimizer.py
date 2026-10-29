import numpy as np
import pytest

from energypilot.core import Forecast
from energypilot.data import generate_scenario
from energypilot.data.forecast import ForecastProvider
from energypilot.instructions import OperationalInstruction, compile_constraints, parse_instruction
from energypilot.optimizer import OptimizerError, diagnose, optimize
from energypilot.simulator import physics


def _setup(config, scenario="normal", step=40, instr=None, seed=0):
    sc = generate_scenario(scenario, config, seed)
    fc = ForecastProvider(sc, config).forecast(step)
    instr = instr or OperationalInstruction(issued_at_minute=step * 15)
    return fc, compile_constraints(instr, config)


def _check_solution(config, res, cons, energy0):
    s = res.strategy
    bat, bs, dt = config.battery, config.base_station, config.simulation.dt_h
    lam, pv = s.forecast.traffic, s.forecast.pv_kw
    load = physics.bs_power_kw(s.capacity_fraction, lam, bs)
    np.testing.assert_allclose(pv - s.pv_curtailed_kw + s.grid_import_kw + s.p_discharge_kw, load + s.p_charge_kw, atol=1e-5)
    np.testing.assert_allclose(
        physics.soc_trajectory_kwh(energy0, s.p_charge_kw, s.p_discharge_kw, dt, bat), s.soc_kwh, atol=1e-5
    )
    assert np.all(s.soc_kwh[1:] >= bat.e_min_kwh - 1e-6) and np.all(s.soc_kwh <= bat.e_max_kwh + 1e-6)
    assert np.all(s.p_charge_kw <= bat.max_charge_kw + 1e-6) and np.all(s.p_discharge_kw <= bat.max_discharge_kw + 1e-6)
    assert not np.any((s.p_charge_kw > 1e-6) & (s.p_discharge_kw > 1e-6))
    t0 = s.start_step
    assert np.all(s.soc_kwh[1:] >= cons.reserve_kwh[t0 + 1 :] - 1e-6)
    assert np.all(s.grid_import_kw <= cons.grid_cap_kw[t0:] + 1e-6)
    assert np.all(s.capacity_fraction >= cons.required_capacity(lam, config) - 1e-6)


def test_feasible_problem_satisfies_all_constraints(config):
    instr = OperationalInstruction.model_validate(
        {
            "reserves": [{"min_soc_pct": 60.0, "start_minute": 1020, "end_minute": 1260}],
            "grid_limits": [{"max_import_kw": 1.5, "start_minute": 1080, "end_minute": 1200}],
            "qos_level": "strict",
            "issued_at_minute": 600,
        }
    )
    fc, cons = _setup(config, instr=instr)
    res = optimize(config, cons, fc, 20.0, terminal_target_kwh=20.0)
    assert res.status == "optimal" and res.feasible
    assert res.model_class == "MILP" and res.n_binaries == len(fc)
    _check_solution(config, res, cons, 20.0)
    assert res.kpis["planned_cost_eur"] > 0


def test_infeasible_problem_is_reported_not_relaxed(config):
    sc = generate_scenario("infeasible_instruction", config, 0)
    instr = parse_instruction(sc.spec.instruction_text, 960, config).instruction
    cons = compile_constraints(instr, config)
    fc = ForecastProvider(sc, config).forecast(sc.issue_step)
    res = optimize(config, cons, fc, 24.0, terminal_target_kwh=24.0)
    assert res.status == "infeasible" and not res.feasible and res.strategy is None
    report = diagnose(config, cons, fc, 24.0, terminal_target_kwh=24.0)
    assert not report.feasible
    assert ["reserve"] in report.conflicting_groups
    assert report.analytic_findings and "grid cap" in report.explanation


def test_reserve_already_violated_now(config):
    instr = OperationalInstruction.model_validate(
        {"reserves": [{"min_soc_pct": 80.0, "start_minute": 600, "end_minute": 720}], "issued_at_minute": 600}
    )
    fc, cons = _setup(config, instr=instr)
    res = optimize(config, cons, fc, 20.0)
    assert res.status == "infeasible" and "below the required reserve" in res.message


def test_objectives_trade_off(config):
    results = {}
    for obj in ("minimize_cost", "minimize_grid_energy", "minimize_peak_import"):
        fc, cons = _setup(config, instr=OperationalInstruction(objective=obj, issued_at_minute=600))
        results[obj] = optimize(config, cons, fc, 20.0, terminal_target_kwh=20.0).strategy
    cost = {k: float(np.sum(s.grid_import_kw * s.forecast.price_eur_per_kwh)) for k, s in results.items()}
    energy = {k: float(np.sum(s.grid_import_kw)) for k, s in results.items()}
    peak = {k: float(np.max(s.grid_import_kw)) for k, s in results.items()}
    assert cost["minimize_cost"] <= min(cost.values()) + 1e-6
    assert energy["minimize_grid_energy"] <= min(energy.values()) + 1e-6
    assert peak["minimize_peak_import"] <= min(peak.values()) + 1e-6


def test_optimizer_not_worse_than_rule_plan(config):
    """On identical forecasts the optimiser's plan costs no more than the rule roll-out."""
    from energypilot.controllers import RuleBasedController
    from energypilot.core import Observation

    fc, cons = _setup(config)
    rule_plan = RuleBasedController(config, cons).plan(
        Observation(40, 10.0, 20.0, 50.0, fc.pv_kw[0], fc.traffic[0], fc.price_eur_per_kwh[0]), fc
    )
    # Use the rule plan's final SoC as the terminal target so both end at the same energy.
    res = optimize(config, cons, fc, 20.0, terminal_target_kwh=float(rule_plan.soc_kwh[-1]))
    rule_cost = float(np.sum(rule_plan.grid_import_kw * fc.price_eur_per_kwh))
    assert res.kpis["planned_cost_eur"] <= rule_cost + 1e-6
    assert res.strategy.soc_kwh[-1] >= rule_plan.soc_kwh[-1] - 1e-6


def test_lp_variant_and_other_solver(config):
    fc, cons = _setup(config)
    base = optimize(config, cons, fc, 20.0, terminal_target_kwh=20.0)
    lp = optimize(config.with_overrides({"optimizer": {"battery_mode_binaries": False}}), cons, fc, 20.0, terminal_target_kwh=20.0)
    assert lp.model_class == "LP" and lp.n_binaries == 0
    assert lp.objective_value <= base.objective_value + 1e-6  # LP relaxation lower bound
    import pulp

    if "HiGHS" in pulp.listSolvers(onlyAvailable=True):
        highs = optimize(config.with_overrides({"optimizer": {"solver": "HiGHS"}}), cons, fc, 20.0, terminal_target_kwh=20.0)
        assert highs.objective_value == pytest.approx(base.objective_value, rel=1e-4)


def test_invalid_inputs_raise_tool_error(config):
    fc, cons = _setup(config)
    bad = Forecast(fc.start_step, fc.pv_kw * np.nan, fc.traffic, fc.price_eur_per_kwh)
    with pytest.raises(OptimizerError):
        optimize(config, cons, bad, 20.0)
    short = Forecast(fc.start_step, fc.pv_kw[:-1], fc.traffic[:-1], fc.price_eur_per_kwh[:-1])
    with pytest.raises(OptimizerError):
        optimize(config, cons, short, 20.0)
    with pytest.raises(OptimizerError):
        optimize(config, cons, fc, 20.0, relax=("everything",))
