import dataclasses
import json

import numpy as np
import pytest

from energypilot.controllers import RuleBasedController
from energypilot.core import ControlAction, Observation
from energypilot.data import generate_scenario
from energypilot.data.forecast import ForecastProvider
from energypilot.instructions import OperationalInstruction, compile_constraints, parse_instruction
from energypilot.optimizer import optimize
from energypilot.safety import AuditLogger, RuntimeShield, SafetyValidator

INSTR = OperationalInstruction.model_validate(
    {
        "reserves": [{"min_soc_pct": 60.0, "start_minute": 1020, "end_minute": 1260}],
        "grid_limits": [{"max_import_kw": 2.0, "start_minute": 1080, "end_minute": 1200}],
        "issued_at_minute": 600,
    }
)


@pytest.fixture(scope="module")
def plan(config):
    sc = generate_scenario("normal", config, 0)
    fc = ForecastProvider(sc, config).forecast(40)
    res = optimize(config, compile_constraints(INSTR, config), fc, 20.0, terminal_target_kwh=20.0)
    assert res.feasible
    return res.strategy


def _tampered(strategy, **changes):
    return dataclasses.replace(strategy, **changes)


def _failed(report):
    return {c.name for c in report.failed()}


def test_accepts_valid_optimizer_plan(config, plan):
    report = SafetyValidator(config).validate(plan, INSTR, current_energy_kwh=20.0, reference_forecast=plan.forecast)
    assert report.accepted, report.reasons()
    assert {c.name for c in report.checks} >= {"reserve", "grid_import_limits", "qos_capacity", "energy_balance"}


def test_accepts_valid_rule_plan(config):
    sc = generate_scenario("normal", config, 0)
    fc = ForecastProvider(sc, config).forecast(40)
    instr = OperationalInstruction(issued_at_minute=600)
    obs = Observation(40, 10.0, 20.0, 50.0, fc.pv_kw[0], fc.traffic[0], fc.price_eur_per_kwh[0])
    rule_plan = RuleBasedController(config, compile_constraints(instr, config)).plan(obs, fc)
    assert SafetyValidator(config).validate(rule_plan, instr, current_energy_kwh=20.0).accepted


def test_rejects_power_limit_violation(config, plan):
    ch = plan.p_charge_kw.copy()
    ch[3] = config.battery.max_charge_kw + 2
    report = SafetyValidator(config).validate(_tampered(plan, p_charge_kw=ch), INSTR)
    assert not report.accepted
    assert {"battery_power_limits", "soc_consistency", "energy_balance"} <= _failed(report)


def test_rejects_claimed_soc_that_does_not_follow_physics(config, plan):
    soc = plan.soc_kwh.copy()
    soc[10:] += 1.0
    report = SafetyValidator(config).validate(_tampered(plan, soc_kwh=soc), INSTR)
    assert _failed(report) == {"soc_consistency"}


def test_rejects_plan_built_for_another_instruction(config, plan):
    relaxed = OperationalInstruction.model_validate({**INSTR.model_dump(mode="json"), "reserves": []})
    report = SafetyValidator(config).validate(_tampered(plan, instruction_fingerprint=relaxed.fingerprint()), INSTR)
    assert "instruction_match" in _failed(report)


def test_rejects_reserve_violation(config, plan):
    dis = plan.p_discharge_kw.copy()
    ch = plan.p_charge_kw.copy()
    k = 1060 // 15 - plan.start_step  # inside the reserve window
    ch[k - 6 : k] = 0.0
    dis[k - 6 : k] = 8.0
    soc = np.concatenate([[plan.soc_kwh[0]], np.zeros(len(ch))])
    from energypilot.simulator import physics

    soc = physics.soc_trajectory_kwh(plan.soc_kwh[0], ch, dis, 0.25, config.battery)
    grid = np.maximum(plan.grid_import_kw - (dis - plan.p_discharge_kw) + (ch - plan.p_charge_kw), 0)
    report = SafetyValidator(config).validate(_tampered(plan, p_discharge_kw=dis, p_charge_kw=ch, soc_kwh=soc, grid_import_kw=grid), INSTR)
    assert "reserve" in _failed(report)
    reserve = next(c for c in report.checks if c.name == "reserve")
    assert reserve.violations and reserve.max_violation > 0


def test_rejects_grid_cap_and_qos_violations(config, plan):
    k = 1100 // 15 - plan.start_step
    grid = plan.grid_import_kw.copy()
    grid[k] += 3.0
    curt = plan.pv_curtailed_kw.copy()
    report = SafetyValidator(config).validate(_tampered(plan, grid_import_kw=grid, pv_curtailed_kw=curt), INSTR)
    assert {"grid_import_limits", "energy_balance"} <= _failed(report)
    cap = plan.capacity_fraction.copy()
    peak = int(np.argmax(plan.forecast.traffic))
    cap[peak] = config.base_station.min_capacity_fraction
    report = SafetyValidator(config).validate(_tampered(plan, capacity_fraction=cap), INSTR)
    assert "qos_capacity" in _failed(report)


def test_rejects_malformed_plan(config, plan):
    ch = plan.p_charge_kw.copy()
    ch[0] = np.nan
    report = SafetyValidator(config).validate(_tampered(plan, p_charge_kw=ch), INSTR)
    assert not report.accepted and _failed(report) == {"parameters_valid"}


def test_detects_forecast_and_initial_soc_mismatch(config, plan):
    other = ForecastProvider(generate_scenario("solar_drop", config, 0), config).forecast(40)
    report = SafetyValidator(config).validate(plan, INSTR, current_energy_kwh=25.0, reference_forecast=other)
    assert {"forecast_integrity", "initial_soc"} <= _failed(report)


def test_infeasible_instruction_rule_plan_rejected(config):
    sc = generate_scenario("infeasible_instruction", config, 0)
    instr = parse_instruction(sc.spec.instruction_text, 960, config).instruction
    fc = ForecastProvider(sc, config).forecast(sc.issue_step)
    obs = Observation(64, 16.0, 26.0, 65.0, fc.pv_kw[0], fc.traffic[0], fc.price_eur_per_kwh[0])
    rule_plan = RuleBasedController(config, compile_constraints(instr, config)).plan(obs, fc)
    report = SafetyValidator(config).validate(rule_plan, instr)
    assert not report.accepted
    assert {"reserve"} <= _failed(report) or {"grid_import_limits"} <= _failed(report)


def test_terminal_soc_is_soft(config, plan):
    report = SafetyValidator(config).validate(plan, INSTR, terminal_target_kwh=config.battery.e_max_kwh)
    assert report.accepted
    assert any(c.name == "terminal_soc" and not c.passed and c.severity == "soft" for c in report.checks)


# ------------------------------------------------------------------ shield


def test_shield_projects_unsafe_actions(config):
    instr = OperationalInstruction.model_validate(
        {
            "reserves": [{"min_soc_pct": 50.0, "start_minute": 0, "end_minute": 1440}],
            "grid_limits": [{"max_import_kw": 4.0, "start_minute": 0, "end_minute": 1440}],
            "qos_level": "strict",
        }
    )
    cons = compile_constraints(instr, config)
    shield = RuntimeShield(config)
    obs = Observation(10, 2.5, 20.5, 51.25, 0.0, 0.63, 0.15)
    safe, notes = shield.filter(obs, ControlAction(p_discharge_kw=10.0, capacity_fraction=0.4), cons)
    assert safe.capacity_fraction == pytest.approx(0.9)  # 0.63 / 0.70
    assert "capacity_raised_for_qos" in notes and "discharge_limited_to_floor" in notes
    assert safe.p_discharge_kw <= (20.5 - 20.0) * 0.95 / 0.25 + 1e-9
    safe, notes = shield.filter(obs, ControlAction(p_charge_kw=10.0, capacity_fraction=0.9), cons)
    assert "charge_reduced_for_grid_cap" in notes
    from energypilot.simulator import physics

    assert physics.bs_power_kw(0.9, 0.63, config.base_station) + safe.p_charge_kw == pytest.approx(4.0)
    safe, notes = shield.filter(obs, ControlAction(p_charge_kw=1.0, p_discharge_kw=0.4, capacity_fraction=0.9), cons)
    assert safe.p_charge_kw == pytest.approx(0.6) and safe.p_discharge_kw == 0 and notes == ["netted_simultaneous_setpoints"]
    untouched, notes = shield.filter(obs, ControlAction(p_charge_kw=0.5, capacity_fraction=0.9), cons)
    assert notes == [] and untouched == ControlAction(p_charge_kw=0.5, capacity_fraction=0.9)


# ------------------------------------------------------------------ audit log


def test_audit_log_chain_and_file(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLogger(run_id="r1", path=path)
    log.log("instruction_received", text="keep 70%", issued_at_minute=np.int64(600))
    log.log("tool_call", tool="run_optimizer", arrays=np.array([1.0, np.inf]), status="ok")
    log.log("decision", status="accepted")
    assert log.verify_chain()
    loaded = AuditLogger.load(path)
    assert [r["event"] for r in loaded] == ["instruction_received", "tool_call", "decision"]
    assert loaded[0]["payload"]["issued_at_minute"] == 600
    assert loaded[1]["payload"]["arrays"] == [1.0, "inf"]
    assert AuditLogger.verify_records(loaded)
    loaded[1]["payload"]["status"] = "error"  # tampering is detected
    assert not AuditLogger.verify_records(loaded)
    assert not AuditLogger.verify_records(loaded[:1] + loaded[2:])  # so is deletion
    json.dumps(log.records)


def test_audit_rejects_unknown_events():
    with pytest.raises(ValueError):
        AuditLogger().log("something_else")
