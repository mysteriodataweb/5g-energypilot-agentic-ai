import json

import numpy as np
import pandas as pd
import pytest

from energypilot.agent import FaultInjection
from energypilot.agent.llm import ScriptedLLMClient, ScriptedTurn
from energypilot.data import generate_scenario
from energypilot.evaluation import approach_label, build_pipeline, mean_ci, paired_differences, run_episode, run_experiment, summarize
from energypilot.evaluation.episode import instruction_input
from energypilot.evaluation.instruction_benchmark import (
    evaluate_interpreter,
    expected_requirements,
    load_benchmark,
    summarize_benchmark,
)
from energypilot.evaluation.metrics import _violations
from energypilot.instructions import OperationalInstruction, compile_constraints, parse_instruction
from energypilot.safety import AuditLogger


def test_end_to_end_reserve_instruction(config, tmp_path):
    """New reserve instruction -> strategy -> independent validation -> closed-loop evaluation."""
    scenario = generate_scenario("reserve_increase", config, seed=3)
    pipeline = build_pipeline("agent", config)
    ep = run_episode(config, scenario, pipeline, audit_dir=tmp_path)

    d = ep.decision
    assert d.status.value == "accepted" and d.selected_controller == "optimizer_mpc"
    assert d.validation["accepted"] and d.validation["failed_checks"] == []
    assert d.parsed_instruction["reserves"] == [{"start_minute": 1080, "end_minute": 1380, "min_soc_pct": 80.0}]
    # The strategy was evaluated in the simulator: reserve held from 18:00 to 23:00.
    window = ep.history[(ep.history.time_h >= 18) & (ep.history.time_h < 23)]
    assert (window.soc_end_pct >= 80.0 - 1e-6).all()
    m = ep.metrics
    assert m["correct_handling"] and m["intent_fully_compliant"] and m["hard_violation_steps"] == 0
    assert m["approach"] == "agent[deterministic-fallback]"
    # Every phase is in the audit log, which is intact on disk.
    records = AuditLogger.load(next(tmp_path.glob("*.jsonl")))
    events = [r["event"] for r in records]
    for event in ("run_started", "instruction_received", "instruction_parsed", "tool_call", "solver_result",
                  "validation_result", "decision", "instruction_activated", "mpc_replan", "run_completed"):  # fmt: skip
        assert event in events
    assert events.index("validation_result") < events.index("instruction_activated")
    assert AuditLogger.verify_records(records)


def test_rejected_instruction_keeps_default_policy(config):
    scenario = generate_scenario("infeasible_instruction", config, seed=0)
    ep = run_episode(config, scenario, build_pipeline("optimizer", config))
    assert ep.decision.status.value == "infeasible"
    assert ep.metrics["correct_handling"] and ep.metrics["selected_controller"] == "none"
    assert ep.history.active_fingerprint.nunique() == 1  # never switched
    assert ep.metrics["hard_violation_steps"] == 0


def test_episode_reproducibility(config):
    a = run_episode(config, generate_scenario("solar_drop", config, 4), build_pipeline("optimizer", config))
    b = run_episode(config, generate_scenario("solar_drop", config, 4), build_pipeline("optimizer", config))
    pd.testing.assert_frame_equal(a.history, b.history)
    for key in ("total_cost_eur", "grid_energy_kwh", "soc_final_pct", "hard_violation_steps"):
        assert a.metrics[key] == b.metrics[key]


def test_metric_calculations(config):
    ep = run_episode(config, generate_scenario("normal", config, 1), build_pipeline("rule_based", config))
    h, m, dt = ep.history, ep.metrics, config.simulation.dt_h
    assert m["total_cost_eur"] == pytest.approx((h.grid_import_kw * h.price_eur_per_kwh * dt).sum())
    assert m["grid_energy_kwh"] == pytest.approx(h.grid_import_kw.sum() * dt)
    assert m["pv_used_kwh"] + m["pv_curtailed_kwh"] == pytest.approx(m["pv_available_kwh"])
    e0, e1 = h.soc_start_kwh.iloc[0], h.soc_end_kwh.iloc[-1]
    assert m["soc_adjusted_cost_eur"] == pytest.approx(m["total_cost_eur"] + (e0 - e1) * h.price_eur_per_kwh.mean())
    # Daily energy balance: PV used + grid + battery net = load served.
    battery_net = ((h.p_discharge_kw - h.p_charge_kw) * dt).sum()
    assert m["pv_used_kwh"] + m["grid_energy_kwh"] + battery_net == pytest.approx(m["load_energy_kwh"] - m["unserved_energy_kwh"])


def test_violation_counting(config):
    instr = OperationalInstruction.model_validate(
        {
            "reserves": [{"min_soc_pct": 50.0, "start_minute": 0, "end_minute": 60}],
            "grid_limits": [{"max_import_kw": 1.0, "start_minute": 0, "end_minute": 60}],
            "qos_level": "strict",
        }
    )
    cons = compile_constraints(instr, config)
    h = pd.DataFrame(
        {
            "soc_end_kwh": [21.0, 19.0, 18.0, 25.0],
            "grid_import_kw": [0.5, 2.0, 1.0, 3.0],
            "traffic": [0.35, 0.35, 0.8, 0.1],
            "capacity_fraction": [0.5, 0.4, 1.0, 0.4],
        }
    )
    v = _violations(config, h, [cons] * 4)
    assert v["reserve_violation_steps"] == 2 and v["reserve_max_deficit_kwh"] == pytest.approx(2.0)
    assert v["grid_cap_violation_steps"] == 2 and v["grid_cap_max_excess_kw"] == pytest.approx(2.0)
    assert v["qos_violation_steps"] == 1  # step 1: 0.35 / 0.4 > 0.70
    assert v["qos_exogenous_steps"] == 1  # step 2: traffic 0.8 > 0.70 even at full capacity
    assert v["compliance_rate"] == pytest.approx(1 - 4 / 8)


def test_stats():
    mean, std, lo, hi, n = mean_ci(np.array([1.0, 2.0, 3.0, np.nan]))
    assert (mean, n) == (2.0, 3) and lo < mean < hi
    assert mean_ci(np.array([1.0, 1.0]))[2] != mean_ci(np.array([1.0, 1.0]))[2]  # NaN CI when n < 3
    df = pd.DataFrame(
        {"scenario": ["s"] * 6, "seed": [0, 1, 2] * 2, "approach": ["a"] * 3 + ["b"] * 3, "x": [1, 2, 3, 2, 4, 6.0]}
    )
    s = summarize(df, metrics=("x",))
    assert set(s.approach) == {"a", "b"} and s[s.approach == "b"]["mean"].iloc[0] == 4.0
    p = paired_differences(df, "a", "x")
    assert p.iloc[0]["mean_diff"] == pytest.approx(2.0) and p.iloc[0]["n"] == 3


def test_run_experiment_and_labels(config):
    df, _ = run_experiment(config, scenarios=["ambiguous_instruction"], seeds=[0], approaches=("rule_based", "agent"),
                           input_modes=("natural_language", "paraphrase_1"))  # fmt: skip
    assert len(df) == 4 and set(df.approach) == {"rule_based", "agent[deterministic-fallback]"}
    assert df.correct_handling.all()
    assert approach_label("agent", ScriptedLLMClient([])) == "agent[TEST-DOUBLE:scripted-test-double/scripted]"


def test_input_modes(config):
    sc = generate_scenario("solar_drop", config, 0)
    assert instruction_input(sc, "structured") == sc.spec.ground_truth
    assert instruction_input(sc, "paraphrase_1") == sc.spec.paraphrases[1]
    assert instruction_input(generate_scenario("ambiguous_instruction", config, 0), "structured").startswith("Keep")
    with pytest.raises(ValueError):
        instruction_input(sc, "telepathy")


def test_robustness_fault_injection(config):
    sc = generate_scenario("solar_drop", config, 0)
    faults = FaultInjection(optimizer_fails=True)
    agent = run_episode(config, sc, build_pipeline("agent", config, faults=faults))
    opt = run_episode(config, sc, build_pipeline("optimizer", config, faults=faults))
    assert agent.decision.status.value == "accepted" and agent.decision.selected_controller == "rule_based"
    assert opt.decision.status.value == "tool_failure" and opt.metrics["correct_handling"] is False
    assert agent.metrics["tool_call_failures"] == 1


def test_scripted_llm_episode_is_labelled(config):
    """An episode driven by the test double must be labelled as such everywhere."""
    sc = generate_scenario("traffic_peak", config, 0)
    llm = ScriptedLLMClient([ScriptedTurn(text="no tool call"), ScriptedTurn(text="still none")])
    ep = run_episode(config, sc, build_pipeline("agent", config, llm=llm))
    assert ep.metrics["approach"].startswith("agent[TEST-DOUBLE")
    assert ep.metrics["llm_is_real"] is False and ep.metrics["fallback_used"]
    assert ep.decision.interpretation_method == "deterministic_rules"


def test_instruction_benchmark(config):
    items = load_benchmark()
    assert {i["split"] for i in items} == {"dev", "test"}
    for item in items:  # every label is itself valid
        expected_requirements(item, config)
    df = evaluate_interpreter("deterministic_rules", parse_instruction, items, config)
    summary = summarize_benchmark(df)
    dev = summary[summary.split == "dev"].iloc[0]
    assert dev.exact_accuracy == 1.0
    assert set(summary.columns) >= {"status_accuracy", "unsafe_acceptance_rate", "over_caution_rate"}

    def broken(text, issued, cfg):
        raise RuntimeError("interpreter crashed")

    crashed = evaluate_interpreter("broken", broken, items[:3], config)
    assert (crashed.status == "error").all() and not crashed.exact_correct.any()
    json.dumps(df.to_dict(orient="records"), default=str)
