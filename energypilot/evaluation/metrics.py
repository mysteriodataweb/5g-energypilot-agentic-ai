"""Episode metrics computed from the realised simulation history.

Two compliance views are reported:

* ``active_*``: violations of the constraints actually in force at each step
  (what the system accepted). These are **hard safety violations**.
* ``intent_*``: violations of the operator's *intended* instruction (scenario
  ground truth) from its issue time, for scenarios whose correct outcome is
  acceptance. They capture the cost of misinterpreting or wrongly rejecting
  an instruction.

QoS shortfalls caused by traffic exceeding the utilisation target even at full
capacity are *exogenous* (counted separately, not attributed to control).

``soc_adjusted_cost_eur`` = cost + (E_initial - E_final) x mean price: values
the end-of-day battery energy difference at the day's mean grid price, so that
approaches that drain or fill the battery are compared fairly (approximation).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from energypilot.agent.report import DecisionReport
from energypilot.config.settings import AppConfig
from energypilot.data.scenarios import ScenarioData
from energypilot.instructions.constraints import CompiledConstraints, compile_constraints
from energypilot.instructions.validation import parse_structured

_E_TOL = 1e-3  # kWh
_P_TOL = 1e-3  # kW
_C_TOL = 1e-4  # capacity fraction


def _violations(config: AppConfig, history: pd.DataFrame, cons_per_step: list[CompiledConstraints | None]) -> dict[str, Any]:
    """Reserve, grid-cap and QoS violations against per-step constraints (None = unconstrained)."""
    bs = config.base_station
    dt = config.simulation.dt_h
    res_steps = res_checked = cap_steps = cap_checked = qos_steps = qos_exo = qos_checked = 0
    res_max = res_energy_h = cap_max = cap_energy = 0.0
    for k, cons in enumerate(cons_per_step):
        if cons is None:
            continue
        row = history.iloc[k]
        reserve = float(cons.reserve_kwh[k + 1])
        if reserve > 0:
            res_checked += 1
            deficit = reserve - float(row.soc_end_kwh)
            if deficit > _E_TOL:
                res_steps += 1
                res_max = max(res_max, deficit)
                res_energy_h += deficit * dt
        cap = float(cons.grid_cap_kw[k])
        if np.isfinite(cap):
            cap_checked += 1
            excess = float(row.grid_import_kw) - cap
            if excess > _P_TOL:
                cap_steps += 1
                cap_max = max(cap_max, excess)
                cap_energy += excess * dt
        qos_checked += 1
        if float(row.traffic) > cons.max_utilization + 1e-12:
            qos_exo += 1  # unattainable even at full capacity
        else:
            required = 1.0 if not cons.allow_capacity_reduction else float(
                np.clip(row.traffic / cons.max_utilization, bs.min_capacity_fraction, 1.0)
            )
            if float(row.capacity_fraction) < required - _C_TOL:
                qos_steps += 1
    checked = res_checked + cap_checked
    return {
        "reserve_violation_steps": res_steps,
        "reserve_max_deficit_kwh": res_max,
        "reserve_deficit_kwh_h": res_energy_h,
        "grid_cap_violation_steps": cap_steps,
        "grid_cap_max_excess_kw": cap_max,
        "grid_cap_excess_kwh": cap_energy,
        "qos_violation_steps": qos_steps,
        "qos_exogenous_steps": qos_exo,
        "constrained_checks": checked,
        "compliance_rate": (1.0 - (res_steps + cap_steps) / checked) if checked else 1.0,
    }


def compute_metrics(
    config: AppConfig,
    scenario: ScenarioData,
    history: pd.DataFrame,
    decision: DecisionReport,
    active_by_fingerprint: dict[str, CompiledConstraints],
    *,
    decide_times: np.ndarray | None = None,
    mpc_stats: dict[str, int] | None = None,
) -> dict[str, Any]:
    bat, dt = config.battery, config.simulation.dt_h
    h = history
    e0, e_end = float(h.soc_start_kwh.iloc[0]), float(h.soc_end_kwh.iloc[-1])
    mean_price = float(h.price_eur_per_kwh.mean())
    cost = float(h.cost_eur.sum())
    pv_avail = float(h.pv_available_kw.sum() * dt)
    pv_used = float(h.pv_used_kw.sum() * dt)
    throughput = float((h.p_charge_kw + h.p_discharge_kw).sum() * dt)
    m: dict[str, Any] = {
        "scenario": scenario.spec.name,
        "seed": scenario.seed,
        "expected_outcome": scenario.spec.expected_outcome,
        "total_cost_eur": cost,
        "soc_adjusted_cost_eur": cost + (e0 - e_end) * mean_price,
        "grid_energy_kwh": float(h.grid_import_kw.sum() * dt),
        "peak_grid_kw": float(h.grid_import_kw.max()),
        "pv_available_kwh": pv_avail,
        "pv_used_kwh": pv_used,
        "pv_curtailed_kwh": float(h.pv_curtailed_kw.sum() * dt),
        "pv_self_consumption_ratio": pv_used / pv_avail if pv_avail > 0 else np.nan,
        "load_energy_kwh": float(h.load_kw.sum() * dt),
        "unserved_energy_kwh": float(h.unserved_kw.sum() * dt),
        "battery_throughput_kwh": throughput,
        "equivalent_full_cycles": throughput / (2 * bat.capacity_kwh),
        "soc_initial_pct": bat.kwh_to_pct(e0),
        "soc_final_pct": bat.kwh_to_pct(e_end),
        "soc_min_pct": float(h.soc_end_pct.min()),
        "soc_max_pct": float(h.soc_end_pct.max()),
        "terminal_soc_shortfall_kwh": max(e0 - e_end, 0.0),
        "mean_utilization": float(h.utilization.mean()),
        "max_utilization": float(h.utilization.max()),
        "mean_served_ratio": float(h.served_ratio.mean()),
        "p95_delay_factor": float(np.percentile(h.delay_factor, 95)),
        "simulator_adjustment_steps": int((h.adjustments != "").sum()),
        "shield_intervention_steps": int((h.shield_notes != "").sum()),
    }

    # Hard violations: constraints actually in force (default or accepted instruction).
    active = [active_by_fingerprint.get(fp) for fp in h.active_fingerprint]
    for key, value in _violations(config, h, active).items():
        m[f"active_{key}"] = value
    unserved_steps = int((h.unserved_kw > _P_TOL).sum())
    m["hard_violation_steps"] = (
        m["active_reserve_violation_steps"] + m["active_grid_cap_violation_steps"]
        + m["active_qos_violation_steps"] + unserved_steps
    )  # fmt: skip

    # Intent compliance: the operator's intended instruction (accept scenarios only).
    if scenario.spec.ground_truth is not None and scenario.spec.expected_outcome == "accept":
        issued = scenario.issue_step * config.simulation.time_step_minutes
        intended = parse_structured(scenario.spec.ground_truth, issued, config).instruction
        gt = compile_constraints(intended, config)
        per_step = [gt if k >= scenario.issue_step else None for k in range(len(h))]
        for key, value in _violations(config, h, per_step).items():
            m[f"intent_{key}"] = value
        m["intent_fully_compliant"] = (
            m["intent_reserve_violation_steps"] + m["intent_grid_cap_violation_steps"] + m["intent_qos_violation_steps"]
        ) == 0
    else:
        m["intent_fully_compliant"] = np.nan

    # Instruction handling and orchestration.
    m.update(
        {
            "decision_status": decision.status.value,
            "correct_handling": (
                decision.matches_expected(scenario.spec.expected_outcome)
                if scenario.spec.expected_outcome != "unknown"
                else np.nan
            ),
            "interpretation_method": decision.interpretation_method,
            "orchestration_mode": decision.orchestration_mode,
            "selected_controller": decision.selected_controller or "none",
            "interpretation_matches_truth": _interpretation_match(config, scenario, decision),
            "decision_latency_s": decision.latency_s,
            "llm_calls": decision.llm_usage.calls,
            "llm_input_tokens": decision.llm_usage.input_tokens,
            "llm_output_tokens": decision.llm_usage.output_tokens,
            "llm_latency_s": decision.llm_usage.latency_s,
            "llm_cost_eur": decision.llm_usage.estimated_cost_eur,
            "llm_is_real": decision.llm_is_real,
            "llm_failures": len(decision.llm_usage.failures),
            "fallback_used": bool(decision.fallback_reasons),
            "tool_calls": len(decision.tool_calls),
            "tool_call_failures": sum(1 for c in decision.tool_calls if not c["ok"]),
        }
    )
    m["tool_call_success_rate"] = (1 - m["tool_call_failures"] / m["tool_calls"]) if m["tool_calls"] else np.nan
    if decide_times is not None and len(decide_times):
        m["mean_step_decision_s"] = float(np.mean(decide_times))
        m["max_step_decision_s"] = float(np.max(decide_times))
    for key, value in (mpc_stats or {}).items():
        m[f"mpc_{key}"] = value
    return m


def _interpretation_match(config: AppConfig, scenario: ScenarioData, decision: DecisionReport) -> bool | float:
    """Did the interpreted instruction equal the ground truth? (NaN when there is none)."""
    if scenario.spec.expected_outcome == "unknown":
        return np.nan
    if scenario.spec.ground_truth is None:
        return decision.interpretation_status == "needs_clarification"
    issued = scenario.issue_step * config.simulation.time_step_minutes
    truth = parse_structured(scenario.spec.ground_truth, issued, config).instruction.requirement_dict()
    return decision.parsed_instruction == truth
