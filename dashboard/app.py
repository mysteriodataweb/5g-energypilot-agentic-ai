"""5G EnergyPilot AI - research dashboard.

Run with:  streamlit run dashboard/app.py

Every number shown on the "Results" and "Decisions & audit" tabs comes from a
closed-loop simulation executed when you press "Run simulation"; the
"Experiment results" tab reads the outputs of scripts/run_evaluation.py.
All data are synthetic. Nothing here controls real equipment.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import charts  # noqa: E402
from energypilot.agent.llm import try_create_llm_client  # noqa: E402
from energypilot.agent.tools import FaultInjection  # noqa: E402
from energypilot.config import load_config  # noqa: E402
from energypilot.data import SCENARIOS, generate_scenario  # noqa: E402
from energypilot.data.scenarios import PROVENANCE  # noqa: E402
from energypilot.evaluation import build_pipeline, run_episode  # noqa: E402
from energypilot.instructions import compile_constraints, default_instruction, parse_instruction, parse_structured  # noqa: E402
from energypilot.instructions.schema import OperationalInstruction  # noqa: E402

RESULTS_DIR = ROOT / "results" / "reference"
STATUS = {
    "accepted": ("✅", "Accepted - valid and feasible", "success"),
    "needs_clarification": ("❓", "Clarification requested - instruction not applied", "warning"),
    "rejected_invalid_instruction": ("⛔", "Rejected - invalid or unsupported instruction", "error"),
    "infeasible": ("⛔", "Rejected - proven infeasible", "error"),
    "rejected_by_validator": ("🛡️", "Rejected by the safety validator", "error"),
    "rejected_by_agent": ("⛔", "Declined by the agent", "error"),
    "tool_failure": ("⚠️", "Tool failure - no validated strategy", "warning"),
}
BAR_METRICS = {
    "soc_adjusted_cost_eur": ("SoC-adjusted cost", "EUR"),
    "total_cost_eur": ("Electricity cost", "EUR"),
    "grid_energy_kwh": ("Grid energy imported", "kWh"),
    "pv_used_kwh": ("Solar energy used locally", "kWh"),
    "peak_grid_kw": ("Peak grid import", "kW"),
}
APPROACH_HELP = {
    "rule_based": "Deterministic parser + interpretable rule-based controller",
    "optimizer": "Deterministic parser + MILP optimiser (MPC), no LLM",
    "agent": "LLM interpretation and tool orchestration if configured, else deterministic agent workflow",
}

st.set_page_config(page_title="5G EnergyPilot AI", page_icon="🔋", layout="wide")
theme = charts.theme_for(getattr(st.context.theme, "type", None))


@st.cache_resource
def _llm():
    return try_create_llm_client(load_config())


@st.cache_data
def _load_csv(name: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / name
    return pd.read_csv(path) if path.exists() else None


def _hhmm(hours: float) -> str:
    minutes = int(round(hours * 60))
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _status_box(status: str, label: str) -> None:
    icon, text, kind = STATUS.get(status, ("•", status, "info"))
    getattr(st, kind)(f"{icon} **{label}** - {text}")


# ------------------------------------------------------------------ sidebar
base_config = load_config()
with st.sidebar:
    st.header("Simulation parameters")
    seed = int(st.number_input("Random seed", min_value=0, max_value=10_000, value=0, step=1))
    step_minutes = st.selectbox("Time step (minutes)", [15, 30, 60], index=0)
    capacity = st.slider("Battery capacity (kWh)", 10.0, 100.0, base_config.battery.capacity_kwh, 5.0)
    power = st.slider("Battery power limit (kW)", 2.0, 20.0, base_config.battery.max_charge_kw, 1.0)
    pv_peak = st.slider("PV peak, clear sky (kW AC)", 0.5, 15.0, base_config.pv.peak_power_kw, 0.5)
    forecast_mode = st.radio(
        "Forecasts available to planners",
        ["persistence_corrected", "perfect"],
        format_func=lambda m: "Persistence-corrected forecast" if m == "persistence_corrected" else "Perfect (oracle reference)",
    )
    override_soc = st.checkbox("Override initial SoC")
    initial_soc = st.slider("Initial SoC (%)", 10.0, 95.0, 60.0, 1.0, disabled=not override_soc)

    st.header("Scenario")
    scenario_name = st.selectbox("Illustrative scenario (synthetic)", list(SCENARIOS), format_func=lambda n: SCENARIOS[n].title)
    spec = SCENARIOS[scenario_name]
    st.caption(spec.description)
    st.caption(f"Instruction issued at {_hhmm(spec.instruction_issue_hour)} · expected handling: **{spec.expected_outcome}**")

    st.header("Approaches")
    approaches = st.multiselect("Compare", list(APPROACH_HELP), default=list(APPROACH_HELP), help="\n".join(f"{k}: {v}" for k, v in APPROACH_HELP.items()))

    st.header("Agent")
    llm, llm_settings, llm_error = _llm()
    use_llm = False
    if llm is not None:
        use_llm = st.checkbox(f"Use configured LLM ({llm.provider}/{llm.model})", value=True)
        if llm_settings.pricing.input_per_mtok is None:
            st.caption("LLM prices not configured: inference cost will be reported as unknown.")
    else:
        st.info(f"No LLM configured{f' ({llm_error})' if llm_error else ''}. The agent runs its deterministic fallback workflow; no LLM is called.")
    inject_fault = st.checkbox("Inject optimiser failure at instruction time", help="Robustness test: the run_optimizer tool fails")

try:
    config = base_config.with_overrides(
        {
            "simulation": {"time_step_minutes": step_minutes},
            "battery": {"capacity_kwh": capacity, "max_charge_kw": power, "max_discharge_kw": power},
            "pv": {"peak_power_kw": pv_peak},
        }
    )
except Exception as exc:  # noqa: BLE001 - surface validation errors to the user
    st.error(f"Invalid parameters: {exc}")
    st.stop()

# ------------------------------------------------------------------- header
st.title("5G EnergyPilot AI")
st.caption("Agentic AI for energy-aware 5G base-station demand response - independent academic research prototype.")
st.warning(
    "**Synthetic data, simulated results.** Profiles are generated, forecasts and outcomes come from a simulator, "
    "and nothing is connected to real telecommunications equipment. Do not read these numbers as real-world results.",
    icon="🧪",
)
tab_run, tab_results, tab_audit, tab_experiments, tab_data = st.tabs(
    ["Instruction & run", "Results", "Decisions & audit", "Experiment results", "Data & assumptions"]
)

# --------------------------------------------------------------- instruction
with tab_run:
    st.subheader("Operational instruction")
    mode = st.radio("Input interface", ["Natural language", "Structured JSON"], horizontal=True)
    issued_minute = int(round(spec.instruction_issue_hour * 60))
    if mode == "Natural language":
        default_text = spec.instruction_text
    else:
        default_text = json.dumps(spec.ground_truth, indent=2) if spec.ground_truth else '{\n  "objective": "minimize_cost"\n}'
    text = st.text_area(
        f"Instruction issued at {_hhmm(spec.instruction_issue_hour)} (simulated clock)",
        value=default_text,
        height=160 if mode == "Structured JSON" else 100,
        key=f"instruction-{scenario_name}-{mode}",
    )
    if spec.paraphrases and mode == "Natural language":
        with st.expander("Held-out paraphrases of this instruction (same intent)"):
            for p in spec.paraphrases:
                st.code(p, language=None)

    st.markdown("**Structured instruction preview** (deterministic interpreter - the LLM, if enabled, runs only when you press *Run*)")
    preview = parse_structured(text, issued_minute, config) if text.lstrip().startswith("{") else parse_instruction(text, issued_minute, config)
    c1, c2 = st.columns([1, 1])
    with c1:
        _status_box({"ok": "accepted", "needs_clarification": "needs_clarification", "rejected": "rejected_invalid_instruction"}[preview.status.value], "Interpretation")
        for issue in preview.issues:
            st.write(f"- `{issue.severity}` {issue.message}")
        if preview.instruction is not None and preview.instruction.defaults_applied:
            st.caption("Documented defaults applied (not stated in the text): " + ", ".join(preview.instruction.defaults_applied))
    with c2:
        if preview.instruction is not None:
            st.json(preview.instruction.requirement_dict())

    run = st.button("Run simulation", type="primary", disabled=not approaches)
    if run:
        scenario = generate_scenario(scenario_name, config, seed)
        if override_soc:
            scenario.initial_soc_pct = float(np.clip(initial_soc, config.battery.soc_min_pct, config.battery.soc_max_pct))
        unchanged = text.strip() == default_text.strip()
        if not unchanged:
            scenario.spec = dataclasses.replace(scenario.spec, instruction_text=text, paraphrases=(), ground_truth=None, expected_outcome="unknown")
        else:
            scenario.spec = dataclasses.replace(scenario.spec, instruction_text=text)
        faults = FaultInjection(optimizer_fails=True) if inject_fault else None
        results = {}
        with st.spinner("Running closed-loop simulations..."):
            for name in approaches:
                pipeline = build_pipeline(name, config, llm=llm if (use_llm and name == "agent") else None,
                                          pricing=llm_settings.pricing if llm_settings else None, faults=faults)  # fmt: skip
                ep = run_episode(config, scenario, pipeline, input_mode="natural_language", forecast_mode=forecast_mode)
                results[pipeline.name] = ep
        st.session_state["results"] = {"episodes": results, "config": config, "scenario": scenario, "custom": not unchanged}
        st.success("Simulation complete - see the *Results* and *Decisions & audit* tabs.")

state = st.session_state.get("results")

# -------------------------------------------------------------------- results
with tab_results:
    if not state:
        st.info("Run a simulation from the *Instruction & run* tab.")
    else:
        episodes, cfg, scenario = state["episodes"], state["config"], state["scenario"]
        dt = cfg.simulation.dt_h
        if state["custom"]:
            st.caption("Custom instruction: no ground-truth label, so handling correctness and intent compliance are not evaluated.")
        cols = st.columns(len(episodes))
        for col, (label, ep) in zip(cols, episodes.items()):
            with col:
                _status_box(ep.decision.status.value, label)
                m = ep.metrics
                st.metric("SoC-adjusted cost (EUR)", f"{m['soc_adjusted_cost_eur']:.2f}", help="cost + (initial - final battery energy) x mean price")
                st.metric("Grid energy (kWh)", f"{m['grid_energy_kwh']:.1f}")
                st.metric("Hard safety violations (steps)", int(m["hard_violation_steps"]))
                st.caption(f"Controller after instruction: {ep.decision.selected_controller or 'previous policy kept'} · "
                           f"interpretation: {ep.decision.interpretation_method} · orchestration: {ep.decision.orchestration_mode}")  # fmt: skip

        accepted = [ep for ep in episodes.values() if ep.decision.parsed_instruction and ep.decision.status.value == "accepted"]
        shown = (
            OperationalInstruction.model_validate({**accepted[0].decision.parsed_instruction, "issued_at_minute": accepted[0].decision.issued_at_minute})
            if accepted
            else default_instruction(0)
        )
        cons = compile_constraints(shown, cfg)
        reserve_pct = np.where(cons.reserve_kwh > 0, 100 * cons.reserve_kwh / cfg.battery.capacity_kwh, 0.0)
        histories = {label: ep.history for label, ep in episodes.items()}
        st.plotly_chart(
            charts.soc_figure(histories, reserve_pct, (cfg.battery.soc_min_pct, cfg.battery.soc_max_pct), cfg.battery.capacity_kwh,
                              dt, theme, issue_hour=scenario.spec.instruction_issue_hour),
            theme=None, width="stretch",
        )  # fmt: skip
        st.caption("Shaded: reserve windows of the accepted instruction (if any). Dashed: BMS SoC limits. Dotted: instruction time.")

        left, right = st.columns(2)
        with left:
            pick = st.selectbox("Power flows for", list(episodes))
            st.plotly_chart(charts.power_figure(episodes[pick].history, cons.grid_cap_kw, dt, theme, pick), theme=None, width="stretch")
        with right:
            metric = st.radio("Compare", list(BAR_METRICS), horizontal=True, format_func=lambda k: BAR_METRICS[k][0])
            title, unit = BAR_METRICS[metric]
            st.plotly_chart(charts.bar_figure({k: ep.metrics[metric] for k, ep in episodes.items()}, title, unit, theme),
                            theme=None, width="stretch")  # fmt: skip
        st.plotly_chart(charts.utilization_figure(histories, cons.max_utilization, dt, theme), theme=None, width="stretch")
        st.caption("Utilisation above the target while capacity is already at 100% is an exogenous shortfall (traffic exceeds the site's capacity), not a control error.")

        st.subheader("Comparison across approaches")
        keys = ["decision_status", "correct_handling", "selected_controller", "total_cost_eur", "soc_adjusted_cost_eur", "grid_energy_kwh",
                "peak_grid_kw", "pv_used_kwh", "pv_curtailed_kwh", "soc_final_pct", "soc_min_pct", "hard_violation_steps",
                "intent_reserve_violation_steps", "intent_grid_cap_violation_steps", "intent_qos_violation_steps",
                "active_qos_exogenous_steps", "shield_intervention_steps", "decision_latency_s", "mean_step_decision_s",
                "tool_calls", "tool_call_failures", "llm_calls", "llm_input_tokens", "llm_output_tokens", "llm_cost_eur"]  # fmt: skip
        table = pd.DataFrame({label: {k: ep.metrics.get(k) for k in keys} for label, ep in episodes.items()})
        st.dataframe(table.map(lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)), width="stretch")
        with st.expander("Per-step data (table view of the charts)"):
            pick_t = st.selectbox("Approach", list(episodes), key="table-approach")
            st.dataframe(episodes[pick_t].history, width="stretch", height=320)

# ----------------------------------------------------------------- decisions
with tab_audit:
    if not state:
        st.info("Run a simulation first.")
    else:
        for label, ep in state["episodes"].items():
            d = ep.decision
            with st.expander(f"{STATUS.get(d.status.value, ('•',))[0]} {label} - {d.status.value}", expanded=True):
                st.markdown(f"**Explanation (deterministic, built from tool outputs):** {d.explanation}")
                if d.llm_explanation:
                    st.markdown(f"**LLM-generated explanation (unverified narrative):** {d.llm_explanation}")
                if d.clarification_question:
                    st.markdown(f"**Clarification requested:** {d.clarification_question}")
                if d.fallback_reasons:
                    st.markdown("**Fallbacks:** " + "; ".join(d.fallback_reasons))
                if d.validation and d.validation.get("reasons"):
                    st.markdown("**Validator rejection reasons:**")
                    for reason in d.validation["reasons"]:
                        st.write(f"- {reason}")
                if d.conflict_report:
                    st.markdown("**Infeasibility diagnosis:**")
                    st.json(d.conflict_report)
                if d.tool_calls:
                    st.markdown("**Tool calls**")
                    st.dataframe(pd.DataFrame(d.tool_calls)[["seq", "tool", "caller", "ok", "error", "duration_s"]], width="stretch")
                shielded = ep.history[ep.history.shield_notes != ""][["time_h", "shield_notes", "p_charge_requested_kw", "p_discharge_requested_kw"]]
                st.markdown(f"**Rejected / corrected actions (runtime shield):** {len(shielded)} step(s)")
                if len(shielded):
                    st.dataframe(shielded, width="stretch", height=200)
                log = pd.DataFrame([{k: r[k] for k in ("seq", "ts", "event")} | {"payload": json.dumps(r["payload"])[:200]} for r in ep.audit.records])
                events = st.multiselect("Audit events", sorted(log.event.unique()), default=[e for e in ("instruction_received", "instruction_parsed", "tool_call", "solver_result", "validation_result", "decision", "instruction_activated") if e in set(log.event)], key=f"ev-{label}")
                st.dataframe(log[log.event.isin(events)], width="stretch", height=240)
                st.caption(f"Audit hash chain valid: {ep.audit.verify_chain()} · {len(ep.audit.records)} records")
                st.download_button("Download audit log (JSONL)", "\n".join(json.dumps(r) for r in ep.audit.records),
                                   file_name=f"{ep.audit.run_id.replace('/', '_')}.jsonl", key=f"dl-{label}")  # fmt: skip

# --------------------------------------------------------------- experiments
with tab_experiments:
    meta_path = RESULTS_DIR / "metadata.json"
    if not meta_path.exists():
        st.info("No experiment results found. Run `python scripts/run_evaluation.py --out results/reference`.")
    else:
        meta = json.loads(meta_path.read_text())
        llm_text = f"{meta['llm']['provider']}/{meta['llm']['model']}" if meta.get("llm") else "none (agent in deterministic fallback mode)"
        st.caption(f"Generated {meta['generated_at_utc']} by scripts/run_evaluation.py · commit {meta['git_commit']} · "
                   f"{len(meta['seeds'])} seeds · LLM: {llm_text}. Synthetic data, simulation only.")  # fmt: skip
        handling = _load_csv("main_handling.csv")
        if handling is not None:
            st.subheader("Instruction handling by approach and input mode")
            st.dataframe(handling, width="stretch")
        summary = _load_csv("main_summary.csv")
        if summary is not None:
            nl = summary[summary.input_mode == "natural_language"]
            accept = [n for n, s in SCENARIOS.items() if s.expected_outcome == "accept"]
            st.plotly_chart(charts.ci_bar_figure(nl[nl.scenario.isin(accept)], "soc_adjusted_cost_eur",
                                                 "SoC-adjusted daily cost, mean and 95% CI over seeds", "EUR", theme),
                            theme=None, width="stretch")  # fmt: skip
            with st.expander("Summary table (all metrics)"):
                st.dataframe(summary, width="stretch", height=320)
        rob = _load_csv("robustness_handling.csv")
        if rob is not None:
            st.subheader("Robustness: optimiser failure injected at instruction time")
            st.dataframe(rob, width="stretch")
        bench = _load_csv("benchmark_summary.csv")
        if bench is not None:
            st.subheader("Instruction-interpretation benchmark")
            st.dataframe(bench, width="stretch")
        report = RESULTS_DIR / "report.md"
        if report.exists():
            with st.expander("Full generated report"):
                st.markdown(report.read_text())

# ---------------------------------------------------------------------- data
with tab_data:
    sc = generate_scenario(scenario_name, config, seed)
    st.subheader("Exogenous profiles for this scenario and seed")
    a, b = st.columns(2)
    with a:
        st.plotly_chart(charts.profile_figure(sc.hours, sc.pv_actual_kw, sc.pv_forecast_kw, "PV power", "kW", theme.pv, theme, config.simulation.dt_h),
                        theme=None, width="stretch")  # fmt: skip
    with b:
        st.plotly_chart(charts.profile_figure(sc.hours, sc.traffic_actual, sc.traffic_forecast, "Normalised traffic (1 = full capacity)", "traffic",
                                              theme.demand, theme, config.simulation.dt_h), theme=None, width="stretch")  # fmt: skip
    st.subheader("Data provenance")
    st.table(pd.DataFrame({"quantity": list(PROVENANCE), "nature": list(PROVENANCE.values())}))
    st.subheader("Time-of-use tariff (illustrative assumption)")
    st.table(pd.DataFrame([p.model_dump() for p in config.tariff.periods]))
    st.caption("Full assumptions: config/default.toml and docs/mathematical_model.md.")
