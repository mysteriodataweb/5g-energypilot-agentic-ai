"""5G EnergyPilot AI - operator view.

A simplified page for the person who *gives* instructions, as opposed to the
research page (app.py) that compares approaches side by side. It runs only the
agent pipeline and walks through the operator's workflow:

  1. write an instruction            2. confirm what the system understood
  3. apply it                        4. read the decision and what to do next
  5. monitor the day: per-constraint checks, alerts, charts, audit record

The confirmation step guards against the failure class found in the evaluation:
an instruction accepted with part of the operator's intent silently dropped.
All data are synthetic and every action runs inside the simulator.
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from pathlib import Path

DASHBOARD = Path(__file__).resolve().parents[1]
ROOT = DASHBOARD.parent
for path in (ROOT, DASHBOARD):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import charts  # noqa: E402
from energypilot.agent.llm import try_create_llm_client  # noqa: E402
from energypilot.config import load_config  # noqa: E402
from energypilot.data import SCENARIOS, generate_scenario  # noqa: E402
from energypilot.evaluation import build_pipeline, run_episode  # noqa: E402
from energypilot.instructions import compile_constraints, parse_instruction, parse_structured  # noqa: E402
from energypilot.instructions.schema import OperationalInstruction  # noqa: E402

OBJECTIVES = {
    "minimize_cost": "Keep the electricity bill as low as possible",
    "minimize_grid_energy": "Buy as little energy from the grid as possible",
    "minimize_peak_import": "Keep the highest grid import as low as possible",
}
# status -> (streamlit box, headline, what the operator should do next)
NEXT_STEP = {
    "accepted": ("success", "Instruction applied",
                 "The system is now controlling the battery automatically according to your instruction. "
                 "Nothing to do: follow the checks below, and send a new instruction if the situation changes."),
    "needs_clarification": ("warning", "Instruction NOT applied - clarification needed",
                            "The previous policy stays in force. Answer the question below by rephrasing your instruction."),
    "rejected_invalid_instruction": ("error", "Instruction NOT applied - invalid or unsupported",
                                     "The previous policy stays in force. Correct the problems listed below and send it again."),
    "infeasible": ("error", "Instruction NOT applied - impossible to satisfy",
                   "Your constraints cannot all be met together. Decide which one to relax (see the conflict below) and send it again."),
    "rejected_by_validator": ("error", "Instruction NOT applied - blocked by the safety validator",
                              "Applying it would have broken a safety limit. Rephrase it, or escalate if it is really needed."),
    "rejected_by_agent": ("error", "Instruction NOT applied - declined by the agent",
                          "Check the reasons below, then rephrase or escalate."),
    "tool_failure": ("warning", "No validated strategy - decision tools failed",
                     "The previous policy stays in force. Inform the technical team."),
}  # fmt: skip
TOL_PCT = 0.1  # tolerance on reserve checks (percentage points)
TOL_KW = 1e-3
ALERT_CAP_SHARE = 0.9  # alert when grid import exceeds this share of an active grid limit
ALERT_RESERVE_MARGIN = 0.02  # "at the reserve" = within 2 % of capacity above it


@st.cache_resource
def _llm():
    return try_create_llm_client(load_config())


def hhmm(minute: float) -> str:
    m = int(round(minute))
    return f"{m // 60:02d}:{m % 60:02d}"


def normalise(req: dict | None, issued_minute: int) -> dict | None:
    """Canonical requirement dict, so two interpretations can be compared."""
    if req is None:
        return None
    return OperationalInstruction.model_validate({**req, "issued_at_minute": issued_minute}).requirement_dict()


def describe(req: dict, defaults: tuple[str, ...] = (), qos_levels: dict[str, float] | None = None) -> list[str]:
    """Plain-language reading of a structured instruction."""
    tag = lambda field: " *(default)*" if field in defaults else ""  # noqa: E731
    lines = [f"🔋 Keep the battery **at or above {r['min_soc_pct']:g} %** from **{hhmm(r['start_minute'])}** to **{hhmm(r['end_minute'])}**"
             for r in req.get("reserves", [])]  # fmt: skip
    lines += [f"🔌 Buy **at most {g['max_import_kw']:g} kW** from the grid from **{hhmm(g['start_minute'])}** to **{hhmm(g['end_minute'])}**"
              for g in req.get("grid_limits", [])]  # fmt: skip
    if not lines:
        lines.append("⚠️ **No battery reserve and no grid limit** - nothing specific will be enforced")
    lines.append(f"🎯 Goal: {OBJECTIVES.get(req['objective'], req['objective'])}{tag('objective')}")
    level = req["qos_level"]
    limit = req.get("max_utilization") or (qos_levels or {}).get(level)
    lines.append(f"📶 Quality of service: **{level}**" + (f" (radio load at most {limit:.0%})" if limit else "") + tag("qos_level"))
    lines.append(("💤 Idle radio capacity may be switched off to save energy" if req["allow_capacity_reduction"]
                  else "📡 All radio capacity stays on") + tag("allow_capacity_reduction"))  # fmt: skip
    return lines


def window_steps(start_min: int, end_min: int, issue_step: int, step_min: int, n: int) -> tuple[int, int]:
    """Same index convention as energypilot.instructions.constraints.compile_constraints."""
    return max(math.floor(start_min / step_min), issue_step), min(math.ceil(end_min / step_min), n)


def compliance(req: dict, issued_minute: int, ep, config) -> pd.DataFrame:
    """One row per constraint of the applied instruction, plus safety and service."""
    h, bat = ep.history, config.battery
    n, step_min, dt = len(h), config.simulation.time_step_minutes, config.simulation.dt_h
    issue_step = min(issued_minute // step_min, n)
    soc_b = np.append(h.soc_start_kwh.to_numpy(), h.soc_end_kwh.iloc[-1]) / bat.capacity_kwh * 100  # % at each boundary
    rows = []
    for r in req.get("reserves", []):
        lo, hi = window_steps(r["start_minute"], r["end_minute"], issue_step, step_min, n)
        if lo > hi:
            continue
        low = float(soc_b[lo : hi + 1].min())
        rows.append({"check": f"Battery ≥ {r['min_soc_pct']:g} % ({hhmm(r['start_minute'])}-{hhmm(r['end_minute'])})",
                     "observed": f"lowest {low:.1f} %", "ok": low >= r["min_soc_pct"] - TOL_PCT})  # fmt: skip
    grid = h.grid_import_kw.to_numpy()
    for g in req.get("grid_limits", []):
        lo, hi = window_steps(g["start_minute"], g["end_minute"], issue_step, step_min, n)
        if lo >= hi:
            continue
        peak = float(grid[lo:hi].max())
        rows.append({"check": f"Grid import ≤ {g['max_import_kw']:g} kW ({hhmm(g['start_minute'])}-{hhmm(g['end_minute'])})",
                     "observed": f"highest {peak:.2f} kW", "ok": peak <= g["max_import_kw"] + TOL_KW})  # fmt: skip
    qos = int(ep.metrics["active_qos_violation_steps"])
    rows.append({"check": "Service quality kept (radio load within the QoS target)",
                 "observed": f"{qos} quarter-hour(s) out of target", "ok": qos == 0})  # fmt: skip
    unserved = float((h.unserved_kw * dt).sum())
    rows.append({"check": "Station fully powered (no unserved load)", "observed": f"{unserved:.2f} kWh unserved", "ok": unserved < 1e-6})
    return pd.DataFrame(rows)


def _periods(mask: np.ndarray, hours: np.ndarray, dt: float) -> list[tuple[float, float]]:
    out, start = [], None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = hours[i]
        if not flag and start is not None:
            out.append((start, hours[i]))
            start = None
    if start is not None:
        out.append((start, hours[-1] + dt))
    return out


def alerts(ep, cons, config) -> pd.DataFrame:
    """Events an operator would want to be told about during the day.

    Sitting *at* the reserve is normal (the cost-optimal plan uses everything above it), so it is only
    flagged while a grid limit is active - the one situation where nothing is left to absorb extra demand.
    """
    h, bat, dt = ep.history, config.battery, config.simulation.dt_h
    hours = h.time_h.to_numpy()
    rows = []
    n = len(h)
    reserve = cons.reserve_kwh[:n] if cons is not None else np.zeros(n)
    cap = cons.grid_cap_kw[:n] if cons is not None else np.full(n, np.inf)
    capped = np.isfinite(cap)
    near_cap = capped & (h.grid_import_kw.to_numpy() >= ALERT_CAP_SHARE * np.where(capped, cap, 0)) & (cap > 0)
    for a, b in _periods(near_cap, hours, dt):
        rows.append({"from": hhmm(a * 60), "to": hhmm(b * 60), "alert": f"Grid import above {ALERT_CAP_SHARE:.0%} of the limit you set"})
    at_reserve = capped & (reserve > 0) & (h.soc_start_kwh.to_numpy() - reserve <= ALERT_RESERVE_MARGIN * bat.capacity_kwh)
    for a, b in _periods(at_reserve, hours, dt):
        rows.append({"from": hhmm(a * 60), "to": hhmm(b * 60),
                     "alert": "Battery at the reserve while the grid limit is active: no margin left for extra demand"})  # fmt: skip
    for a, b in _periods(h.unserved_kw.to_numpy() > 1e-6, hours, dt):
        rows.append({"from": hhmm(a * 60), "to": hhmm(b * 60), "alert": "Part of the station load could not be served"})
    for _, r in h[h.shield_notes != ""].iterrows():
        rows.append({"from": hhmm(r.time_h * 60), "to": hhmm((r.time_h + dt) * 60), "alert": f"Safety shield corrected a command: {r.shield_notes}"})
    return pd.DataFrame(rows, columns=["from", "to", "alert"]).sort_values("from", kind="stable")


# ------------------------------------------------------------------ page
st.set_page_config(page_title="Operator view - 5G EnergyPilot AI", page_icon="🔋", layout="wide")
theme = charts.theme_for(getattr(st.context.theme, "type", None))
config = load_config()

with st.sidebar:
    st.header("Site situation")
    scenario_name = st.selectbox("Situation of the day (synthetic)", list(SCENARIOS), format_func=lambda n: SCENARIOS[n].title)
    spec = SCENARIOS[scenario_name]
    st.caption(spec.description)
    with st.expander("Simulation settings"):
        seed = int(st.number_input("Random seed", min_value=0, max_value=10_000, value=0, step=1))
    st.header("Assistant")
    llm, llm_settings, llm_error = _llm()
    use_llm = False
    if llm is not None:
        use_llm = st.checkbox(f"Use the language model ({llm.provider}/{llm.model})", value=True,
                              help="The LLM interprets your instruction and orchestrates the tools. "
                                   "An independent safety validator always has the final word.")  # fmt: skip
    else:
        st.info("No language model configured: the assistant uses its deterministic interpreter.")

st.title("🔋 Operator console")
st.caption("Give an energy instruction for the base station, confirm what the system understood, then follow its execution.")
st.warning("**Simulation only** - synthetic data; no real equipment is controlled.", icon="🧪")

issued_minute = int(round(spec.instruction_issue_hour * 60))

# 1. instruction --------------------------------------------------------------
st.subheader("1 · Your instruction")
text = st.text_area(
    f"Instruction sent at {hhmm(issued_minute)} (simulated clock)",
    value=spec.instruction_text,
    height=90,
    key=f"op-instruction-{scenario_name}",
    help="Plain English, e.g. 'Keep at least 80% battery until 23:00 and limit grid import to 3 kW from 18:00 to 20:00'. "
    "Press Ctrl+Enter to update the preview.",
)

# 2. what the system understood -----------------------------------------------
st.subheader("2 · What the system understood")
preview = parse_structured(text, issued_minute, config) if text.lstrip().startswith("{") else parse_instruction(text, issued_minute, config)
preview_ok = preview.status.value == "ok" and preview.instruction is not None
preview_req = preview.instruction.requirement_dict() if preview_ok else None
with st.container(border=True):
    if preview_ok:
        for line in describe(preview_req, tuple(preview.instruction.defaults_applied), config.qos.max_utilization_by_level):
            st.markdown(f"- {line}")
        for issue in preview.issues:
            st.caption(f"Note: {issue.message}")
        if use_llm:
            st.caption("This preview comes from the deterministic interpreter. The language model makes the final "
                       "interpretation when you apply; you will be warned if it differs.")  # fmt: skip
    else:
        st.warning("The system could not turn this into enforceable constraints:")
        for issue in preview.issues:
            st.markdown(f"- {issue.message}")
        st.caption("Say what to keep or limit, by how much and when - e.g. *Keep at least 60% battery between 18:00 and 20:00*.")

confirm_label = "Yes, this is what I mean" if preview_ok else "Apply anyway and let the language model interpret it"
confirmed = st.checkbox(confirm_label, key=f"op-confirm-{scenario_name}-{hash(text)}", disabled=not (preview_ok or use_llm))
apply = st.button("Apply instruction", type="primary", disabled=not confirmed)

if apply:
    scenario = generate_scenario(scenario_name, config, seed)
    if text.strip() != spec.instruction_text.strip():
        scenario.spec = dataclasses.replace(scenario.spec, instruction_text=text, paraphrases=(), ground_truth=None, expected_outcome="unknown")
    pipeline = build_pipeline("agent", config, llm=llm if use_llm else None, pricing=llm_settings.pricing if llm_settings else None)
    with st.spinner("The assistant is interpreting your instruction and planning the day..."):
        ep = run_episode(config, scenario, pipeline, input_mode="natural_language")
    st.session_state["op_result"] = {"ep": ep, "text": text, "confirmed": preview_req, "issued": issued_minute, "scenario": scenario_name}

res = st.session_state.get("op_result")
if not res or res["scenario"] != scenario_name:
    st.info("Confirm the interpretation above, then press **Apply instruction**.")
    st.stop()

ep, d = res["ep"], res["ep"].decision
if res["text"].strip() != text.strip():
    st.caption(f"Results below are for the instruction you applied: *{res['text']}*")

# 3. decision -------------------------------------------------------------------
st.subheader("3 · Decision")
status = d.status.value
kind, headline, todo = NEXT_STEP.get(status, ("info", status, ""))
getattr(st, kind)(f"**{headline}.** {todo}")

applied = normalise(d.parsed_instruction, d.issued_at_minute) if status == "accepted" else None
if applied is not None and res["confirmed"] is not None and applied != res["confirmed"]:
    st.warning("⚠️ **The system applied a different interpretation from the one you confirmed.** Check it below; "
               "if it is not what you meant, send a corrected instruction.")  # fmt: skip
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**You confirmed**")
        for line in describe(res["confirmed"]):
            st.markdown(f"- {line}")
    with c2:
        st.markdown("**The system applied**")
        for line in describe(applied):
            st.markdown(f"- {line}")
elif applied is not None and res["confirmed"] is None:
    st.markdown("**The system applied:**")
    for line in describe(applied, (), config.qos.max_utilization_by_level):
        st.markdown(f"- {line}")

if d.clarification_question:
    st.markdown(f"**Question from the system:** {d.clarification_question}")
if d.validation and d.validation.get("reasons") and status != "accepted":
    st.markdown("**Why it was blocked:**")
    for reason in d.validation["reasons"]:
        st.markdown(f"- {reason}")
if d.conflict_report:
    with st.expander("Conflict between your constraints", expanded=True):
        st.json(d.conflict_report)
if d.interpretation_issues and status != "accepted":
    st.markdown("**Problems found in the instruction:**")
    for issue in d.interpretation_issues:
        st.markdown(f"- {issue.get('message', issue) if isinstance(issue, dict) else issue}")
if d.fallback_reasons:
    st.info("Backup mode used: " + "; ".join(d.fallback_reasons))
st.markdown(f"**Explanation:** {d.explanation}")
if d.llm_explanation:
    with st.expander("Language-model explanation (not verified)"):
        st.markdown(d.llm_explanation)
st.caption(f"Interpreted by: {d.interpretation_method} · orchestrated by: {d.orchestration_mode} · "
           f"decision time: {d.latency_s:.2f} s" + (f" · controller: {d.selected_controller}" if d.selected_controller else ""))  # fmt: skip

# 4. monitoring --------------------------------------------------------------------
st.subheader("4 · Day monitoring")
if applied is None:
    st.caption("Your instruction was not applied: the day below ran under the previous policy.")
m = ep.metrics
k1, k2, k3, k4 = st.columns(4)
k1.metric("Electricity cost", f"{m['total_cost_eur']:.2f} EUR")
k2.metric("Energy bought from the grid", f"{m['grid_energy_kwh']:.1f} kWh")
k3.metric("Solar energy used", f"{m['pv_used_kwh']:.1f} kWh")
k4.metric("Battery at end of day", f"{m['soc_final_pct']:.0f} %")

checks = compliance(applied or {}, d.issued_at_minute, ep, config)
st.markdown("**Is everything under control?**")
for _, row in checks.iterrows():
    st.markdown(f"{'✅' if row.ok else '❌'} {row.check} — {row.observed}")

cons = compile_constraints(OperationalInstruction.model_validate({**applied, "issued_at_minute": d.issued_at_minute}), config) if applied else None
alert_df = alerts(ep, cons, config)
st.markdown("**Alerts during the day**")
if alert_df.empty:
    st.caption("No alerts.")
else:
    st.dataframe(alert_df, hide_index=True, width="stretch")

dt, bat = config.simulation.dt_h, config.battery
reserve_pct = 100 * cons.reserve_kwh / bat.capacity_kwh if cons is not None else np.zeros(len(ep.history) + 1)
grid_cap = cons.grid_cap_kw if cons is not None else np.full(len(ep.history), np.inf)
label = ep.decision.approach
st.plotly_chart(charts.soc_figure({label: ep.history}, reserve_pct, (bat.soc_min_pct, bat.soc_max_pct), bat.capacity_kwh, dt, theme,
                                  issue_hour=d.issued_at_minute / 60), theme=None, width="stretch")  # fmt: skip
st.caption("Shaded: battery reserve you requested. Dashed: battery safety limits. Dotted: time the instruction was sent.")
st.plotly_chart(charts.power_figure(ep.history, grid_cap, dt, theme, label), theme=None, width="stretch")
st.caption("When station demand is above solar + grid, the battery supplies the difference.")

# 5. record --------------------------------------------------------------------------
st.subheader("5 · Record")
st.caption(f"Tamper-evident audit log: {len(ep.audit.records)} entries, integrity check "
           f"{'passed ✅' if ep.audit.verify_chain() else 'FAILED ❌'}.")  # fmt: skip
st.download_button("Download the audit log (JSONL)", "\n".join(json.dumps(r) for r in ep.audit.records),
                   file_name=f"{ep.audit.run_id.replace('/', '_')}.jsonl")  # fmt: skip
