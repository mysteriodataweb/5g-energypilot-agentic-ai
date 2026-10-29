# Architecture

## Layout

```
5g-energypilot-agentic-ai/
├── config/default.toml          all parameters (mirrors energypilot/config/settings.py)
├── data/                        labelled instruction benchmark; generated profiles (git-ignored)
├── energypilot/                 the Python package
│   ├── config/                  validated settings (pydantic) + TOML loader
│   ├── core/                    shared domain types: ControlAction, Observation, Forecast, Strategy
│   ├── data/                    synthetic profiles, scenario catalogue, forecast provider
│   ├── simulator/               physics (battery, base-station power, QoS) + closed-loop environment
│   ├── instructions/            strict schema, semantic validation, deterministic parser, compiled constraints
│   ├── controllers/             rule-based controller, MPC execution of the optimiser, forecast roll-outs
│   ├── optimizer/               MILP scheduler (PuLP/CBC) + infeasibility diagnosis
│   ├── safety/                  independent validator, runtime shield, hash-chained audit log
│   ├── agent/                   LLM interface + adapters, typed tools, interpretation, orchestrator, reports
│   └── evaluation/              episodes, metrics, statistics, benchmark, report generation
├── dashboard/                   Streamlit app + Plotly figure builders
├── scripts/                     CLI entry points (data, single run, evaluation, benchmark)
├── tests/                       pytest suite
└── docs/                        model, architecture, protocol, data provenance, technical note
```

Python code lives in one importable package (`energypilot`) whose sub-packages mirror the suggested
module names; top-level packages named `config` or `data` would collide with other installed
packages. Non-code assets (`config/`, `data/`, `dashboard/`, `docs/`, `scripts/`, `tests/`) stay at
the repository root.

## Dependency layers (no cycles)

```
config → core → data → simulator → instructions → controllers / optimizer → safety → agent → evaluation → dashboard, scripts
```

* The **simulator** knows nothing about instructions, controllers or LLMs; every approach drives it
  through `observe()` / `step(ControlAction)`.
* The **safety** layer imports physics and instruction compilation, never `energypilot.agent`.
* LLM providers are isolated behind `agent/llm/base.py`; SDKs are imported lazily by the adapters,
  so the package runs without them.

## Decision flow for one instruction

```
instruction text / JSON
   │
   ├─ interpretation ── deterministic parser | strict JSON schema | LLM + grounding check
   │        └─ safety screen + semantic validation  →  needs_clarification / rejected  (stop)
   │
   ├─ orchestration ── deterministic workflow | LLM tool loop (bounded)  ── circuit breaker → fallback
   │        tools: get_current_state, get_active_constraints, run_optimizer,
   │               run_rule_controller, validate_strategy, diagnose_infeasibility
   │
   ├─ final gate ── independent SafetyValidator on the selected strategy
   │
   └─ DecisionReport (status, tool calls, solver status, validation, diagnosis, explanation, usage)
          + audit log entries; accepted → controller activated in the simulator
```

Outcome statuses: `accepted`, `needs_clarification`, `rejected_invalid_instruction`, `infeasible`,
`rejected_by_validator`, `rejected_by_agent`, `tool_failure`. A rejected instruction never replaces
the active one.

## Closed loop

At every step: observe (measured PV, traffic, SoC) → forecast (persistence-corrected) → controller
decision → runtime shield → simulator (physics, adjustments recorded) → history. The MPC controller
re-plans hourly and validates each plan before adopting it.

## Approaches as configurations

| Approach | Interpretation | Orchestration | Controller |
|---|---|---|---|
| `rule_based` | deterministic parser | fixed: rule roll-out → validate | rule-based |
| `optimizer` | deterministic parser | fixed: optimise → validate; diagnose if infeasible | MPC |
| `agent` | LLM if configured, else deterministic | LLM tool loop if configured, else deterministic workflow with rule fallback on tool failure | MPC or rule-based |

All three share the simulator, forecasts, validator, shield and metrics, so differences are
attributable to interpretation and orchestration only.

## Design trade-offs

* **Plan-then-execute with MPC** rather than a single open-loop plan: forecast errors (scenarios 2, 3,
  5) are real in the simulator, and hourly re-planning is the standard remedy.
* **MILP with one binary per step** rather than a pure LP: guarantees no simultaneous charge and
  discharge at negligible solve-time cost; `battery_mode_binaries = false` gives the LP variant.
* **LLM only at instruction time**: bounded latency and cost; the loop never depends on an external API.
* **Small tool surface, IDs instead of payloads**: the LLM cannot pass arbitrary set-points; the
  audit log captures every call.
* **Strict schema + grounding** rather than trusting structured output: values must be quoted from
  the operator's text; documented defaults are recorded in `defaults_applied`.
