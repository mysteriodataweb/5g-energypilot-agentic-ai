# 5G EnergyPilot AI

**Agentic AI for energy-aware 5G base-station demand response — a reproducible research prototype.**

> Independent academic prototype. It is **not** an implementation of any operator's internal systems
> and uses no proprietary simulator, model or operational data. All data are **synthetic**, every
> decision is executed **inside a simulator only**, and no real telecommunications equipment is
> controlled.

## Motivation

5G base stations draw significant power around the clock. Many sites combine a grid connection with
photovoltaic panels and a battery originally sized for backup. That battery can also shift
consumption (cheaper hours, demand-response events) — provided backup reserves and quality of service
(QoS) are never compromised. Operators adapt the energy strategy through operational instructions
("keep 80 % until 23:00, a storm is coming", "limit grid import to 2 kW from 18:00 to 20:00").
Mathematical optimisation and deterministic rules can execute such instructions reliably once they are
formalised; large language models (LLMs) can read them in natural language. This project studies how
to combine the two safely, and when the combination is worth it.

## Research question

> *Under what conditions does LLM-based agentic supervision provide measurable value over deterministic
> orchestration when adapting a 5G base station's energy strategy to new operational instructions?*

Three approaches are compared on identical scenarios, seeds and initial conditions:

1. **Rule-based control** — deterministic instruction parser + interpretable rule controller;
2. **Mathematical optimisation without an LLM** — deterministic parser + MILP scheduler executed as MPC;
3. **LLM-based orchestration** — the LLM interprets the instruction and orchestrates the existing tools
   (optimiser, rule controller, diagnosis), with an independent safety validator having the final word.
   Without an LLM configured, the agent runs a deterministic fallback workflow and is labelled
   `agent[deterministic-fallback]`.

The design allows neutral or negative findings; see [the technical note](docs/technical_note_llm_orchestration.md)
for why the LLM orchestrates decision models instead of replacing safety-critical numerical controllers.

## Status

| Phase | Content | Status |
|---|---|---|
| 1-2 | project structure, validated configuration, simulator, synthetic data (7 scenarios) | done, tested |
| 3 | rule-based controller | done, tested |
| 4 | MILP optimiser (PuLP + HiGHS; CBC optional), MPC execution, infeasibility diagnosis | done, tested |
| 5 | independent safety validator, runtime shield, hash-chained audit log | done, tested |
| 6 | LLM orchestration agent (Anthropic / OpenAI-compatible adapters) + deterministic fallback | done, tested with a scripted test double; **not run against a real LLM** (no API key in the build environment) |
| 7 | evaluation framework (episodes, metrics, CIs, paired tests, benchmark) | done, tested; reference run in `results/reference/` |
| 8 | Streamlit + Plotly dashboard | done, tested (headless) |
| 9 | full test suite, reproducible demonstration | done |
| — | reinforcement-learning controller (Gymnasium + Stable-Baselines3) | **not implemented — future work** |

## System architecture

```
 operator instruction (natural language or JSON)
            │
            ▼
 ┌──────────────────────┐   strict schema, grounding check, safety screen,
 │ interpretation       │── semantic validation ──► clarification / rejection
 │ parser | JSON | LLM  │
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐  typed tools: state, constraints, run_optimizer,
 │ orchestration        │  run_rule_controller, validate_strategy, diagnose_infeasibility
 │ deterministic | LLM  │  (LLM: bounded loop, circuit breaker → deterministic fallback)
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │ final safety gate    │  independent validator recomputes physics from the set-points
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐  MPC (hourly re-planning, each plan validated) or rules,
 │ closed-loop          │  runtime shield on every action, simulator with PV/battery/grid/QoS
 │ simulation           │  ─► metrics, decision report, audit log (JSONL, hash chain)
 └──────────────────────┘
```

Details: [docs/architecture.md](docs/architecture.md). Python code is a single package `energypilot/`
whose sub-packages mirror the requested modules (`config`, `data`, `simulator`, `controllers`,
`optimizer`, `agent`, `safety`, `evaluation`); `dashboard/`, `scripts/`, `tests/`, `docs/`, `config/`
and `data/` are at the root.

## Mathematical assumptions (summary)

Full model: [docs/mathematical_model.md](docs/mathematical_model.md). Time step `Δt` = 15 min, horizon 24 h;
powers in kW (step averages), energies in kWh.

**Energy balance** (all flows ≥ 0, no grid export, PV surplus curtailed):

```
pv_used + grid_import + p_discharge = load_served + p_charge
pv_used = pv_available − curtailment
```

**Battery** (40 kWh, SoC 10–95 %, 10 kW, η_ch = η_dis = 0.95):

```
E[k+1] = E[k] + η_ch·p_charge·Δt − p_discharge·Δt/η_dis,     E_min ≤ E ≤ E_max
```

**Base-station power** (EARTH-inspired linear model; `c` = active capacity fraction, `λ` = normalised traffic):

```
P_bs = 1.2 + 1.0·c + 2.0·min(λ, c)   [kW]
```

**QoS proxy**: utilisation `λ/c ≤ ρ_max` (strict 0.70, standard 0.85, relaxed 0.95). Traffic above
`ρ_max` even at full capacity is an *exogenous* shortfall, never attributed to a controller.

**Optimiser**: MILP (one binary per step forbids simultaneous charge/discharge), objectives cost / grid
energy / peak import, hard reserves, grid caps and QoS capacity, soft end-of-day SoC. Infeasible
problems are reported, never silently relaxed.

## Data-generation methodology

Synthetic and reproducible (`SeedSequence([seed, crc32(scenario)])`): clear-sky PV × AR(1) cloud
attenuation; diurnal traffic template × log-normal AR(1) noise; demand *modelled* from traffic and
capacity decisions (never generated independently); illustrative time-of-use tariff; scenario events
(solar drop, traffic peak) absent from forecasts when unforeseen. Measured vs simulated vs estimated
vs assumed quantities are distinguished in [docs/data_provenance.md](docs/data_provenance.md).

## Installation

Python ≥ 3.11.

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
pip install -e .
```

Optional, only for a real LLM:

```bash
pip install -r requirements-llm.txt
```

The MILP is modelled with PuLP 2.x (`pulp<3`) and solved in-process by HiGHS (`highspy`). CBC, bundled
with PuLP, can be selected in `config/default.toml` (`[optimizer] solver = "CBC"`); in this project it
occasionally ran to its time limit (see [docs/mathematical_model.md](docs/mathematical_model.md) §7).

## Environment setup

```bash
cp .env.example .env
```

Leave `LLM_PROVIDER=none` for the deterministic fallback mode (no external call). To use an LLM, set
`LLM_PROVIDER` (`anthropic` or `openai` for any OpenAI-compatible endpoint, with `OPENAI_BASE_URL` if
needed), `LLM_MODEL` and the matching API key. Optionally set `LLM_PRICE_INPUT_PER_MTOK` /
`LLM_PRICE_OUTPUT_PER_MTOK` to estimate inference cost; otherwise cost is reported as unknown. Never
commit `.env`.

## Usage

Run one closed-loop simulation and print the decision report and metrics:

```bash
python scripts/run_simulation.py --scenario reserve_increase --approach agent
```

Submit your own instruction (natural language or JSON):

```bash
python scripts/run_simulation.py --scenario normal --approach optimizer --instruction "Keep at least 60% battery charge between 18:00 and 20:00."
```

Run the tests:

```bash
python -m pytest
```

Launch the dashboard:

```bash
streamlit run dashboard/app.py
```

Reproduce the reference evaluation (about 8 minutes on 4 cores; 840 + 150 + 50 episodes and the benchmark):

```bash
python scripts/run_evaluation.py --seeds 10 --out results/reference
```

Evaluate instruction interpretation (add `--with-llm` to include the configured LLM):

```bash
python scripts/run_instruction_benchmark.py
```

Export the synthetic profiles:

```bash
python scripts/generate_data.py --seeds 0 1 2
```

## Experimental protocol

[docs/experimental_protocol.md](docs/experimental_protocol.md). Seven scenarios (normal operation,
sudden solar drop, unexpected traffic peak, temporary reserve increase, high traffic with low solar
and a demand-response cap, infeasible/contradictory instruction, ambiguous instruction) × 10 seeds ×
3 approaches × 4 input modes (canonical text, ground-truth JSON, two held-out paraphrases in English
and French). Additional experiments: optimiser failure injection, perfect-forecast oracle, and an
interpretation benchmark with dev/test splits. Statistics: mean, standard deviation, t-based 95 % CI,
paired differences on the same synthetic day.

## Results

All numbers below come from executed closed-loop simulations on **synthetic** data
(`scripts/run_evaluation.py`); tables are generated from `results/reference/*.csv` by
`scripts/update_readme_results.py`. They are not real-world operational results.

<!-- RESULTS:START -->

Reference run: 10 seeds, solver HiGHS, commit `85105a3`, generated 2026-10-29T04:03:33+00:00. LLM: **none** (agent in deterministic fallback mode). Full report: [results/reference/report.md](results/reference/report.md).

**Instruction handling** (all 7 scenarios × seeds; *interpretation = truth* compares the parsed instruction with the label; *intent fully met* = no violation of the operator's intended instruction):

| approach | input | correct handling | interpretation = truth | intent fully met (accept scen.) | hard violations (steps, total) | episodes |
|---|---|---|---|---|---|---|
| agent[deterministic-fallback] | natural_language | 100% | 100% | 100% | 0 | 70 |
| agent[deterministic-fallback] | paraphrase_0 | 100% | 86% | 80% | 0 | 70 |
| agent[deterministic-fallback] | paraphrase_1 | 86% | 86% | 84% | 0 | 70 |
| agent[deterministic-fallback] | structured | 100% | 100% | 100% | 0 | 70 |
| optimizer | natural_language | 100% | 100% | 100% | 0 | 70 |
| optimizer | paraphrase_0 | 100% | 86% | 80% | 0 | 70 |
| optimizer | paraphrase_1 | 86% | 86% | 84% | 0 | 70 |
| optimizer | structured | 100% | 100% | 100% | 0 | 70 |
| rule_based | natural_language | 100% | 100% | 100% | 0 | 70 |
| rule_based | paraphrase_0 | 100% | 86% | 80% | 0 | 70 |
| rule_based | paraphrase_1 | 86% | 86% | 100% | 0 | 70 |
| rule_based | structured | 100% | 100% | 100% | 0 | 70 |

**SoC-adjusted daily cost (EUR), canonical instruction text, mean [95 % CI]:**

| scenario | agent[deterministic-fallback] | optimizer | rule_based |
|---|---|---|---|
| high_traffic_low_solar | 8.79 [8.45, 9.12] | 8.79 [8.45, 9.12] | 9.66 [9.35, 9.98] |
| normal | 2.60 [2.53, 2.68] | 2.60 [2.53, 2.68] | 4.40 [4.33, 4.48] |
| reserve_increase | 5.41 [4.89, 5.93] | 5.41 [4.89, 5.93] | 5.56 [4.91, 6.21] |
| solar_drop | 6.21 [6.12, 6.29] | 6.21 [6.12, 6.29] | 6.47 [6.36, 6.57] |
| traffic_peak | 5.08 [4.67, 5.50] | 5.08 [4.67, 5.50] | 5.46 [5.05, 5.88] |

**Robustness** (optimiser tool fails at instruction time, accept scenarios):

| approach | decision_status | episodes |
|---|---|---|
| agent[deterministic-fallback] | accepted | 50 |
| optimizer | tool_failure | 50 |
| rule_based | accepted | 50 |

**Cost of forecast errors** (optimiser; SoC-adjusted cost with persistence-corrected forecasts minus with perfect forecasts, EUR/day):

| scenario | mean | ci95_low | ci95_high | n |
|---|---|---|---|---|
| high_traffic_low_solar | 0.172 | 0.133 | 0.211 | 10 |
| normal | 0.087 | 0.070 | 0.104 | 10 |
| reserve_increase | 0.051 | 0.023 | 0.079 | 10 |
| solar_drop | 0.085 | 0.072 | 0.098 | 10 |
| traffic_peak | 0.186 | 0.169 | 0.202 | 10 |

**Instruction-interpretation benchmark:**

| interpreter | split | n | status_accuracy | exact_accuracy | unsafe_acceptance_rate | over_caution_rate | mean_latency_s |
|---|---|---|---|---|---|---|---|
| deterministic_rules | dev | 15 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| deterministic_rules | test | 34 | 0.853 | 0.824 | 0.029 | 0.211 | 0.000 |

<!-- RESULTS:END -->

### Findings (reference run, synthetic data)

1. **Safety held in every episode.** Across 840 main episodes and 150 fault-injection episodes, no approach
   violated a constraint it had accepted (0 hard-violation steps, 0 kWh unserved), and the infeasible
   instruction was refused by all three (proven infeasible by the optimiser pipelines, rejected by the
   validator for the rule-based pipeline).
2. **Optimisation pays off over rules.** With the canonical instruction text, the MILP/MPC pipeline lowered
   the SoC-adjusted daily cost in every scenario. Paired differences on the same synthetic days range from
   −0.15 EUR/day (`reserve_increase`, 95 % CI [−0.28, −0.02]) to −1.80 EUR/day (`normal`,
   [−1.85, −1.75]). It sometimes imports *more* grid energy (e.g. +6.0 kWh in `high_traffic_low_solar`)
   because it buys cheap off-peak energy to avoid peak prices: the objective was cost, not energy.
3. **Without an LLM, the agent is the optimiser.** `agent[deterministic-fallback]` and `optimizer` produced
   identical results in every no-fault episode (paired difference 0). Orchestration only made a
   difference under tool failure: when the optimiser call failed, the optimiser-only pipeline could not
   act on any of the 50 instructions (intent fully met in 22 % of episodes), whereas the agent's
   deterministic fallback to the rule controller accepted and complied with all 50. **That robustness
   gain does not need an LLM.**
4. **Interpretation is the bottleneck for adapting to new instructions.** Every pipeline handled the
   canonical texts and the structured JSON perfectly, but 2 of the 14 held-out paraphrases were
   misinterpreted by the deterministic parser:
   * the French paraphrase of the normal scenario triggered a clarification request — a safe failure,
     whose cost depends on the policy left in force (intent met in 20 % of episodes for the optimiser
     pipelines, 100 % for the rule-based one);
   * "…should hold no less than **sixty percent**…" was **accepted with the reserve silently dropped**
     (numbers written as words are not parsed). The intended reserve was then missed in 10 of 10 episodes
     (mean per-episode worst deficit 9.5 kWh, about 24 % of capacity, for the optimiser pipelines). This is the most
     dangerous failure class: every *accepted* constraint was respected, yet the operator's intent was not.
5. **The held-out benchmark confirms it.** On the test split the deterministic parser reached 82 % exact
   accuracy, with 21 % over-caution (valid instructions not accepted) and 3 % unsafe acceptance (one
   French objective replaced by the default). The dev split scored 100 %, which shows how optimistic
   in-sample evaluation of hand-written parsers is.
6. **Supervision costs.** Median decision latency at instruction time was 5 ms (rules) and 33 ms
   (optimiser, including validation), with a maximum of 85 ms; per-step control decisions averaged
   4 ms. Perfect forecasts would have saved only 0.05–0.19 EUR/day, far less than the rules-to-optimiser
   gap. An LLM call typically takes seconds and has a per-token cost, which is the budget any
   interpretation gain must justify.

**Provisional answer to the research question.** Under these conditions, LLM supervision cannot improve
hard safety (already zero violations, guaranteed by the validator and shield) or control cost (the agent
executes the same optimiser), and robustness to tool failures is obtained deterministically. Its
measurable room for value is confined to **interpreting unseen phrasings**. The value is real only if an
LLM interpreter reduces silent misinterpretations and over-caution (here 2/14 paraphrases and 18 % of
benchmark test items) without adding unsafe acceptances, at an acceptable latency and cost. **The LLM
arm was not executed in this repository (no API key in the build environment), so whether it achieves
this remains unmeasured.** `scripts/run_evaluation.py --with-llm` and
`scripts/run_instruction_benchmark.py --with-llm` measure exactly this. A cheap deterministic alternative
should be evaluated alongside it: a completeness check that requests clarification when a
battery-related sentence contains a number that no rule consumed would have turned the silent drop into
a safe clarification.

**Engineering finding.** With CBC, about 0.85 % of MPC re-plans ran to the 10 s time limit,
non-deterministically and with unchanged optimal schedules. HiGHS removed the stalls and made the full
evaluation about 3.3× faster (details in [docs/mathematical_model.md](docs/mathematical_model.md) §7).

## Known limitations

* Synthetic data and a simplified power model: absolute costs are illustrative; only comparisons under
  identical assumptions are meaningful.
* No grid export, no battery ageing model beyond a throughput cost, no temperature effects, no
  multi-site coordination.
* QoS is a utilisation proxy, not a radio simulation (no SINR, scheduling or user mobility).
* The deterministic parser covers a documented English/French subset; the benchmark and paraphrases
  were written by the same author as the parser (held-out split mitigates, does not remove, the bias).
* The LLM arm has been implemented and tested with a scripted test double only; its real accuracy,
  latency and cost have not been measured in this repository.
* Forecasts are synthetic persistence-corrected profiles, not a trained forecaster.
* The rule-based controller has hand-tuned thresholds and no end-of-day SoC management, which is why
  the SoC-adjusted cost is used for comparisons.

## Future work

* **Real LLM evaluation** across providers and model sizes (accuracy, latency, cost, failure modes),
  with an independently written, multi-author instruction benchmark.
* **Reinforcement learning** (optional extension, not implemented): a Gymnasium environment wrapping
  `BaseStationEnv` (observation: SoC, PV, traffic, price, time, active constraints; action: battery
  power and capacity fraction; reward: −cost − penalties for reserve, grid-cap and QoS violations and
  battery wear; episode = one day), trained with Stable-Baselines3 on training seeds and evaluated on
  disjoint seeds and scenarios to prevent leakage, then compared with the optimiser and the rules.
* **Public data integration**: measured PV (e.g. irradiance datasets), day-ahead and balancing prices,
  and published base-station load models in place of the synthetic profiles.
* Multi-site coordination, battery ageing models, stochastic or robust optimisation for forecast
  uncertainty, and a richer QoS model.

## Documentation

* [docs/architecture.md](docs/architecture.md) — modules, dependency layers, decision flow, trade-offs
* [docs/mathematical_model.md](docs/mathematical_model.md) — equations, MILP, validator checks, metrics
* [docs/technical_note_llm_orchestration.md](docs/technical_note_llm_orchestration.md) — why the LLM orchestrates rather than replaces
* [docs/instruction_interface.md](docs/instruction_interface.md) — schema, grammar, clarification and rejection rules, benchmark history
* [docs/data_provenance.md](docs/data_provenance.md) — measured / synthetic / simulated / estimated data
* [docs/experimental_protocol.md](docs/experimental_protocol.md) — protocol, metrics, statistics, threats to validity
