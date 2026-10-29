# Experimental protocol

## Research question

> Under what conditions does LLM-based agentic supervision provide measurable value over
> deterministic orchestration when adapting a 5G base station's energy strategy to new operational
> instructions?

The protocol is designed so that neutral or negative answers are possible and visible.

## Approaches

| Label | Interpretation | Orchestration | Controller after acceptance |
|---|---|---|---|
| `rule_based` | deterministic parser | rule roll-out → validation | rule-based |
| `optimizer` | deterministic parser | MILP → validation (diagnosis if infeasible), no fallback | MPC |
| `agent[deterministic-fallback]` | deterministic parser | agent workflow without LLM: MILP → validation → diagnosis; rule controller on tool failure | MPC or rule-based |
| `agent[llm:<provider>/<model>]` | LLM + grounding + safety screen | bounded LLM tool loop, final validation gate, circuit breaker | MPC or rule-based |

The LLM row is only produced when an LLM is configured (`--with-llm` and `.env`); without it, no LLM
result is reported. A scripted test double exists for tests only and is labelled `TEST-DOUBLE`.

## Scenarios (synthetic)

| # | Scenario | Instruction (issue time) | Expected handling | Stress |
|---|---|---|---|---|
| 1 | `normal` | 70 % reserve for 2 h, minimise cost (00:00) | accept | nominal |
| 2 | `solar_drop` | 60 % reserve 17–21 h (08:00) | accept | unforeseen −80 % PV 11–16 h |
| 3 | `traffic_peak` | strict QoS today (08:00) | accept | unforeseen +0.45 traffic 18–21:30 |
| 4 | `reserve_increase` | 80 % reserve 18–23 h (14:00) | accept | pre-charging needed |
| 5 | `high_traffic_low_solar` | 2 kW grid cap 18–20 h (10:00) | accept | overcast, traffic +25 % unforeseen |
| 6 | `infeasible_instruction` | 90 % reserve + 0 kW grid 17–21 h (16:00) | reject | contradictory requirements |
| 7 | `ambiguous_instruction` | "reasonably high for a while" (10:00) | clarify | missing information |

## Input modes

* `natural_language` — the canonical text (seen during parser development);
* `paraphrase_0`, `paraphrase_1` — held-out rephrasings (EN, FR) with the same ground truth, written
  after the parser was developed; the parser was not tuned on them;
* `structured` — the ground-truth JSON (isolates control performance from interpretation; scenario 7
  has no ground truth and is submitted as text).

## Experiments (`scripts/run_evaluation.py`)

* **E1 main**: 7 scenarios × N seeds × 3 approaches × 4 input modes, same initial conditions.
* **E2 robustness**: accept scenarios with the optimiser tool failing at instruction time.
* **E3 oracle**: optimiser with perfect forecasts, to quantify the cost of forecast errors.
* **E4 benchmark**: interpretation accuracy on `data/instruction_benchmark.json` (dev/test splits).

## Metrics

See `docs/mathematical_model.md` §12. Primary outcomes:

1. **Correct instruction handling** (accept / clarify / refuse as labelled);
2. **Intent compliance**: violations of the *intended* instruction (ground truth) on accept scenarios;
3. **Hard safety violations** of the constraints actually in force (should be zero for every approach);
4. **SoC-adjusted cost**, grid energy, PV self-consumption;
5. **Decision latency**, LLM calls/tokens/cost, tool-call failures (cost of supervision).

Hard violations (safety) and soft penalties (cost, end-of-day SoC, exogenous QoS shortfall) are
reported separately.

## Statistics

Seeds give independent synthetic days (cloud and traffic noise, initial SoC). For each metric:
mean, sample standard deviation and a t-based 95 % confidence interval when n ≥ 3 and the variance is
non-zero. Comparisons between approaches use **paired differences on the same seed** (same synthetic
day), which removes between-day variance. Deterministic outcomes (e.g. a parser that always fails on a
phrasing) have zero variance and no CI: they are reported as rates.

## Threats to validity

* **Author bias in the benchmark**: instructions, paraphrases and labels were written by the developer
  of the deterministic parser. The held-out split mitigates but does not remove this; an independently
  written benchmark (several operators) is needed for strong claims.
* **Synthetic data**: the simulator, the power model and the profiles are simplified; absolute costs
  are illustrative. Relative comparisons under identical assumptions are the object of study.
* **Small scenario set**: seven scenarios cannot represent the diversity of operational instructions.
* **No LLM results in the reference run**: the build environment had no API key, so the LLM arm was
  not executed; its value is therefore *not yet measured*.
* **MPC = optimiser in fallback mode**: without an LLM or faults, `agent[deterministic-fallback]`
  and `optimizer` produce identical decisions; differences appear only under tool failures.

## Running with an LLM

```bash
cp .env.example .env              # set LLM_PROVIDER, LLM_MODEL and the API key
pip install -r requirements-llm.txt
python scripts/run_instruction_benchmark.py --with-llm
python scripts/run_evaluation.py --seeds 5 --with-llm --out results/llm_run
```

Set `LLM_PRICE_INPUT_PER_MTOK` / `LLM_PRICE_OUTPUT_PER_MTOK` to obtain cost estimates; otherwise the
cost is reported as unknown.
