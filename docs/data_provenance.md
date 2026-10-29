# Data provenance and generation methodology

**All data in this prototype are synthetic.** No proprietary simulator, model or operational data of
any operator is used. The prototype is an independent academic exercise.

## Categories

| Category | Meaning in this project | Examples |
|---|---|---|
| **Measured** (real) | values recorded on real equipment | *none* |
| **Synthetic measurement** | generated values that play the role of measurements in the simulator | realised PV `pv_actual_kw`, realised traffic `traffic_actual` |
| **Synthetic forecast** | generated values available to planners in advance | `pv_forecast_kw`, `traffic_forecast` |
| **Simulated** | outputs of the simulator given decisions | grid import, SoC trajectory, curtailment, QoS indicators, costs |
| **Modelled** | quantities computed by a documented model | base-station demand from traffic and capacity |
| **Estimated** | approximations used for comparison or reporting | SoC-adjusted cost, delay factor, LLM cost (only with user-supplied prices) |
| **Assumed** | parameters chosen by the author | tariff, battery and PV sizes, efficiencies, power-model coefficients |
| **Externally sourced** | values taken from outside sources | *none in v1* (planned: public PV and price data) |

Each `ScenarioData` carries a `provenance` dictionary with these labels; the dashboard displays it.

## Generation (reproducible)

`energypilot/data/scenarios.py::generate_scenario(name, config, seed)` uses
`numpy.random.SeedSequence([seed, crc32(name)])` and spawns independent streams for clouds, traffic
and the initial SoC, so each (scenario, seed) pair is reproducible across processes and platforms
(Python's salted `hash` is never used).

* **PV**: clear-sky AC profile `P_peak · sin(π x)^1.5` between sunrise (06:30) and sunset (20:30),
  multiplied by a cloud attenuation following an AR(1) process around the weather-class mean
  (sunny 0.95 ± 0.03, partly cloudy 0.70 ± 0.15, overcast 0.30 ± 0.08), clipped to [0.05, 1].
  Forecast = clear-sky × class mean.
* **Traffic**: baseline + three circular Gaussian peaks (09:00, 13:30, 20:30) × scenario scale ×
  mean-one log-normal AR(1) noise; clipped to [0, 1.3]. Forecast = template × forecast scale.
* **Demand**: never generated independently — modelled by the simulator from traffic and the
  controller's capacity decision, so demand and traffic stay physically consistent.
* **Price**: illustrative time-of-use tariff (0.15 / 0.21 / 0.32 EUR/kWh), not a real offer.
* **Events**: localised modifications with linear ramps (e.g. PV × 0.2 from 11:00 to 16:00; traffic
  +0.45 from 18:00 to 21:30), flagged as foreseen or unforeseen; unforeseen events are absent from
  forecasts.
* **Initial SoC**: 60 % ± 5 % uniform jitter per seed, unless the scenario fixes it.

Regenerate CSV files with `python scripts/generate_data.py --seeds 0 1 2` (written to
`data/generated/`, git-ignored, with a JSON sidecar describing provenance).

## Instruction benchmark

`data/instruction_benchmark.json` contains operator instructions written by the project author (EN
and FR) with labels. The `dev` split was used while developing the deterministic parser; the `test`
split was written afterwards, and the parser's interpretation coverage was not tuned on it (see
`docs/instruction_interface.md` for the one safety-screen fix and its before/after numbers).
