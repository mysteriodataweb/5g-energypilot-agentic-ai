# Experiment results

Generated 2026-10-29T04:03:33+00:00 by `scripts/run_evaluation.py` (commit 85105a3). Seeds: 10. Solver: HiGHS. LLM: none configured - the agent ran in deterministic fallback mode; no LLM results are reported.

> All numbers come from executed closed-loop simulations on **synthetic** data. They are not measurements of a real base station and must not be read as real-world operational results.

## E1 - Instruction handling by approach and input mode

Correct handling = accepted when the label is *accept*, clarification when *clarify*, any refusal (invalid, infeasible, validator rejection) when *reject*. Intent compliance is measured against the operator's intended instruction on *accept* scenarios.

| approach | input_mode | episodes | correct_handling_rate | interpretation_match_rate | accepted_rate | intent_fully_compliant_rate | hard_violation_steps_total | median_decision_latency_s | p95_decision_latency_s | tool_call_failures | llm_calls |
|---|---|---|---|---|---|---|---|---|---|---|---|
| agent[deterministic-fallback] | natural_language | 70 | 1.000 | 1.000 | 0.714 | 1.000 | 0 | 0.033 | 0.049 | 0 | 0 |
| agent[deterministic-fallback] | paraphrase_0 | 70 | 1.000 | 0.857 | 0.714 | 0.800 | 0 | 0.033 | 0.047 | 0 | 0 |
| agent[deterministic-fallback] | paraphrase_1 | 70 | 0.857 | 0.857 | 0.571 | 0.840 | 0 | 0.020 | 0.039 | 0 | 0 |
| agent[deterministic-fallback] | structured | 70 | 1.000 | 1.000 | 0.714 | 1.000 | 0 | 0.033 | 0.048 | 0 | 0 |
| optimizer | natural_language | 70 | 1.000 | 1.000 | 0.714 | 1.000 | 0 | 0.033 | 0.048 | 0 | 0 |
| optimizer | paraphrase_0 | 70 | 1.000 | 0.857 | 0.714 | 0.800 | 0 | 0.033 | 0.047 | 0 | 0 |
| optimizer | paraphrase_1 | 70 | 0.857 | 0.857 | 0.571 | 0.840 | 0 | 0.019 | 0.040 | 0 | 0 |
| optimizer | structured | 70 | 1.000 | 1.000 | 0.714 | 1.000 | 0 | 0.033 | 0.048 | 0 | 0 |
| rule_based | natural_language | 70 | 1.000 | 1.000 | 0.714 | 1.000 | 0 | 0.005 | 0.008 | 0 | 0 |
| rule_based | paraphrase_0 | 70 | 1.000 | 0.857 | 0.714 | 0.800 | 0 | 0.005 | 0.007 | 0 | 0 |
| rule_based | paraphrase_1 | 70 | 0.857 | 0.857 | 0.571 | 1.000 | 0 | 0.004 | 0.006 | 0 | 0 |
| rule_based | structured | 70 | 1.000 | 1.000 | 0.714 | 1.000 | 0 | 0.005 | 0.007 | 0 | 0 |

## E1 - SoC-adjusted daily cost (EUR), canonical instruction text, mean [95% CI] over seeds

| scenario | agent[deterministic-fallback] | optimizer | rule_based |
|---|---|---|---|
| high_traffic_low_solar | 8.79 [8.45, 9.12] | 8.79 [8.45, 9.12] | 9.66 [9.35, 9.98] |
| normal | 2.60 [2.53, 2.68] | 2.60 [2.53, 2.68] | 4.40 [4.33, 4.48] |
| reserve_increase | 5.41 [4.89, 5.93] | 5.41 [4.89, 5.93] | 5.56 [4.91, 6.21] |
| solar_drop | 6.21 [6.12, 6.29] | 6.21 [6.12, 6.29] | 6.47 [6.36, 6.57] |
| traffic_peak | 5.08 [4.67, 5.50] | 5.08 [4.67, 5.50] | 5.46 [5.05, 5.88] |

## E1 - Grid energy imported (kWh), canonical text

| scenario | agent[deterministic-fallback] | optimizer | rule_based |
|---|---|---|---|
| high_traffic_low_solar | 55.7 [54.1, 57.2] | 55.7 [54.1, 57.2] | 49.7 [47.9, 51.4] |
| normal | 17.1 [16.6, 17.5] | 17.1 [16.6, 17.5] | 23.9 [23.6, 24.1] |
| reserve_increase | 34.0 [31.1, 36.9] | 34.0 [31.1, 36.9] | 34.9 [32.1, 37.8] |
| solar_drop | 35.2 [34.9, 35.4] | 35.2 [34.9, 35.4] | 33.4 [32.5, 34.2] |
| traffic_peak | 32.6 [30.0, 35.3] | 32.6 [30.0, 35.3] | 28.4 [25.9, 30.8] |

## E1 - Hard safety violations (steps, constraints in force), canonical text

| scenario | agent[deterministic-fallback] | optimizer | rule_based |
|---|---|---|---|
| ambiguous_instruction | 0.0 | 0.0 | 0.0 |
| high_traffic_low_solar | 0.0 | 0.0 | 0.0 |
| infeasible_instruction | 0.0 | 0.0 | 0.0 |
| normal | 0.0 | 0.0 | 0.0 |
| reserve_increase | 0.0 | 0.0 | 0.0 |
| solar_drop | 0.0 | 0.0 | 0.0 |
| traffic_peak | 0.0 | 0.0 | 0.0 |

## E1 - Exogenous QoS shortfall steps (traffic above target even at full capacity)

| scenario | agent[deterministic-fallback] | optimizer | rule_based |
|---|---|---|---|
| ambiguous_instruction | 0.0 | 0.0 | 0.0 |
| high_traffic_low_solar | 4.7 [2.9, 6.5] | 4.7 [2.9, 6.5] | 4.7 [2.9, 6.5] |
| infeasible_instruction | 0.0 | 0.0 | 0.0 |
| normal | 0.0 | 0.0 | 0.0 |
| reserve_increase | 0.0 | 0.0 | 0.0 |
| solar_drop | 0.0 | 0.0 | 0.0 |
| traffic_peak | 13.0 | 13.0 | 13.0 |

## E1 - Paired differences on the same synthetic days (approach - baseline), canonical text

| scenario | approach | baseline | metric | mean_diff | ci95_low | ci95_high | n |
|---|---|---|---|---|---|---|---|
| ambiguous_instruction | agent[deterministic-fallback] | rule_based | soc_adjusted_cost_eur | -0.355 | -0.494 | -0.215 | 10 |
| high_traffic_low_solar | agent[deterministic-fallback] | rule_based | soc_adjusted_cost_eur | -0.877 | -0.922 | -0.832 | 10 |
| infeasible_instruction | agent[deterministic-fallback] | rule_based | soc_adjusted_cost_eur | -0.426 | -0.485 | -0.367 | 10 |
| normal | agent[deterministic-fallback] | rule_based | soc_adjusted_cost_eur | -1.801 | -1.848 | -1.754 | 10 |
| reserve_increase | agent[deterministic-fallback] | rule_based | soc_adjusted_cost_eur | -0.150 | -0.281 | -0.019 | 10 |
| solar_drop | agent[deterministic-fallback] | rule_based | soc_adjusted_cost_eur | -0.258 | -0.283 | -0.233 | 10 |
| traffic_peak | agent[deterministic-fallback] | rule_based | soc_adjusted_cost_eur | -0.378 | -0.416 | -0.339 | 10 |
| ambiguous_instruction | optimizer | rule_based | soc_adjusted_cost_eur | -0.355 | -0.494 | -0.215 | 10 |
| high_traffic_low_solar | optimizer | rule_based | soc_adjusted_cost_eur | -0.877 | -0.922 | -0.832 | 10 |
| infeasible_instruction | optimizer | rule_based | soc_adjusted_cost_eur | -0.426 | -0.485 | -0.367 | 10 |
| normal | optimizer | rule_based | soc_adjusted_cost_eur | -1.801 | -1.848 | -1.754 | 10 |
| reserve_increase | optimizer | rule_based | soc_adjusted_cost_eur | -0.150 | -0.281 | -0.019 | 10 |
| solar_drop | optimizer | rule_based | soc_adjusted_cost_eur | -0.258 | -0.283 | -0.233 | 10 |
| traffic_peak | optimizer | rule_based | soc_adjusted_cost_eur | -0.378 | -0.416 | -0.339 | 10 |
| ambiguous_instruction | agent[deterministic-fallback] | rule_based | grid_energy_kwh | 2.035 | -0.865 | 4.936 | 10 |
| high_traffic_low_solar | agent[deterministic-fallback] | rule_based | grid_energy_kwh | 6.013 | 5.382 | 6.644 | 10 |
| infeasible_instruction | agent[deterministic-fallback] | rule_based | grid_energy_kwh | 3.880 | 2.596 | 5.163 | 10 |
| normal | agent[deterministic-fallback] | rule_based | grid_energy_kwh | -6.803 | -7.202 | -6.404 | 10 |
| reserve_increase | agent[deterministic-fallback] | rule_based | grid_energy_kwh | -0.887 | -1.033 | -0.740 | 10 |
| solar_drop | agent[deterministic-fallback] | rule_based | grid_energy_kwh | 1.816 | 1.040 | 2.592 | 10 |
| traffic_peak | agent[deterministic-fallback] | rule_based | grid_energy_kwh | 4.284 | 3.658 | 4.910 | 10 |
| ambiguous_instruction | optimizer | rule_based | grid_energy_kwh | 2.035 | -0.865 | 4.936 | 10 |
| high_traffic_low_solar | optimizer | rule_based | grid_energy_kwh | 6.013 | 5.382 | 6.644 | 10 |
| infeasible_instruction | optimizer | rule_based | grid_energy_kwh | 3.880 | 2.596 | 5.163 | 10 |
| normal | optimizer | rule_based | grid_energy_kwh | -6.803 | -7.202 | -6.404 | 10 |
| reserve_increase | optimizer | rule_based | grid_energy_kwh | -0.887 | -1.033 | -0.740 | 10 |
| solar_drop | optimizer | rule_based | grid_energy_kwh | 1.816 | 1.040 | 2.592 | 10 |
| traffic_peak | optimizer | rule_based | grid_energy_kwh | 4.284 | 3.658 | 4.910 | 10 |
| ambiguous_instruction | agent[deterministic-fallback] | optimizer | soc_adjusted_cost_eur | 0.000 | - | - | 10 |
| high_traffic_low_solar | agent[deterministic-fallback] | optimizer | soc_adjusted_cost_eur | 0.000 | - | - | 10 |
| infeasible_instruction | agent[deterministic-fallback] | optimizer | soc_adjusted_cost_eur | 0.000 | - | - | 10 |
| normal | agent[deterministic-fallback] | optimizer | soc_adjusted_cost_eur | 0.000 | - | - | 10 |
| reserve_increase | agent[deterministic-fallback] | optimizer | soc_adjusted_cost_eur | 0.000 | - | - | 10 |
| solar_drop | agent[deterministic-fallback] | optimizer | soc_adjusted_cost_eur | 0.000 | - | - | 10 |
| traffic_peak | agent[deterministic-fallback] | optimizer | soc_adjusted_cost_eur | 0.000 | - | - | 10 |
| ambiguous_instruction | rule_based | optimizer | soc_adjusted_cost_eur | 0.355 | 0.215 | 0.494 | 10 |
| high_traffic_low_solar | rule_based | optimizer | soc_adjusted_cost_eur | 0.877 | 0.832 | 0.922 | 10 |
| infeasible_instruction | rule_based | optimizer | soc_adjusted_cost_eur | 0.426 | 0.367 | 0.485 | 10 |
| normal | rule_based | optimizer | soc_adjusted_cost_eur | 1.801 | 1.754 | 1.848 | 10 |
| reserve_increase | rule_based | optimizer | soc_adjusted_cost_eur | 0.150 | 0.019 | 0.281 | 10 |
| solar_drop | rule_based | optimizer | soc_adjusted_cost_eur | 0.258 | 0.233 | 0.283 | 10 |
| traffic_peak | rule_based | optimizer | soc_adjusted_cost_eur | 0.378 | 0.339 | 0.416 | 10 |

## E1 - Decision latency at instruction time (s), canonical text: median and 95th percentile

Wall-clock times on the machine that ran the experiment; medians are robust to scheduling outliers.

| scenario | approach | median | p95 | max |
|---|---|---|---|---|
| ambiguous_instruction | agent[deterministic-fallback] | 0.0003 | 0.0004 | 0.0004 |
| ambiguous_instruction | optimizer | 0.0003 | 0.0005 | 0.0006 |
| ambiguous_instruction | rule_based | 0.0003 | 0.0004 | 0.0004 |
| high_traffic_low_solar | agent[deterministic-fallback] | 0.0179 | 0.0217 | 0.0226 |
| high_traffic_low_solar | optimizer | 0.0181 | 0.0216 | 0.0224 |
| high_traffic_low_solar | rule_based | 0.0052 | 0.0054 | 0.0055 |
| infeasible_instruction | agent[deterministic-fallback] | 0.0372 | 0.0533 | 0.0653 |
| infeasible_instruction | optimizer | 0.0377 | 0.0399 | 0.0403 |
| infeasible_instruction | rule_based | 0.0040 | 0.0045 | 0.0045 |
| normal | agent[deterministic-fallback] | 0.0467 | 0.0529 | 0.0547 |
| normal | optimizer | 0.0474 | 0.0514 | 0.0515 |
| normal | rule_based | 0.0075 | 0.0187 | 0.0272 |
| reserve_increase | agent[deterministic-fallback] | 0.0179 | 0.0190 | 0.0192 |
| reserve_increase | optimizer | 0.0181 | 0.0202 | 0.0210 |
| reserve_increase | rule_based | 0.0045 | 0.0046 | 0.0046 |
| solar_drop | agent[deterministic-fallback] | 0.0331 | 0.0387 | 0.0392 |
| solar_drop | optimizer | 0.0330 | 0.0403 | 0.0415 |
| solar_drop | rule_based | 0.0057 | 0.0063 | 0.0067 |
| traffic_peak | agent[deterministic-fallback] | 0.0344 | 0.0384 | 0.0393 |
| traffic_peak | optimizer | 0.0348 | 0.0384 | 0.0385 |
| traffic_peak | rule_based | 0.0057 | 0.0070 | 0.0071 |

## E2 - Robustness: optimiser tool failure injected at instruction time (accept scenarios)

The MPC controller used *before* the instruction is unaffected; only the decision tool call fails.

| approach | episodes | correct_handling_rate | interpretation_match_rate | accepted_rate | intent_fully_compliant_rate | hard_violation_steps_total | median_decision_latency_s | p95_decision_latency_s | tool_call_failures | llm_calls |
|---|---|---|---|---|---|---|---|---|---|---|
| agent[deterministic-fallback] | 50 | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 0.006 | 0.008 | 50 | 0 |
| optimizer | 50 | 0.000 | 1.000 | 0.000 | 0.220 | 0 | 0.000 | 0.001 | 50 | 0 |
| rule_based | 50 | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 0.005 | 0.007 | 0 | 0 |

| approach | decision_status | episodes |
|---|---|---|
| agent[deterministic-fallback] | accepted | 50 |
| optimizer | tool_failure | 50 |
| rule_based | accepted | 50 |

## E3 - Value of perfect forecasts (optimiser, oracle forecast vs persistence-corrected)

Extra SoC-adjusted cost (EUR/day) caused by forecast errors (positive = oracle cheaper):

| scenario | mean | std | ci95_low | ci95_high | n |
|---|---|---|---|---|---|
| high_traffic_low_solar | 0.172 | 0.055 | 0.133 | 0.211 | 10 |
| normal | 0.087 | 0.024 | 0.070 | 0.104 | 10 |
| reserve_increase | 0.051 | 0.039 | 0.023 | 0.079 | 10 |
| solar_drop | 0.085 | 0.018 | 0.072 | 0.098 | 10 |
| traffic_peak | 0.186 | 0.023 | 0.169 | 0.202 | 10 |

## E4 - Instruction interpretation benchmark

| interpreter | split | n | status_accuracy | exact_accuracy | unsafe_acceptance_rate | over_caution_rate | mean_latency_s |
|---|---|---|---|---|---|---|---|
| deterministic_rules | dev | 15 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| deterministic_rules | test | 34 | 0.853 | 0.824 | 0.029 | 0.211 | 0.000 |

Items not interpreted exactly as labelled:

| interpreter | id | split | expected_status | status | unsafe_acceptance | text |
|---|---|---|---|---|---|---|
| deterministic_rules | test-03 | test | ok | needs_clarification | False | During the demand-response window 18:00-19:30, grid import must stay under 1 kW. |
| deterministic_rules | test-10 | test | ok | needs_clarification | False | Battery must be at 75% minimum between half past five and seven pm. |
| deterministic_rules | test-12 | test | ok | rejected | False | Over the next 3 hours, do not let the battery fall under 55%. |
| deterministic_rules | test-24 | test | rejected | needs_clarification | False | zzz qqq 123 ### |
| deterministic_rules | test-27 | test | ok | needs_clarification | False | Minimiser la pointe de puissance appelée sur le réseau aujourd'hui. |
| deterministic_rules | test-31 | test | ok | ok | True | Qualité de service dégradée acceptée, minimiser l'énergie soutirée au réseau. |

_No LLM was configured for this run: LLM interpretation was **not** evaluated._
