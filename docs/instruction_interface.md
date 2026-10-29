# Operational instruction interface

## Schema (`energypilot/instructions/schema.py`)

```json
{
  "objective": "minimize_cost | minimize_grid_energy | minimize_peak_import",
  "reserves": [{"min_soc_pct": 70.0, "start_minute": 720, "end_minute": 840}],
  "grid_limits": [{"max_import_kw": 2.0, "start_minute": 1080, "end_minute": 1200}],
  "qos_level": "strict | standard | relaxed",
  "max_utilization": null,
  "allow_capacity_reduction": true
}
```

* Times are minutes since 00:00 of the simulated day; windows are `[start, end)`.
* Numeric and boolean fields are **strict**: `"70"` or `"yes"` are rejected, unknown keys are rejected.
* Only `objective` (cost), `qos_level` (standard) and `allow_capacity_reduction` (true) have documented
  defaults; when used, they are listed in `defaults_applied`. Reserves and grid limits are never defaulted.
* A new accepted instruction replaces the active one entirely.

## Three interpreters, one validation path

| Interface | Used by | Notes |
|---|---|---|
| Structured JSON | any approach (reliable fallback) | `parse_structured`; JSON text is routed here automatically |
| Deterministic parser | `rule_based`, `optimizer`, agent fallback | regex grammar, English + French subset |
| LLM interpretation | `agent` with an LLM configured | `record_instruction` tool + grounding check |

All three end in the same semantic validation (`instructions/validation.py`):

| Outcome | Triggered by |
|---|---|
| `rejected` | schema violation, reserve above the BMS maximum, window beyond the horizon or already elapsed, unsafe or unsupported request (switching equipment off, bypassing safety, grid export, direct battery set-points, upper SoC targets, minimum grid import), contradictory request (reserve + full discharge), malformed text |
| `needs_clarification` | missing level or window, vague wording ("reasonably high", "for a while", "tonight"), relative targets without an absolute value, several objectives, conflicting QoS levels, conflicting reserves for one window, ungrounded LLM values |
| `ok` (+ warnings) | e.g. window that started before the issue time (applies from the issue time), reserve not above the BMS minimum |

Feasibility is *not* an interpretation question: a well-formed but infeasible instruction is accepted by
the interpreter and then refused by the optimiser (`infeasible`, with diagnosis) or the validator.

## Deterministic grammar (summary)

* reserve: lower-bound word (`at least`, `minimum`, `above`, `not … below`, `au moins`, `au-dessus de`, …)
  + `N %` + battery context word, with a window;
* grid cap: cap word + `grid/import/réseau/soutirage` + `N kW`, or zero-import phrases (`no grid import`,
  `aucun soutirage`);
* windows: `between/from HH[:MM] and/to HH[:MM]`, `de 18h à 20h`, `next N hours/minutes`,
  `les N prochaines heures`, `for N hours`, `until HH:MM`, `starting at HH for N minutes`,
  `from HH onwards`, `today / rest of the day`, `the same period`;
* QoS: level word attached to a QoS term (`strict QoS`, `QoS can be relaxed`, `qualité de service stricte`),
  `preserve the required quality of service`, `utilisation below N %`, `full capacity / no sleep modes`.

## LLM grounding check (`agent/interpretation.py`)

Every non-default value returned by the LLM must come with a verbatim quote from the instruction
(`evidence`), and numeric values must appear in that quote (digits or small number words in English or
French). Otherwise the value is treated as invented and clarification is requested. The deterministic
safety screen runs on the original text whatever the LLM says.

## Interpretation benchmark and its history

`data/instruction_benchmark.json`: 15 `dev` items (phrasings used during development) and 34 `test`
items written afterwards. Rates reported by `scripts/run_instruction_benchmark.py`: status accuracy,
exact accuracy, **unsafe acceptance** (accepted although it should not be, or accepted with wrong
content) and **over-caution** (valid instruction not accepted).

Transparency log for the deterministic parser on the `test` split:

| Version | Exact accuracy | Unsafe acceptance | Over-caution | Change |
|---|---|---|---|---|
| v1 (first held-out run) | 0.794 (27/34) | 0.059 (2/34) | 0.211 (4/19) | — |
| v1.1 (current) | 0.824 (28/34) | 0.029 (1/34) | 0.211 (4/19) | safety screen also rejects "disable/deactivate/turn off … safety/validator/limits" (item `test-22`) |

The fix closed a safety gap revealed by the held-out set; it means that one item of the `test` split is
no longer strictly held out. No interpretation-coverage change was made after seeing the `test` split:
the remaining misses (e.g. "half past five", "minimiser la pointe", "do not let the battery fall under",
"l'énergie soutirée au réseau" interpreted with the default objective) are kept as measured limitations.
