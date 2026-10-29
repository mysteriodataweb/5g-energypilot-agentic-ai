# Technical note: why the LLM orchestrates decision models instead of replacing them

*5G EnergyPilot AI — design rationale.*

## The question

A base station's energy strategy must respect physical limits (battery power, state of charge, grid
connection), operational commitments (backup reserves, demand-response caps) and service quality.
Operators express new requirements in natural language and at irregular times. Large language models
are good at reading such text, but should the model *compute* the schedule?

## Why not let the LLM produce set-points

1. **Numerical correctness is not a language task.** A day-ahead schedule is a coupled optimisation
   problem: 96 steps linked by battery dynamics, efficiency losses, and window constraints. A MILP
   solver returns either a provably optimal schedule or a proof that none exists. An LLM can only
   approximate it, and nothing in its output indicates when the approximation is wrong.
2. **Hard constraints need proofs, not plausibility.** "Keep at least 80 % between 18:00 and 23:00"
   is satisfied or violated; a fluent explanation of a schedule that violates it is worse than no
   schedule. The validator recomputes the consequences of every plan from the physics.
3. **Infeasibility must be detected, not hidden.** When requirements conflict (scenario 6: a 90 %
   reserve together with zero grid import over the evening peak), the right outcome is a refusal with
   an explanation of which requirements conflict. A generative model tends to produce *some* answer;
   a solver proves infeasibility and the diagnosis names the conflicting groups.
4. **Auditability and reproducibility.** Safety-relevant decisions must be traceable. Solver status,
   validator checks and a hash-chained audit log can be replayed; free-text reasoning cannot.
5. **Latency, cost and availability.** The control loop runs every 15 minutes, indefinitely. LLM calls
   add seconds of latency, a per-token cost and an external dependency that can time out. The
   prototype therefore never calls the LLM inside the control loop, and a deterministic fallback path
   always exists.

## What the LLM is allowed to do

The agent sits *above* the numerical tools:

| The LLM may | The LLM may not |
|---|---|
| interpret a natural-language instruction into the strict schema | invent missing values (every value must be quoted verbatim from the text) |
| ask for clarification, explain a refusal | change the battery state or emit set-points |
| read the state and the active constraints | edit a strategy, relax a constraint or change the validated instruction |
| call the optimiser, the rule controller, the diagnosis | skip validation: the final gate re-validates whatever it chooses |
| choose among strategies that passed validation | act on real equipment (everything runs in the simulator) |

Every LLM output passes through deterministic code: schema validation with strict types, a grounding
check (quotes must appear in the instruction and contain the numbers), a safety screen for unsafe
requests, semantic range checks, and finally the independent validator. If the LLM fails (timeout,
refusal, malformed tool call, a decision that cannot be applied), a circuit breaker switches the rest
of that decision to the deterministic workflow, and the report says so.

## Where an LLM can add value

The deterministic pipeline is reliable on the phrasings it was written for and has zero inference
cost. Its weakness is coverage: on the held-out benchmark split it asked for clarification on several
valid instructions, and it silently applied a default objective when it did not recognise the
operator's wording (see `results/reference/report.md`). These are exactly the situations where
language understanding helps:

* paraphrases, other languages, informal time expressions ("half past five");
* clarification questions that propose a concrete alternative (e.g. the earliest feasible time to
  reach a reserve, taken from the diagnosis);
* explaining a refusal to an operator in plain language.

The hypothesis to test is therefore not "the LLM controls better than the optimiser" — by
construction the agent executes the optimiser's schedules — but **when LLM interpretation and
orchestration improve correct instruction handling (and hence intent compliance) enough to justify
their latency, cost and new failure modes.** The evaluation framework measures exactly these
quantities and allows a neutral or negative answer.

## Failure modes that remain

* A grounded but *wrong* interpretation (e.g. a correct quote mapped to the wrong window) passes the
  grounding check; it is then executed safely (constraints are respected) but does not meet the
  operator's intent. The intent-compliance metric captures this.
* An LLM can be over-cautious (unnecessary clarification requests or refusals). This is safe but
  costly in operator time and in compliance with the intended instruction; it is measured too.
* Prompt injection inside an instruction is treated as data; unsafe requests are screened
  deterministically, independent of the model.
