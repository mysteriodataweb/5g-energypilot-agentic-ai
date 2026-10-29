# Mathematical model and assumptions

All parameters live in `config/default.toml` (validated by `energypilot/config/settings.py`).
Every value is a **modelling assumption** for a synthetic study, not a measurement of a real site.

## 1. Time and units

* Discrete time, step length `Δt` = 15 min (configurable: any divisor of the horizon), horizon 24 h,
  step `k = 0 … N-1` starts at `k·Δt` (step 0 = 00:00), `N = 96` by default.
* **Power** quantities (kW) are averages over a step; **energy** quantities (kWh) are power × `Δt` (h).
  Battery state of charge is an energy at step *boundaries* (`E[0] … E[N]`, kWh), reported in % of
  the capacity `C` (kWh).
* Prices in EUR/kWh, costs in EUR.

## 2. Energy balance (sign convention)

All flows are non-negative; the direction is in the name.

```
pv_used[k] + grid[k] + p_dis[k] = load_served[k] + p_ch[k]          (AC bus, kW)
pv_used[k] = pv_avail[k] - curt[k],   0 <= curt[k] <= pv_avail[k]
load_served[k] = load[k] - unserved[k]
```

* `p_ch`: power drawn from the bus into the battery; `p_dis`: power delivered by the battery to the bus.
* Grid export is **not** modelled: PV surplus that cannot be stored is curtailed (`curt`).
* `0 <= grid[k] <= G_max` (physical connection, 15 kW). If the connection limit is reached the
  simulator first reduces charging, then sheds load (`unserved`, a hard violation).

## 3. Battery

```
E[k+1] = r · E[k] + η_ch · p_ch[k] · Δt - p_dis[k] · Δt / η_dis
E_min = C · SoC_min,  E_max = C · SoC_max,   E_min <= E[k] <= E_max
0 <= p_ch[k] <= P_ch_max,   0 <= p_dis[k] <= P_dis_max,   not (p_ch > 0 and p_dis > 0)
```

| Parameter | Default | Note |
|---|---|---|
| `C` | 40 kWh | usable nameplate energy |
| `SoC_min`, `SoC_max` | 10 %, 95 % | BMS limits (hard) |
| `P_ch_max`, `P_dis_max` | 10 kW | AC-side limits (0.25 C) |
| `η_ch`, `η_dis` | 0.95, 0.95 | round trip ≈ 0.90 |
| `r` | `1 - σ·Δt/24`, `σ = 0` | self-discharge retention (disabled by default) |
| degradation cost | 0.02 EUR/kWh throughput | used by the optimiser only |

The simulator nets simultaneous charge/discharge requests and clips set-points to power and SoC
limits, recording each adjustment.

## 4. Base-station power model (EARTH-inspired, linear)

Traffic `λ[k]` is normalised: 1.0 = the traffic the site can carry with **all** capacity layers active.
The controller chooses the active capacity fraction `c[k] ∈ [c_min, 1]` (energy-saving features such
as carrier / MIMO-layer sleep, abstracted as a continuous fraction).

```
load[k] = P_fix + P_cap · c[k] + P_traffic · min(λ[k], c[k])
```

| Parameter | Default | Meaning |
|---|---|---|
| `P_fix` | 1.2 kW | baseband, cooling, transport, rectifier losses |
| `P_cap` | 1.0 kW | idle power of the radio units at full capacity |
| `P_traffic` | 2.0 kW | load-dependent amplifier power at full served traffic |
| `c_min` | 0.4 | deepest energy-saving state (coverage layer only) |

So a fully loaded site draws 4.2 kW and a quiet night site ≈ 1.8 kW. The structure follows the
load-dependent models popularised by the EARTH project (fixed part + part proportional to the radiated
load); the numbers are illustrative assumptions, not vendor data. If `λ[k] > c[k]` the simulator models
the RAN's own load-based wake-up and raises `c[k]` to `min(1, λ[k])`.

## 5. QoS proxy

The QoS requirement is a maximum radio utilisation `ρ_max` (`λ/c`): strict 0.70, standard 0.85,
relaxed 0.95, or an explicit value from the instruction.

```
c[k] >= c_req[k] = clip(λ[k] / ρ_max, c_min, 1)            (hard; c[k] = 1 if capacity reduction is forbidden)
```

If `λ[k] > ρ_max` the target is unattainable even at full capacity: such steps are reported as
**exogenous** (capacity-limited) QoS shortfalls and are never attributed to a controller. Reported
indicators: utilisation, served-traffic ratio `min(λ,c)/λ`, and an indicative M/M/1-style delay factor
`1/(1-min(u, 0.99))` (not used for optimisation).

## 6. Operational instructions → constraints

A validated instruction (`energypilot/instructions/schema.py`) is compiled per step:

* reserve `R` over window `[a, b)` → `E[j] >= C·R/100` for every boundary `j = ⌊a/Δt⌋ … ⌈b/Δt⌉`
  (conservative: the energy varies linearly within a step);
* grid-import cap `G` over `[a, b)` → `grid[k] <= G` for every step overlapping the window;
* QoS level / utilisation cap / capacity-reduction permission → `c_req[k]`;
* nothing is constrained before the issue time.

## 7. MILP scheduler (`energypilot/optimizer/milp.py`)

Model class: **mixed-integer linear programme** (PuLP + CBC; HiGHS optional). For the remaining
horizon `k = 0 … n-1` starting at the current step, with forecasts `pv[k]`, `λ[k]`, `price[k]`:

Variables: `p_ch, p_dis ∈ [0, P_max]`, `grid ∈ [0, G_max]`, `curt ∈ [0, pv]`, `c ∈ [c_req, 1]`,
`E[k+1] ∈ [E_min, E_max]`, `b[k] ∈ {0,1}`, `s_term >= 0`.

Constraints:

```
pv[k] - curt[k] + grid[k] + p_dis[k] = P_fix + P_cap·c[k] + P_traffic·min(λ[k],1) + p_ch[k]   (balance)
E[k+1] = r·E[k] + η_ch·Δt·p_ch[k] - (Δt/η_dis)·p_dis[k],  E[0] = measured energy             (dynamics)
p_ch[k] <= P_ch_max·b[k],  p_dis[k] <= P_dis_max·(1 - b[k])                                     (exclusivity)
E[k+1] >= reserve[t0+k+1]                                                                       (reserves)
grid[k] <= cap[t0+k]                                                                            (DR caps)
E[n] + s_term >= E_target                                                                       (soft terminal SoC)
```

`min(λ,1)` is a constant because the QoS constraint `c >= min(1, λ/ρ_max) >= min(λ, 1)` guarantees all
offered traffic up to capacity is served; the model therefore stays linear.

Objectives (selected by the instruction):

* `minimize_cost`: `Σ price·grid·Δt + c_deg·Σ(p_ch+p_dis)·Δt + c_curt·Σ curt·Δt + w_term·s_term`
* `minimize_grid_energy`: `Σ grid·Δt + 0.01·(cost terms) + 1.25·s_term`
* `minimize_peak_import`: `peak + 0.01·(cost terms) + 1.25·s_term`, `peak >= grid[k]`

with `w_term = 0.40 EUR/kWh` (above the highest price divided by `η_dis`, so draining the battery to
the horizon end never pays) and `c_curt = 0.001` (tie-breaker). The end-of-day target is the SoC at
00:00 (soft, so a demanding instruction is never declared infeasible because of it).

Size for a full day: 673 variables (96 binaries), ~390 constraints; CBC solves it in < 0.1 s.

**No silent relaxation.** If the hard constraints cannot all be met, the status is `infeasible`
(proved by the solver, or by a pre-solve check when the current SoC is already below an active
reserve). The diagnosis module (`optimizer/diagnosis.py`) then explains the conflict with analytic
necessary conditions (reserve reachability at `P_ch_max`, discharge power and energy needed to respect
a grid cap) and a group-level deletion filter (re-solving without `{reserve}`, `{grid_cap}`, `{qos}`,
then pairs). Relaxed solutions are diagnostic only and never executed.

## 8. Receding-horizon execution (MPC)

The optimiser re-plans every `replan_interval_steps` (4 = 1 h) from the measured SoC and an updated
forecast. Each new plan must pass the safety validator before it replaces the current one. Between
re-plans, deterministic real-time corrections handle forecast errors without relaxing anything:
capacity raised to the measured QoS requirement, unexpected PV surplus stored, charging reduced and
discharge increased (down to the floor) to respect an active grid cap. If no valid plan covers the
current step, the rule-based controller decides that step (logged).

## 9. Forecasts

At step `t` the current PV and traffic are measured (nowcast). For `k >= 1`:
`f_adj[t+k] = f[t+k] · (1 + (r_t - 1)·d^k)` with `r_t = actual[t]/f[t]` and decay `d` (0.85 PV,
0.90 traffic). `forecast.mode = "perfect"` gives the realised profiles (oracle reference only).

## 10. Rule-based controller (`controllers/rule_based.py`)

Interpretable, reactive, forecast-free rules evaluated in order: R1 capacity from the QoS target;
R2 charge at the minimal constant power (× 1.3 margin) that reaches every upcoming reserve in time;
R3 store PV surplus; R4 off-peak grid charging to 50 % (cost objective); R5 discharge above the floor
(cost: at peak price; grid-energy: whenever there is a deficit; peak: above 3 kW); R6 respect grid
caps by reducing charging, then discharging. Limitations: myopic, hand-tuned thresholds, no
feasibility guarantee, grid cap prioritised over reserves when they conflict, no end-of-day SoC
management.

## 11. Safety validation (`safety/validator.py`)

Independent of the LLM and of the optimiser's internal model: it recomputes the SoC trajectory and the
load from the plan's set-points with the simulator's physics functions and checks, against the
*validated* instruction:

| Check | Kind |
|---|---|
| parameters well-formed, plan covers the horizon | hard |
| plan fingerprint = validated instruction fingerprint | hard |
| plan uses the reference forecast; starts from the measured SoC | hard |
| power limits, no simultaneous charge/discharge, capacity range | hard |
| claimed SoC = recomputed SoC; SoC within BMS limits | hard |
| energy balance with recomputed load; flow bounds | hard |
| reserves, grid-import caps, QoS capacity | hard |
| exogenous QoS shortfall | informational |
| end-of-day SoC target | soft |

A runtime shield additionally projects every executed action onto a safe set (power/SoC limits,
reserve floor, QoS capacity for measured traffic, charging that would breach a grid cap).

## 12. Metrics (`evaluation/metrics.py`)

* cost, **SoC-adjusted cost** = cost + (E_start − E_end) × mean price (fair comparison when approaches
  end the day with different SoC; an approximation);
* grid energy, peak import, PV used / curtailed, self-consumption ratio, battery throughput, cycles;
* **hard violations**: reserve, grid-cap and control-induced QoS violations of the constraints *in force*,
  plus unserved energy (the shield and simulator should keep this at zero);
* **intent compliance**: the same violations measured against the operator's *intended* instruction
  (scenario ground truth) from the issue time — the cost of misinterpreting or wrongly refusing;
* decision outcome and correctness, interpretation match, decision latency, per-step decision time,
  LLM calls / tokens / latency / estimated cost (only with user-supplied prices), tool-call failures.
