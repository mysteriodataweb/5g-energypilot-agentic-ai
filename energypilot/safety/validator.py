"""Independent safety validation of proposed strategies.

The validator is deliberately independent of the LLM layer (it imports nothing
from ``energypilot.agent``) and of the optimiser's internal model: it
re-derives SoC, load and energy balance from the plan's set-points with the
simulator's physics functions, and checks them against the *validated*
instruction, not against constraints claimed by the plan.

Hard checks (any failure => rejection):
    parameters_valid, instruction_match, forecast_integrity, initial_soc,
    battery_power_limits, capacity_range, soc_consistency, soc_limits,
    energy_balance, reserve, grid_import_limits, qos_capacity
Soft / informational checks (never reject):
    terminal_soc, qos_capacity_limited
"""

from __future__ import annotations

import time
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from energypilot.config.settings import AppConfig
from energypilot.core.types import Forecast, Strategy
from energypilot.instructions.constraints import compile_constraints
from energypilot.instructions.schema import OperationalInstruction
from energypilot.simulator import physics

_MAX_LISTED = 8


class CheckResult(BaseModel):
    name: str
    passed: bool
    severity: Literal["hard", "soft", "info"]
    message: str
    violations: list[dict] = Field(default_factory=list)
    max_violation: float = 0.0


class ValidationReport(BaseModel):
    strategy_id: str
    strategy_source: str
    instruction_fingerprint: str
    accepted: bool
    checks: list[CheckResult]
    duration_s: float = 0.0

    def failed(self, severity: str = "hard") -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == severity]

    def reasons(self) -> list[str]:
        return [f"{c.name}: {c.message}" for c in self.failed()]

    def summary(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "strategy_source": self.strategy_source,
            "accepted": self.accepted,
            "failed_checks": [c.name for c in self.failed()],
            "reasons": self.reasons(),
            "warnings": [f"{c.name}: {c.message}" for c in self.checks if not c.passed and c.severity != "hard"],
        }


def _steps_where(mask: np.ndarray, values: np.ndarray, start: int, label: str) -> list[dict]:
    idx = np.flatnonzero(mask)[:_MAX_LISTED]
    return [{"step": int(start + i), label: round(float(values[i]), 4)} for i in idx]


class SafetyValidator:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def validate(
        self,
        strategy: Strategy,
        instruction: OperationalInstruction,
        *,
        current_energy_kwh: float | None = None,
        reference_forecast: Forecast | None = None,
        terminal_target_kwh: float | None = None,
    ) -> ValidationReport:
        started = time.perf_counter()
        checks = self._run(strategy, instruction, current_energy_kwh, reference_forecast, terminal_target_kwh)
        accepted = all(c.passed for c in checks if c.severity == "hard")
        return ValidationReport(
            strategy_id=strategy.strategy_id,
            strategy_source=strategy.source,
            instruction_fingerprint=instruction.fingerprint(),
            accepted=accepted,
            checks=checks,
            duration_s=time.perf_counter() - started,
        )

    # ------------------------------------------------------------------
    def _run(self, s, instruction, current_energy, reference_forecast, terminal_target) -> list[CheckResult]:
        cfg = self.config
        bat, bs, sim, tol = cfg.battery, cfg.base_station, cfg.simulation, cfg.safety
        checks: list[CheckResult] = []

        # 1. Structural validity: nothing below can be trusted otherwise.
        n = len(s.p_charge_kw)
        arrays = {
            "p_charge_kw": s.p_charge_kw,
            "p_discharge_kw": s.p_discharge_kw,
            "capacity_fraction": s.capacity_fraction,
            "grid_import_kw": s.grid_import_kw,
            "pv_curtailed_kw": s.pv_curtailed_kw,
        }
        problems = []
        if not 0 <= s.start_step < sim.n_steps or s.start_step + n != sim.n_steps:
            problems.append("plan does not cover the remaining horizon")
        if len(s.forecast) != n or s.forecast.start_step != s.start_step:
            problems.append("forecast does not match the plan steps")
        if len(s.soc_kwh) != n + 1:
            problems.append("SoC trajectory length mismatch")
        for name, arr in arrays.items():
            arr = np.asarray(arr, dtype=float)
            if arr.shape != (n,):
                problems.append(f"{name} has the wrong length")
            elif not np.all(np.isfinite(arr)) or np.any(arr < -tol.power_tolerance_kw):
                problems.append(f"{name} contains non-finite or negative values")
        if not np.all(np.isfinite(s.soc_kwh)):
            problems.append("SoC trajectory contains non-finite values")
        checks.append(
            CheckResult(
                name="parameters_valid",
                passed=not problems,
                severity="hard",
                message="all plan parameters are well-formed" if not problems else "; ".join(problems),
            )
        )
        if problems:
            return checks

        # 2. The plan must have been computed for exactly the validated instruction.
        expected_fp = instruction.fingerprint()
        checks.append(
            CheckResult(
                name="instruction_match",
                passed=s.instruction_fingerprint == expected_fp,
                severity="hard",
                message=(
                    "plan corresponds to the validated instruction"
                    if s.instruction_fingerprint == expected_fp
                    else f"plan computed for instruction {s.instruction_fingerprint}, validated instruction is {expected_fp}"
                ),
            )
        )
        if reference_forecast is not None:
            same = (
                reference_forecast.start_step == s.forecast.start_step
                and len(reference_forecast) == n
                and np.allclose(reference_forecast.pv_kw, s.forecast.pv_kw)
                and np.allclose(reference_forecast.traffic, s.forecast.traffic)
                and np.allclose(reference_forecast.price_eur_per_kwh, s.forecast.price_eur_per_kwh)
            )
            checks.append(
                CheckResult(
                    name="forecast_integrity",
                    passed=bool(same),
                    severity="hard",
                    message="plan uses the reference forecast" if same else "plan uses a forecast that differs from the reference",
                )
            )
        if current_energy is not None:
            diff = abs(float(s.soc_kwh[0]) - current_energy)
            checks.append(
                CheckResult(
                    name="initial_soc",
                    passed=diff <= tol.energy_tolerance_kwh,
                    severity="hard",
                    message=f"plan starts from the measured SoC (|diff| = {diff:.4f} kWh)",
                    max_violation=diff,
                )
            )

        ch = np.asarray(s.p_charge_kw, dtype=float)
        dis = np.asarray(s.p_discharge_kw, dtype=float)
        cap = np.asarray(s.capacity_fraction, dtype=float)
        grid = np.asarray(s.grid_import_kw, dtype=float)
        curt = np.asarray(s.pv_curtailed_kw, dtype=float)
        pv = np.asarray(s.forecast.pv_kw, dtype=float)
        lam = np.asarray(s.forecast.traffic, dtype=float)
        p_tol, e_tol = tol.power_tolerance_kw, tol.energy_tolerance_kwh

        # 3. Battery power limits and mode exclusivity.
        over_ch = ch - bat.max_charge_kw
        over_dis = dis - bat.max_discharge_kw
        both = (ch > p_tol) & (dis > p_tol)
        bad = (over_ch > p_tol) | (over_dis > p_tol) | both
        checks.append(
            CheckResult(
                name="battery_power_limits",
                passed=not np.any(bad),
                severity="hard",
                message=(
                    f"charge <= {bat.max_charge_kw:g} kW, discharge <= {bat.max_discharge_kw:g} kW, never simultaneous"
                    if not np.any(bad)
                    else f"{int(np.sum(bad))} step(s) exceed power limits or charge and discharge simultaneously"
                ),
                violations=_steps_where(bad, np.maximum(over_ch, over_dis), s.start_step, "excess_kw"),
                max_violation=float(max(np.max(over_ch), np.max(over_dis), 0.0)),
            )
        )

        # 4. Capacity within the hardware range.
        out = (cap < bs.min_capacity_fraction - tol.capacity_tolerance) | (cap > 1 + tol.capacity_tolerance)
        checks.append(
            CheckResult(
                name="capacity_range",
                passed=not np.any(out),
                severity="hard",
                message="capacity fractions within the hardware range"
                if not np.any(out)
                else f"{int(np.sum(out))} step(s) outside [{bs.min_capacity_fraction:g}, 1]",
                violations=_steps_where(out, cap, s.start_step, "capacity"),
            )
        )

        # 5. Independent SoC recomputation.
        soc = physics.soc_trajectory_kwh(float(s.soc_kwh[0]), ch, dis, sim.dt_h, bat)
        drift = np.abs(soc - s.soc_kwh)
        checks.append(
            CheckResult(
                name="soc_consistency",
                passed=float(np.max(drift)) <= e_tol,
                severity="hard",
                message=f"claimed SoC matches the battery model (max drift {np.max(drift):.2e} kWh)",
                max_violation=float(np.max(drift)),
            )
        )
        low = bat.e_min_kwh - soc[1:]
        high = soc[1:] - bat.e_max_kwh
        bad = (low > e_tol) | (high > e_tol)
        checks.append(
            CheckResult(
                name="soc_limits",
                passed=not np.any(bad),
                severity="hard",
                message=(
                    f"SoC stays within [{bat.soc_min_pct:g}%, {bat.soc_max_pct:g}%]"
                    if not np.any(bad)
                    else f"SoC leaves the permitted range at {int(np.sum(bad))} boundary(ies)"
                ),
                violations=_steps_where(bad, np.maximum(low, high), s.start_step + 1, "excess_kwh"),
                max_violation=float(max(np.max(low), np.max(high), 0.0)),
            )
        )

        # 6. Energy balance with an independently recomputed load.
        load = physics.bs_power_kw(cap, lam, bs)
        residual = pv - curt + grid + dis - load - ch
        bad = (
            (np.abs(residual) > max(p_tol, 1e-3))
            | (curt > pv + p_tol)
            | (grid > cfg.grid.max_import_kw + p_tol)
        )
        checks.append(
            CheckResult(
                name="energy_balance",
                passed=not np.any(bad),
                severity="hard",
                message="pv_used + grid + discharge = load + charge at every step"
                if not np.any(bad)
                else f"energy balance or flow bounds violated at {int(np.sum(bad))} step(s)",
                violations=_steps_where(bad, residual, s.start_step, "residual_kw"),
                max_violation=float(np.max(np.abs(residual))),
            )
        )

        # 7. Operational constraints from the *validated* instruction.
        cons = compile_constraints(instruction, cfg)
        reserve = cons.reserve_kwh[s.start_step :]
        deficit = reserve - soc
        bad = (reserve > 0) & (deficit > e_tol)
        checks.append(
            CheckResult(
                name="reserve",
                passed=not np.any(bad),
                severity="hard",
                message=(
                    "required battery reserve maintained throughout every reserve window"
                    if not np.any(bad)
                    else f"reserve missed at {int(np.sum(bad))} boundary(ies), worst deficit "
                    f"{float(np.max(deficit[bad])):.2f} kWh ({bat.kwh_to_pct(float(np.max(deficit[bad]))):.1f}% of capacity)"
                ),
                violations=_steps_where(bad, deficit, s.start_step, "deficit_kwh"),
                max_violation=float(max(np.max(np.where(reserve > 0, deficit, 0.0)), 0.0)),
            )
        )
        caps = cons.grid_cap_kw[s.start_step :]
        excess = grid - caps
        bad = np.isfinite(caps) & (excess > p_tol)
        checks.append(
            CheckResult(
                name="grid_import_limits",
                passed=not np.any(bad),
                severity="hard",
                message="grid import within every requested limit"
                if not np.any(bad)
                else f"grid limit exceeded at {int(np.sum(bad))} step(s), worst excess {float(np.max(excess[bad])):.2f} kW",
                violations=_steps_where(bad, excess, s.start_step, "excess_kw"),
                max_violation=float(max(np.max(np.where(np.isfinite(caps), excess, 0.0)), 0.0)),
            )
        )
        required = np.asarray(cons.required_capacity(lam, cfg), dtype=float)
        short = required - cap
        bad = short > tol.capacity_tolerance
        checks.append(
            CheckResult(
                name="qos_capacity",
                passed=not np.any(bad),
                severity="hard",
                message=(
                    f"active capacity keeps forecast utilisation <= {cons.max_utilization:g}"
                    if not np.any(bad)
                    else f"capacity below the QoS requirement at {int(np.sum(bad))} step(s)"
                ),
                violations=_steps_where(bad, short, s.start_step, "capacity_shortfall"),
                max_violation=float(max(np.max(short), 0.0)),
            )
        )
        limited = physics.capacity_limited(lam, cons.max_utilization)
        checks.append(
            CheckResult(
                name="qos_capacity_limited",
                passed=not np.any(limited),
                severity="info",
                message="QoS target attainable at every step"
                if not np.any(limited)
                else f"forecast traffic exceeds the QoS target even at full capacity at {int(np.sum(limited))} step(s) "
                "(exogenous; not attributable to the strategy)",
                violations=_steps_where(limited, lam, s.start_step, "traffic"),
            )
        )
        if terminal_target is not None:
            shortfall = terminal_target - soc[-1]
            checks.append(
                CheckResult(
                    name="terminal_soc",
                    passed=shortfall <= e_tol,
                    severity="soft",
                    message=f"end-of-horizon SoC {bat.kwh_to_pct(soc[-1]):.1f}% vs target {bat.kwh_to_pct(terminal_target):.1f}%",
                    max_violation=float(max(shortfall, 0.0)),
                )
            )
        return checks
