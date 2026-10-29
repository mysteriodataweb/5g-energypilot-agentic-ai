"""Aggregation across seeds: mean, standard deviation and t-based 95% CIs.

Confidence intervals are only reported when n >= 3 and the sample variance is
non-zero; otherwise they are NaN. Paired differences use the same seed (same
synthetic day) for both approaches, which removes between-day variance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_METRICS = (
    "total_cost_eur",
    "soc_adjusted_cost_eur",
    "grid_energy_kwh",
    "peak_grid_kw",
    "pv_used_kwh",
    "pv_self_consumption_ratio",
    "soc_final_pct",
    "hard_violation_steps",
    "intent_reserve_violation_steps",
    "intent_grid_cap_violation_steps",
    "intent_qos_violation_steps",
    "active_qos_exogenous_steps",
    "correct_handling",
    "decision_latency_s",
    "mean_step_decision_s",
    "tool_call_failures",
    "llm_calls",
)


def mean_ci(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float, float, int]:
    """(mean, std, ci_low, ci_high, n) ignoring NaNs."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return (np.nan, np.nan, np.nan, np.nan, 0)
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if n > 1 else np.nan
    if n < 3 or not std > 0:
        return (mean, std, np.nan, np.nan, n)
    half = float(stats.t.ppf(0.5 + confidence / 2, n - 1) * std / np.sqrt(n))
    return (mean, std, mean - half, mean + half, n)


def summarize(df: pd.DataFrame, by=("scenario", "approach"), metrics=DEFAULT_METRICS) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(list(by), sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for metric in metrics:
            if metric not in group:
                continue
            mean, std, lo, hi, n = mean_ci(group[metric].astype(float).to_numpy())
            rows.append({**dict(zip(by, keys)), "metric": metric, "mean": mean, "std": std, "ci95_low": lo, "ci95_high": hi, "n": n})
    return pd.DataFrame(rows)


def paired_differences(df: pd.DataFrame, baseline: str, metric: str, by: str = "scenario") -> pd.DataFrame:
    """Per-scenario paired difference (approach - baseline) over seeds, with 95% CI."""
    rows = []
    keys = [by, "seed", "input_mode"] if "input_mode" in df else [by, "seed"]
    base = df[df.approach == baseline].set_index(keys)[metric]
    for approach in sorted(set(df.approach) - {baseline}):
        other = df[df.approach == approach].set_index(keys)[metric]
        joined = pd.concat([other.rename("x"), base.rename("b")], axis=1, join="inner").dropna()
        for scenario, group in joined.groupby(level=0):
            mean, std, lo, hi, n = mean_ci((group.x - group.b).astype(float).to_numpy())
            rows.append({by: scenario, "approach": approach, "baseline": baseline, "metric": metric,
                         "mean_diff": mean, "std": std, "ci95_low": lo, "ci95_high": hi, "n": n})  # fmt: skip
    return pd.DataFrame(rows)
