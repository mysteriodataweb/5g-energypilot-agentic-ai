"""Markdown report built exclusively from executed experiment outputs."""

from __future__ import annotations

import math
from typing import Iterable

import pandas as pd


def _fmt(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(df: pd.DataFrame, columns: Iterable[str] | None = None, digits: int = 3) -> str:
    columns = list(columns or df.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for _, row in df[columns].iterrows():
        lines.append("| " + " | ".join(_fmt(row[c], digits) for c in columns) + " |")
    return "\n".join(lines)


def mean_with_ci(summary: pd.DataFrame, metric: str, index: str = "scenario", columns: str = "approach",
                 digits: int = 2) -> pd.DataFrame:  # fmt: skip
    """Pivot 'mean [ci_low, ci_high]' strings for one metric."""
    sub = summary[summary.metric == metric].copy()

    def cell(r):
        if math.isnan(r["mean"]):
            return "-"
        if math.isnan(r["ci95_low"]):
            return f"{r['mean']:.{digits}f}"
        return f"{r['mean']:.{digits}f} [{r['ci95_low']:.{digits}f}, {r['ci95_high']:.{digits}f}]"

    sub["cell"] = sub.apply(cell, axis=1)
    return sub.pivot(index=index, columns=columns, values="cell").reset_index()


def handling_table(runs: pd.DataFrame, by: tuple[str, ...] = ("approach", "input_mode")) -> pd.DataFrame:
    g = runs.groupby(list(by))
    return pd.DataFrame(
        {
            "episodes": g.size(),
            "correct_handling_rate": g.correct_handling.mean(),
            "interpretation_match_rate": g.interpretation_matches_truth.apply(lambda s: s.astype(float).mean()),
            "accepted_rate": g.decision_status.apply(lambda s: (s == "accepted").mean()),
            "intent_fully_compliant_rate": g.intent_fully_compliant.apply(lambda s: s.dropna().astype(float).mean()),
            "hard_violation_steps_total": g.hard_violation_steps.sum(),
            "mean_decision_latency_s": g.decision_latency_s.mean(),
            "tool_call_failures": g.tool_call_failures.sum(),
            "llm_calls": g.llm_calls.sum(),
        }
    ).reset_index()
