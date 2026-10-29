"""Regenerate the README's results tables from results/reference (no manual transcription).

The block between ``<!-- RESULTS:START -->`` and ``<!-- RESULTS:END -->`` is replaced.

Example:
    python scripts/update_readme_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from energypilot.data import SCENARIOS
from energypilot.evaluation.report import markdown_table, mean_with_ci

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "results" / "reference"
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
ACCEPT = [n for n, s in SCENARIOS.items() if s.expected_outcome == "accept"]


def _rate_table(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (approach, mode), g in runs.groupby(["approach", "input_mode"]):
        acc = g[g.expected_outcome == "accept"]
        rows.append(
            {
                "approach": approach,
                "input": mode,
                "correct handling": f"{g.correct_handling.mean():.0%}",
                "interpretation = truth": f"{g.interpretation_matches_truth.astype(float).mean():.0%}",
                "intent fully met (accept scen.)": f"{acc.intent_fully_compliant.astype(float).mean():.0%}",
                "hard violations (steps, total)": int(g.hard_violation_steps.sum()),
                "episodes": len(g),
            }
        )
    return pd.DataFrame(rows)


def build() -> str:
    meta = json.loads((REF / "metadata.json").read_text())
    runs = pd.read_csv(REF / "main_runs.csv")
    summary = pd.read_csv(REF / "main_summary.csv")
    nl = summary[(summary.input_mode == "natural_language") & summary.scenario.isin(ACCEPT)]
    parts = [
        f"Reference run: {len(meta['seeds'])} seeds, solver {meta['solver']}, commit `{meta['git_commit']}`, "
        f"generated {meta['generated_at_utc']}. LLM: "
        + (f"{meta['llm']['provider']}/{meta['llm']['model']}" if meta.get("llm") else "**none** (agent in deterministic fallback mode)")
        + ". Full report: [results/reference/report.md](results/reference/report.md).",
        "**Instruction handling** (all 7 scenarios × seeds; *interpretation = truth* compares the parsed "
        "instruction with the label; *intent fully met* = no violation of the operator's intended instruction):",
        markdown_table(_rate_table(runs)),
        "**SoC-adjusted daily cost (EUR), canonical instruction text, mean [95 % CI]:**",
        markdown_table(mean_with_ci(nl, "soc_adjusted_cost_eur")),
    ]
    rob_path = REF / "robustness_runs.csv"
    if rob_path.exists():
        rob = pd.read_csv(rob_path)
        counts = rob.groupby(["approach", "decision_status"]).size().reset_index(name="episodes")
        parts += ["**Robustness** (optimiser tool fails at instruction time, accept scenarios):", markdown_table(counts)]
    gap_path = REF / "oracle_gap.csv"
    if gap_path.exists():
        gap = pd.read_csv(gap_path)
        parts += [
            "**Cost of forecast errors** (optimiser; SoC-adjusted cost with persistence-corrected forecasts minus "
            "with perfect forecasts, EUR/day):",
            markdown_table(gap, ["scenario", "mean", "ci95_low", "ci95_high", "n"]),
        ]
    bench_path = REF / "benchmark_summary.csv"
    if bench_path.exists():
        bench = pd.read_csv(bench_path)
        parts += ["**Instruction-interpretation benchmark:**", markdown_table(bench)]
    return "\n\n".join(parts)


def main() -> None:
    readme = ROOT / "README.md"
    text = readme.read_text()
    if START not in text or END not in text:
        raise SystemExit("README markers not found")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(f"{head}{START}\n\n{build()}\n\n{END}{tail}")
    print("README results updated")


if __name__ == "__main__":
    main()
