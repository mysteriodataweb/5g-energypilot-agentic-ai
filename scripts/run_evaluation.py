"""Reproducible comparative evaluation of all approaches.

Experiments (all on synthetic data, inside the simulator):
  main        all scenarios x seeds x approaches x input modes
              (canonical text, ground-truth JSON, two held-out paraphrases)
  robustness  accept-scenarios with an injected optimiser failure
  oracle      MPC optimiser with perfect forecasts (reference bound)
  benchmark   instruction-interpretation benchmark (data/instruction_benchmark.json)

Example:
    python scripts/run_evaluation.py --seeds 10 --out results/latest
    python scripts/run_evaluation.py --seeds 3 --experiments main benchmark --with-llm

``--with-llm`` uses the LLM configured in .env for the agent; without it (or
without configuration) the agent is reported as "agent[deterministic-fallback]".
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from energypilot import __version__
from energypilot.agent.interpretation import interpret_with_llm
from energypilot.agent.llm import try_create_llm_client
from energypilot.agent.tools import FaultInjection
from energypilot.config import load_config
from energypilot.data import SCENARIOS
from energypilot.evaluation import paired_differences, run_experiment, summarize
from energypilot.evaluation.instruction_benchmark import evaluate_interpreter, load_benchmark, summarize_benchmark
from energypilot.evaluation.report import handling_table, markdown_table, mean_with_ci
from energypilot.instructions import parse_instruction

ACCEPT_SCENARIOS = [n for n, s in SCENARIOS.items() if s.expected_outcome == "accept"]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=10, help="number of seeds (0..N-1)")
    parser.add_argument("--experiments", nargs="+", default=["main", "robustness", "oracle", "benchmark"],
                        choices=["main", "robustness", "oracle", "benchmark"])  # fmt: skip
    parser.add_argument("--input-modes", nargs="+", default=["natural_language", "structured", "paraphrase_0", "paraphrase_1"])
    parser.add_argument("--with-llm", action="store_true", help="use the LLM configured in .env for the agent")
    parser.add_argument("--out", default="results/latest")
    parser.add_argument("--audit", action="store_true", help="write one audit JSONL per episode under <out>/audit")
    args = parser.parse_args()

    config = load_config()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))
    llm, settings, llm_error = (None, None, None)
    if args.with_llm:
        llm, settings, llm_error = try_create_llm_client(config)
        if llm is None:
            print(f"[warning] no LLM available ({llm_error or 'LLM_PROVIDER=none'}); agent runs in fallback mode")
    pricing = settings.pricing if settings else None
    audit_dir = out / "audit" if args.audit else None
    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "energypilot_version": __version__,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "seeds": seeds,
        "experiments": args.experiments,
        "input_modes": args.input_modes,
        "llm": {"provider": llm.provider, "model": llm.model, "is_real_llm": llm.is_real_llm} if llm else None,
        "solver": config.optimizer.solver,
        "data": "synthetic (see data provenance); simulation only",
    }
    sections: list[str] = []
    timings: dict[str, float] = {}

    if "main" in args.experiments:
        t = time.perf_counter()
        runs, _ = run_experiment(config, seeds=seeds, input_modes=args.input_modes, llm=llm, pricing=pricing,
                                 audit_dir=audit_dir, progress=True)  # fmt: skip
        timings["main_s"] = time.perf_counter() - t
        runs.to_csv(out / "main_runs.csv", index=False)
        summary = summarize(runs, by=("scenario", "approach", "input_mode"))
        summary.to_csv(out / "main_summary.csv", index=False)
        handling = handling_table(runs)
        handling.to_csv(out / "main_handling.csv", index=False)
        nl = runs[runs.input_mode == "natural_language"]
        nl_summary = summarize(nl, by=("scenario", "approach"))
        paired = pd.concat(
            [paired_differences(nl, "rule_based", m) for m in ("soc_adjusted_cost_eur", "grid_energy_kwh")]
            + [paired_differences(nl, "optimizer", "soc_adjusted_cost_eur")],
            ignore_index=True,
        )
        paired.to_csv(out / "main_paired.csv", index=False)
        accept_nl = nl_summary[nl_summary.scenario.isin(ACCEPT_SCENARIOS)]
        sections += [
            "## E1 - Instruction handling by approach and input mode",
            "Correct handling = accepted when the label is *accept*, clarification when *clarify*, any refusal "
            "(invalid, infeasible, validator rejection) when *reject*. Intent compliance is measured against the "
            "operator's intended instruction on *accept* scenarios.",
            markdown_table(handling),
            "## E1 - SoC-adjusted daily cost (EUR), canonical instruction text, mean [95% CI] over seeds",
            markdown_table(mean_with_ci(accept_nl, "soc_adjusted_cost_eur")),
            "## E1 - Grid energy imported (kWh), canonical text",
            markdown_table(mean_with_ci(accept_nl, "grid_energy_kwh", digits=1)),
            "## E1 - Hard safety violations (steps, constraints in force), canonical text",
            markdown_table(mean_with_ci(nl_summary, "hard_violation_steps", digits=1)),
            "## E1 - Exogenous QoS shortfall steps (traffic above target even at full capacity)",
            markdown_table(mean_with_ci(nl_summary, "active_qos_exogenous_steps", digits=1)),
            "## E1 - Paired differences on the same synthetic days (approach - baseline), canonical text",
            markdown_table(paired, ["scenario", "approach", "baseline", "metric", "mean_diff", "ci95_low", "ci95_high", "n"]),
            "## E1 - Decision latency (s) at instruction time, canonical text",
            markdown_table(mean_with_ci(nl_summary, "decision_latency_s", digits=3)),
        ]

    if "robustness" in args.experiments:
        t = time.perf_counter()
        rob, _ = run_experiment(config, scenarios=ACCEPT_SCENARIOS, seeds=seeds, llm=llm, pricing=pricing,
                                faults=FaultInjection(optimizer_fails=True), progress=True)  # fmt: skip
        timings["robustness_s"] = time.perf_counter() - t
        rob.to_csv(out / "robustness_runs.csv", index=False)
        rob_table = handling_table(rob, by=("approach",))
        rob_table.to_csv(out / "robustness_handling.csv", index=False)
        sections += [
            "## E2 - Robustness: optimiser tool failure injected at instruction time (accept scenarios)",
            "The MPC controller used *before* the instruction is unaffected; only the decision tool call fails.",
            markdown_table(rob_table),
            markdown_table(
                rob.groupby(["approach", "decision_status"]).size().reset_index(name="episodes"),
            ),
        ]

    if "oracle" in args.experiments:
        t = time.perf_counter()
        oracle, _ = run_experiment(config, scenarios=ACCEPT_SCENARIOS, seeds=seeds, approaches=("optimizer",),
                                   forecast_mode="perfect", progress=True)  # fmt: skip
        timings["oracle_s"] = time.perf_counter() - t
        oracle.to_csv(out / "oracle_runs.csv", index=False)
        sections.append("## E3 - Value of perfect forecasts (optimiser, oracle forecast vs persistence-corrected)")
        if "main" in args.experiments:
            base = runs[(runs.approach == "optimizer") & (runs.input_mode == "natural_language") & runs.scenario.isin(ACCEPT_SCENARIOS)]
            merged = base.merge(oracle, on=["scenario", "seed"], suffixes=("", "_oracle"))
            merged["forecast_error_cost_eur"] = merged.soc_adjusted_cost_eur - merged.soc_adjusted_cost_eur_oracle
            gap = summarize(merged.assign(approach="optimizer"), by=("scenario",), metrics=("forecast_error_cost_eur",))
            gap.to_csv(out / "oracle_gap.csv", index=False)
            sections.append("Extra SoC-adjusted cost (EUR/day) caused by forecast errors (positive = oracle cheaper):")
            sections.append(markdown_table(gap, ["scenario", "mean", "std", "ci95_low", "ci95_high", "n"]))

    if "benchmark" in args.experiments:
        items = load_benchmark()
        bench = [evaluate_interpreter("deterministic_rules", parse_instruction, items, config)]
        if llm is not None:
            def llm_interpreter(text, issued, cfg):
                result, _ = interpret_with_llm(llm, text, issued, cfg, max_tokens=settings.max_tokens, timeout_s=settings.timeout_s)
                return result

            bench.append(evaluate_interpreter(f"llm:{llm.provider}/{llm.model}", llm_interpreter, items, config))
        bench_df = pd.concat(bench, ignore_index=True)
        bench_df.to_csv(out / "benchmark_items.csv", index=False)
        bench_summary = summarize_benchmark(bench_df)
        bench_summary.to_csv(out / "benchmark_summary.csv", index=False)
        failures = bench_df[~bench_df.exact_correct]
        sections += [
            "## E4 - Instruction interpretation benchmark",
            markdown_table(bench_summary),
            "Items not interpreted exactly as labelled:",
            markdown_table(failures, ["interpreter", "id", "split", "expected_status", "status", "unsafe_acceptance", "text"]),
        ]
        if llm is None:
            sections.append("_No LLM was configured for this run: LLM interpretation was **not** evaluated._")

    meta["timings_s"] = {k: round(v, 1) for k, v in timings.items()}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    llm_line = (
        f"LLM: {meta['llm']['provider']}/{meta['llm']['model']}" if meta["llm"]
        else "LLM: none configured - the agent ran in deterministic fallback mode; no LLM results are reported"
    )  # fmt: skip
    header = [
        "# Experiment results",
        f"Generated {meta['generated_at_utc']} by `scripts/run_evaluation.py` (commit {meta['git_commit']}). "
        f"Seeds: {len(seeds)}. Solver: {meta['solver']}. {llm_line}.",
        "> All numbers come from executed closed-loop simulations on **synthetic** data. They are not "
        "measurements of a real base station and must not be read as real-world operational results.",
    ]
    (out / "report.md").write_text("\n\n".join(header + sections) + "\n")
    print(f"\nwrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
