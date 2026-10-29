"""Run one closed-loop episode and print the decision report and key metrics.

Examples:
    python scripts/run_simulation.py --scenario reserve_increase --approach agent
    python scripts/run_simulation.py --scenario normal --approach optimizer --instruction "Keep at least 60% battery charge between 18:00 and 20:00."
    python scripts/run_simulation.py --scenario solar_drop --approach rule_based --input-mode structured

The agent uses the LLM configured in .env (LLM_PROVIDER); without one it runs
its deterministic fallback workflow and is labelled accordingly.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

from energypilot.agent.llm import try_create_llm_client
from energypilot.config import load_config
from energypilot.data import SCENARIOS, generate_scenario
from energypilot.evaluation import build_pipeline, run_episode
from energypilot.evaluation.runner import APPROACHES

KEY_METRICS = (
    "decision_status", "correct_handling", "selected_controller", "total_cost_eur", "soc_adjusted_cost_eur",
    "grid_energy_kwh", "peak_grid_kw", "pv_used_kwh", "pv_curtailed_kwh", "soc_final_pct", "soc_min_pct",
    "hard_violation_steps", "intent_reserve_violation_steps", "intent_grid_cap_violation_steps",
    "intent_qos_violation_steps", "active_qos_exogenous_steps", "shield_intervention_steps",
    "decision_latency_s", "mean_step_decision_s", "llm_calls", "llm_cost_eur",
)  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", choices=list(SCENARIOS), default="normal")
    parser.add_argument("--approach", choices=list(APPROACHES), default="agent")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-mode", default="natural_language", help="natural_language | structured | paraphrase_0 | paraphrase_1")
    parser.add_argument("--instruction", default=None, help="override the scenario's instruction text")
    parser.add_argument("--forecast", choices=["persistence_corrected", "perfect"], default=None)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    config = load_config()
    llm, settings, llm_error = try_create_llm_client(config) if args.approach == "agent" else (None, None, None)
    if llm_error:
        print(f"[warning] LLM not available ({llm_error}); the agent runs in deterministic fallback mode.")
    scenario = generate_scenario(args.scenario, config, args.seed)
    if args.instruction:
        # Custom text: no label, so handling-correctness and intent metrics are not applicable.
        scenario.spec = dataclasses.replace(
            scenario.spec, instruction_text=args.instruction, paraphrases=(), ground_truth=None, expected_outcome="unknown"
        )
    pipeline = build_pipeline(args.approach, config, llm=llm, pricing=settings.pricing if settings else None)
    episode = run_episode(config, scenario, pipeline, input_mode=args.input_mode, forecast_mode=args.forecast,
                          audit_dir=Path(args.runs_dir))  # fmt: skip

    print("=" * 78)
    print(f"SIMULATION (synthetic data) - scenario '{args.scenario}', seed {args.seed}, approach {pipeline.name}")
    print("=" * 78)
    report = episode.decision.model_dump(mode="json")
    report.pop("tool_calls")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("-" * 78)
    for key in KEY_METRICS:
        print(f"{key:34s} {episode.metrics.get(key)}")
    print(f"\naudit log: {episode.audit.path} (hash chain valid: {episode.audit.verify_chain()})")


if __name__ == "__main__":
    main()
