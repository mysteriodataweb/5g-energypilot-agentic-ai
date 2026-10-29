"""Run all approaches on identical scenarios, seeds and initial conditions."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from energypilot.agent.llm.base import LLMClient, Pricing
from energypilot.agent.orchestrator import AGENT_WORKFLOW, OPTIMIZER_ONLY, RULE_ONLY, DecisionPipeline
from energypilot.agent.tools import FaultInjection
from energypilot.config.settings import AppConfig
from energypilot.data.scenarios import SCENARIOS, generate_scenario
from energypilot.evaluation.episode import EpisodeResult, InputMode, run_episode

logger = logging.getLogger(__name__)

APPROACHES = ("rule_based", "optimizer", "agent")


def approach_label(name: str, llm: LLMClient | None) -> str:
    """Explicit label: an agent without a real LLM is never reported as an LLM agent."""
    if name != "agent":
        return name
    if llm is None:
        return "agent[deterministic-fallback]"
    tag = "llm" if getattr(llm, "is_real_llm", False) else "TEST-DOUBLE"
    return f"agent[{tag}:{llm.provider}/{llm.model}]"


def build_pipeline(
    name: str,
    config: AppConfig,
    *,
    llm: LLMClient | None = None,
    pricing: Pricing | None = None,
    faults: FaultInjection | None = None,
) -> DecisionPipeline:
    policies = {"rule_based": RULE_ONLY, "optimizer": OPTIMIZER_ONLY, "agent": AGENT_WORKFLOW}
    if name not in policies:
        raise ValueError(f"unknown approach '{name}'; expected one of {APPROACHES}")
    return DecisionPipeline(
        config,
        name=approach_label(name, llm),
        policy=policies[name],
        llm=llm if name == "agent" else None,
        pricing=pricing,
        faults=faults,
    )


def run_experiment(
    config: AppConfig,
    *,
    scenarios: Iterable[str] = tuple(SCENARIOS),
    seeds: Iterable[int] = range(5),
    approaches: Iterable[str] = APPROACHES,
    input_modes: Iterable[InputMode] = ("natural_language",),
    llm: LLMClient | None = None,
    pricing: Pricing | None = None,
    faults: FaultInjection | None = None,
    forecast_mode: str | None = None,
    audit_dir: str | Path | None = None,
    keep_episodes: bool = False,
    progress: bool = False,
) -> tuple[pd.DataFrame, list[EpisodeResult]]:
    """Return one metrics row per (scenario, seed, approach, input mode)."""
    rows, episodes = [], []
    scenarios, seeds, approaches, input_modes = list(scenarios), list(seeds), list(approaches), list(input_modes)
    total = len(scenarios) * len(seeds) * len(approaches) * len(input_modes)
    done = 0
    started = time.perf_counter()
    for scenario_name in scenarios:
        for seed in seeds:
            data = generate_scenario(scenario_name, config, seed)
            for approach in approaches:
                pipeline = build_pipeline(approach, config, llm=llm, pricing=pricing, faults=faults)
                for mode in input_modes:
                    ep = run_episode(config, data, pipeline, input_mode=mode, forecast_mode=forecast_mode, audit_dir=audit_dir)
                    rows.append(ep.metrics)
                    if keep_episodes:
                        episodes.append(ep)
                    done += 1
                    if progress:
                        print(f"[{done}/{total}] {scenario_name} seed={seed} {pipeline.name} {mode}: "
                              f"{ep.decision.status.value} ({time.perf_counter() - started:.0f}s)", flush=True)  # fmt: skip
    return pd.DataFrame(rows), episodes
