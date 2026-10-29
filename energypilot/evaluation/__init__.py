"""Experimental evaluation: episodes, metrics, statistics, instruction benchmark."""

from energypilot.evaluation.episode import EpisodeResult, run_episode
from energypilot.evaluation.runner import APPROACHES, approach_label, build_pipeline, run_experiment
from energypilot.evaluation.stats import mean_ci, paired_differences, summarize

__all__ = [
    "APPROACHES",
    "EpisodeResult",
    "approach_label",
    "build_pipeline",
    "mean_ci",
    "paired_differences",
    "run_episode",
    "run_experiment",
    "summarize",
]
