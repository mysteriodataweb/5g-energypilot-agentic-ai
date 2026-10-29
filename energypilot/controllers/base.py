"""Controller interface and ex-ante roll-out of a policy into a Strategy."""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from energypilot.config.settings import AppConfig
from energypilot.core.types import ControlAction, Forecast, Observation, Strategy
from energypilot.simulator.environment import BaseStationEnv


class Controller(Protocol):
    """A closed-loop policy: one decision per step from the current observation.

    ``forecast`` covers the remaining horizon as known at decision time.
    """

    name: str

    def decide(self, obs: Observation, forecast: Forecast) -> ControlAction: ...


Policy = Callable[[Observation], ControlAction]


def rollout_strategy(
    policy: Policy,
    forecast: Forecast,
    energy_kwh: float,
    config: AppConfig,
    *,
    source: str,
    instruction_fingerprint: str,
    metadata: dict | None = None,
) -> Strategy:
    """Simulate ``policy`` on the forecast (simulator physics) and return the plan."""
    env = BaseStationEnv.from_forecast(config, forecast, energy_kwh)
    while not env.done:
        env.step(policy(env.observe()))
    hist = env.history()  # only the roll-out steps (start_step .. n-1)
    soc = np.concatenate([[hist["soc_start_kwh"].iloc[0]], hist["soc_end_kwh"].to_numpy()])
    adjustments = [a for a in hist["adjustments"] if a]
    return Strategy(
        source=source,
        start_step=forecast.start_step,
        p_charge_kw=hist["p_charge_kw"].to_numpy(),
        p_discharge_kw=hist["p_discharge_kw"].to_numpy(),
        capacity_fraction=hist["capacity_fraction"].to_numpy(),
        grid_import_kw=hist["grid_import_kw"].to_numpy(),
        pv_curtailed_kw=hist["pv_curtailed_kw"].to_numpy(),
        soc_kwh=soc,
        forecast=forecast,
        instruction_fingerprint=instruction_fingerprint,
        metadata={"rollout_adjustment_steps": len(adjustments), **(metadata or {})},
    )
