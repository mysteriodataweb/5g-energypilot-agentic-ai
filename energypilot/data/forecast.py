"""Forecasts available to planners at decision time.

Planners never see the realised future. At step ``t`` they receive:

* index 0: the measured current values (PV, traffic) - nowcast assumption;
* indices k >= 1: the scenario's forecast corrected by the currently observed
  forecast error, decaying geometrically (persistence correction):

      f_adj[t+k] = f[t+k] * (1 + (r_t - 1) * decay**k),   r_t = actual[t] / f[t]

``mode = "perfect"`` returns the realised profiles (an oracle used only as a
reference bound in experiments, clearly labelled as such).
"""

from __future__ import annotations

import numpy as np

from energypilot.config.settings import AppConfig
from energypilot.core.types import Forecast
from energypilot.data.scenarios import ScenarioData


class ForecastProvider:
    def __init__(self, scenario: ScenarioData, config: AppConfig, mode: str | None = None) -> None:
        self.scenario = scenario
        self.config = config
        self.mode = mode or config.forecast.mode
        if self.mode not in ("persistence_corrected", "perfect"):
            raise ValueError(f"unknown forecast mode '{self.mode}'")

    def forecast(self, step: int) -> Forecast:
        """Forecast for steps ``step .. n_steps - 1`` as known at ``step``."""
        sc = self.scenario
        n = len(sc.hours)
        if not 0 <= step < n:
            raise IndexError(f"step {step} outside horizon")
        price = sc.price_eur_per_kwh[step:].copy()
        if self.mode == "perfect":
            return Forecast(step, sc.pv_actual_kw[step:].copy(), sc.traffic_actual[step:].copy(), price)

        k = np.arange(n - step)
        pv_min = 0.05 * self.config.pv.peak_power_kw
        pv_fc = sc.pv_forecast_kw[step:].copy()
        if pv_fc[0] > pv_min:
            r_pv = float(np.clip(sc.pv_actual_kw[step] / pv_fc[0], 0.0, 3.0))
            pv_fc = pv_fc * (1.0 + (r_pv - 1.0) * self.config.forecast.pv_decay_per_step**k)
        pv_fc[0] = sc.pv_actual_kw[step]

        tr_fc = sc.traffic_forecast[step:].copy()
        if tr_fc[0] > 1e-6:
            r_tr = float(np.clip(sc.traffic_actual[step] / tr_fc[0], 0.0, 3.0))
            tr_fc = tr_fc * (1.0 + (r_tr - 1.0) * self.config.forecast.traffic_decay_per_step**k)
        tr_fc[0] = sc.traffic_actual[step]
        tr_fc = np.clip(tr_fc, 0.0, self.config.traffic.max_traffic)
        return Forecast(step, np.clip(pv_fc, 0.0, None), tr_fc, price)
