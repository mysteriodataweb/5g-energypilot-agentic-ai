"""Shared fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from energypilot.config import AppConfig, load_config
from energypilot.simulator import BaseStationEnv


@pytest.fixture(scope="session")
def config() -> AppConfig:
    return load_config()


@pytest.fixture()
def flat_env(config: AppConfig):
    """Factory for an environment with constant, hand-picked profiles."""

    def _make(pv_kw: float = 0.0, traffic: float = 0.5, price: float = 0.2, soc_pct: float = 50.0, cfg=None):
        cfg = cfg or config
        n = cfg.simulation.n_steps
        return BaseStationEnv(
            cfg,
            pv_kw=np.full(n, pv_kw),
            traffic=np.full(n, traffic),
            price_eur_per_kwh=np.full(n, price),
            initial_soc_pct=soc_pct,
        )

    return _make
