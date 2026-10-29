import sys
from pathlib import Path

import numpy as np
import pytest

DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard"
sys.path.insert(0, str(DASHBOARD))

import charts  # noqa: E402

from energypilot.data import generate_scenario  # noqa: E402
from energypilot.evaluation import build_pipeline, run_episode  # noqa: E402
from energypilot.instructions import compile_constraints, parse_instruction  # noqa: E402


@pytest.fixture(scope="module")
def episodes(config):
    sc = generate_scenario("high_traffic_low_solar", config, 0)
    return {p.name: run_episode(config, sc, p) for p in (build_pipeline(n, config) for n in ("rule_based", "optimizer", "agent"))}


def test_figures_build_in_both_themes(config, episodes):
    instr = parse_instruction(generate_scenario("high_traffic_low_solar", config, 0).spec.instruction_text, 600, config).instruction
    cons = compile_constraints(instr, config)
    histories = {k: e.history for k, e in episodes.items()}
    for mode in ("light", "dark"):
        theme = charts.theme_for(mode)
        soc = charts.soc_figure(histories, np.zeros(97), (10, 95), 40.0, 0.25, theme, issue_hour=10)
        assert len(soc.data) == 3 and soc.data[2].line.dash == "dash"  # agent dashed (secondary encoding)
        assert {t.line.color for t in soc.data} == set(theme.approaches)
        power = charts.power_figure(histories["optimizer"], cons.grid_cap_kw, 0.25, theme, "optimizer")
        assert [t.name for t in power.data] == ["PV available", "Station demand", "Grid import"]
        assert any("grid cap 2 kW" in a.text for a in power.layout.annotations)
        bars = charts.bar_figure({k: e.metrics["total_cost_eur"] for k, e in episodes.items()}, "cost", "EUR", theme)
        assert list(bars.data[0].marker.color) == list(theme.approaches)
        util = charts.utilization_figure(histories, 0.85, 0.25, theme)
        assert len(util.data) == 3


def test_colour_identity_follows_approach():
    theme = charts.LIGHT
    assert charts.approach_color("agent[llm:x/y]", theme) == charts.approach_color("agent[deterministic-fallback]", theme)
    assert charts.approach_color("optimizer", charts.DARK) == charts.DARK.approaches[1]
    assert charts.theme_for(None) is charts.LIGHT


def test_app_runs_a_simulation():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(DASHBOARD / "app.py"), default_timeout=240)
    at.run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert any("Simulation complete" in s.value for s in at.success)
    assert len(at.get("plotly_chart")) >= 5
