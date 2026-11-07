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


def _operator_page():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(DASHBOARD / "pages" / "1_Operator_view.py"), default_timeout=240)
    at.run()
    for cb in at.sidebar.checkbox:  # never call a real LLM from the test suite
        cb.uncheck()
    return at.run()


def _confirm_and_apply(at, text):
    at.text_area[0].set_value(text).run()
    assert at.button[0].disabled  # nothing is applied before the operator confirms the interpretation
    next(c for c in at.checkbox if c.label.startswith("Yes")).check().run()
    at.button[0].click().run()
    assert not at.exception
    return [m.value for m in at.markdown if m.value[:1] in "✅❌"]


def test_operator_view_confirms_applies_and_checks():
    at = _operator_page()
    assert not at.exception
    checks = _confirm_and_apply(at, "Keep at least 80% battery until 23:00 and limit grid import to 3 kW from 18:00 to 20:00")
    assert any("Instruction applied" in s.value for s in at.success)
    assert any("Battery ≥ 80 %" in c for c in checks) and any("Grid import ≤ 3 kW" in c for c in checks)
    assert all(c.startswith("✅") for c in checks)
    assert len(at.get("plotly_chart")) == 2


def test_operator_view_flags_a_missed_limit():
    # Accepted on forecasts, but the realised day overshoots this very tight limit: the view must show it.
    checks = _confirm_and_apply(_operator_page(), "Keep at least 80% battery until 23:00 and limit grid import to 0.5 kW from 20:00 to 22:00")
    assert any(c.startswith("❌") and "0.5 kW" in c for c in checks)


def test_operator_view_blocks_vague_instruction_without_llm():
    at = _operator_page()
    at.text_area[0].set_value("Continuously monitor network traffic and battery").run()
    assert any("could not turn this into enforceable constraints" in w.value for w in at.warning)
    assert all(c.disabled for c in at.checkbox if not c.label.startswith("Use the language model"))
    assert at.button[0].disabled
