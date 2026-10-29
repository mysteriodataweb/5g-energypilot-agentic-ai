"""Plotly figure builders for the dashboard (pure functions, testable without Streamlit).

Colour roles (validated with the data-viz palette validator, light and dark):
* approaches - categorical slots 1-3 (blue, orange, aqua); identity follows the
  approach in every chart;
* power flows - PV yellow, demand violet, grid magenta, with distinct dash
  styles as a secondary encoding (some light-mode steps are below 3:1 contrast,
  so legends, direct labels and data tables are always provided).
Single y-axis per chart (SoC in % and power in kW are never mixed).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go

APPROACH_ORDER = ("rule_based", "optimizer", "agent")


@dataclass(frozen=True)
class Theme:
    mode: str
    surface: str
    ink: str
    ink_secondary: str
    muted: str
    grid: str
    axis: str
    approaches: tuple[str, str, str]
    pv: str
    demand: str
    grid_import: str
    band: str  # reserve / constraint windows


LIGHT = Theme("light", "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7",
              ("#2a78d6", "#eb6834", "#1baf7a"), "#eda100", "#6250d6", "#e87ba4", "rgba(137,135,129,0.14)")  # fmt: skip
DARK = Theme("dark", "#1a1a19", "#f0efec", "#c3c2b7", "#898781", "#2c2c2a", "#383835",
             ("#3987e5", "#d95926", "#199e70"), "#c98500", "#9085e9", "#d55181", "rgba(195,194,183,0.12)")  # fmt: skip


def theme_for(mode: str | None) -> Theme:
    return DARK if mode == "dark" else LIGHT


def approach_key(label: str) -> str:
    """'agent[deterministic-fallback]' -> 'agent'."""
    return label.split("[", 1)[0]


def approach_dash(label: str) -> str:
    """Secondary encoding: the agent is dashed so it stays visible when it coincides with the optimiser."""
    return "dash" if approach_key(label) == "agent" else "solid"


def approach_color(label: str, theme: Theme) -> str:
    key = approach_key(label)
    return theme.approaches[APPROACH_ORDER.index(key)] if key in APPROACH_ORDER else theme.muted


def _hours_ticks() -> dict:
    vals = list(range(0, 25, 3))
    return {"tickvals": vals, "ticktext": [f"{h:02d}:00" for h in vals], "range": [0, 24]}


def _base(theme: Theme, title: str, y_title: str, height: int = 360) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title={"text": title, "font": {"size": 15, "color": theme.ink}, "x": 0, "xanchor": "left"},
        height=height,
        margin={"l": 56, "r": 16, "t": 48, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=theme.surface,
        font={"family": "system-ui, -apple-system, Segoe UI, sans-serif", "color": theme.ink_secondary, "size": 12},
        legend={"orientation": "h", "y": -0.18, "x": 0, "font": {"color": theme.ink_secondary}},
        hovermode="x unified",
        hoverlabel={"bgcolor": theme.surface, "font": {"color": theme.ink}},
    )
    fig.update_xaxes(title=None, showgrid=False, linecolor=theme.axis, tickfont={"color": theme.muted}, **_hours_ticks())
    fig.update_yaxes(title={"text": y_title, "font": {"color": theme.muted}}, gridcolor=theme.grid, zeroline=False,
                     linecolor=theme.axis, tickfont={"color": theme.muted})  # fmt: skip
    return fig


def _windows(mask: np.ndarray, hours: np.ndarray, dt: float) -> list[tuple[float, float]]:
    spans, start = [], None
    for k, on in enumerate(mask):
        if on and start is None:
            start = hours[k]
        if not on and start is not None:
            spans.append((start, hours[k]))
            start = None
    if start is not None:
        spans.append((start, hours[-1] + dt))
    return spans


def soc_figure(histories: dict[str, pd.DataFrame], reserve_pct: np.ndarray, soc_limits: tuple[float, float],
               capacity_kwh: float, dt: float, theme: Theme, issue_hour: float | None = None) -> go.Figure:  # fmt: skip
    """SoC per approach; reserve windows shaded; BMS limits as dashed rules."""
    fig = _base(theme, "Battery state of charge", "SoC (%)")
    hours = np.arange(len(reserve_pct) - 1) * dt
    for start, end in _windows(reserve_pct[1:] > 0, hours, dt):
        level = float(np.max(reserve_pct[1:][(hours >= start) & (hours < end)]))
        fig.add_vrect(x0=start, x1=end, fillcolor=theme.band, line_width=0, layer="below",
                      annotation_text=f"reserve ≥ {level:g}%", annotation_position="top left",
                      annotation_font={"color": theme.ink_secondary, "size": 11})  # fmt: skip
    for value, name in ((soc_limits[0], "SoC min"), (soc_limits[1], "SoC max")):
        fig.add_hline(y=value, line={"color": theme.muted, "width": 1, "dash": "dash"},
                      annotation_text=name, annotation_font={"color": theme.muted, "size": 11})  # fmt: skip
    if issue_hour is not None:
        fig.add_vline(x=issue_hour, line={"color": theme.muted, "width": 1, "dash": "dot"}, annotation_text="instruction",
                      annotation_position="bottom right", annotation_font={"color": theme.muted, "size": 11})  # fmt: skip
    for label, h in histories.items():
        x = np.concatenate([[0.0], h.time_h.to_numpy() + dt])
        y = np.concatenate([[100.0 * h.soc_start_kwh.iloc[0] / capacity_kwh], h.soc_end_pct.to_numpy()])
        fig.add_trace(go.Scatter(x=x, y=y, name=label, mode="lines", line={"color": approach_color(label, theme), "width": 2, "dash": approach_dash(label)},
                                 hovertemplate="%{y:.1f}%"))  # fmt: skip
    fig.update_yaxes(range=[0, 100])
    return fig


def power_figure(h: pd.DataFrame, grid_cap_kw: np.ndarray, dt: float, theme: Theme, label: str) -> go.Figure:
    """PV available, station demand and grid import for one approach (kW, one axis)."""
    fig = _base(theme, f"Power flows - {label}", "Power (kW)")
    x = h.time_h.to_numpy() + dt / 2
    capped = np.isfinite(grid_cap_kw)
    for start, end in _windows(capped, h.time_h.to_numpy(), dt):
        level = float(np.min(grid_cap_kw[(h.time_h >= start) & (h.time_h < end)]))
        fig.add_shape(type="line", x0=start, x1=end, y0=level, y1=level, line={"color": theme.ink_secondary, "width": 1.5, "dash": "dash"})
        fig.add_annotation(x=start, y=level, text=f"grid cap {level:g} kW", showarrow=False, yshift=10, xanchor="left",
                           font={"color": theme.ink_secondary, "size": 11})  # fmt: skip
    for col, name, color, dash in (
        ("pv_available_kw", "PV available", theme.pv, "solid"),
        ("load_kw", "Station demand", theme.demand, "dash"),
        ("grid_import_kw", "Grid import", theme.grid_import, "dot"),
    ):
        fig.add_trace(go.Scatter(x=x, y=h[col], name=name, mode="lines", line={"color": color, "width": 2, "dash": dash},
                                 hovertemplate="%{y:.2f} kW"))  # fmt: skip
    fig.update_yaxes(rangemode="tozero")
    return fig


def utilization_figure(histories: dict[str, pd.DataFrame], target: float, dt: float, theme: Theme) -> go.Figure:
    fig = _base(theme, "Radio utilisation vs QoS target", "Utilisation (traffic / active capacity)")
    fig.add_hline(y=target, line={"color": theme.ink_secondary, "width": 1.5, "dash": "dash"},
                  annotation_text=f"QoS target {target:g}", annotation_font={"color": theme.ink_secondary, "size": 11})  # fmt: skip
    for label, h in histories.items():
        fig.add_trace(go.Scatter(x=h.time_h + dt / 2, y=h.utilization, name=label, mode="lines",
                                 line={"color": approach_color(label, theme), "width": 2, "dash": approach_dash(label)},
                                 hovertemplate="%{y:.2f}"))  # fmt: skip
    fig.update_yaxes(rangemode="tozero")
    return fig


def bar_figure(values: dict[str, float], title: str, unit: str, theme: Theme, decimals: int = 2) -> go.Figure:
    labels = list(values)
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=[values[k] for k in labels],
            marker={"color": [approach_color(k, theme) for k in labels], "cornerradius": 4, "line": {"width": 0}},
            text=[f"{values[k]:.{decimals}f} {unit}" for k in labels],
            textposition="outside",
            textfont={"color": theme.ink},
            hovertemplate="%{x}: %{y:." + str(decimals) + "f} " + unit + "<extra></extra>",
            width=0.5,
        )
    )
    fig.update_layout(
        title={"text": title, "font": {"size": 15, "color": theme.ink}, "x": 0, "xanchor": "left"},
        height=320, margin={"l": 56, "r": 16, "t": 48, "b": 40}, bargap=0.4,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=theme.surface,
        font={"family": "system-ui, -apple-system, Segoe UI, sans-serif", "color": theme.ink_secondary, "size": 12},
        showlegend=False,
    )  # fmt: skip
    fig.update_xaxes(linecolor=theme.axis, tickfont={"color": theme.ink_secondary})
    fig.update_yaxes(title={"text": unit, "font": {"color": theme.muted}}, gridcolor=theme.grid, zeroline=True,
                     zerolinecolor=theme.axis, tickfont={"color": theme.muted}, rangemode="tozero")  # fmt: skip
    top = max([abs(v) for v in values.values()] + [1e-9])
    fig.update_yaxes(range=[0, top * 1.2])
    return fig


def ci_bar_figure(summary: pd.DataFrame, metric: str, title: str, unit: str, theme: Theme) -> go.Figure:
    """Grouped bars (scenario x approach) of an experiment metric with 95% CI error bars."""
    sub = summary[summary.metric == metric]
    fig = go.Figure()
    for label in sorted(sub.approach.unique(), key=lambda a: APPROACH_ORDER.index(approach_key(a)) if approach_key(a) in APPROACH_ORDER else 9):
        g = sub[sub.approach == label]
        err_hi = (g.ci95_high - g["mean"]).fillna(0)
        err_lo = (g["mean"] - g.ci95_low).fillna(0)
        fig.add_trace(go.Bar(
            x=g.scenario, y=g["mean"], name=label,
            marker={"color": approach_color(label, theme), "cornerradius": 4, "line": {"width": 0}},
            error_y={"type": "data", "symmetric": False, "array": err_hi, "arrayminus": err_lo,
                     "color": theme.ink_secondary, "thickness": 1.2, "width": 3},
            hovertemplate="%{x}<br>" + label + ": %{y:.2f} " + unit + "<extra></extra>",
        ))  # fmt: skip
    fig.update_layout(
        title={"text": title, "font": {"size": 15, "color": theme.ink}, "x": 0, "xanchor": "left"},
        barmode="group", bargap=0.25, bargroupgap=0.08, height=380, margin={"l": 56, "r": 16, "t": 48, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=theme.surface,
        font={"family": "system-ui, -apple-system, Segoe UI, sans-serif", "color": theme.ink_secondary, "size": 12},
        legend={"orientation": "h", "y": -0.2, "x": 0, "font": {"color": theme.ink_secondary}},
    )  # fmt: skip
    fig.update_xaxes(linecolor=theme.axis, tickfont={"color": theme.ink_secondary})
    fig.update_yaxes(title={"text": unit, "font": {"color": theme.muted}}, gridcolor=theme.grid,
                     tickfont={"color": theme.muted}, rangemode="tozero")  # fmt: skip
    return fig


def profile_figure(hours: np.ndarray, actual: np.ndarray, forecast: np.ndarray, title: str, unit: str,
                   color: str, theme: Theme, dt: float) -> go.Figure:  # fmt: skip
    """Synthetic 'measured' profile vs the forecast available to planners."""
    fig = _base(theme, title, unit, height=300)
    x = hours + dt / 2
    fig.add_trace(go.Scatter(x=x, y=forecast, name="forecast (known to planners)", mode="lines",
                             line={"color": theme.muted, "width": 2, "dash": "dash"}, hovertemplate="%{y:.2f}"))  # fmt: skip
    fig.add_trace(go.Scatter(x=x, y=actual, name="realised (synthetic measurement)", mode="lines",
                             line={"color": color, "width": 2}, hovertemplate="%{y:.2f}"))  # fmt: skip
    fig.update_yaxes(rangemode="tozero")
    return fig
