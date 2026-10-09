"""State-space model blocks: structural components (trend, seasonality, cycle,
autoregression, regression) or a SARIMAX specification wired into one model, fitted
with the Kalman filter (statsmodels)."""
from __future__ import annotations

from ai_made_easy.core.blocks._dsl import P
from ai_made_easy.core.palette import family_color
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import BlockDefinition, PortSpec

CATEGORY = "State-Space Models"
COMPONENT_OUT = PortSpec("out", dtype="config", role="component")
TREND_KINDS = ("local level", "local linear trend", "smooth trend", "random walk with drift",
               "deterministic trend", "fixed intercept", "random walk")
COMPONENTS = ("ssm.trend", "ssm.seasonal", "ssm.cycle", "ssm.autoregressive",
              "ssm.regression")
SSM_DATA = ("data.structural_series", "data.timeseries_csv")


def _component(type_id: str, name: str, params: tuple, desc: str,
               family: str = "model") -> BlockDefinition:
    return BlockDefinition(
        type_id=type_id, display_name=name, category=CATEGORY, color=family_color(family),
        params=params, outputs=(COMPONENT_OUT,), library="statsmodels", description=desc,
        meta={"ssm": "component"})


def _blocks() -> list[BlockDefinition]:
    return [
        _component("ssm.trend", "Level / Trend",
                   (P("kind", "enum", "local linear trend", options=TREND_KINDS,
                      help="local level: a wandering mean; local linear trend: a wandering "
                           "mean and slope; smooth trend: a fixed level with a wandering "
                           "slope; deterministic trend: a straight line"),),
                   "The series' underlying level and slope, allowed to drift over time."),
        _component("ssm.seasonal", "Seasonal Component",
                   (P("period", "int", 12, lo=2, hi=10000, help="Steps per season"),
                    P("harmonics", "int", 0, lo=0, hi=500,
                      help="0: one effect per step of the season; n: n sine / cosine pairs "
                           "(smooth shapes, long periods)"),
                    P("stochastic", "bool", True, help="Let the seasonal shape change")),
                   "A pattern that repeats every period (days of the week, months of the "
                   "year)."),
        _component("ssm.cycle", "Cycle",
                   (P("stochastic", "bool", True), P("damped", "bool", True),
                    P("min_period", "float", 1.5, lo=1.5,
                      help="Shortest cycle length searched (steps)"),
                    P("max_period", "float", 0.0, lo=0.0,
                      help="Longest cycle length (0: the series length / 2)")),
                   "A quasi-periodic swing without a fixed period (business cycles)."),
        _component("ssm.autoregressive", "Autoregressive Component",
                   (P("order", "int", 1, lo=1, hi=20),),
                   "Short-term memory: today's deviation carries over to the next steps."),
        _component("ssm.regression", "Regression Component",
                   (P("columns", "str", "", help="Explanatory columns of the data (their "
                                                 "future values are needed to forecast)"),
                    P("time_varying", "bool", False,
                      help="Let the coefficients drift over time")),
                   "Effects of explanatory columns (price, temperature, holidays).",
                   family="model"),
        _component("ssm.arima", "SARIMAX Specification",
                   (P("p", "int", 1, lo=0, hi=20), P("d", "int", 1, lo=0, hi=3),
                    P("q", "int", 1, lo=0, hi=20),
                    P("seasonal_p", "int", 0, lo=0, hi=5), P("seasonal_d", "int", 0, lo=0, hi=2),
                    P("seasonal_q", "int", 0, lo=0, hi=5),
                    P("season", "int", 0, lo=0, hi=1000, help="Seasonal period (0: none)"),
                    P("trend", "enum", "c", options=("n", "c", "t", "ct"),
                      help="n: none, c: constant, t: linear, ct: both")),
                   "(Seasonal) ARIMA in state-space form: p autoregressive, d differences, "
                   "q moving-average terms."),
        BlockDefinition(
            type_id="ssm.model", display_name="State-Space Model", category=CATEGORY,
            color=family_color("model"), library="statsmodels",
            inputs=(PortSpec("components", dtype="config", role="component", multi=True),),
            params=(P("target_column", "str", "",
                      help="Series column (empty: the data block's first target column)"),
                    P("horizon", "int", 24, lo=1, hi=100000,
                      help="Steps held out for testing and forecast when serving"),
                    P("interval", "float", 0.9, lo=0.5, hi=0.999,
                      help="Coverage of the prediction intervals"),
                    P("irregular", "bool", True, help="Observation noise (structural models)"),
                    P("max_iterations", "int", 200, lo=1, hi=100000)),
            description="Fits the wired components with the Kalman filter by maximum "
                        "likelihood: forecasts with intervals and the smoothed components "
                        "(trend, season, cycle).",
            meta={"ssm": "model"}),
        BlockDefinition(
            type_id="data.structural_series", display_name="Structural Series",
            category="Data", color=family_color("data"), library="NumPy",
            meta={"modality": "timeseries", "ssm": "data"},
            params=(P("length", "int", 240, lo=20, hi=1_000_000),
                    P("slope", "float", 0.05, lo=-1e6, hi=1e6, help="Average change per step"),
                    P("level_drift", "float", 0.02, lo=0.0,
                      help="Random-walk noise of the slope"),
                    P("season_length", "int", 12, lo=0, hi=10000, help="0: no seasonality"),
                    P("season_amplitude", "float", 2.0, lo=0.0),
                    P("cycle_amplitude", "float", 0.0, lo=0.0,
                      help="A slow quasi-periodic swing (0: none)"),
                    P("noise", "float", 0.4, lo=0.0), P("seed", "int", 0, lo=0)),
            description="A series built from known parts (trend + season + cycle + noise): "
                        "check that the model recovers them."),
    ]


def register_all() -> None:
    reg = get_registry()
    for defn in _blocks():
        reg.register(defn)
