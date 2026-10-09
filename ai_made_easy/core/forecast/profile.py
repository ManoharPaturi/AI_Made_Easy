"""Data workspace profiles for forecasting datasets (long-format tables, synthetic series).

Reads the series with the training script's own reader (``core.forecast.runtime``)
and reports what matters for forecasting: series count and lengths, whether each
series is long enough for the window and backtest, missing values, irregular time
steps, zero-heavy / count data (negbin head) and the dominant seasonal period.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ai_made_easy.core.data.health import Finding
from ai_made_easy.core.data.lints import LOCAL_BLOCKS
from ai_made_easy.core.data.profile import DataProfile, register_profiler, resolve_path
from ai_made_easy.core.forecast.blocks import FORECAST_DATA

MAX_LAG = 400   # longest seasonal period searched


def season_of(y: np.ndarray, max_lag: int = MAX_LAG) -> int:
    """Lag (≥ 2) of the strongest autocorrelation peak, or 1 when nothing stands out."""
    y = np.asarray(y, float)
    y = y[~np.isnan(y)]
    if len(y) < 8:
        return 1
    t = np.arange(len(y))
    y = y - np.polyval(np.polyfit(t, y, 1), t)   # remove a linear trend
    denom = float(np.dot(y, y)) or 1.0
    lags = list(range(2, min(max_lag, len(y) // 2) + 1))
    acf = np.array([np.dot(y[:-k], y[k:]) / denom for k in lags])
    peaks = [i for i in range(1, len(acf) - 1)
             if acf[i] >= acf[i - 1] and acf[i] >= acf[i + 1] and acf[i] > 0.2]
    if not peaks:
        return 1
    best = max(acf[i] for i in peaks)
    return int(lags[next(i for i in peaks if acf[i] >= 0.6 * best)])   # not a harmonic


def _irregular(times) -> int:
    """Series steps that differ from the most common spacing (0 when not datetimes)."""
    import pandas as pd

    try:
        stamps = pd.to_datetime(pd.Series(times), errors="raise")
    except (ValueError, TypeError):
        return 0
    if len(stamps) < 3:
        return 0
    deltas = stamps.diff().dropna()
    return int((deltas != deltas.mode().iloc[0]).sum())


def profile_forecast(type_id: str, params: dict, base=None) -> DataProfile:
    from ai_made_easy.core.forecast.runtime import namespace

    ns = namespace()
    p = {"block": type_id, **params}
    if type_id == "data.forecast_csv":
        p["path"] = str(resolve_path(str(p["path"]), base))
        if not Path(p["path"]).exists():
            return DataProfile(type_id, "table", source=p["path"],
                               error=f"{p['path']} does not exist")
    try:
        series = ns["load_series"](p)
    except ImportError:
        return DataProfile(type_id, "table", error="reading tables needs pandas: "
                                                   "pip install pandas")
    except (ValueError, KeyError, OSError) as exc:
        return DataProfile(type_id, "table", source=str(p.get("path", "")), error=str(exc))
    kind = "synthetic" if type_id == "data.synthetic_series" else "table"
    profile = DataProfile(type_id, kind, source=str(p.get("path") or "generated"),
                          format=str(p.get("format", "")), task="forecasting")
    if not series:
        profile.error = "no series found"
        return profile
    lengths = np.array([len(s["target"]) for s in series])
    profile.rows = int(lengths.sum())
    findings: list[Finding] = []
    window, horizon = int(p["window"]), int(p["horizon"])
    held = horizon * (int(p["val_windows"]) + int(p["test_windows"]))
    need = window + horizon + held
    short = [s["id"] for s, n in zip(series, lengths, strict=True) if n < need]
    if short:
        findings.append(Finding(
            "warning", f"{len(short)} of {len(series)} series are shorter than the {need} steps "
                       f"a window, a horizon and the backtest need (e.g. {short[0]})",
            "lower window or val_windows / test_windows, or drop the short series"))
    y = np.concatenate([s["target"] for s in series])
    missing = int(np.isnan(y).sum())
    if missing:
        findings.append(Finding("warning", f"{missing:,} missing target values "
                                           f"({missing / len(y):.1%})",
                                "fill them (forward-fill or interpolate) before training"))
    finite = y[~np.isnan(y)]
    integer = bool(len(finite)) and bool(np.all(finite >= 0)) and \
        bool(np.allclose(finite, np.round(finite)))
    zeros = float(np.mean(finite == 0)) if len(finite) else 0.0
    if integer:
        findings.append(Finding("info", f"the target is non-negative counts ({zeros:.0%} zeros)",
                                "the negbin head fits counts and never predicts below zero"))
    elif len(finite) and finite.min() < 0:
        findings.append(Finding("info", "the target has negative values",
                                "avoid the negbin head; point, quantile or student_t fit"))
    if type_id == "data.forecast_csv" and p.get("time_column"):
        gaps = sum(_irregular(s["time"]) for s in series)
        if gaps:
            findings.append(Finding("warning", f"{gaps} time steps break the regular spacing",
                                    "resample to a fixed frequency and fill the gaps"))
    season = season_of(series[0]["target"])
    configured = int(p["season_length"])
    if season > 1 and season != configured:
        findings.append(Finding("info", f"the series repeat every {season} steps but "
                                        f"season_length is {configured}",
                                f"set season_length to {season} for a fairer MASE baseline"))
    if season > window:
        findings.append(Finding("warning", f"the window ({window}) is shorter than one season "
                                           f"({season})", f"set window to at least {season}"))
    channels = 1 + series[0]["past"].shape[1] + series[0]["future"].shape[1]
    profile.details = [
        f"length {lengths.min():,}–{lengths.max():,} steps",
        f"target {np.nanmin(y):.4g} … {np.nanmax(y):.4g} (mean {np.nanmean(y):.4g})",
        f"{series[0]['past'].shape[1]} past · {series[0]['future'].shape[1]} known-future "
        f"covariate(s) → Input [{window}, {channels}]",
        f"seasonal period ≈ {season}" if season > 1 else "no clear seasonal period",
    ]
    profile.summary = (f"{len(series):,} series · {profile.rows:,} steps · "
                       f"window {window} → horizon {horizon}")
    profile.findings = findings
    return profile


def register() -> None:
    for type_id in FORECAST_DATA:
        register_profiler(type_id, lambda params, base, _t=type_id: profile_forecast(
            _t, params, base))
        LOCAL_BLOCKS.add(type_id)
