"""Runtime code shared by generated forecasting scripts and the app (numpy; pandas for files).

Like ``core.vision.runtime``: the training script embeds these strings and the
app executes the same source for profiling and tests.
"""
from __future__ import annotations

DATA_CODE = r'''
# ======================================================================== forecasting data
def _columns(spec) -> list:
    return [c.strip() for c in str(spec or "").split(",") if c.strip()]


def read_forecast_table(path, fmt, time_column, target_column, id_column="", past="",
                        future="") -> list:
    """One dict per series: id, time, target [T], past [T, P], future [T, F]."""
    import pandas as pd

    path = Path(str(path)).expanduser()
    frame = pd.read_parquet(path) if fmt == "parquet" else pd.read_csv(
        path, sep="\t" if fmt == "tsv" else ",")
    past_cols, future_cols = _columns(past), _columns(future)
    needed = [c for c in (time_column, target_column, id_column, *past_cols, *future_cols) if c]
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise ValueError(f"columns {missing} not found; the file has {list(frame.columns)}")
    groups = frame.groupby(id_column, sort=True) if id_column else [("series", frame)]
    out = []
    for key, part in groups:
        if time_column:
            part = part.sort_values(time_column)
        out.append({
            "id": str(key), "time": part[time_column].astype(str).to_numpy() if time_column
            else np.arange(len(part)).astype(str),
            "target": part[target_column].to_numpy(np.float32),
            "past": part[past_cols].to_numpy(np.float32) if past_cols
            else np.zeros((len(part), 0), np.float32),
            "future": part[future_cols].to_numpy(np.float32) if future_cols
            else np.zeros((len(part), 0), np.float32)})
    return out


def synthetic_series(n_series: int, length: int, season_length: int, trend: float,
                     noise: float, counts: bool, promotions: bool, seed: int) -> list:
    """Seasonal series with trend, noise and an optional known-future promotion flag."""
    rng = np.random.default_rng(seed)
    t = np.arange(length, dtype=np.float32)
    out = []
    for i in range(n_series):
        level = rng.uniform(20, 80)
        phase = rng.uniform(0, 2 * np.pi)
        season = np.sin(2 * np.pi * t / max(season_length, 1) + phase)
        season += 0.5 * np.sin(4 * np.pi * t / max(season_length, 1) + 2 * phase)
        mean = level * (1 + 0.3 * season) * (1 + trend * t / max(length, 1))
        promo = (rng.random(length) < 0.06).astype(np.float32) if promotions else \
            np.zeros(length, np.float32)
        mean = mean * (1 + 0.5 * promo)
        y = rng.poisson(np.maximum(mean, 0)) if counts else \
            mean * (1 + noise * rng.standard_normal(length))
        out.append({"id": f"series_{i}", "time": t.astype(int).astype(str),
                    "target": np.asarray(y, np.float32),
                    "past": np.zeros((length, 0), np.float32),
                    "future": promo[:, None] if promotions else np.zeros((length, 0), np.float32)})
    return out


def load_series(dataset: dict) -> list:
    if dataset["block"] == "data.synthetic_series":
        return synthetic_series(int(dataset["n_series"]), int(dataset["length"]),
                                int(dataset["season_length"]), float(dataset["trend"]),
                                float(dataset["noise"]), bool(dataset["counts"]),
                                bool(dataset["promotions"]), int(dataset["seed"]))
    return read_forecast_table(dataset["path"], dataset["format"], dataset["time_column"],
                               dataset["target_column"], dataset["id_column"],
                               dataset["past_covariates"], dataset["future_covariates"])


def regions(length: int, horizon: int, val_windows: int, test_windows: int) -> dict:
    """Chronological split of one series: train | val origins | test origins."""
    test_start = length - test_windows * horizon
    val_start = test_start - val_windows * horizon
    return {"train": (0, val_start), "val": (val_start, test_start), "test": (test_start, length)}


def features(s: dict) -> np.ndarray:
    """[T, C]: target, past covariates, future covariates (as observed in the history)."""
    return np.concatenate([s["target"][:, None], s["past"], s["future"]], axis=1)


def seasonal_scale(y: np.ndarray, season: int) -> float:
    """In-sample seasonal-naive MAE: the MASE denominator."""
    m = max(1, season)
    if len(y) <= m:
        return float(np.mean(np.abs(np.diff(y)))) if len(y) > 1 else 1.0
    return float(np.mean(np.abs(y[m:] - y[:-m]))) or 1.0


def make_windows(series: list, window: int, horizon: int, val_windows: int, test_windows: int,
                 stride: int, split: str, season: int) -> dict:
    """Arrays for one split: x [N, L, C], future [N, H, F], y [N, H], mase scale [N],
    series index [N] and the forecast origin [N]."""
    xs, fs, ys, scales, ids, origins = [], [], [], [], [], []
    for i, s in enumerate(series):
        data = features(s)
        bounds = regions(len(data), horizon, val_windows, test_windows)
        lo, hi = bounds[split]
        scale = seasonal_scale(s["target"][: bounds["train"][1]], season)
        if split == "train":
            starts = range(window, hi - horizon + 1, max(1, stride))
        else:
            starts = range(lo, hi - horizon + 1, horizon)
        for origin in starts:
            if origin - window < 0:
                continue
            xs.append(data[origin - window:origin])
            fs.append(s["future"][origin:origin + horizon])
            ys.append(s["target"][origin:origin + horizon])
            scales.append(scale)
            ids.append(i)
            origins.append(origin)
    channels = series[0]["past"].shape[1] + series[0]["future"].shape[1] + 1 if series else 1
    fut = series[0]["future"].shape[1] if series else 0
    if not xs:
        return {"x": np.zeros((0, window, channels), np.float32),
                "future": np.zeros((0, horizon, fut), np.float32),
                "y": np.zeros((0, horizon), np.float32), "mase_scale": np.zeros(0, np.float32),
                "series": np.zeros(0, int), "origin": np.zeros(0, int)}
    return {"x": np.stack(xs).astype(np.float32), "future": np.stack(fs).astype(np.float32),
            "y": np.stack(ys).astype(np.float32), "mase_scale": np.asarray(scales, np.float32),
            "series": np.asarray(ids), "origin": np.asarray(origins)}
'''

METRICS_CODE = r'''
# ======================================================================== forecasting metrics
def point_metrics(y: np.ndarray, pred: np.ndarray, mase_scale: np.ndarray) -> dict:
    err = pred - y
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mase": float(np.mean(np.abs(err).mean(1) / np.maximum(mase_scale, 1e-8))),
        "smape": float(100 * np.mean(2 * np.abs(err) / np.maximum(np.abs(y) + np.abs(pred),
                                                                 1e-8))),
        "wape": float(np.abs(err).sum() / max(float(np.abs(y).sum()), 1e-8)),
    }


def pinball(y: np.ndarray, q_pred: np.ndarray, levels) -> float:
    """Mean quantile (pinball) loss; q_pred [N, H, Q]."""
    levels = np.asarray(levels, np.float32)
    diff = y[..., None] - q_pred
    return float(np.mean(np.maximum(levels * diff, (levels - 1) * diff)))


def crps_from_quantiles(y: np.ndarray, q_pred: np.ndarray, levels) -> float:
    """CRPS approximated by twice the pinball loss averaged over the quantile levels."""
    return 2 * pinball(y, q_pred, levels)


def crps_from_samples(y: np.ndarray, samples: np.ndarray) -> float:
    """Sample CRPS: E|X - y| - E|X - X'| / 2 (samples [S, N, H])."""
    first = np.mean(np.abs(samples - y[None]), axis=0)
    ordered = np.sort(samples, axis=0)
    s = samples.shape[0]
    weights = (2 * np.arange(1, s + 1) - s - 1).reshape(-1, 1, 1)
    spread = np.sum(weights * ordered, axis=0) / (s * s)
    return float(np.mean(first - spread))


def coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((y >= lower) & (y <= upper)))
'''

PLOT_CODE = r'''
# ======================================================================== plots
def plot_forecast(history, actual, forecast, lower=None, upper=None, title: str = "",
                  size=(560, 220)) -> "Image.Image":
    """History (grey), actual future (green), forecast (red) and its interval band."""
    w, h = size
    pad = 28
    img = Image.new("RGB", size, (24, 26, 30))
    draw = ImageDraw.Draw(img)
    series = [np.asarray(history, float), np.asarray(actual, float), np.asarray(forecast, float)]
    values = np.concatenate([*series] + [np.asarray(v, float) for v in (lower, upper)
                                          if v is not None])
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    hi = hi if hi > lo else lo + 1
    total = len(series[0]) + len(series[1])

    def xy(i, v):
        return (pad + i * (w - 2 * pad) / max(total - 1, 1),
                h - pad - (v - lo) / (hi - lo) * (h - 2 * pad))

    start = len(series[0])
    if lower is not None and upper is not None:
        band = [xy(start + i, v) for i, v in enumerate(upper)]
        band += [xy(start + i, v) for i, v in reversed(list(enumerate(lower)))]
        if len(band) > 2:
            draw.polygon(band, fill=(90, 45, 45))
    for values_, offset, color in ((series[0], 0, (150, 150, 150)), (series[1], start,
                                   (80, 220, 120)), (series[2], start, (240, 80, 70))):
        points = [xy(offset + i, v) for i, v in enumerate(values_)]
        if offset and len(series[0]):
            points.insert(0, xy(start - 1, series[0][-1]))
        if len(points) > 1:
            draw.line(points, fill=color, width=2)
    draw.line([xy(start - 0.5, lo)[0], pad, xy(start - 0.5, lo)[0], h - pad], fill=(70, 70, 70))
    draw.text((pad, 6), title, fill=(220, 220, 220))
    x = w - 190
    for word, color in (("history", (150, 150, 150)), ("actual", (80, 220, 120)),
                        ("forecast", (240, 80, 70))):
        draw.text((x, 6), word, fill=color)
        x += 8 * len(word) + 10
    return img
'''


def namespace() -> dict:
    import json
    from pathlib import Path

    import numpy as np

    try:   # plots only; reading data and metrics work without pillow
        from PIL import Image, ImageDraw
    except ImportError:
        Image = ImageDraw = None
    ns: dict = {"np": np, "json": json, "Path": Path, "Image": Image, "ImageDraw": ImageDraw}
    for code in (DATA_CODE, METRICS_CODE, PLOT_CODE):
        exec(compile(code, "<forecast-runtime>", "exec"), ns)  # noqa: S102 — our own source
    return ns
