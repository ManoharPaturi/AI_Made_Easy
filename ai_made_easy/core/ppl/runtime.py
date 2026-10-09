"""Runtime code shared by generated PyMC scripts and the app (numpy / scipy / pandas).

Convergence diagnostics are implemented here (rank-normalised split R-hat and bulk / tail
effective sample size, Vehtari et al. 2021) so scripts do not depend on ArviZ's
changing API; the tests compare them with ArviZ.
"""
from __future__ import annotations

DATA_CODE = r'''
# ======================================================================== data
def synthetic_groups(n_groups: int, rows_per_group: int, intercept: float, group_sd: float,
                     slope: float, noise: float, outcome: str, seed: int):
    """Grouped rows (group, x, y) with varying intercepts; returns (frame, true values)."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    effects = rng.normal(intercept, group_sd, n_groups)
    groups, xs = [], []
    for g in range(n_groups):
        n = max(2, int(rng.poisson(rows_per_group)))
        groups += [f"g{g}"] * n
        xs.append(rng.normal(0, 1, n))
    x = np.concatenate(xs)
    codes = np.array([int(g[1:]) for g in groups])
    eta = effects[codes] + slope * x
    if outcome == "binary":
        y = (rng.random(len(x)) < 1 / (1 + np.exp(-eta))).astype(int)
    elif outcome == "count":
        y = rng.poisson(np.exp(0.3 * eta))
    else:
        y = eta + rng.normal(0, noise, len(x))
    truth = {"intercept": intercept, "group_sd": group_sd, "slope": slope, "noise": noise,
             "group_effects": effects.round(4).tolist()}
    return pd.DataFrame({"group": groups, "x": x, "y": y}), truth
'''

DIAGNOSTICS_CODE = r'''
# ======================================================================== diagnostics
def _split(x: np.ndarray) -> np.ndarray:
    """[chains, draws] -> [2 * chains, draws / 2]: detects drift within chains."""
    half = x.shape[1] // 2
    return np.concatenate([x[:, :half], x[:, -half:]], axis=0)


def _rank_normalize(x: np.ndarray) -> np.ndarray:
    from scipy.stats import norm, rankdata

    ranks = rankdata(x, axis=None).reshape(x.shape)
    return norm.ppf((ranks - 0.375) / (x.size + 0.25))


def _rhat_raw(x: np.ndarray) -> float:
    m, n = x.shape
    within = x.var(axis=1, ddof=1).mean()
    between = n * x.mean(axis=1).var(ddof=1)
    if within == 0:
        return float("nan")
    return float(np.sqrt(((n - 1) / n * within + between / n) / within))


def rhat(x: np.ndarray) -> float:
    """Rank-normalised split R-hat (max of bulk and folded / tail); ~1.00 is converged."""
    if x.shape[0] * (x.shape[1] // 2) < 4 or np.ptp(x) == 0:
        return float("nan")
    bulk = _rhat_raw(_rank_normalize(_split(x)))
    tail = _rhat_raw(_rank_normalize(_split(np.abs(x - np.median(x)))))
    return max(bulk, tail)


def _autocov(chain: np.ndarray) -> np.ndarray:
    n = len(chain)
    centred = chain - chain.mean()
    f = np.fft.rfft(centred, 2 * n)
    return np.fft.irfft(f * np.conj(f))[:n] / n


def _ess_raw(x: np.ndarray) -> float:
    chains, n = x.shape
    if n < 4 or np.ptp(x) == 0:
        return float("nan")
    acov = np.stack([_autocov(c) for c in x])
    mean_var = acov[:, 0].mean() * n / (n - 1)
    var_plus = mean_var * (n - 1) / n
    if chains > 1:
        var_plus += x.mean(axis=1).var(ddof=1)
    rho = np.zeros(n)
    rho[0] = 1.0
    even, odd = 1.0, 1.0 - (mean_var - acov[:, 1].mean()) / var_plus
    rho[1] = odd
    t = 1
    while t < n - 3 and even + odd > 0:
        even = 1.0 - (mean_var - acov[:, t + 1].mean()) / var_plus
        odd = 1.0 - (mean_var - acov[:, t + 2].mean()) / var_plus
        if even + odd >= 0:
            rho[t + 1], rho[t + 2] = even, odd
        t += 2
    max_t = t - 2
    if even > 0:
        rho[max_t + 1] = even
    t = 1          # Geyer's initial monotone sequence
    while t <= max_t - 2:
        if rho[t + 1] + rho[t + 2] > rho[t - 1] + rho[t]:
            rho[t + 1] = rho[t + 2] = (rho[t - 1] + rho[t]) / 2
        t += 2
    total = chains * n
    tau = -1 + 2 * rho[: max_t + 1].sum() + rho[max_t + 1: max_t + 2].sum()
    return float(total / max(tau, 1 / np.log10(total)))


def ess_bulk(x: np.ndarray) -> float:
    return _ess_raw(_rank_normalize(_split(x)))


def ess_tail(x: np.ndarray) -> float:
    lo, hi = np.quantile(x, [0.05, 0.95])
    return min(_ess_raw(_split((x <= lo).astype(float))),
               _ess_raw(_split((x <= hi).astype(float))))


def summarize(draws: dict) -> list[dict]:
    """One row per scalar (or vector element) of each variable: name -> [chains, draws, ...]."""
    rows = []
    for name, arr in draws.items():
        flat = arr.reshape(arr.shape[0], arr.shape[1], -1)
        for k in range(flat.shape[2]):
            x = flat[:, :, k]
            label = name if flat.shape[2] == 1 else f"{name}[{k}]"
            lo, hi = np.quantile(x, [0.03, 0.97])
            rows.append({"name": label, "mean": float(x.mean()), "sd": float(x.std()),
                         "lower_94": float(lo), "upper_94": float(hi),
                         "ess_bulk": ess_bulk(x), "ess_tail": ess_tail(x), "r_hat": rhat(x)})
    return rows
'''

PLOT_CODE = r'''
# ======================================================================== plots
PALETTE = [(90, 150, 240), (240, 120, 80), (110, 200, 120), (220, 190, 80),
           (190, 120, 220), (90, 200, 200)]


def posterior_grid(draws: dict, max_panels: int = 12, size=(240, 130)) -> "Image.Image":
    """Histogram of each scalar's posterior with its 94% interval."""
    panels = []
    for name, arr in draws.items():
        flat = arr.reshape(-1, int(np.prod(arr.shape[2:])) if arr.ndim > 2 else 1)
        for k in range(flat.shape[1]):
            panels.append((name if flat.shape[1] == 1 else f"{name}[{k}]", flat[:, k]))
    panels = panels[:max_panels]
    cols = min(4, max(1, len(panels)))
    rows = (len(panels) + cols - 1) // cols
    w, h = size
    img = Image.new("RGB", (cols * w, max(rows, 1) * h), (24, 26, 30))
    draw = ImageDraw.Draw(img)
    for i, (label, values) in enumerate(panels):
        ox, oy = (i % cols) * w, (i // cols) * h
        counts, edges = np.histogram(values, bins=30)
        top = counts.max() or 1
        for j, c in enumerate(counts):
            x0 = ox + 10 + j * (w - 20) / 30
            draw.rectangle([x0, oy + h - 20 - c / top * (h - 45), x0 + (w - 20) / 30 - 1,
                            oy + h - 20], fill=(90, 150, 240))
        lo, hi = np.quantile(values, [0.03, 0.97])
        span = edges[-1] - edges[0] or 1
        for q in (lo, hi):
            xq = ox + 10 + (q - edges[0]) / span * (w - 20)
            draw.line([xq, oy + 22, xq, oy + h - 20], fill=(240, 170, 60))
        draw.text((ox + 8, oy + 4), f"{label}  {values.mean():.3g} ± {values.std():.2g}",
                  fill=(225, 225, 225))
        draw.text((ox + 8, oy + h - 16), f"{edges[0]:.3g}", fill=(150, 150, 150))
        draw.text((ox + w - 60, oy + h - 16), f"{edges[-1]:.3g}", fill=(150, 150, 150))
    return img


def trace_plot(draws: dict, max_panels: int = 6, size=(720, 90)) -> "Image.Image":
    """Draw-by-draw values of each chain: well-mixed chains overlap like noise."""
    panels = [(n, a.reshape(a.shape[0], a.shape[1], -1)[:, :, 0]) for n, a in draws.items()]
    panels = panels[:max_panels]
    w, h = size
    img = Image.new("RGB", (w, h * max(len(panels), 1)), (24, 26, 30))
    draw = ImageDraw.Draw(img)
    for i, (label, x) in enumerate(panels):
        oy = i * h
        lo, hi = float(x.min()), float(x.max())
        hi = hi if hi > lo else lo + 1
        for c in range(x.shape[0]):
            pts = [(10 + t * (w - 20) / max(x.shape[1] - 1, 1),
                    oy + h - 8 - (v - lo) / (hi - lo) * (h - 24)) for t, v in enumerate(x[c])]
            draw.line(pts, fill=PALETTE[c % len(PALETTE)], width=1)
        draw.text((10, oy + 2), label, fill=(225, 225, 225))
    return img


def ppc_plot(observed: np.ndarray, mean: np.ndarray, lower: np.ndarray, upper: np.ndarray,
             title: str, size=(420, 420)) -> "Image.Image":
    """Observed vs predicted (posterior predictive mean, 94% interval as a whisker)."""
    w, h = size
    img = Image.new("RGB", size, (24, 26, 30))
    draw = ImageDraw.Draw(img)
    lo = float(min(observed.min(), lower.min()))
    hi = float(max(observed.max(), upper.max()))
    hi = hi if hi > lo else lo + 1

    def xy(a, b):
        return 30 + (a - lo) / (hi - lo) * (w - 50), h - 30 - (b - lo) / (hi - lo) * (h - 60)

    draw.line([xy(lo, lo), xy(hi, hi)], fill=(90, 90, 90))
    idx = np.random.default_rng(0).permutation(len(observed))[:400]
    for i in idx:
        x0, y0 = xy(observed[i], lower[i])
        _x1, y1 = xy(observed[i], upper[i])
        draw.line([x0, y0, x0, y1], fill=(70, 90, 130))
        xm, ym = xy(observed[i], mean[i])
        draw.ellipse([xm - 2, ym - 2, xm + 2, ym + 2], fill=(240, 120, 80))
    draw.text((10, 6), title, fill=(225, 225, 225))
    draw.text((w - 120, h - 18), "observed →", fill=(150, 150, 150))
    draw.text((6, 22), "predicted ↑", fill=(150, 150, 150))
    return img
'''


def namespace() -> dict:
    import json
    from pathlib import Path

    import numpy as np

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        Image = ImageDraw = None
    ns: dict = {"np": np, "json": json, "Path": Path, "Image": Image, "ImageDraw": ImageDraw}
    for code in (DATA_CODE, DIAGNOSTICS_CODE, PLOT_CODE):
        exec(compile(code, "<ppl-runtime>", "exec"), ns)  # noqa: S102 — our own source
    return ns
