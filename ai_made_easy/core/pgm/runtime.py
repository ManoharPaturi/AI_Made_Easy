"""Runtime code shared by generated graphical-model scripts and the app (numpy / pandas).

Like ``core.forecast.runtime``: the training script embeds these strings and the app
executes the same source for profiling and tests. Scores are computed here from the
fitted tables, so scripts do not depend on pgmpy's (changing) metrics modules.
"""
from __future__ import annotations

DATA_CODE = r'''
# ======================================================================== data
def regime_series(n_steps: int, n_regimes: int, stickiness: float, seed: int):
    """A series switching between regimes with their own mean and volatility."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    means = np.linspace(-2.0, 2.0, n_regimes) if n_regimes > 1 else np.zeros(1)
    scales = np.linspace(0.4, 1.2, n_regimes)
    state, states, values = 0, [], []
    for _ in range(n_steps):
        if rng.random() > stickiness:
            state = int(rng.integers(n_regimes))
        states.append(state)
        values.append(rng.normal(means[state], scales[state]))
    return pd.DataFrame({"value": values, "regime": states})


def forward_sample(variables: dict, n: int, seed: int):
    """Rows drawn from the designed CPDs in topological order (all tables must be set)."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    order, done = [], set()
    while len(order) < len(variables):
        ready = [v for v in variables if v not in done
                 and all(p in done for p in variables[v]["parents"])]
        if not ready:
            raise SystemExit("the network has a cycle: it cannot be sampled")
        for v in sorted(ready):
            order.append(v)
            done.add(v)
    data: dict = {}
    for name in order:
        var = variables[name]
        if var["table"] is None:
            raise SystemExit(f"Network Sample needs every probability table: {name} has none")
        table = np.asarray(var["table"], float)
        col = np.zeros(n, int)
        for parent in var["parents"]:
            states = variables[parent]["states"]
            col = col * len(states) + np.array([states.index(s) for s in data[parent]])
        u = rng.random(n)
        cum = np.cumsum(table[:, col], axis=0)
        idx = (u[None] > cum).sum(0).clip(max=len(var["states"]) - 1)
        data[name] = [var["states"][i] for i in idx]
    return pd.DataFrame(data)[list(variables)]
'''

SCORES_CODE = r'''
# ======================================================================== scores
def cpd_arrays(model, variables: dict) -> dict:
    """name -> (table [states, parent combos], parents) from a fitted discrete model, in the
    design's state order."""
    out = {}
    for name, var in variables.items():
        cpd = model.get_cpds(name)
        parents = list(cpd.variables[1:])
        values = cpd.get_values()
        names = cpd.state_names
        own = [names[name].index(s) for s in var["states"]]
        shape = [len(names[p]) for p in parents]
        values = values[own]
        if parents:
            full = values.reshape([len(own), *shape])
            for axis, p in enumerate(parents, start=1):
                order = [names[p].index(s) for s in variables[p]["states"]]
                full = np.take(full, order, axis=axis)
            values = full.reshape(len(own), -1)
        out[name] = (values, parents)
    return out


def log_likelihood(arrays: dict, variables: dict, data) -> np.ndarray:
    """Per-row log P(row) of fully observed discrete data."""
    total = np.zeros(len(data))
    for name, (table, parents) in arrays.items():
        if name not in data:
            continue
        col = np.zeros(len(data), int)
        for p in parents:
            states = variables[p]["states"]
            col = col * len(states) + data[p].map({s: i for i, s in enumerate(states)}).to_numpy()
        row = data[name].map({s: i for i, s in enumerate(variables[name]["states"])}).to_numpy()
        total += np.log(np.clip(table[row, col], 1e-12, None))
    return total


def free_parameters(arrays: dict) -> int:
    return int(sum((t.shape[0] - 1) * t.shape[1] for t, _p in arrays.values()))


def gaussian_log_likelihood(cpds: dict, data) -> np.ndarray:
    """Per-row log density under linear-Gaussian CPDs: name -> (intercept, weights, std, parents)."""
    total = np.zeros(len(data))
    for name, (intercept, weights, std, parents) in cpds.items():
        mean = intercept + sum(w * data[p].to_numpy() for w, p in zip(weights, parents))
        total += -0.5 * np.log(2 * np.pi * std ** 2) - (data[name].to_numpy() - mean) ** 2 / (
            2 * std ** 2)
    return total


def shd(design: list, learned: list) -> int:
    """Structural Hamming distance: missing + extra + reversed edges."""
    a, b = set(map(tuple, design)), set(map(tuple, learned))
    reversed_ = {(u, v) for u, v in a if (v, u) in b}
    missing = {e for e in a - b if (e[1], e[0]) not in b}
    extra = {e for e in b - a if (e[1], e[0]) not in a}
    return len(missing) + len(extra) + len(reversed_)


def skeleton_errors(design: list, learned: list) -> int:
    """Edges present in one graph but not the other, ignoring direction."""
    a = {frozenset(e) for e in design}
    b = {frozenset(e) for e in learned}
    return len(a ^ b)


def best_label_accuracy(true, pred, k: int) -> float:
    """Accuracy of decoded states under the best matching of state labels (Hungarian)."""
    from scipy.optimize import linear_sum_assignment

    true, pred = np.asarray(true), np.asarray(pred)
    m = max(k, int(true.max()) + 1)
    counts = np.zeros((m, m))
    np.add.at(counts, (pred, true), 1)
    rows, cols = linear_sum_assignment(-counts)
    return float(counts[rows, cols].sum() / max(len(true), 1))
'''

PLOT_CODE = r'''
# ======================================================================== plots
def draw_network(nodes: list, edges: list, extra=(), missing=(), title: str = "",
                 notes: dict | None = None, size=(720, 460), reversed_=()) -> "Image.Image":
    """Nodes in topological layers with arrows; extra (learned-only) edges green, missing
    (design-only) edges red and dashed, reversed edges orange (learned direction); notes:
    name -> text under the node."""
    w, h = size
    img = Image.new("RGB", size, (24, 26, 30))
    draw = ImageDraw.Draw(img)
    parents = {n: [u for u, v in (*edges, *extra, *reversed_) if v == n] for n in nodes}
    layer: dict = {}
    for _ in range(len(nodes) + 1):
        for n in nodes:
            layer[n] = 1 + max((layer.get(p, 0) for p in parents[n]), default=-1)
    layers: dict = {}
    for n in nodes:
        layers.setdefault(layer.get(n, 0), []).append(n)
    pos = {}
    depth = max(layers) + 1 if layers else 1
    for li, members in layers.items():
        for i, n in enumerate(sorted(members)):
            pos[n] = ((i + 1) * w / (len(members) + 1), 50 + li * (h - 100) / max(depth - 1, 1))

    def arrow(u, v, color, dashed=False):
        (x1, y1), (x2, y2) = pos[u], pos[v]
        d = max(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5, 1)
        ux, uy = (x2 - x1) / d, (y2 - y1) / d
        a, b = (x1 + ux * 30, y1 + uy * 18), (x2 - ux * 30, y2 - uy * 18)
        if dashed:
            steps = int(d // 12)
            for i in range(0, steps, 2):
                p = (a[0] + (b[0] - a[0]) * i / steps, a[1] + (b[1] - a[1]) * i / steps)
                q = (a[0] + (b[0] - a[0]) * (i + 1) / steps, a[1] + (b[1] - a[1]) * (i + 1) / steps)
                draw.line([p, q], fill=color, width=2)
        else:
            draw.line([a, b], fill=color, width=2)
        left = (b[0] - ux * 10 - uy * 6, b[1] - uy * 10 + ux * 6)
        right = (b[0] - ux * 10 + uy * 6, b[1] - uy * 10 - ux * 6)
        draw.polygon([b, left, right], fill=color)

    for u, v in edges:
        if u in pos and v in pos:
            arrow(u, v, (170, 170, 180))
    for u, v in extra:
        if u in pos and v in pos:
            arrow(u, v, (80, 220, 120))
    for u, v in missing:
        if u in pos and v in pos:
            arrow(u, v, (240, 90, 80), dashed=True)
    for u, v in reversed_:
        if u in pos and v in pos:
            arrow(u, v, (240, 170, 60))
    for n, (x, y) in pos.items():
        draw.rounded_rectangle([x - 34, y - 16, x + 34, y + 16], 8, fill=(52, 72, 110),
                               outline=(120, 150, 210))
        draw.text((x - 4 * len(n[:9]), y - 6), n[:9], fill=(240, 240, 240))
        if notes and n in notes:
            draw.text((x - 34, y + 20), notes[n][:28], fill=(190, 200, 160))
    draw.text((10, 8), title, fill=(220, 220, 220))
    return img


def draw_regimes(values: np.ndarray, states: np.ndarray, title: str = "",
                 size=(760, 240)) -> "Image.Image":
    w, h = size
    img = Image.new("RGB", size, (24, 26, 30))
    draw = ImageDraw.Draw(img)
    palette = [(90, 150, 240), (240, 120, 80), (110, 200, 120), (220, 190, 80),
               (190, 120, 220), (90, 200, 200)]
    lo, hi = float(np.min(values)), float(np.max(values))
    hi = hi if hi > lo else lo + 1
    n = len(values)
    xs = [10 + i * (w - 20) / max(n - 1, 1) for i in range(n)]
    ys = [h - 20 - (v - lo) / (hi - lo) * (h - 50) for v in values]
    for i in range(n - 1):
        draw.line([(xs[i], ys[i]), (xs[i + 1], ys[i + 1])],
                  fill=palette[int(states[i]) % len(palette)], width=1)
    draw.text((10, 6), title, fill=(220, 220, 220))
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
    for code in (DATA_CODE, SCORES_CODE, PLOT_CODE):
        exec(compile(code, "<pgm-runtime>", "exec"), ns)  # noqa: S102 — our own source
    return ns
