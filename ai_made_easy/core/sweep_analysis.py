"""Which settings mattered in a sweep (3.6): parameter importance and the data for a
parallel-coordinates chart. Pure Python, shared by both apps.

Importance is the share of the score's variance each parameter explains on its own,
scaled so the parameters add up to 1. From 10 finished trials (and with scikit-learn) it
comes from functional ANOVA on a random forest fitted to the trials, the idea behind
fANOVA: each setting's own effect, plus how much each pair of settings explains together
beyond that, and the forest's predicted score over the strongest pair (a heatmap). With
fewer trials it is the correlation ratio (eta squared, numbers grouped into up to four
quantile bins), a quick model-free reading that ignores interactions. Either way, few
trials make it noisy, which the result says.

``narrowed`` (3.7) proposes a follow-up sweep around the best trials.
"""
from __future__ import annotations

import math
from typing import Any

MIN_TRIALS = 4
_BINS = 4


def _number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _label(key: str) -> str:
    """"opt.lr" -> "lr (opt)": the parameter first, then its block."""
    block, _, name = key.rpartition(".")
    return f"{name} ({block})" if block else key


def _eta_squared(groups: list[list[float]]) -> float:
    values = [v for g in groups for v in g]
    mean = sum(values) / len(values)
    total = sum((v - mean) ** 2 for v in values)
    if total <= 0:
        return 0.0
    between = sum(len(g) * (sum(g) / len(g) - mean) ** 2 for g in groups if g)
    return between / total


def _bins(values: list[float]) -> list[int]:
    """Up to four quantile bins (equal values stay together)."""
    distinct = sorted(set(values))
    if len(distinct) <= _BINS:
        return [distinct.index(v) for v in values]
    edges = [distinct[int(len(distinct) * q / _BINS)] for q in range(1, _BINS)]
    return [sum(v >= e for e in edges) for v in values]


def _numeric_trend(groups: dict[int, list[float]], spans: dict[int, list[float]],
                   direction: str) -> str:
    """Where the best scores were: at the low or high end of the range, or in between."""
    if _eta_squared(list(groups.values())) < 0.1 or len(groups) < 2:
        return ""
    means = {b: sum(g) / len(g) for b, g in groups.items()}
    best = min(means, key=means.get) if direction == "min" else max(means, key=means.get)
    order = sorted(groups)
    if best == order[0]:
        return "lower values scored better"
    if best == order[-1]:
        return "higher values scored better"
    low, high = min(spans[best]), max(spans[best])
    return f"best at {low:.3g}" if low == high else f"best between {low:.3g} and {high:.3g}"


def analyze(record: Any) -> dict:
    """``record``: a SweepRecord or its dict. Returns the axes and rows of a parallel-
    coordinates chart, and each parameter's importance (largest first)."""
    data = record if isinstance(record, dict) else record.to_dict()
    spec = data.get("spec") or {}
    metric = spec.get("metric", "score")
    direction = spec.get("direction") or ""
    if direction not in ("min", "max"):
        from ai_made_easy.core.runs.history import lower_is_better

        direction = "min" if lower_is_better(metric) else "max"
    trials = [t for t in data.get("trials") or []
              if t.get("state") == "finished" and _number(t.get("score"))]
    best = (data.get("best") or {}).get("number")
    keys = list(dict.fromkeys(k for t in trials for k in (t.get("values") or {})))
    axes = []
    for key in keys:
        values = [t["values"].get(key) for t in trials]
        if all(_number(v) for v in values if v is not None) and any(v is not None
                                                                     for v in values):
            nums = [float(v) for v in values if v is not None]
            low, high = min(nums), max(nums)
            axes.append({"key": key, "label": _label(key), "kind": "number", "min": low,
                         "max": high, "log": low > 0 and high / low >= 20})
        else:
            choices = list(dict.fromkeys(str(v) for v in values if v is not None))
            axes.append({"key": key, "label": _label(key), "kind": "choice",
                         "choices": choices})
    scores = [float(t["score"]) for t in trials]
    rows = [{"number": t["number"], "run_id": t.get("run_id", ""),
             "values": t.get("values") or {}, "score": float(t["score"]),
             "best": t["number"] == best} for t in trials]
    out = {"metric": metric, "direction": direction, "trials": len(trials), "axes": axes,
           "score_axis": ({"min": min(scores), "max": max(scores)} if scores else None),
           "rows": rows, "importance": [], "note": "", "method": "eta2",
           "interactions": [], "heatmap": None}
    if len(trials) < MIN_TRIALS:
        out["note"] = (f"Importance needs at least {MIN_TRIALS} finished trials "
                       f"({len(trials)} so far).")
        return out
    raw = []
    for axis in axes:
        values = [t["values"].get(axis["key"]) for t in trials]
        if axis["kind"] == "number":
            known = [(float(v), s) for v, s in zip(values, scores, strict=True) if v is not None]
            if len({v for v, _ in known}) < 2:
                continue
            bins = _bins([v for v, _ in known])
            groups: dict[int, list[float]] = {}
            spans: dict[int, list[float]] = {}
            for b, (v, s) in zip(bins, known, strict=True):
                groups.setdefault(b, []).append(s)
                spans.setdefault(b, []).append(v)
            trend = _numeric_trend(groups, spans, direction)
        else:
            groups = {}
            for v, s in zip(values, scores, strict=True):
                if v is not None:          # AutoML: a setting only some recipes have
                    groups.setdefault(str(v), []).append(s)
            if len(groups) < 2:
                continue
            means = {k: sum(g) / len(g) for k, g in groups.items()}
            pick = min(means, key=means.get) if direction == "min" else max(means,
                                                                             key=means.get)
            trend = f"best with {pick}"
        raw.append((axis, _eta_squared(list(groups.values())), trend))
    forest = forest_importance(axes, trials, scores)
    if forest is not None:
        main = dict(zip((a["key"] for a in forest["axes"]), forest["main"], strict=True))
        raw = [(axis, main.get(axis["key"], 0.0), trend) for axis, _r, trend in raw]
        out["method"] = "forest"
        out["interactions"] = forest["pairs"]
        out["heatmap"] = forest["heatmap"]
    total = sum(r for _, r, _ in raw)
    out["importance"] = sorted(
        ({"key": a["key"], "label": a["label"], "importance": (r / total) if total else 0.0,
          "explained": r, "trend": trend} for a, r, trend in raw),
        key=lambda d: -d["importance"])
    if len(trials) < 10:
        out["note"] = (f"Based on {len(trials)} trials: a rough guide; more trials make it "
                       "steadier.")
    return out


# ------------------------------------------------------------------ interactions (3.7)

FOREST_MIN = 10        # trials before the forest-based reading replaces the simple one
_GRID = 8              # grid points per setting for the marginal predictions


def _encode(axes: list[dict], trials: list[dict]):
    """One numeric column per axis: numbers (log10 when the axis is log), choices as their
    index; a missing value (AutoML: a setting only some recipes have) as the column's
    median for numbers or an extra "missing" index for choices."""
    import numpy as np

    cols = []
    for axis in axes:
        values = [t["values"].get(axis["key"]) for t in trials]
        if axis["kind"] == "number":
            nums = [math.log10(v) if axis["log"] else float(v) if v is not None else None
                    for v in values]
            known = [v for v in nums if v is not None]
            fill = sorted(known)[len(known) // 2]
            cols.append([fill if v is None else v for v in nums])
        else:
            choices = axis["choices"]
            cols.append([float(choices.index(str(v))) if v is not None else float(len(choices))
                         for v in values])
    return np.array(cols, dtype=float).T


def _grid(column) -> list[float]:
    import numpy as np

    distinct = np.unique(column)
    if len(distinct) <= _GRID:
        return [float(v) for v in distinct]
    return [float(v) for v in np.quantile(column, np.linspace(0, 1, _GRID))]


def _marginal(forest, x, cols: tuple[int, ...], grids: list[list[float]]):
    """The forest's average prediction with ``cols`` fixed to every grid combination
    (the other settings as they were in the trials)."""
    import itertools

    import numpy as np

    out = np.zeros([len(g) for g in grids])
    for idx in itertools.product(*(range(len(g)) for g in grids)):
        probe = x.copy()
        for c, g, i in zip(cols, grids, idx, strict=True):
            probe[:, c] = g[i]
        out[idx] = forest.predict(probe).mean()
    return out


def forest_importance(axes: list[dict], trials: list[dict], scores: list[float],
                      seed: int = 0) -> dict | None:
    """Functional ANOVA on a random forest fitted to the trials (the idea behind fANOVA,
    Hutter et al., 2014): the share of the predicted score's variance that each setting
    explains on its own, and that each pair explains together beyond that. None when
    scikit-learn is missing or there are too few usable trials."""
    try:
        import numpy as np
        from sklearn.ensemble import RandomForestRegressor
    except ImportError:
        return None
    usable = [i for i, a in enumerate(axes) if a["kind"] == "number" and a["max"] > a["min"]
              or a["kind"] == "choice" and len(a["choices"]) > 1]
    if len(trials) < FOREST_MIN or not usable:
        return None
    axes = [axes[i] for i in usable]
    x = _encode(axes, trials)
    y = np.array(scores, dtype=float)
    forest = RandomForestRegressor(n_estimators=64, min_samples_leaf=2, random_state=seed)
    forest.fit(x, y)
    total = float(np.var(forest.predict(x)))
    if total <= 1e-12:
        return {"main": [0.0] * len(axes), "pairs": [], "axes": axes, "heatmap": None}
    grids = [_grid(x[:, c]) for c in range(len(axes))]
    singles = [_marginal(forest, x, (c,), [grids[c]]) for c in range(len(axes))]
    main = [float(np.var(s)) / total for s in singles]
    order = sorted(range(len(axes)), key=lambda c: -main[c])[:5]
    pairs, tables = [], {}
    for a_i, a in enumerate(order):
        for b in order[a_i + 1:]:
            table = _marginal(forest, x, (a, b), [grids[a], grids[b]])
            joint = float(np.var(table))
            alone = float(np.var(table.mean(axis=1))) + float(np.var(table.mean(axis=0)))
            pairs.append((a, b, max(joint - alone, 0.0) / total))
            tables[(a, b)] = table
    pairs.sort(key=lambda p: -p[2])
    heatmap = None
    if len(order) >= 2:
        a, b = (pairs[0][0], pairs[0][1]) if pairs and pairs[0][2] >= 0.05 else sorted(order[:2])
        table = tables.get((a, b)) if (a, b) in tables else tables.get((b, a)).T
        heatmap = {"x": _axis_values(axes[a], grids[a]), "y": _axis_values(axes[b], grids[b]),
                   "x_key": axes[a]["key"], "y_key": axes[b]["key"],
                   "x_label": axes[a]["label"], "y_label": axes[b]["label"],
                   "z": [[float(v) for v in row] for row in table.T]}   # z[y][x]
    return {"main": main, "axes": axes, "heatmap": heatmap,
            "pairs": [{"keys": [axes[a]["key"], axes[b]["key"]],
                       "labels": [axes[a]["label"], axes[b]["label"]], "share": s}
                      for a, b, s in pairs]}


def _axis_values(axis: dict, grid: list[float]) -> list:
    if axis["kind"] == "choice":
        names = [*axis["choices"], "(missing)"]
        return [names[int(round(v))] for v in grid]
    return [float(f"{10 ** v if axis['log'] else v:.4g}") for v in grid]


# ------------------------------------------------------------------ sweep again (3.7)

TOP_SHARE = 0.25       # the best quarter of the finished trials


def narrowed(record: Any) -> dict:
    """A follow-up sweep around the best trials: each number range shrinks to where the
    best quarter (at least 3) of the finished trials were, padded by 10% of that span (in
    log space for log ranges) and kept inside the old range; choices keep only the values
    those trials used. Returns ``{"spec", "graph", "changes", "note"}``; the caller shows
    it for review before starting."""
    data = record if isinstance(record, dict) else record.to_dict()
    spec = dict(data.get("spec") or {})
    if spec.get("automl"):
        raise ValueError("AutoML searches recipes, not ranges: start a sweep from the best "
                         "design instead")
    info = analyze(data)
    rows = info["rows"]
    if len(rows) < MIN_TRIALS:
        raise ValueError(f"Sweeping again needs at least {MIN_TRIALS} finished trials "
                         f"({len(rows)} so far)")
    ranked = sorted(rows, key=lambda r: r["score"], reverse=info["direction"] == "max")
    top = ranked[:max(3, math.ceil(len(ranked) * TOP_SHARE))]
    dims, changes = [], []
    for dim in spec.get("dimensions") or []:
        dim = dict(dim)
        key = f"{dim['node']}.{dim['param']}"
        label = _label(key)
        values = [r["values"].get(key) for r in top if r["values"].get(key) is not None]
        if dim.get("kind") == "choice" or dim.get("values"):
            kept = [v for v in (dim.get("values") or []) if v in values] or list(dim["values"])
            if len(kept) < len(dim["values"]):
                changes.append({"key": key, "label": label, "before": list(dim["values"]),
                                "after": kept})
            dim["values"] = kept
        elif values and all(_number(v) for v in values):
            lo, hi = float(dim["low"]), float(dim["high"])
            log = bool(dim.get("log")) and lo > 0
            f = math.log10 if log else (lambda v: v)
            inv = (lambda v: 10 ** v) if log else (lambda v: v)
            a, b = f(min(values)), f(max(values))
            pad = 0.1 * (b - a) if b > a else 0.1 * (f(hi) - f(lo))
            new_lo, new_hi = max(f(lo), a - pad), min(f(hi), b + pad)
            new_lo, new_hi = inv(new_lo), inv(new_hi)
            if dim.get("kind") == "int":
                new_lo, new_hi = math.floor(new_lo), math.ceil(new_hi)
            else:
                new_lo, new_hi = float(f"{new_lo:.4g}"), float(f"{new_hi:.4g}")
            if (new_lo, new_hi) != (lo, hi):
                changes.append({"key": key, "label": label, "before": [lo, hi],
                                "after": [new_lo, new_hi]})
            dim["low"], dim["high"] = new_lo, new_hi
        dims.append(dim)
    spec["dimensions"] = dims
    note = (f"Ranges narrowed to the best {len(top)} of {len(rows)} trials."
            if changes else "The best trials already span the whole ranges: nothing to narrow.")
    return {"spec": spec, "graph": data.get("graph") or {}, "changes": changes, "note": note}
