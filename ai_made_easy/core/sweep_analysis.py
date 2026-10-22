"""Which settings mattered in a sweep (3.6): parameter importance and the data for a
parallel-coordinates chart. Pure Python, shared by both apps.

Importance is the share of the score's variance each parameter explains on its own (the
correlation ratio, eta squared: numbers are grouped into up to four quantile bins first),
scaled so the parameters add up to 1. It is a quick, model-free reading for a handful of
trials, not fANOVA: it ignores interactions, and with few trials it is noisy, which the
result says.
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
    return f"best between {min(spans[best]):.3g} and {max(spans[best]):.3g}"


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
           "rows": rows, "importance": [], "note": ""}
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
    total = sum(r for _, r, _ in raw)
    out["importance"] = sorted(
        ({"key": a["key"], "label": a["label"], "importance": (r / total) if total else 0.0,
          "explained": r, "trend": trend} for a, r, trend in raw),
        key=lambda d: -d["importance"])
    if len(trials) < 10:
        out["note"] = (f"Based on {len(trials)} trials: a rough guide; more trials make it "
                       "steadier.")
    return out
