"""Design rules for state-space models (the ``ssm`` family's validator)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ai_made_easy.core.graph import Graph, ValidationIssue
from ai_made_easy.core.ssm.blocks import COMPONENTS
from ai_made_easy.core.ssm.template import (
    _names,
    dataset_of,
    model_node,
    target_of,
    wired,
)

SLOW_LENGTH = 50_000


def _issue(severity: str, message: str, node_id: str | None = None) -> ValidationIssue:
    return ValidationIssue(severity, message, node_id)


def _label(node) -> str:  # noqa: ANN001
    return node.definition().display_name


@lru_cache(maxsize=16)
def _header(path: str, mtime: float) -> tuple[tuple[str, ...], int]:
    import pandas as pd

    frame = pd.read_csv(path, usecols=None)
    return tuple(str(c) for c in frame.columns), len(frame)


def _table(node) -> tuple[tuple[str, ...], int] | None:  # noqa: ANN001
    """(columns, rows) of a time-series CSV, or None when it cannot be read."""
    path = Path(str(node.resolved_params()["path"])).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.is_file():
        return None
    try:
        return _header(str(path), path.stat().st_mtime)
    except Exception:  # noqa: BLE001 — unreadable files are reported by the data profile
        return None


def ssm_issues(graph: Graph) -> list[ValidationIssue]:
    out: list[ValidationIssue] = []
    models = [n for n in graph.nodes.values() if n.type_id == "ssm.model"]
    if not models:
        anchor = next((n for n in graph.nodes.values() if n.type_id.startswith("ssm.")), None)
        return [_issue("error", "add a State-Space Model block and wire components into it",
                       anchor.instance_id if anchor else None)]
    if len(models) > 1:
        out.append(_issue("error", "one State-Space Model per design", models[1].instance_id))
    model = model_node(graph)
    p = model.resolved_params()
    parts = wired(graph)
    if not parts:
        return out + [_issue("error", "wire a Level / Trend (or a SARIMAX Specification) into "
                                      "the State-Space Model", model.instance_id)]
    wired_ids = {n.instance_id for n in parts}
    for node in graph.nodes.values():
        if node.type_id in (*COMPONENTS, "ssm.arima") and node.instance_id not in wired_ids:
            out.append(_issue("warning", f"{_label(node)} is not wired into the State-Space "
                                         "Model: it has no effect", node.instance_id))
    seen: dict[str, str] = {}
    for node in parts:
        if node.type_id in seen:
            out.append(_issue("error", f"one {_label(node)} per model: combine them into "
                                       "one block", node.instance_id))
        seen[node.type_id] = node.instance_id
    if "ssm.arima" in seen:
        mixed = [n for n in parts if n.type_id not in ("ssm.arima", "ssm.regression")]
        for node in mixed:
            out.append(_issue("error", f"{_label(node)} is a structural component: a SARIMAX "
                                       "Specification cannot be combined with it (keep one "
                                       "kind of model)", node.instance_id))
        a = graph.nodes[seen["ssm.arima"]].resolved_params()
        if int(a["season"]) <= 1 and any(int(a[k]) for k in ("seasonal_p", "seasonal_d",
                                                             "seasonal_q")):
            out.append(_issue("warning", "seasonal terms need a season: set season to the "
                                         "period (e.g. 12)", seen["ssm.arima"]))
    elif "ssm.trend" not in seen:
        out.append(_issue("info", "no Level / Trend wired: the model uses a fixed intercept",
                          model.instance_id))

    data = dataset_of(graph)
    other = next((n for n in graph.nodes.values()
                  if n.type_id in ("data.synthetic_series", "data.csv")), None)
    if data is None and other is not None:
        out.append(_issue("error", "state-space models fit one series: use Structural Series "
                                   "or a Time-Series CSV", other.instance_id))
    exog = (_names(graph.nodes[seen["ssm.regression"]].resolved_params()["columns"])
            if "ssm.regression" in seen else [])
    if "ssm.regression" in seen and not exog:
        out.append(_issue("error", "list the explanatory columns of the Regression Component",
                          seen["ssm.regression"]))
    length = None
    if data is not None and data.type_id == "data.structural_series":
        dp = data.resolved_params()
        length = int(dp["length"])
        if exog:
            out.append(_issue("error", "Structural Series has no explanatory columns: remove "
                                       "the Regression Component or use a Time-Series CSV",
                              seen["ssm.regression"]))
    elif data is not None:
        table = _table(data)
        if table is not None:
            columns, length = table
            target = target_of(graph)
            if target not in columns:
                out.append(_issue("error", f"the series column {target!r} is not in the data: "
                                           f"set target_column to one of {list(columns)[:8]}",
                                  model.instance_id))
            missing = [c for c in exog if c not in columns]
            if missing:
                out.append(_issue("error", f"regression column(s) {missing} are not in the "
                                           "data", seen["ssm.regression"]))
    if length is None and data is None:
        length = 240
    horizon = int(p["horizon"])
    if length is not None:
        if length <= horizon + 10:
            out.append(_issue("error", f"the series has {length} steps: set horizon to "
                                       f"{max(1, length // 5)}", model.instance_id))
        if length > SLOW_LENGTH:
            out.append(_issue("warning", f"{length:,} steps: the Kalman filter is O(n) per "
                                         "likelihood evaluation, fitting may take minutes",
                              model.instance_id))
        if "ssm.seasonal" in seen:
            period = int(graph.nodes[seen["ssm.seasonal"]].resolved_params()["period"])
            if period * 2 > length - horizon:
                out.append(_issue("warning", f"a {period}-step season needs at least two "
                                             "seasons of training data",
                                  seen["ssm.seasonal"]))
    if "ssm.seasonal" in seen:
        s = graph.nodes[seen["ssm.seasonal"]].resolved_params()
        if int(s["harmonics"]) * 2 > int(s["period"]):
            out.append(_issue("warning", f"set harmonics to {int(s['period']) // 2}: more than "
                                         "period / 2 harmonics repeat each other",
                              seen["ssm.seasonal"]))
        if int(s["period"]) > 100 and int(s["harmonics"]) == 0:
            out.append(_issue("info", "a long season with one effect per step adds "
                                      f"{s['period']} states: set harmonics to 6 for a smooth "
                                      "shape", seen["ssm.seasonal"]))
    return out
