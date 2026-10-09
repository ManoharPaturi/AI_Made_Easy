"""Design rules for probabilistic programs (the ``ppl`` family's validator)."""
from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path

from ai_made_easy.core.graph import Graph, ValidationIssue
from ai_made_easy.core.ppl.blocks import BY_ID
from ai_made_easy.core.ppl.model import ModelError, parse_expression, resolve, variables

SYNTHETIC_COLUMNS = {"group": "category", "x": "real", "y": None}
LINKS = {"exp": "positive", "softplus": "positive", "sqrt": "positive", "abs": "positive",
         "invlogit": "unit", "sigmoid": "unit", "invprobit": "unit"}
FITS = {"positive": ("positive", "unit"), "unit": ("unit",), "count": ("count", "binary"),
        "real": ("real", "positive", "unit", "count", "binary", "category"),
        "vector": ("vector",)}


def _issue(severity: str, message: str, node_id: str | None = None) -> ValidationIssue:
    return ValidationIssue(severity, message, node_id)


@lru_cache(maxsize=16)
def _sample(path: str, fmt: str, _mtime: float):
    import pandas as pd

    if fmt == "parquet":
        return pd.read_parquet(path).head(2000)
    return pd.read_csv(path, sep="\t" if fmt == "tsv" else ",", nrows=2000)


def data_frame(node):  # noqa: ANN001, ANN201
    """The first rows of a table dataset (None when unknown or unreadable)."""
    if node is None or node.type_id != "data.csv":
        return None
    p = node.resolved_params()
    path = Path(str(p["path"])).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.is_file() or p.get("format") not in ("csv", "tsv", "parquet"):
        return None
    try:
        return _sample(str(path), p["format"], path.stat().st_mtime)
    except Exception:  # noqa: BLE001 — unreadable files are reported by the data profile
        return None


def _support_of(var) -> str:  # noqa: ANN001
    if var.kind == "distribution":
        return BY_ID[var.type_id].support
    try:
        tree, _ = parse_expression(var.expression)
    except ModelError:
        return "real"
    body = tree.body
    if isinstance(body, ast.Call) and isinstance(body.func, ast.Name):
        return LINKS.get(body.func.id, "real")
    return "real"


def _column_kind(values) -> str:  # noqa: ANN001
    import pandas as pd

    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        return "category"
    if set(numeric.unique()) <= {0, 1}:
        return "binary"
    if (numeric >= 0).all() and (numeric == numeric.round()).all():
        return "count"
    if (numeric > 0).all() and (numeric < 1).all():
        return "unit"
    if (numeric > 0).all():
        return "positive"
    return "real"


def ppl_issues(graph: Graph) -> list[ValidationIssue]:
    from ai_made_easy.core.ppl.tasks import dataset_of

    out: list[ValidationIssue] = []
    samplers = [n for n in graph.nodes.values() if n.type_id == "ppl.sampler"]
    if len(samplers) > 1:
        out.append(_issue("error", "only one Sampler block is allowed", samplers[1].instance_id))
    def node_of(exc: ModelError) -> str | None:
        return next((n.instance_id for n in graph.nodes.values()
                     if exc.variable and str(n.resolved_params().get("name", "")).strip()
                     == exc.variable), None)

    try:
        vars_ = variables(graph)
    except ModelError as exc:
        return out + [_issue("error", str(exc), node_of(exc))]
    if not vars_:
        anchor = samplers[0].instance_id if samplers else None
        return out + [_issue("error", "add distribution blocks: priors, and a likelihood with "
                                      "observed set to a data column", anchor)]
    by_name = {v.name: v for v in vars_}
    data = dataset_of(graph)
    frame = data_frame(data)
    columns = None
    if frame is not None:
        columns = set(frame.columns)
    elif data is not None and data.type_id == "data.synthetic_groups":
        columns = set(SYNTHETIC_COLUMNS)
    try:
        info = resolve(vars_, columns)
    except ModelError as exc:
        return out + [_issue("error", str(exc), node_of(exc))]
    if (info["observed"] or info["columns"]) and data is None:
        out.append(_issue("error", "observed variables need data: add a table or Synthetic "
                                   "Groups", next(v.node_id for v in vars_ if v.observed)
                          if info["observed"] else None))
    if not info["observed"]:
        out.append(_issue("warning", "no variable is observed (set observed to a data column "
                                     "on the likelihood): sampling would show the prior only",
                          vars_[-1].node_id))
    for g in info["groups"]:
        if columns is not None and g not in columns:
            node = next(v.node_id for v in vars_ if v.group == g)
            out.append(_issue("error", f"group column {g!r} is not in the data", node))
    for v in vars_:
        if v.kind != "distribution":
            continue
        dist = BY_ID[v.type_id]
        for pname, _default, support in dist.params:
            kind, value = v.params[pname]
            if kind != "var":
                continue
            parent_support = _support_of(by_name[value])
            if support in FITS and parent_support not in FITS[support]:
                hint = {"positive": "use HalfNormal / Exponential / Gamma, or exp() / "
                                    "softplus() in a Deterministic",
                        "unit": "use a Beta prior, or invlogit() in a Deterministic",
                        "count": "use a count distribution"}.get(support, "")
                out.append(_issue("warning", f"{v.name}.{pname} must be {support} but "
                                             f"{value} can be {parent_support}: {hint}",
                                  v.node_id))
        if v.observed:
            if columns is not None and v.observed not in columns:
                out.append(_issue("error", f"observed column {v.observed!r} is not in the data",
                                  v.node_id))
            elif frame is not None:
                have = _column_kind(frame[v.observed])
                need = dist.support
                if need == "category":
                    continue
                if need in ("count", "binary") and have not in (
                        ("count", "binary") if need == "count" else ("binary",)):
                    out.append(_issue("error", f"{dist.label} explains "
                                               f"{'0 / 1 values' if need == 'binary' else 'counts'}"
                                               f" but {v.observed!r} has {have} values",
                                      v.node_id))
                elif need in ("positive", "unit") and have not in FITS[need]:
                    out.append(_issue("error", f"{dist.label} needs {need} values but "
                                               f"{v.observed!r} has {have} values", v.node_id))
        elif dist.support in ("count", "binary", "category") and samplers and \
                samplers[0].resolved_params()["method"] == "nuts":
            out.append(_issue("info", f"{v.name} is a discrete latent variable: PyMC samples "
                                      "it with Metropolis steps alongside NUTS (slower "
                                      "mixing)", v.node_id))
    return out
