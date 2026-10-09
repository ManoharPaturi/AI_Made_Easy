"""The network a graphical-model design describes: variables, their ordered parents and
states, and the layout of each conditional probability table (CPD).

Shared by code generation, the design rules and the CPD table editors (desktop and web).
CPD tables follow pgmpy: one row per state of the variable, one column per combination
of parent states, the last parent varying fastest.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field

from ai_made_easy.core.pgm.blocks import VARIABLES, _names

MAX_COLUMNS = 10_000   # CPD columns before a table becomes impractical to fill or learn


@dataclass
class Variable:
    node_id: str
    name: str
    kind: str                      # discrete | gaussian
    states: list[str]
    parents: list[str] = field(default_factory=list)   # parent variable names, ordered
    table: str = ""                # raw CPD param (JSON)
    latent: bool = False
    lagged: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)


def variables(graph) -> dict[str, Variable]:  # noqa: ANN001
    """Variables by name; parents sorted by name so tables have a stable column order."""
    out: dict[str, Variable] = {}
    by_id = {}
    for node in graph.nodes.values():
        if node.type_id not in VARIABLES:
            continue
        p = dict(node.resolved_params())
        var = Variable(node.instance_id, str(p["name"]).strip(),
                       "discrete" if node.type_id == "pgm.variable" else "gaussian",
                       _names(p.get("states")), table=str(p.get("cpd") or ""),
                       latent=bool(p.get("latent")), lagged=_names(p.get("lagged_parents")),
                       params=p)
        by_id[node.instance_id] = var
        out.setdefault(var.name, var)
    for var in by_id.values():
        parents = {by_id[e.source_id].name for e in graph.incoming(var.node_id)
                   if e.source_id in by_id}
        var.parents = sorted(parents)
    return out


def edges(graph) -> list[tuple[str, str]]:  # noqa: ANN001
    by_id = {n.instance_id: str(n.resolved_params()["name"]).strip()
             for n in graph.nodes.values() if n.type_id in VARIABLES}
    return sorted({(by_id[e.source_id], by_id[e.target_id]) for e in graph.edges
                   if e.source_id in by_id and e.target_id in by_id})


def column_labels(parents: list[tuple[str, list[str]]]) -> list[str]:
    if not parents:
        return ["P"]
    combos = itertools.product(*[states for _n, states in parents])
    return [", ".join(f"{n}={s}" for (n, _st), s in zip(parents, combo, strict=True))
            for combo in combos]


def parse_table(text: str) -> list[list[float]] | None:
    """A CPD param -> rows of floats (None when empty); raises ValueError when malformed."""
    if not str(text or "").strip():
        return None
    try:
        rows = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"the table is not valid JSON ({exc.msg})") from None
    if isinstance(rows, list) and rows and not isinstance(rows[0], list):
        rows = [[v] for v in rows]          # a root variable written as one column
    if not (isinstance(rows, list) and rows and all(isinstance(r, list) for r in rows)):
        raise ValueError("the table must be a list of rows")
    try:
        return [[float(v) for v in row] for row in rows]
    except (TypeError, ValueError):
        raise ValueError("table entries must be numbers") from None


def layout(graph, node_id: str) -> dict:  # noqa: ANN001
    """What a CPD editor shows for one variable: row / column labels and current values
    (uniform when the table is empty or no longer fits the wiring)."""
    vars_ = variables(graph)
    var = next((v for v in vars_.values() if v.node_id == node_id), None)
    if var is None or var.kind != "discrete":
        return {"error": "only discrete variables have probability tables"}
    if not var.states:
        return {"error": "set this variable's states first (e.g. low, high)"}
    parents = []
    for name in var.parents:
        parent = vars_.get(name)
        if parent is None or parent.kind != "discrete" or not parent.states:
            return {"error": f"set the states of parent '{name}' first"}
        parents.append((name, parent.states))
    columns = column_labels(parents)
    if len(columns) > MAX_COLUMNS:
        return {"error": f"{len(columns):,} parent combinations: too many to fill by hand; "
                         "learn this table from data"}
    rows = len(var.states)
    try:
        values = parse_table(var.table)
    except ValueError:
        values = None
    fits = values is not None and len(values) == rows and all(len(r) == len(columns)
                                                               for r in values)
    if not fits:
        values = [[round(1.0 / rows, 6)] * len(columns) for _ in range(rows)]
    return {"variable": var.name, "rows": var.states, "columns": columns,
            "parents": var.parents, "values": values, "fits": fits,
            "learned": not var.table.strip()}


def table_problems(var: Variable, columns: list[str]) -> list[str]:
    """Shape and normalisation problems of a filled CPD."""
    values = parse_table(var.table)
    if values is None:
        return []
    if len(values) != len(var.states):
        return [f"the table has {len(values)} row(s) but {var.name} has {len(var.states)} "
                "state(s)"]
    bad = [i for i, r in enumerate(values) if len(r) != len(columns)]
    if bad:
        return [f"the table has {len(values[bad[0]])} column(s) but the parents give "
                f"{len(columns)} combination(s)"]
    out = []
    for j, label in enumerate(columns):
        column = [values[i][j] for i in range(len(values))]
        if any(v < 0 for v in column):
            out.append(f"column '{label}' has a negative probability")
        elif abs(sum(column) - 1) > 1e-4:
            out.append(f"column '{label}' sums to {sum(column):.4g}, not 1")
    return out


def normalized(text: str) -> str:
    """The table with every column rescaled to sum to 1 (negative entries clipped)."""
    values = parse_table(text) or []
    if not values:
        return text
    cols = len(values[0])
    for j in range(cols):
        total = sum(max(values[i][j], 0.0) for i in range(len(values)))
        for i in range(len(values)):
            values[i][j] = round(max(values[i][j], 0.0) / total, 8) if total else \
                round(1.0 / len(values), 8)
    return json.dumps(values)
