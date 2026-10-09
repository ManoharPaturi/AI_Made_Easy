"""Design rules for graphical models (the ``pgm`` family's validator).

Messages follow the app's Quick Fix phrasing where a fix exists ("Normalize the table"
for CPD columns that do not sum to one).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ai_made_easy.core.graph import Graph, ValidationIssue
from ai_made_easy.core.pgm import network
from ai_made_easy.core.pgm.blocks import CONFIG, VARIABLES, _names

EXPLOSION = 10_000


def _issue(severity: str, message: str, node_id: str | None = None) -> ValidationIssue:
    return ValidationIssue(severity, message, node_id)


@lru_cache(maxsize=16)
def _columns(path: str, fmt: str, _mtime: float) -> tuple[str, ...]:
    import pandas as pd

    if fmt == "parquet":
        return tuple(pd.read_parquet(path).columns)
    return tuple(pd.read_csv(path, sep="\t" if fmt == "tsv" else ",", nrows=5).columns)


def data_columns(node) -> tuple[str, ...] | None:  # noqa: ANN001
    """Column names of a table dataset (None when unknown or unreadable)."""
    if node is None or node.type_id != "data.csv":
        return None
    p = node.resolved_params()
    path = Path(str(p["path"])).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.is_file() or p.get("format") not in ("csv", "tsv", "parquet"):
        return None
    try:
        return _columns(str(path), p["format"], path.stat().st_mtime)
    except Exception:  # noqa: BLE001 — unreadable files are reported by the data profile
        return None


def _cycles(graph: Graph, edges: list) -> list[str]:
    parents: dict = {}
    for u, v in edges:
        parents.setdefault(v, set()).add(u)
    state: dict = {}

    def visit(n, path):
        if state.get(n) == 1:
            return path[path.index(n):] + [n]
        if state.get(n) == 2:
            return None
        state[n] = 1
        for p in sorted(parents.get(n, ())):
            found = visit(p, path + [n])
            if found:
                return found
        state[n] = 2
        return None

    for n in sorted({x for e in edges for x in e}):
        found = visit(n, [])
        if found:
            return list(reversed(found))
    return []


def pgm_issues(graph: Graph) -> list[ValidationIssue]:
    from ai_made_easy.core.pgm.tasks import dataset_of

    out: list[ValidationIssue] = []
    nodes = list(graph.nodes.values())
    configs: dict = {}
    for n in nodes:
        if n.type_id in CONFIG:
            if n.type_id in configs:
                out.append(_issue("error", f"only one {n.definition().display_name} block "
                                           "is allowed", n.instance_id))
            configs.setdefault(n.type_id, n)
    hmm = [n for n in nodes if n.type_id == "pgm.hmm"]
    var_nodes = [n for n in nodes if n.type_id in VARIABLES]
    data = dataset_of(graph)
    if hmm:
        if var_nodes:
            out.append(_issue("error", "a Hidden Markov Model stands alone: remove the "
                                       "variable blocks (or the HMM)", hmm[0].instance_id))
        if len(hmm) > 1:
            out.append(_issue("error", "one Hidden Markov Model per design",
                              hmm[1].instance_id))
        if data is None:
            out.append(_issue("error", "a Hidden Markov Model learns from data: add a table "
                                       "or a Regime Series", hmm[0].instance_id))
        cols = data_columns(data)
        if cols is not None:
            p = hmm[0].resolved_params()
            missing = [c for c in _names(p["observations"]) if c not in cols]
            if missing:
                out.append(_issue("error", f"observation column(s) {missing} are not in the "
                                           f"data ({', '.join(cols[:8])})", hmm[0].instance_id))
        return out
    if not var_nodes:
        anchor = next((n for n in nodes if n.type_id.startswith("pgm.")), None)
        return out + [_issue("error", "add Discrete or Gaussian Variable blocks and wire "
                                      "parents into children",
                             anchor.instance_id if anchor else None)]
    model = configs.get("pgm.model")
    kind = model.resolved_params()["kind"] if model else "bayesian_network"
    variables = network.variables(graph)
    seen: dict = {}
    for n in var_nodes:
        name = str(n.resolved_params()["name"]).strip()
        if not name or not name.replace("_", "").isalnum():
            out.append(_issue("error", f"variable name {name!r} must be letters, digits or _",
                              n.instance_id))
        if name in seen:
            out.append(_issue("error", f"two variables are named {name!r}: names identify "
                                       "data columns and must be unique", n.instance_id))
        seen[name] = n
    edges = network.edges(graph)
    kinds = {v.kind for v in variables.values()}
    if len(kinds) > 1:
        out.append(_issue("error", "mixing discrete and Gaussian variables is not supported: "
                                   "discretise the continuous ones or make them all "
                                   "Gaussian"))
    if kind != "markov_network":
        loop = _cycles(graph, edges)
        if loop:
            out.append(_issue("error", f"a {kind.replace('_', ' ')} must be acyclic: "
                                       f"{' → '.join(loop)} is a cycle (a Markov network "
                                       "allows loops)", seen[loop[0]].instance_id))
    structure = configs.get("pgm.structure_learning")
    params = configs.get("pgm.parameter_learning")
    estimator = params.resolved_params()["estimator"] if params else None
    for name, var in variables.items():
        node = seen[name]
        if var.kind == "gaussian":
            coefs = _names(var.params.get("coefficients"))
            if coefs and len(coefs) != len(var.parents):
                out.append(_issue("error", f"{name} has {len(var.parents)} parent(s) but "
                                           f"{len(coefs)} coefficient(s)", node.instance_id))
            if data is None and not coefs and var.parents:
                out.append(_issue("error", f"{name}: give one coefficient per parent, or add a "
                                           "dataset to learn them", node.instance_id))
            continue
        if var.latent:
            out.append(_issue("info", f"{name} is hidden: EM learns its states only up to "
                                      "relabelling (which learned state is "
                                      f"'{var.states[0] if var.states else ''}' is arbitrary)",
                              node.instance_id))
        if var.latent and estimator not in (None, "em"):
            out.append(_issue("error", f"{name} is hidden: learn it with the em estimator",
                              params.instance_id))
        parent_vars = [variables.get(p) for p in var.parents]
        if var.table.strip():
            if not var.states:
                out.append(_issue("error", f"{name} has a probability table but no states",
                                  node.instance_id))
                continue
            if any(p is None or not p.states for p in parent_vars):
                out.append(_issue("error", f"{name}'s parents need states before its table "
                                           "can be checked", node.instance_id))
                continue
            columns = network.column_labels([(p.name, p.states) for p in parent_vars])
            try:
                problems = network.table_problems(var, columns)
            except ValueError as exc:
                problems = [str(exc)]
            for problem in problems[:3]:
                hint = ": Normalize the table" if "sums to" in problem else ""
                out.append(_issue("error", f"{name}: {problem}{hint}", node.instance_id))
        elif data is None:
            out.append(_issue("error", f"{name} has no probability table and there is no "
                                       "dataset to learn it from", node.instance_id))
        combos = 1
        for p in parent_vars:
            combos *= max(len(p.states), 2) if p is not None else 2
        if combos > EXPLOSION:
            out.append(_issue("warning", f"{name} has {combos:,} parent-state combinations: "
                                         "its table needs a lot of data (consider fewer "
                                         "parents or states)", node.instance_id))
        for lag in var.lagged:
            if lag not in variables:
                out.append(_issue("error", f"lagged parent {lag!r} of {name} is not a "
                                           "variable", node.instance_id))
        if var.lagged and kind != "dynamic_bn":
            out.append(_issue("warning", f"{name} has lagged parents: they only apply to "
                                         "dynamic networks (set the model kind to "
                                         "dynamic_bn)", node.instance_id))
    connected = {x for e in edges for x in e}
    if kind in ("bayesian_network", "markov_network") and len(variables) > 1 \
            and structure is None:
        for name in variables:
            if name not in connected:
                out.append(_issue("warning", f"{name} is not connected: it is independent of "
                                             "every other variable", seen[name].instance_id))
    if kind == "naive_bayes":
        target = str(model.resolved_params()["class_variable"]).strip()
        if target not in variables:
            out.append(_issue("error", "naive Bayes needs the class variable: set "
                                       "class_variable to one of the variables",
                              model.instance_id))
        elif data is None:
            out.append(_issue("error", "naive Bayes learns from data: add a dataset",
                              model.instance_id))
    if kind == "markov_network" and data is None:
        out.append(_issue("error", "a Markov network learns its factors from data: add a "
                                   "dataset", model.instance_id))
    if structure is not None and any(v.table.strip() for v in variables.values()) and \
            (data is None or data.type_id != "data.network_sample"):
        out.append(_issue("info", "with structure learning every probability table is "
                                  "learned from the data; the tables in the design are "
                                  "ignored", structure.instance_id))
    if structure is not None and data is None:
        out.append(_issue("error", "structure learning needs data: add a dataset",
                          structure.instance_id))
    if data is not None and data.type_id == "data.network_sample":
        empty = [n for n, v in variables.items() if not v.table.strip()]
        if empty:
            out.append(_issue("error", f"Network Sample draws rows from the designed tables: "
                                       f"fill the table of {', '.join(empty[:4])}",
                              data.instance_id))
    cols = data_columns(data)
    if cols is not None:
        missing = [n for n, v in variables.items() if not v.latent and n not in cols]
        if missing:
            out.append(_issue("error", f"no data column for {', '.join(missing[:5])} (the data "
                                       f"has {', '.join(cols[:8])})", data.instance_id))
    def known(name: str) -> str:
        """Dynamic networks query 'X[t]' / 'X[t-1]'; the base name is the variable."""
        if kind == "dynamic_bn" and name.endswith(("[t]", "[t-1]")):
            return name[: name.index("[")]
        return name

    for n in nodes:
        if n.type_id != "pgm.query":
            continue
        p = n.resolved_params()
        for v in _names(p["variables"]):
            if kind == "dynamic_bn" and known(v) == v:
                out.append(_issue("error", f"in a dynamic network name the time step: "
                                           f"{v}[t] or {v}[t-1]", n.instance_id))
            elif known(v) not in variables:
                out.append(_issue("error", f"query variable {v!r} is not in the network",
                                  n.instance_id))
        for part in _names(p["evidence"]):
            if "=" not in part:
                out.append(_issue("error", f"evidence '{part}' must look like Variable=state",
                                  n.instance_id))
                continue
            key, value = (x.strip() for x in part.split("=", 1))
            var = variables.get(known(key))
            if var is None:
                out.append(_issue("error", f"evidence variable {key!r} is not in the network",
                                  n.instance_id))
            elif var.kind == "discrete" and var.states and value not in var.states:
                out.append(_issue("error", f"{key} has no state {value!r} (states: "
                                           f"{', '.join(var.states)})", n.instance_id))
            if key in _names(p["variables"]):
                out.append(_issue("error", f"{key} is both queried and observed",
                                  n.instance_id))
    return out
