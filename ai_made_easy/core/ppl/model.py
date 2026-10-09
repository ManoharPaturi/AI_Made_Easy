"""From a probabilistic-program design to PyMC model code.

Variables are the distribution / deterministic blocks; a parameter port that is wired
takes the parent variable, otherwise the block's constant. Expressions are parsed with
``ast`` (arithmetic, a fixed set of functions, variable and data-column names only) and
rewritten to PyMC. A variable with ``group`` has one value per level of that column;
used inside a row-level variable it is indexed by the rows' group codes.
"""
from __future__ import annotations

import ast
import keyword
import re
from dataclasses import dataclass, field

from ai_made_easy.core.ppl.blocks import BY_ID

FUNCTIONS = {"exp": "pm.math.exp", "log": "pm.math.log", "sqrt": "pm.math.sqrt",
             "abs": "pm.math.abs", "invlogit": "pm.math.invlogit",
             "sigmoid": "pm.math.invlogit", "logit": "pm.math.logit",
             "softplus": "pm.math.log1pexp", "invprobit": "pm.math.invprobit"}
RESERVED = {"pm", "np", "pd", "data", "model", "rows", "math", *FUNCTIONS}
_ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
            ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd)


class ModelError(ValueError):
    """The design cannot be turned into a PyMC model (message for the user); ``variable``
    names the variable at fault when there is one."""

    def __init__(self, message: str, variable: str | None = None) -> None:
        super().__init__(message)
        self.variable = variable


@dataclass
class Var:
    node_id: str
    name: str
    kind: str                         # distribution | deterministic
    type_id: str
    support: str
    params: dict = field(default_factory=dict)       # param -> ("const", v) | ("var", name)
    expression: str = ""
    names: set = field(default_factory=set)          # names an expression references
    observed: str = ""
    group: str = ""
    size: int = 0
    non_centered: bool = False
    parents: list = field(default_factory=list)
    row_level: bool = False


def identifier(column: str) -> str:
    return "data_" + re.sub(r"\W+", "_", column).strip("_")


def parse_expression(text: str) -> tuple[ast.Expression, set[str]]:
    try:
        tree = ast.parse(text.strip() or "0", mode="eval")
    except SyntaxError as exc:
        raise ModelError(f"the expression is not valid: {exc.msg}") from None
    names: set[str] = set()
    for sub in ast.walk(tree):
        if not isinstance(sub, _ALLOWED):
            raise ModelError(f"'{ast.unparse(sub)}' is not allowed in an expression "
                             "(arithmetic, functions and names only)")
        if isinstance(sub, ast.Call):
            if not isinstance(sub.func, ast.Name) or sub.func.id not in FUNCTIONS:
                raise ModelError(f"unknown function in '{ast.unparse(sub)}'; available: "
                                 f"{', '.join(sorted(FUNCTIONS))}")
        elif isinstance(sub, ast.Name) and sub.id not in FUNCTIONS:
            names.add(sub.id)
    return tree, names


def variables(graph) -> list[Var]:  # noqa: ANN001
    """Variables in dependency order (parents first)."""
    nodes = {n.instance_id: n for n in graph.nodes.values()
             if n.type_id in BY_ID or n.type_id == "ppl.deterministic"}
    out: dict[str, Var] = {}
    for nid, node in nodes.items():
        p = dict(node.resolved_params())
        if node.type_id == "ppl.deterministic":
            try:
                _tree, names = parse_expression(str(p["expression"]))
            except ModelError as exc:
                raise ModelError(f"{p['name']}: {exc}", str(p["name"]).strip()) from None
            var = Var(nid, str(p["name"]).strip(), "deterministic", node.type_id, "real",
                      expression=str(p["expression"]), names=names,
                      group=str(p.get("group") or "").strip())
        else:
            dist = BY_ID[node.type_id]
            params = {}
            for pname, default, _support in dist.params:
                params[pname] = ("const", float(p.get(pname, default)))
            var = Var(nid, str(p["name"]).strip(), "distribution", node.type_id, dist.support,
                      params=params, observed=str(p.get("observed") or "").strip(),
                      group=str(p.get("group") or "").strip(),
                      size=int(p.get("size", 0) or 0),
                      non_centered=bool(p.get("non_centered", False)))
        out[nid] = var
    for e in graph.edges:
        if e.source_id in out and e.target_id in out:
            child, parent = out[e.target_id], out[e.source_id]
            if child.kind == "distribution" and e.target_port in child.params:
                child.params[e.target_port] = ("var", parent.name)
            child.parents.append(parent.name)
    order, done = [], set()
    pending = dict(out)
    while pending:
        ready = [v for v in pending.values()
                 if all(nodes_name in done for nodes_name in v.parents)]
        if not ready:
            raise ModelError("the variables form a cycle")
        for v in sorted(ready, key=lambda v: v.name):
            order.append(v)
            done.add(v.name)
            del pending[v.node_id]
    return order


def resolve(vars_: list[Var], columns: set[str] | None = None) -> dict:
    """Row-level flags, referenced data columns and group columns (raises ModelError)."""
    by_name = {v.name: v for v in vars_}
    seen = set()
    for v in vars_:
        if not v.name.isidentifier() or keyword.iskeyword(v.name) or v.name in RESERVED:
            raise ModelError(f"variable name {v.name!r} must be an identifier and not one of "
                             f"{', '.join(sorted(RESERVED))}", v.name)
        if v.name in seen:
            raise ModelError(f"two variables are named {v.name!r}", v.name)
        seen.add(v.name)
    data_columns: set[str] = set()
    groups: set[str] = set()
    for v in vars_:
        if v.group:
            groups.add(v.group)
        if v.observed:
            data_columns.add(v.observed)
        if v.kind == "deterministic":
            for name in v.names:
                if name in by_name:
                    if name not in v.parents:
                        raise ModelError(f"{v.name} uses {name}: wire {name} into it", v.name)
                else:
                    data_columns.add(name)
                    if columns is not None and name not in columns:
                        raise ModelError(f"{v.name} uses {name!r}, which is neither a wired "
                                         f"variable nor a data column", v.name)
        parents = [by_name[p] for p in v.parents]
        v.row_level = bool(v.observed) or any(p.row_level for p in parents) or (
            v.kind == "deterministic" and any(n not in by_name for n in v.names))
        for p in parents:
            if p.group and not v.row_level and v.group != p.group:
                raise ModelError(f"{p.name} has one value per {p.group!r} but {v.name} does "
                                 f"not: give {v.name} group = {p.group} or use it per row",
                                 v.name)
        if v.row_level and v.group:
            raise ModelError(f"{v.name} is computed per data row, so it cannot also have one "
                             f"value per group: clear its group", v.name)
    return {"columns": sorted(data_columns), "groups": sorted(groups),
            "observed": [v.name for v in vars_ if v.observed]}


def _ref(name: str, by_name: dict, row_level: bool) -> str:
    parent = by_name[name]
    if parent.group and row_level:
        return f"{name}[{identifier(parent.group)}_idx]"
    return name


class _Rewrite(ast.NodeTransformer):
    def __init__(self, by_name: dict, row_level: bool) -> None:
        self.by_name, self.row_level = by_name, row_level

    def visit_Name(self, node: ast.Name):  # noqa: N802, ANN201
        if node.id in FUNCTIONS:
            return node
        target = (_ref(node.id, self.by_name, self.row_level) if node.id in self.by_name
                  else identifier(node.id))
        return ast.parse(target, mode="eval").body

    def visit_Call(self, node: ast.Call):  # noqa: N802, ANN201
        node.args = [self.visit(a) for a in node.args]
        node.func = ast.parse(FUNCTIONS[node.func.id], mode="eval").body
        return node


def model_code(vars_: list[Var], info: dict) -> list[str]:
    """Lines of the ``build_model`` body (inside ``with pm.Model(coords=...)``)."""
    by_name = {v.name: v for v in vars_}
    lines = ["rows = pm.Data(\"rows\", data[\"_rows\"])"]
    for col in info["columns"]:
        if col in info["observed_columns"]:
            continue
        lines.append(f"{identifier(col)} = pm.Data(\"{identifier(col)}\", data[{col!r}])")
    for g in info["groups"]:
        lines.append(f"{identifier(g)}_idx = pm.Data(\"{identifier(g)}_idx\", "
                     f"data[{g + '__code'!r}])")
    for v in vars_:
        if v.kind == "deterministic":
            tree, _names = parse_expression(v.expression)
            expr = ast.unparse(_Rewrite(by_name, v.row_level).visit(tree))
            lines.append(f"{v.name} = pm.Deterministic(\"{v.name}\", {expr})")
            continue
        dist = BY_ID[v.type_id]
        if (v.non_centered and v.group and not v.observed
                and all(v.params[k][0] == "var" for k in ("mu", "sigma"))):
            mu = _ref(v.params["mu"][1], by_name, False)
            sigma = _ref(v.params["sigma"][1], by_name, False)
            dims = f"dims=\"{identifier(v.group)}\""
            lines.append(f"{v.name}_offset = pm.Normal(\"{v.name}_offset\", mu=0.0, sigma=1.0, "
                         f"{dims})")
            lines.append(f"{v.name} = pm.Deterministic(\"{v.name}\", {mu} + {sigma} * "
                         f"{v.name}_offset, {dims})")
            continue
        args = []
        for pname, _default, _support in dist.params:
            kind, value = v.params[pname]
            if kind == "var":
                args.append(f"{pname}={_ref(value, by_name, v.row_level)}")
            elif v.type_id == "ppl.dirichlet":
                args.append(f"{pname}=np.full({v.size}, {value!r})")
            elif v.type_id == "ppl.categorical":    # uniform over the observed categories
                k = f"len(STATE['categories'][{v.observed!r}])"
                args.append(f"{pname}=np.full({k}, 1 / {k})")
            elif v.type_id == "ppl.binomial" and pname == "n":
                args.append(f"n={int(value)}")
            else:
                args.append(f"{pname}={value!r}")
        if v.group:
            args.append(f"dims=\"{identifier(v.group)}\"")
        if v.observed:
            args.append(f"observed=data[{v.observed!r}]")
            args.append("shape=rows.shape[0]")
        lines.append(f"{v.name} = pm.{dist.pm}(\"{v.name}\", {', '.join(args)})")
    return lines


def compile_design(graph, columns: set[str] | None = None) -> dict:  # noqa: ANN001
    """Everything the script template needs (raises ModelError for invalid designs)."""
    vars_ = variables(graph)
    if not vars_:
        raise ModelError("add distribution blocks: priors, and a likelihood with observed set")
    info = resolve(vars_, columns)
    observed_columns = {v.observed for v in vars_ if v.observed}
    info["observed_columns"] = sorted(observed_columns)
    categorical = {v.observed for v in vars_ if v.type_id == "ppl.categorical" and v.observed}
    return {"vars": vars_, "info": info, "categorical": sorted(categorical)}
