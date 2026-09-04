"""Code generation: turn the Graph IR into real, runnable framework code.

Two independent backends over one IR. Fragment contract (see BlockDefinition
docstring): per-framework module constructors + call expressions with
``{self_var}``/``{i0}``/``{i1}``/param/shape-context placeholders.

Channels-first IR -> Keras channels-last translation happens here and only
here: Input shapes are reversed, axis params remapped, Permute/Reshape
targets translated.
"""
from __future__ import annotations

import re
from pathlib import Path

from jinja2 import Environment

from ai_made_easy.core.graph import Graph, NodeInstance
from ai_made_easy.core.spec import shape_volume

_env = Environment(trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)

FRAMEWORKS = ("pytorch", "keras")

_IDENT_RE = re.compile(r"[^0-9a-zA-Z_]+")

_UNSUPPORTED_KERAS_MSG = "{name} does not support Keras export yet"


class CodegenError(ValueError):
    """The graph cannot be rendered for the requested framework."""


def sanitize_identifier(name: str) -> str:
    ident = _IDENT_RE.sub("_", name).strip("_")
    if not ident or ident[0].isdigit():
        ident = "x" + ident
    return ident


def class_name_for(graph_name: str) -> str:
    parts = re.split(r"[^0-9a-zA-Z]+", graph_name)
    joined = "".join(p.capitalize() for p in parts if p)
    return joined or "AimModel"


def _base_name(node: NodeInstance) -> str:
    return sanitize_identifier(node.type_id.rsplit(".", 1)[-1]).lower()


def resolve_expr(expr: str, context: dict) -> str:
    def sub(match: re.Match) -> str:
        key = match.group(1)
        if key not in context:
            raise CodegenError(
                f"code fragment references unknown placeholder {key!r} in: {expr!r}"
            )
        val = context[key]
        if isinstance(val, bool) or isinstance(val, (int, float)):
            return repr(val)
        return str(val)  # strings/enum values insert raw

    return re.sub(r"\{(\w+)\}", sub, expr)


def _shape_context(in_shapes: list[list[int]], params: dict) -> dict:
    """Shape-derived placeholders shared by all fragments."""
    ctx: dict = {}
    if in_shapes:
        first = in_shapes[0]
        rank = len(first)
        ctx["in_features"] = shape_volume(first)
        ctx["in_channels"] = first[0] if rank >= 3 else first[-1]
        ctx["num_features"] = ctx["in_channels"]
        ctx["input_size"] = first[-1]
        ctx["input_shape"] = first
        ctx["normalized_shape"] = "[" + ", ".join(str(d) for d in first) + "]"
        ctx["in_rank"] = rank
        ctx["in_shapes"] = in_shapes
    # Axis params are IR axes (no batch dim): pre-translate per framework.
    # Both spellings are exposed because fragments use them interchangeably.
    for key in ("axis", "dim"):
        a = params.get(key)
        if isinstance(a, int):
            t = a + 1 if a >= 0 else a
            k = -1 if a < 0 else ((a - 1) % len(in_shapes[0])) + 1
            ctx["torch_axis"] = ctx["torch_dim"] = t
            ctx["keras_axis"] = k
    return ctx


def _node_inputs(graph: Graph, node: NodeInstance, vars: dict[str, str]) -> list[str]:
    """Input variable names in port order."""
    names = []
    for port in node.definition().inputs:
        edge = graph.input_edge_for(node.instance_id, port.name)
        if edge is None or edge.source_id not in vars:
            raise CodegenError(
                f"{node.instance_id}: input '{port.name}' has no resolved producer"
            )
        names.append(vars[edge.source_id])
    return names


def emit_dag(graph: Graph) -> tuple[list[dict], list[int], str]:
    """Render every model node into per-framework code fragments.

    Returns (nodes, input_shape, output_var) where each node dict carries:
    var / torch_module / torch_expr / keras_expr.
    """
    shapes = graph.infer_shapes()
    chain = graph.model_nodes()
    head = chain[0]
    input_shape = shapes[head.instance_id]

    torch_vars: dict[str, str] = {head.instance_id: "x"}
    keras_vars: dict[str, str] = {head.instance_id: "inputs"}
    used: dict[str, int] = {}
    emitted: list[dict] = []
    output_var_torch = "x"
    output_var_keras = "inputs"

    for node in chain[1:]:
        defn = node.definition()
        base = _base_name(node)
        used[base] = used.get(base, 0) + 1
        var = f"v_{base}_{used[base]}"
        kvar = f"x_{base}_{used[base]}"
        params = dict(node.resolved_params())
        in_names_t = _node_inputs(graph, node, torch_vars)
        in_names_k = _node_inputs(graph, node, keras_vars)
        in_shapes = [shapes[e.source_id] for e in
                     (graph.input_edge_for(node.instance_id, p.name)
                      for p in defn.inputs) if e is not None]
        ctx = dict(params)
        ctx.update(_shape_context(in_shapes, params))
        ctx["self_var"] = var[2:]  # attribute name without the v_ prefix

        if node.type_id == "core.output":
            output_var_torch = in_names_t[0]
            output_var_keras = in_names_k[0]
            continue

        # --- PyTorch -----------------------------------------------------
        module = ""
        if defn.pytorch_layer:
            module = f"self.{ctx['self_var']} = {resolve_expr(defn.pytorch_layer, ctx)}"
        tctx = dict(ctx)
        for i, name in enumerate(in_names_t):
            tctx[f"i{i}"] = name
        if callable(defn.pytorch_expr):
            expr = defn.pytorch_expr(tctx)
        else:
            template = defn.pytorch_expr or "self.{self_var}({i0})"
            expr = resolve_expr(template, tctx)
        torch_vars[node.instance_id] = var

        # --- Keras ---------------------------------------------------------
        if defn.keras_layer or defn.keras_expr:
            kctx = dict(ctx)
            kctx["keras_layer"] = (
                resolve_expr(defn.keras_layer, kctx) if defn.keras_layer else ""
            )
            for i, name in enumerate(in_names_k):
                kctx[f"i{i}"] = name
            if callable(defn.keras_expr):
                kexpr = defn.keras_expr(kctx)
            else:
                ktemplate = defn.keras_expr or "{keras_layer}({i0})"
                kexpr = resolve_expr(ktemplate, kctx)
        else:
            raise CodegenError(
                f"{defn.display_name} ({node.instance_id}) "
                f"cannot be exported to Keras yet"
            )
        keras_vars[node.instance_id] = kvar

        emitted.append(
            {
                "var": var,
                "kvar": kvar,
                "torch_module": module,
                "torch_expr": f"{var} = {expr}",
                "keras_expr": f"{kvar} = {kexpr}",
            }
        )

    return emitted, input_shape, (output_var_torch, output_var_keras)


def _keras_input_shape(ir_shape: list[int]) -> str:
    dims = ", ".join(str(d) for d in reversed(ir_shape))
    return f"({dims},)"


def generate(graph: Graph, framework: str) -> str:
    if framework not in FRAMEWORKS:
        raise ValueError(f"unknown framework {framework!r}; expected one of {FRAMEWORKS}")
    if errors := [i for i in graph.validate() if i.severity == "error"]:
        raise CodegenError(
            "graph has validation errors:\n" + "\n".join(f"  - {e}" for e in errors)
        )
    nodes, input_shape, (out_t, out_k) = emit_dag(graph)
    ctx = {
        "graph_name": graph.name,
        "class_name": class_name_for(graph.name),
        "input_shape": input_shape,
        "input_dims": ", ".join(str(d) for d in input_shape),
        "keras_input_shape": _keras_input_shape(input_shape),
        "nodes": nodes,
        "output_var": out_t,
        "keras_output_var": out_k,
        "modules": [n for n in nodes if n["torch_module"]],
    }
    if framework == "pytorch":
        from ai_made_easy.core.codegen.pytorch_gen import PYTORCH_TEMPLATE

        return _env.from_string(PYTORCH_TEMPLATE).render(**ctx)
    from ai_made_easy.core.codegen.keras_gen import KERAS_TEMPLATE

    return _env.from_string(KERAS_TEMPLATE).render(**ctx)


def export(graph: Graph, framework: str, out_dir: str | Path) -> Path:
    out = Path(out_dir) / f"{sanitize_identifier(graph.name)}_{framework}.py"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(generate(graph, framework))
    return out


def export_training(graph: Graph, framework: str, out_dir: str | Path) -> Path:
    """Write the self-contained training script for the graph."""
    from ai_made_easy.core.codegen.training_gen import generate_training

    out = Path(out_dir) / f"{sanitize_identifier(graph.name)}_train_{framework}.py"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(generate_training(graph, framework))
    return out
