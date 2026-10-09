"""Code generation: turn the Graph IR into runnable framework code.

Two independent backends over one IR:

* **PyTorch** mirrors the IR exactly (channels-first, batch-first sequences).
* **Keras 3** tracks a layout per tensor. Channels-first IR tensors consumed
  by conv/pool/normalization layers live channels-last in Keras ("cl");
  sequences and axis-addressed ops keep the IR order ("ir"); elementwise
  layers accept either. The emitter inserts the minimal ``Permute`` layers
  where a block needs a different layout than its input has.

A block that one backend cannot express never blocks the other backend:
unsupported blocks are collected and reported as one clear error when that
framework is requested.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from jinja2 import Environment

from ai_made_easy.core.graph import Graph, NodeInstance
from ai_made_easy.core.spec import shape_volume

_env = Environment(trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)

FRAMEWORKS = ("pytorch", "keras")
CL, IR = "cl", "ir"

_IDENT_RE = re.compile(r"[^0-9a-zA-Z_]+")


class CodegenError(ValueError):
    """The graph cannot be rendered for the requested framework."""


class KerasUnsupported(CodegenError):
    """Raised by a Keras fragment for a configuration Keras cannot express."""


# ----------------------------------------------------------------- naming

def sanitize_identifier(name: str) -> str:
    ident = _IDENT_RE.sub("_", name).strip("_")
    if not ident or ident[0].isdigit():
        ident = "x" + ident
    return ident


def class_name_for(graph_name: str) -> str:
    parts = re.split(r"[^0-9a-zA-Z]+", graph_name)
    joined = "".join(p[:1].upper() + p[1:] for p in parts if p)
    if not joined or joined[0].isdigit():
        joined = "Model" + joined
    return joined


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


def _render(fragment, ctx: dict) -> str:
    """A fragment is a placeholder template or a callable(ctx) -> str."""
    return fragment(ctx) if callable(fragment) else resolve_expr(fragment, ctx)


# ----------------------------------------------------------------- layouts

def keras_shape(ir_shape: list[int], layout: str) -> list[int]:
    """The Keras (batch-excluded) shape of an IR tensor in ``layout``."""
    if layout == CL and len(ir_shape) >= 2:
        return [*ir_shape[1:], ir_shape[0]]
    return list(ir_shape)


def _keras_axis(axis: int, rank: int, layout: str) -> int:
    """Map an IR axis (no batch dim) to a Keras axis (batch at 0)."""
    idx = axis if axis >= 0 else rank + axis
    if layout == CL and rank >= 2:
        return -1 if idx == 0 else idx
    return idx + 1


def _permute_expr(expr: str, rank: int, src: str, dst: str) -> str:
    if src == dst or rank < 2:
        return expr
    if src == CL and dst == IR:
        dims = (rank, *range(1, rank))
    else:
        dims = (*range(2, rank + 1), 1)
    return f"layers.Permute({dims})({expr})"


def _shape_context(in_shapes: list[list[int]], params: dict,
                   layout: str = IR) -> dict:
    """Shape-derived placeholders shared by all fragments."""
    ctx: dict = {}
    if in_shapes:
        first = in_shapes[0]
        rank = len(first)
        ctx["in_features"] = shape_volume(first)
        ctx["in_channels"] = first[0]          # channels-first IR
        ctx["num_features"] = first[0]
        ctx["input_size"] = first[-1]          # batch-first sequences [L, C]
        ctx["features"] = first[-1]
        ctx["seq_len"] = first[0]
        ctx["input_shape"] = first
        ctx["normalized_shape"] = "[" + ", ".join(str(d) for d in first) + "]"
        ctx["in_rank"] = rank
        ctx["spatial_rank"] = max(rank - 1, 0)
        ctx["in_shapes"] = in_shapes
        ctx["keras_layout"] = layout
        if len(in_shapes) > 1:
            ctx["in2_features"] = shape_volume(in_shapes[1])
            ctx["in2_channels"] = in_shapes[1][0]
            ctx["in2_size"] = in_shapes[1][-1]
    for key in ("axis", "dim"):
        a = params.get(key)
        if isinstance(a, int) and not isinstance(a, bool) and in_shapes:
            rank = len(in_shapes[0])
            ctx["torch_axis"] = ctx["torch_dim"] = a + 1 if a >= 0 else a
            ctx["keras_axis"] = _keras_axis(a, rank, layout)
    return ctx


# ----------------------------------------------------------------- emission

@dataclass
class EmissionPlan:
    """Everything the templates need, for both frameworks."""

    nodes: list[dict] = field(default_factory=list)
    input_shape: list[int] = field(default_factory=list)
    input_dtype: str = "float32"
    output_shape: list[int] = field(default_factory=list)
    torch_output: str = "x"
    keras_output: str = "x"
    keras_input_layout: str = IR
    keras_output_layout: str = IR
    torch_helpers: list[str] = field(default_factory=list)
    keras_helpers: list[str] = field(default_factory=list)
    keras_unsupported: list[str] = field(default_factory=list)
    torch_unsupported: list[str] = field(default_factory=list)

    @property
    def keras_input_shape(self) -> list[int]:
        return keras_shape(self.input_shape, self.keras_input_layout)

    @property
    def keras_output_shape(self) -> list[int]:
        return keras_shape(self.output_shape, self.keras_output_layout)

    @property
    def keras_input_transpose(self) -> tuple[int, ...] | None:
        """np.transpose axes taking (N, *IR shape) data to the Keras input."""
        rank = len(self.input_shape)
        if self.keras_input_layout != CL or rank < 2:
            return None
        return (0, *range(2, rank + 1), 1)


class _Namer:
    """Variable names for one backend: straight chains reuse ``x``; values
    that are still needed by a later block get their own name."""

    def __init__(self, graph: Graph, chain: list[NodeInstance], head_var: str):
        ids = {n.instance_id for n in chain}
        self.pending = {nid: 0 for nid in ids}
        for e in graph.edges:
            if e.source_id in ids and e.target_id in ids:
                self.pending[e.source_id] += 1
        self.var: dict[str, str] = {chain[0].instance_id: head_var}
        self.holder = chain[0].instance_id if head_var == "x" else None

    def consume(self, producer_ids: list[str]) -> None:
        for pid in producer_ids:
            self.pending[pid] -= 1

    def assign(self, node_id: str, fresh: str) -> str:
        x_free = self.holder is None or self.pending.get(self.holder, 0) <= 0
        if x_free:
            self.var[node_id] = "x"
            self.holder = node_id
        else:
            self.var[node_id] = fresh
        return self.var[node_id]


def _producers(graph: Graph, node: NodeInstance) -> list[str]:
    out = []
    for port in node.definition().inputs:
        edge = graph.input_edge_for(node.instance_id, port.name)
        if edge is None:
            raise CodegenError(
                f"{node.instance_id}: input '{port.name}' is not connected")
        out.append(edge.source_id)
    return out


def _choose_input_layout(graph: Graph, head: NodeInstance, rank: int) -> str:
    if rank < 2:
        return IR
    wants = []
    for e in graph.outgoing(head.instance_id):
        defn = graph.nodes[e.target_id].definition()
        if e.target_id in graph.nodes and graph.nodes[e.target_id].type_id != "core.output":
            wants.append(defn.keras_layout)
    if CL in wants:
        return CL
    if IR in wants:
        return IR
    return CL if rank >= 3 else IR


def emit_graph(graph: Graph) -> EmissionPlan:
    """Render every model node into PyTorch and Keras code fragments."""
    shapes = graph.infer_shapes()
    chain = graph.model_nodes()
    head = chain[0]
    plan = EmissionPlan()
    plan.input_shape = list(shapes[head.instance_id])
    plan.input_dtype = str(head.resolved_params().get("dtype", "float32"))
    plan.keras_input_layout = _choose_input_layout(graph, head, len(plan.input_shape))

    t_names = _Namer(graph, chain, "x")
    k_names = _Namer(graph, chain, "inputs")
    k_layout: dict[str, str] = {head.instance_id: plan.keras_input_layout}
    used: dict[str, int] = {}

    for node in chain[1:]:
        defn = node.definition()
        producers = _producers(graph, node)

        if node.type_id == "core.output":
            plan.torch_output = t_names.var[producers[0]]
            plan.keras_output = k_names.var[producers[0]]
            plan.keras_output_layout = k_layout[producers[0]]
            plan.output_shape = list(shapes[producers[0]])
            continue

        base = _base_name(node)
        used[base] = used.get(base, 0) + 1
        attr = f"{base}_{used[base]}"
        params = dict(node.resolved_params())
        in_shapes = [shapes[p] for p in producers]
        out_rank = len(shapes[node.instance_id])

        # --- PyTorch ------------------------------------------------------
        t_inputs = [t_names.var[p] for p in producers]
        t_names.consume(producers)
        module = ""
        torch_line = None
        if defn.supports("pytorch"):
            tctx = dict(params)
            tctx.update(_shape_context(in_shapes, params))
            tctx["self_var"] = attr
            for i, name in enumerate(t_inputs):
                tctx[f"i{i}"] = name
            if defn.pytorch_layer:
                module = f"self.{attr} = {_render(defn.pytorch_layer, tctx)}"
            if callable(defn.pytorch_expr):
                expr = defn.pytorch_expr(tctx)
            else:
                expr = resolve_expr(defn.pytorch_expr or "self.{self_var}({i0})", tctx)
            out_var = t_names.assign(node.instance_id, attr)
            torch_line = f"{out_var} = {expr}"
            plan.torch_helpers += list(defn.pytorch_helpers)
        else:
            t_names.assign(node.instance_id, attr)
            plan.torch_unsupported.append(f"{defn.display_name} ({node.instance_id})")

        # --- Keras --------------------------------------------------------
        src_layouts = [k_layout[p] for p in producers]
        if defn.keras_layout == "any":
            target = src_layouts[0] if src_layouts else IR
        else:
            target = defn.keras_layout
        k_inputs = []
        for p, src in zip(producers, src_layouts, strict=True):
            rank = len(shapes[p])
            k_inputs.append(_permute_expr(k_names.var[p], rank, src, target))
        k_names.consume(producers)
        keras_line = None
        kexpr, reason = None, ""
        if defn.supports("keras"):
            kctx = dict(params)
            kctx.update(_shape_context(in_shapes, params, target))
            kctx["self_var"] = attr
            for i, name in enumerate(k_inputs):
                kctx[f"i{i}"] = name
            try:
                kctx["keras_layer"] = (
                    _render(defn.keras_layer, kctx) if defn.keras_layer else "")
                if callable(defn.keras_expr):
                    kexpr = defn.keras_expr(kctx)
                else:
                    kexpr = resolve_expr(defn.keras_expr or "{keras_layer}({i0})", kctx)
            except KerasUnsupported as exc:
                kexpr, reason = None, f": {exc}"
        out_var = k_names.assign(node.instance_id, attr)
        if kexpr is not None:
            keras_line = f"{out_var} = {kexpr}"
            plan.keras_helpers += list(defn.keras_helpers)
        else:
            plan.keras_unsupported.append(
                f"{defn.display_name} ({node.instance_id}){reason}")
        k_layout[node.instance_id] = target if out_rank >= 2 else IR

        plan.nodes.append({
            "id": node.instance_id,
            "var": t_names.var[node.instance_id],
            "kvar": k_names.var[node.instance_id],
            "torch_module": module,
            "torch_expr": torch_line,
            "keras_expr": keras_line,
        })
    return plan


def emit_dag(graph: Graph) -> tuple[list[dict], list[int], tuple[str, str]]:
    """Back-compat view of :func:`emit_graph` used by the script generators."""
    plan = emit_graph(graph)
    return plan.nodes, plan.input_shape, (plan.torch_output, plan.keras_output)


def keras_output_shape(graph: Graph) -> list[int]:
    return emit_graph(graph).keras_output_shape


def unsupported_blocks(graph: Graph, framework: str) -> list[str]:
    plan = emit_graph(graph)
    return plan.keras_unsupported if framework == "keras" else plan.torch_unsupported


# ----------------------------------------------------------------- render

def _validated(graph: Graph) -> None:
    if errors := [i for i in graph.validate() if i.severity == "error"]:
        raise CodegenError(
            "graph has validation errors:\n" + "\n".join(f"  - {e}" for e in errors)
        )


def require_framework(plan: EmissionPlan, framework: str) -> None:
    missing = plan.keras_unsupported if framework == "keras" else plan.torch_unsupported
    if missing:
        label = "Keras" if framework == "keras" else "PyTorch"
        raise CodegenError(
            f"{len(missing)} block(s) have no {label} equivalent: "
            + ", ".join(missing))


def _param_count(graph: Graph) -> str:
    try:
        from ai_made_easy.core.summary import summarize

        return f"{summarize(graph).total_params:,}"
    except Exception:  # noqa: BLE001 — the docstring line is optional
        return ""


def _dims(shape: list[int]) -> str:
    return ", ".join(str(d) for d in shape)


def template_context(graph: Graph, plan: EmissionPlan) -> dict:
    """Context shared by the model templates and the training templates."""
    from ai_made_easy.core.codegen.helpers import (
        KERAS_HELPERS,
        PYTORCH_HELPERS,
        render_helpers,
    )

    int_input = plan.input_dtype.startswith("int")
    torch_lines = [n["torch_expr"] for n in plan.nodes if n["torch_expr"]]
    keras_lines = [n["keras_expr"] for n in plan.nodes if n["keras_expr"]]
    return {
        "graph_name": graph.name,
        "class_name": class_name_for(graph.name),
        "model_name": sanitize_identifier(graph.name),
        "input_shape": plan.input_shape,
        "input_dims": _dims(plan.input_shape),
        "output_dims": _dims(plan.output_shape),
        "keras_input_shape": "(" + _dims(plan.keras_input_shape)
        + ("," if len(plan.keras_input_shape) == 1 else "") + ")",
        "keras_input_dims": _dims(plan.keras_input_shape),
        "keras_output_dims": _dims(plan.keras_output_shape),
        "keras_channels_last": plan.keras_input_layout == CL,
        "int_input": int_input,
        "keras_input_dtype": "int32" if int_input else "float32",
        "nodes": plan.nodes,
        "modules": [n for n in plan.nodes if n["torch_module"]],
        "output_var": plan.torch_output,
        "keras_output_var": plan.keras_output,
        "torch_helpers": render_helpers(plan.torch_helpers, PYTORCH_HELPERS),
        "keras_helpers": render_helpers(plan.keras_helpers, KERAS_HELPERS),
        "uses_functional": any("F." in line for line in torch_lines),
        "uses_ops": any("ops." in line for line in keras_lines)
        or any("ops." in h for h in render_helpers(plan.keras_helpers, KERAS_HELPERS)),
        "total_params": _param_count(graph),
    }


def _reject_classic(graph: Graph) -> None:
    from ai_made_easy.core.classic.generate import is_classic

    if is_classic(graph):
        raise CodegenError("this project is a scikit-learn pipeline; export it with the "
                           "scikit-learn target")


def generate(graph: Graph, framework: str) -> str:
    if framework not in FRAMEWORKS:
        raise ValueError(f"unknown framework {framework!r}; expected one of {FRAMEWORKS}")
    _reject_classic(graph)
    _validated(graph)
    plan = emit_graph(graph)
    require_framework(plan, framework)
    ctx = template_context(graph, plan)
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
    """Write the self-contained training script for the graph.

    Classic (scikit-learn) projects always produce a scikit-learn script.
    """
    from ai_made_easy.core.classic.generate import generate_classic, is_classic
    from ai_made_easy.core.codegen.training_gen import generate_training
    from ai_made_easy.core.families import generator_for

    if is_classic(graph):
        framework = "sklearn"
    out = Path(out_dir) / f"{sanitize_identifier(graph.name)}_train_{framework}.py"
    out.parent.mkdir(parents=True, exist_ok=True)
    own = generator_for(graph, framework)
    code = own(graph) if own is not None else (
        generate_classic(graph) if framework == "sklearn" else generate_training(graph, framework))
    out.write_text(code)
    return out
