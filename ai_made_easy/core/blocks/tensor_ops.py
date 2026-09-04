"""Tensor-op blocks: merges (Add/Multiply/Concatenate) and shape surgery
(Reshape/Permute/Squeeze/Unsqueeze/Lambda).

Merge blocks have two explicit input ports (in1, in2) so edge order stays
deterministic; chain a second merge block for 3+ inputs.
"""
from __future__ import annotations

from ai_made_easy.core.blocks import _shape
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import (
    BlockDefinition,
    ParamSpec,
    PortSpec,
    ShapeError,
    shape_volume,
)
from ai_made_easy.core.blocks._palette import family_color

reg = get_registry()

_OP_COLOR = family_color("tensor")

_MERGE_PORTS = (PortSpec("in1"), PortSpec("in2"))
_axis_param = ParamSpec(name="axis", type="int", default=-1, minimum=-4,
                        help="IR axis (no batch dim); -1 = last")


def _same_shapes(in_shapes, name):
    a, b = in_shapes
    if a != b:
        raise ShapeError(f"{name} inputs must have identical shapes: {a} vs {b}")
    return list(a)


def _concat_shape(in_shapes, params):
    a, b = in_shapes
    axis = params["axis"]
    axis = len(a) + axis if axis < 0 else axis
    if len(a) != len(b):
        raise ShapeError(f"Concatenate ranks differ: {a} vs {b}")
    for d in range(len(a)):
        if d != axis and a[d] != b[d]:
            raise ShapeError(
                f"Concatenate dims must match except along axis {axis}: {a} vs {b}"
            )
    out = list(a)
    out[axis] = a[axis] + b[axis]
    return out


reg.register(
    BlockDefinition(
        type_id="core.add",
        display_name="Add (skip connection)",
        category="Tensor Ops",
        color=_OP_COLOR,
        inputs=_MERGE_PORTS,
        outputs=(PortSpec("out"),),
        shape_fn=lambda s, p: _same_shapes(s, "Add"),
        pytorch_expr="{i0} + {i1}",
        keras_expr="layers.Add()([{i0}, {i1}])",
    )
)

reg.register(
    BlockDefinition(
        type_id="core.multiply",
        display_name="Multiply (gating)",
        category="Tensor Ops",
        color=_OP_COLOR,
        inputs=_MERGE_PORTS,
        outputs=(PortSpec("out"),),
        shape_fn=lambda s, p: _same_shapes(s, "Multiply"),
        pytorch_expr="{i0} * {i1}",
        keras_expr="layers.Multiply()([{i0}, {i1}])",
    )
)

reg.register(
    BlockDefinition(
        type_id="core.concatenate",
        display_name="Concatenate",
        category="Tensor Ops",
        color=_OP_COLOR,
        params=(_axis_param,),
        inputs=_MERGE_PORTS,
        outputs=(PortSpec("out"),),
        shape_fn=_concat_shape,
        pytorch_expr="torch.cat([{i0}, {i1}], dim={torch_dim})",
        keras_expr="layers.Concatenate(axis={keras_axis})([{i0}, {i1}])",
    )
)


# ------------------------------------------------------------- shape surgery

def _reshape_shape(in_shapes, params):
    (s,) = in_shapes
    dims = _shape.parse_target(params["target"])
    return _shape.resolve_target(dims, shape_volume(s))


def _reshape_torch(ctx):
    dims = _shape.parse_target(ctx["target"])
    _shape.resolve_target(dims, shape_volume(ctx["in_shapes"][0]))
    return f"torch.reshape({ctx['i0']}, ({ctx['i0']}.shape[0], {tuple(dims)}))"


def _reshape_keras(ctx):
    dims = _shape.parse_target(ctx["target"])
    _shape.resolve_target(dims, shape_volume(ctx["in_shapes"][0]))
    keras_dims = tuple(reversed(dims)) if len(dims) > 1 else tuple(dims)
    return f"layers.Reshape({keras_dims})({ctx['i0']})"


reg.register(
    BlockDefinition(
        type_id="core.reshape",
        display_name="Reshape",
        category="Tensor Ops",
        color=_OP_COLOR,
        params=(ParamSpec(name="target", type="str", default="784",
                          help="Target sample shape, e.g. '784' or '28,28,1'; "
                               "batch dim is preserved automatically"),),
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        shape_fn=_reshape_shape,
        pytorch_expr=_reshape_torch,
        keras_expr=_reshape_keras,
    )
)


def _permute_shape(in_shapes, params):
    (s,) = in_shapes
    order = _shape.parse_order(params["order"], len(s))
    return [s[d] for d in order]


def _permute_torch(ctx):
    shape = ctx["in_shapes"][0]
    order = _shape.parse_order(ctx["order"], len(shape))
    dims = ", ".join(["0"] + [str(d + 1) for d in order])
    return f"{ctx['i0']}.permute({dims})"


def _permute_keras(ctx):
    shape = ctx["in_shapes"][0]
    rank = len(shape)
    order = _shape.parse_order(ctx["order"], rank)
    # IR dim k sits at NHWC index (k-1) % rank (0-based); Keras Permute is 1-based.
    dims = ", ".join(str(((d - 1) % rank) + 1) for d in order)
    return f"layers.Permute(dims=({dims}))({ctx['i0']})"


reg.register(
    BlockDefinition(
        type_id="core.permute",
        display_name="Permute",
        category="Tensor Ops",
        color=_OP_COLOR,
        params=(ParamSpec(name="order", type="str", default="1, 0",
                          help="Sample-dim order, e.g. '2, 0, 1' for [C,H,W]"),),
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        shape_fn=_permute_shape,
        pytorch_expr=_permute_torch,
        keras_expr=_permute_keras,
    )
)


def _squeeze_shape(in_shapes, params):
    (s,) = in_shapes
    dim = params["dim"]
    dim = len(s) + dim if dim < 0 else dim
    if not 0 <= dim < len(s):
        raise ShapeError(f"squeeze dim {params['dim']} out of range for {s}")
    if s[dim] != 1:
        raise ShapeError(f"cannot squeeze dim {dim} of {s}: size is {s[dim]}, not 1")
    out = [d for i, d in enumerate(s) if i != dim]
    if not out:
        raise ShapeError("cannot squeeze the only dim; use Unsqueeze elsewhere")
    return out


def _squeeze_torch(ctx):
    dim = ctx["dim"]
    rank = len(ctx["in_shapes"][0])
    tdim = dim + 1 if dim >= 0 else rank + dim + 1
    return f"torch.squeeze({ctx['i0']}, dim={tdim})"


reg.register(
    BlockDefinition(
        type_id="core.squeeze",
        display_name="Squeeze",
        category="Tensor Ops",
        color=_OP_COLOR,
        params=(ParamSpec(name="dim", type="int", default=0, minimum=-4,
                          help="IR axis to remove (must have size 1)"),),
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        shape_fn=_squeeze_shape,
        pytorch_expr=_squeeze_torch,
    )
)


def _unsqueeze_shape(in_shapes, params):
    (s,) = in_shapes
    dim = params["dim"]
    dim = len(s) + dim + 1 if dim < 0 else dim
    if not 0 <= dim <= len(s):
        raise ShapeError(f"unsqueeze dim {params['dim']} out of range for {s}")
    return [*s[:dim], 1, *s[dim:]]


def _unsqueeze_torch(ctx):
    dim = ctx["dim"]
    rank = len(ctx["in_shapes"][0])
    tdim = dim + 1 if dim >= 0 else rank + dim + 2
    return f"torch.unsqueeze({ctx['i0']}, dim={tdim})"


def _unsqueeze_keras(ctx):
    shape = list(ctx["in_shapes"][0])
    dim = ctx["dim"]
    dim = len(shape) + dim + 1 if dim < 0 else dim
    new_shape = [*shape[:dim], 1, *shape[dim:]]
    return f"layers.Reshape({tuple(reversed(new_shape))})({ctx['i0']})"


reg.register(
    BlockDefinition(
        type_id="core.unsqueeze",
        display_name="Unsqueeze",
        category="Tensor Ops",
        color=_OP_COLOR,
        params=(ParamSpec(name="dim", type="int", default=0, minimum=-4,
                          help="IR axis where a size-1 dim is inserted"),),
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        shape_fn=_unsqueeze_shape,
        pytorch_expr=_unsqueeze_torch,
        keras_expr=_unsqueeze_keras,
    )
)


def _lambda_checks(p: dict) -> list[tuple[str, str]]:
    """Compile-only syntax check of the custom expression (never executes)."""
    expr = str(p.get("expression", ""))
    if not expr.strip():
        return [("error", "the expression is empty — write something like 't * 2.0'")]
    try:
        compile(expr, "<lambda>", "eval")
    except SyntaxError as exc:
        return [("error",
                 f"the expression has a Python typo on line {exc.lineno}: "
                 f"{exc.msg} — check quotes, brackets and colons")]
    if "t" not in expr:
        return [("warning",
                 "the expression doesn't use 't' — it ignores the tensor "
                 "coming in")]
    return []


reg.register(
    BlockDefinition(
        type_id="core.lambda",
        display_name="Lambda (custom expr)",
        category="Tensor Ops",
        color=_OP_COLOR,
        checks_fn=_lambda_checks,
        params=(ParamSpec(name="expression", type="str", default="t * 2.0",
                          help="Python expression over 't' (torch/Keras tensor)"),),
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        shape_fn=lambda s, p: list(s[0]),
        pytorch_expr="(lambda t: {expression})({i0})",
        keras_expr="layers.Lambda(lambda t: {expression})({i0})",
    )
)
