"""Tensor operations: reshaping, axis surgery, reductions and math."""
from __future__ import annotations

from ai_made_easy.core.blocks import _shape
from ai_made_easy.core.blocks._dsl import P, nn_block, norm_axis, passthrough, tuple_literal
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import ShapeError, shape_volume

reg = get_registry()


def _t_dim(axis: int, rank: int) -> int:
    """IR axis -> torch dim (batch at 0), always non-negative."""
    return norm_axis(axis, rank, "axis") + 1


# ------------------------------------------------------------------ reshape

reg.register(nn_block(
    "core.flatten", "Flatten", "Tensor Ops", family="tensor",
    shape=lambda s, p: [shape_volume(s[0])],
    torch="nn.Flatten()", keras="layers.Flatten()",
    desc="Collapses every non-batch axis into one feature vector.",
))


def _reshape_shape(in_shapes, params):
    dims = _shape.parse_target(params["target"])
    return _shape.resolve_target(dims, shape_volume(in_shapes[0]))


reg.register(nn_block(
    "core.reshape", "Reshape", "Tensor Ops", family="tensor",
    params=(P("target", "str", "784",
              help="Target per-sample shape, e.g. '16,49'; total size must match"),),
    shape=_reshape_shape, layout="ir",
    torch_expr=lambda c: (f"{c['i0']}.reshape({c['i0']}.shape[0], "
                          f"{', '.join(str(d) for d in _shape.parse_target(c['target']))})"),
    keras_expr=lambda c: (f"layers.Reshape({tuple_literal(_shape.parse_target(c['target']))})"
                          f"({c['i0']})"),
    desc="Reinterprets the sample with a new shape of equal size.",
))


def _permute_shape(in_shapes, params):
    s = in_shapes[0]
    order = _shape.parse_order(params["order"], len(s))
    return [s[d] for d in order]


reg.register(nn_block(
    "core.permute", "Permute", "Tensor Ops", family="tensor",
    params=(P("order", "str", "1, 0", help="New axis order, e.g. '1, 0' or '2, 0, 1'"),),
    shape=_permute_shape, layout="ir",
    torch_expr=lambda c: (f"{c['i0']}.permute(0, " + ", ".join(
        str(d + 1) for d in _shape.parse_order(c["order"], c["in_rank"])) + ")"),
    keras_expr=lambda c: (f"layers.Permute({tuple_literal(d + 1 for d in _shape.parse_order(c['order'], c['in_rank']))})"
                          f"({c['i0']})"),
    desc="Reorders the sample axes.",
))


def _transpose_shape(in_shapes, params):
    s = list(in_shapes[0])
    a = norm_axis(int(params["dim0"]), len(s), "Transpose")
    b = norm_axis(int(params["dim1"]), len(s), "Transpose")
    s[a], s[b] = s[b], s[a]
    return s


def _transpose_keras(c):
    rank = c["in_rank"]
    order = list(range(rank))
    a, b = norm_axis(int(c["dim0"]), rank, "T"), norm_axis(int(c["dim1"]), rank, "T")
    order[a], order[b] = order[b], order[a]
    return f"layers.Permute({tuple_literal(d + 1 for d in order)})({c['i0']})"


reg.register(nn_block(
    "core.transpose", "Transpose", "Tensor Ops", family="tensor",
    params=(P("dim0", "int", 0, lo=-4, hi=3), P("dim1", "int", 1, lo=-4, hi=3)),
    shape=_transpose_shape, layout="ir",
    torch_expr=lambda c: (f"{c['i0']}.transpose({_t_dim(int(c['dim0']), c['in_rank'])}, "
                          f"{_t_dim(int(c['dim1']), c['in_rank'])})"),
    keras_expr=_transpose_keras,
    desc="Swaps two sample axes (e.g. [C, L] ↔ [L, C]).",
))


def _squeeze_shape(in_shapes, params):
    s = in_shapes[0]
    dim = norm_axis(int(params["dim"]), len(s), "Squeeze")
    if s[dim] != 1:
        raise ShapeError(f"Squeeze: axis {params['dim']} of {s} has size {s[dim]}, not 1")
    out = [d for i, d in enumerate(s) if i != dim]
    if not out:
        raise ShapeError("Squeeze cannot remove the only axis")
    return out


reg.register(nn_block(
    "core.squeeze", "Squeeze", "Tensor Ops", family="tensor",
    params=(P("dim", "int", 0, lo=-4, hi=3, help="Axis of size 1 to remove"),),
    shape=_squeeze_shape, layout="ir",
    torch_expr=lambda c: f"{c['i0']}.squeeze({_t_dim(int(c['dim']), c['in_rank'])})",
    keras_expr=lambda c: f"ops.squeeze({c['i0']}, axis={_t_dim(int(c['dim']), c['in_rank'])})",
    desc="Removes an axis of size 1.",
))


def _unsqueeze_shape(in_shapes, params):
    s = in_shapes[0]
    dim = int(params["dim"])
    idx = dim if dim >= 0 else len(s) + dim + 1
    if not 0 <= idx <= len(s):
        raise ShapeError(f"Unsqueeze: axis {dim} is out of range for {s}")
    return [*s[:idx], 1, *s[idx:]]


def _unsq_dim(c) -> int:
    dim = int(c["dim"])
    return (dim if dim >= 0 else c["in_rank"] + dim + 1) + 1


reg.register(nn_block(
    "core.unsqueeze", "Unsqueeze", "Tensor Ops", family="tensor",
    params=(P("dim", "int", 0, lo=-5, hi=4, help="Where to insert the new size-1 axis"),),
    shape=_unsqueeze_shape, layout="ir",
    torch_expr=lambda c: f"{c['i0']}.unsqueeze({_unsq_dim(c)})",
    keras_expr=lambda c: f"ops.expand_dims({c['i0']}, axis={_unsq_dim(c)})",
    desc="Inserts an axis of size 1.",
))


def _repeat_shape(in_shapes, params):
    s = in_shapes[0]
    if len(s) != 1:
        raise ShapeError(f"Repeat Vector expects a vector [F], got {s}")
    return [int(params["times"]), s[0]]


reg.register(nn_block(
    "core.repeat_vector", "Repeat Vector", "Tensor Ops", family="tensor",
    params=(P("times", "int", 8, lo=1, help="Sequence length to produce"),),
    shape=_repeat_shape, layout="ir",
    torch_expr="{i0}.unsqueeze(1).expand(-1, {times}, -1)",
    keras="layers.RepeatVector({times})",
    desc="Repeats a vector [F] into a sequence [times, F].",
))


def _slice_shape(in_shapes, params):
    s = list(in_shapes[0])
    axis = norm_axis(int(params["dim"]), len(s), "Slice")
    start, end = int(params["start"]), int(params["end"])
    size = s[axis]
    lo = start if start >= 0 else size + start
    hi = (end if end > 0 else size + end) if end != 0 else size
    if not 0 <= lo < hi <= size:
        raise ShapeError(f"Slice [{start}:{end}] is empty or out of range for axis "
                         f"{params['dim']} of size {size}")
    s[axis] = hi - lo
    return s


def _slice_expr(c):
    rank = c["in_rank"]
    axis = norm_axis(int(c["dim"]), rank, "Slice")
    end = int(c["end"])
    sl = f"{c['start']}:{end if end != 0 else ''}"
    parts = [":"] * (rank + 1)
    parts[axis + 1] = sl
    return f"{c['i0']}[{', '.join(parts)}]"


reg.register(nn_block(
    "core.slice", "Slice", "Tensor Ops", family="tensor",
    params=(P("dim", "int", 0, lo=-4, hi=3), P("start", "int", 0, lo=-100000),
            P("end", "int", 0, lo=-100000, help="0 = to the end; negatives count back")),
    shape=_slice_shape, layout="ir", torch_expr=_slice_expr, keras_expr=_slice_expr,
    desc="Takes a contiguous range of one axis.",
))


# ---------------------------------------------------------------- reductions

_REDUCE = {"mean": ("mean", "mean"), "sum": ("sum", "sum"), "max": ("amax", "max"),
           "min": ("amin", "min"), "prod": ("prod", "prod"), "std": ("std", "std"),
           "var": ("var", "var")}


def _reduce_shape(in_shapes, params):
    s = list(in_shapes[0])
    axis = norm_axis(int(params["dim"]), len(s), "Reduce")
    if params["keepdim"]:
        s[axis] = 1
        return s
    out = [d for i, d in enumerate(s) if i != axis]
    if not out:
        raise ShapeError("Reduce would remove the only axis; enable keepdim")
    return out


def _reduce_torch(c):
    op = _REDUCE[c["op"]][0]
    dim = _t_dim(int(c["dim"]), c["in_rank"])
    keep = ", keepdim=True" if c["keepdim"] else ""
    extra = ", correction=0" if op in ("std", "var") else ""
    if op == "prod":
        return f"torch.prod({c['i0']}, dim={dim}{keep})"
    return f"torch.{op}({c['i0']}, dim={dim}{keep}{extra})"


def _reduce_keras(c):
    op = _REDUCE[c["op"]][1]
    dim = _t_dim(int(c["dim"]), c["in_rank"])
    keep = ", keepdims=True" if c["keepdim"] else ""
    return f"ops.{op}({c['i0']}, axis={dim}{keep})"


reg.register(nn_block(
    "core.reduce", "Reduce", "Tensor Ops", family="tensor",
    params=(P("op", "enum", "mean", options=tuple(_REDUCE)),
            P("dim", "int", -1, lo=-4, hi=3), P("keepdim", "bool", False)),
    shape=_reduce_shape, layout="ir", torch_expr=_reduce_torch, keras_expr=_reduce_keras,
    desc="Mean / sum / max / min / product / std / variance over one axis.",
))


# ---------------------------------------------------------------------- math

def _clamp_checks(p):
    if float(p["min"]) > float(p["max"]):
        return [("error", f"min {p['min']} is greater than max {p['max']}")]
    return []


reg.register(nn_block(
    "core.clamp", "Clamp", "Tensor Ops", family="tensor",
    params=(P("min", "float", -1.0, lo=-1e12), P("max", "float", 1.0, lo=-1e12)),
    shape=passthrough, checks=_clamp_checks,
    torch_expr="torch.clamp({i0}, min={min}, max={max})",
    keras_expr="ops.clip({i0}, {min}, {max})",
    desc="Limits values to [min, max].",
))

reg.register(nn_block(
    "core.scale", "Scale & Shift", "Tensor Ops", family="tensor",
    params=(P("scale", "float", 1.0, lo=-1e12), P("shift", "float", 0.0, lo=-1e12)),
    shape=passthrough,
    torch_expr="{i0} * {scale} + {shift}", keras_expr="{i0} * {scale} + {shift}",
    desc="Affine transform with constants: x · scale + shift.",
))

_UNARY = {"abs": ("torch.abs", "ops.abs"), "exp": ("torch.exp", "ops.exp"),
          "log": ("torch.log", "ops.log"), "log1p": ("torch.log1p", "ops.log1p"),
          "sqrt": ("torch.sqrt", "ops.sqrt"), "square": ("torch.square", "ops.square"),
          "negative": ("torch.neg", "ops.negative"), "reciprocal": ("torch.reciprocal", "ops.reciprocal"),
          "sin": ("torch.sin", "ops.sin"), "cos": ("torch.cos", "ops.cos"),
          "sign": ("torch.sign", "ops.sign")}

reg.register(nn_block(
    "core.math", "Math (unary)", "Tensor Ops", family="tensor",
    params=(P("op", "enum", "abs", options=tuple(_UNARY)),),
    shape=passthrough,
    torch_expr=lambda c: f"{_UNARY[c['op']][0]}({c['i0']})",
    keras_expr=lambda c: f"{_UNARY[c['op']][1]}({c['i0']})",
    desc="Elementwise math function.",
))


def _l2_shape(in_shapes, params):
    norm_axis(int(params["dim"]), len(in_shapes[0]), "L2 Normalize")
    return list(in_shapes[0])


reg.register(nn_block(
    "core.l2_normalize", "L2 Normalize", "Tensor Ops", family="tensor",
    params=(P("dim", "int", -1, lo=-4, hi=3),), shape=_l2_shape,
    torch_expr="nn.functional.normalize({i0}, p=2.0, dim={torch_dim})",
    keras_expr="layers.UnitNormalization(axis={keras_axis})({i0})",
    desc="Scales vectors along an axis to unit Euclidean length.",
))


def _lambda_checks(p):
    expr = str(p.get("expression", ""))
    if not expr.strip():
        return [("error", "expression is empty")]
    try:
        compile(expr, "<lambda>", "eval")
    except SyntaxError as exc:
        return [("error", f"expression syntax error: {exc.msg} (line {exc.lineno})")]
    if "t" not in expr:
        return [("warning", "expression does not reference the input tensor 't'")]
    return []


reg.register(nn_block(
    "core.lambda", "Lambda", "Tensor Ops", family="tensor",
    params=(P("expression", "str", "t * 2.0",
              help="Python expression over the input tensor 't'. The output shape must "
                   "equal the input shape. Use operators for portability."),),
    shape=passthrough, checks=_lambda_checks,
    torch_expr="(lambda t: {expression})({i0})",
    keras_expr="layers.Lambda(lambda t: {expression})({i0})",
    desc="Custom shape-preserving expression.",
))
