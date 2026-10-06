"""Merge layers combining two tensors."""
from __future__ import annotations

from ai_made_easy.core.blocks._dsl import P, ensure_same, nn_block, norm_axis
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import ShapeError

reg = get_registry()

_PORTS = ("in1", "in2")

_ELEMENTWISE = [
    ("core.add", "Add", "{i0} + {i1}", "Add", "Elementwise sum (residual / skip connections)."),
    ("core.subtract", "Subtract", "{i0} - {i1}", "Subtract", "Elementwise difference in1 − in2."),
    ("core.multiply", "Multiply", "{i0} * {i1}", "Multiply", "Elementwise product (gating)."),
    ("core.average", "Average", "({i0} + {i1}) / 2", "Average", "Elementwise mean."),
    ("core.maximum", "Maximum", "torch.maximum({i0}, {i1})", "Maximum", "Elementwise maximum."),
    ("core.minimum", "Minimum", "torch.minimum({i0}, {i1})", "Minimum", "Elementwise minimum."),
]

for _tid, _name, _torch, _keras, _desc in _ELEMENTWISE:
    reg.register(nn_block(
        _tid, _name, "Merge", family="merge", inputs=_PORTS,
        shape=lambda s, p, n=_name: ensure_same(s, n),
        torch_expr=_torch, keras_expr=f"layers.{_keras}()([{{i0}}, {{i1}}])",
        desc=_desc,
    ))


def _concat_shape(in_shapes, params):
    a, b = in_shapes
    if len(a) != len(b):
        raise ShapeError(f"Concatenate: input ranks differ ({a} vs {b})")
    axis = norm_axis(int(params["axis"]), len(a), "Concatenate")
    for d in range(len(a)):
        if d != axis and a[d] != b[d]:
            raise ShapeError(f"Concatenate: shapes must match except on axis "
                             f"{params['axis']} ({a} vs {b})")
    out = list(a)
    out[axis] = a[axis] + b[axis]
    return out


reg.register(nn_block(
    "core.concatenate", "Concatenate", "Merge", family="merge", inputs=_PORTS,
    params=(P("axis", "int", -1, lo=-4, hi=3,
              help="Axis without the batch dim: 0 = channels (images) / time (sequences)"),),
    shape=_concat_shape,
    torch_expr="torch.cat([{i0}, {i1}], dim={torch_dim})",
    keras_expr="layers.Concatenate(axis={keras_axis})([{i0}, {i1}])",
    desc="Joins two tensors along an axis.",
))


def _dot_shape(in_shapes, params):
    a, b = in_shapes
    if len(a) != 1 or a != b:
        raise ShapeError(f"Dot expects two vectors of equal length [F]; got {a} and {b}")
    return [1]


reg.register(nn_block(
    "core.dot", "Dot Product", "Merge", family="merge", inputs=_PORTS,
    params=(P("normalize", "bool", False, help="Cosine similarity instead of dot product"),),
    shape=_dot_shape, layout="ir",
    torch_expr=lambda c: (
        f"nn.functional.cosine_similarity({c['i0']}, {c['i1']}, dim=-1).unsqueeze(-1)"
        if c["normalize"] else f"({c['i0']} * {c['i1']}).sum(dim=-1, keepdim=True)"),
    keras_expr=lambda c: (f"layers.Dot(axes=-1, normalize={bool(c['normalize'])})"
                          f"([{c['i0']}, {c['i1']}])"),
    desc="Dot product (or cosine similarity) of two vectors → [1].",
))
