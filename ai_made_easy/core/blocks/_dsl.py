"""Concise constructors for model-flow blocks.

``nn_block`` builds a BlockDefinition with the conventions every tensor
block shares (one ``in``/``out`` port unless told otherwise, family colour,
library badge) so block modules read as a table of layer specs.
"""
from __future__ import annotations

from typing import Any, Callable

from ai_made_easy.core.blocks._palette import family_color
from ai_made_easy.core.spec import (
    BlockDefinition,
    ParamSpec,
    PortSpec,
    ShapeError,
    require_rank,
)


def P(name: str, type_: str, default: Any = None, *, options: tuple = (),
      lo: Any = None, hi: Any = None, help: str = "") -> ParamSpec:  # noqa: A002
    """Short ParamSpec constructor."""
    return ParamSpec(name=name, type=type_, default=default, options=tuple(options),
                     minimum=lo, maximum=hi, help=help)


def nn_block(type_id: str, name: str, category: str, *, family: str,
             shape: Callable, params: tuple = (), inputs: tuple[str, ...] = ("in",),
             torch: Any = "", torch_expr: Any = "", keras: Any = "",
             keras_expr: Any = "", layout: str = "any", desc: str = "",
             param_fn: Callable | None = None, checks: Callable | None = None,
             torch_helpers: tuple[str, ...] = (), keras_helpers: tuple[str, ...] = (),
             input_dtype: str = "float") -> BlockDefinition:
    libs = [lib for lib, ok in (("PyTorch", torch or torch_expr),
                                ("Keras", keras or keras_expr)) if ok]
    return BlockDefinition(
        type_id=type_id,
        display_name=name,
        category=category,
        color=family_color(family),
        params=tuple(params),
        inputs=tuple(PortSpec(n) for n in inputs),
        outputs=(PortSpec("out"),),
        shape_fn=shape,
        param_fn=param_fn,
        checks_fn=checks,
        pytorch_layer=torch,
        pytorch_expr=torch_expr,
        keras_layer=keras,
        keras_expr=keras_expr,
        keras_layout=layout,
        pytorch_helpers=tuple(torch_helpers),
        keras_helpers=tuple(keras_helpers),
        description=desc,
        library=" · ".join(libs),
        input_dtype=input_dtype,
    )


# ----------------------------------------------------------- shape helpers

def passthrough(in_shapes, params):
    return list(in_shapes[0])


def same_rank(*ranks: int, name: str = "block", layout: str = ""):
    """Passthrough that requires one of ``ranks``."""
    names = {1: "[F]", 2: "[C, L]" if layout == "cl" else "[L, C]",
             3: "[C, H, W]", 4: "[C, D, H, W]"}

    def fn(in_shapes, params):
        s = in_shapes[0]
        if len(s) not in ranks:
            want = " or ".join(names.get(r, f"rank {r}") for r in ranks)
            raise ShapeError(f"{name} expects {want} input, got {s}")
        return list(s)

    return fn


def rank_exact(rank: int, name: str):
    def fn(in_shapes, params):
        require_rank(in_shapes[0], rank, name)
        return list(in_shapes[0])

    return fn


def ensure_same(in_shapes, name: str) -> list[int]:
    first = in_shapes[0]
    for other in in_shapes[1:]:
        if other != first:
            raise ShapeError(f"{name} inputs must have identical shapes: "
                             f"{first} vs {other}")
    return list(first)


def norm_axis(axis: int, rank: int, name: str) -> int:
    idx = axis if axis >= 0 else rank + axis
    if not 0 <= idx < rank:
        raise ShapeError(f"{name}: axis {axis} is out of range for a rank-{rank} input")
    return idx


def tuple_literal(values) -> str:
    vals = list(values)
    return "(" + ", ".join(str(v) for v in vals) + ("," if len(vals) == 1 else "") + ")"
