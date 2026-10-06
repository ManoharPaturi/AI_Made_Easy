"""Pooling layers: windowed, adaptive, global, and sequence reductions."""
from __future__ import annotations

from ai_made_easy.core.blocks import _shape
from ai_made_easy.core.blocks._dsl import P, nn_block
from ai_made_easy.core.codegen import KerasUnsupported
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import ShapeError

reg = get_registry()

_RANK_NAMES = {1: "[C, L]", 2: "[C, H, W]", 3: "[C, D, H, W]"}


def _pool_shape(rank: int, name: str):
    def fn(in_shapes, params):
        s = in_shapes[0]
        if len(s) != rank + 1:
            raise ShapeError(f"{name} expects {_RANK_NAMES[rank]} input, got {s}")
        k, p = int(params["kernel_size"]), int(params["padding"])
        if 2 * p > k:
            raise ShapeError(f"{name}: padding {p} must be at most half the kernel "
                             f"size {k}")
        return _shape.pool_nd(s, rank, params)

    return fn


def _keras_pool(rank: int, kind: str):
    cls = f"layers.{'Max' if kind == 'max' else 'Average'}Pooling{rank}D"

    def fn(c):
        k, s, p = int(c["kernel_size"]), int(c["stride"]), int(c["padding"])
        inp = f"layers.ZeroPadding{rank}D({p})({c['i0']})" if p else c["i0"]
        return f"{cls}(pool_size={k}, strides={s}, padding=\"valid\")({inp})"

    return fn


_POOL_PARAMS = (
    P("kernel_size", "int", 2, lo=1, help="Pooling window size"),
    P("stride", "int", 2, lo=1),
    P("padding", "int", 0, lo=0, help="At most half the kernel size"),
)

for _rank in (1, 2, 3):
    for _kind, _torch in (("max", "MaxPool"), ("avg", "AvgPool")):
        _name = f"{_torch}{_rank}D"
        reg.register(nn_block(
            f"core.{_kind}pool{_rank}d", _name, "Pooling", family="pool",
            params=_POOL_PARAMS, shape=_pool_shape(_rank, _name), layout="cl",
            torch=(f"nn.{_torch}{_rank}d(kernel_size={{kernel_size}}, stride={{stride}}, "
                   f"padding={{padding}})"),
            keras_expr=_keras_pool(_rank, _kind),
            desc=f"{'Maximum' if _kind == 'max' else 'Average'} over sliding "
                 f"{_rank}-D windows.",
        ))


# --------------------------------------------------------------- adaptive

def _adaptive_params(rank: int) -> tuple:
    return (P("output_size", "int", 1, lo=1, help="Target spatial size per axis"),)


def _adaptive_shape(rank: int, name: str):
    def fn(in_shapes, params):
        s = in_shapes[0]
        if len(s) != rank + 1:
            raise ShapeError(f"{name} expects {_RANK_NAMES[rank]} input, got {s}")
        out = int(params["output_size"])
        return [s[0], *([out] * rank)]

    return fn


def _keras_adaptive(rank: int, kind: str):
    def fn(c):
        out = int(c["output_size"])
        spatial = c["input_shape"][1:]
        if out == 1:
            cls = "GlobalMaxPooling" if kind == "max" else "GlobalAveragePooling"
            return f"layers.{cls}{rank}D(keepdims=True)({c['i0']})"
        if any(d % out for d in spatial):
            raise KerasUnsupported(f"input size {spatial} is not divisible by "
                                   f"output_size {out}")
        if len(set(spatial)) != 1:
            pool = "(" + ", ".join(str(d // out) for d in spatial) + ")"
        else:
            pool = str(spatial[0] // out)
        cls = "MaxPooling" if kind == "max" else "AveragePooling"
        return f"layers.{cls}{rank}D(pool_size={pool}, strides={pool})({c['i0']})"

    return fn


for _rank in (1, 2, 3):
    for _kind, _torch in (("avg", "AdaptiveAvgPool"), ("max", "AdaptiveMaxPool")):
        _name = f"{_torch}{_rank}D"
        reg.register(nn_block(
            f"core.adaptive_{_kind}pool{_rank}d", _name, "Pooling", family="pool",
            params=_adaptive_params(_rank), shape=_adaptive_shape(_rank, _name),
            layout="cl", torch=f"nn.{_torch}{_rank}d(output_size={{output_size}})",
            keras_expr=_keras_adaptive(_rank, _kind),
            desc=f"Pools any {_RANK_NAMES[_rank]} input to a fixed output size.",
        ))


# ----------------------------------------------------------------- global

def _global_shape(rank: int, name: str):
    def fn(in_shapes, params):
        s = in_shapes[0]
        if len(s) != rank + 1:
            raise ShapeError(f"{name} expects {_RANK_NAMES[rank]} input, got {s}")
        return [s[0]]

    return fn


_GLOBAL_IDS = {  # keep historical ids for the 2-D variants
    ("avg", 2): "core.global_avgpool2d", ("max", 2): "core.global_maxpool2d",
}

for _rank in (1, 2, 3):
    _dims = "(" + ", ".join(str(d) for d in range(2, _rank + 2)) + ("," if _rank == 1 else "") + ")"
    for _kind, _op, _kcls in (("avg", "mean", "GlobalAveragePooling"),
                              ("max", "amax", "GlobalMaxPooling")):
        _tid = _GLOBAL_IDS.get((_kind, _rank), f"core.global_{_kind}pool{_rank}d")
        _name = f"Global{'Avg' if _kind == 'avg' else 'Max'}Pool{_rank}D"
        reg.register(nn_block(
            _tid, _name, "Pooling", family="pool",
            shape=_global_shape(_rank, _name), layout="cl",
            torch_expr=f"torch.{_op}({{i0}}, dim={_dims})",
            keras=f"layers.{_kcls}{_rank}D()",
            desc=f"{'Mean' if _kind == 'avg' else 'Maximum'} over all spatial "
                 f"positions of {_RANK_NAMES[_rank]} → [C].",
        ))


def _over_time(name: str):
    def fn(in_shapes, params):
        s = in_shapes[0]
        if len(s) != 2:
            raise ShapeError(f"{name} expects a sequence [L, C], got {s}")
        return [s[1]]

    return fn


reg.register(nn_block(
    "core.mean_over_time", "Mean Over Time", "Pooling", family="pool",
    shape=_over_time("Mean Over Time"), layout="ir",
    torch_expr="torch.mean({i0}, dim=1)", keras="layers.GlobalAveragePooling1D()",
    desc="Average a batch-first sequence [L, C] over time → [C].",
))
reg.register(nn_block(
    "core.max_over_time", "Max Over Time", "Pooling", family="pool",
    shape=_over_time("Max Over Time"), layout="ir",
    torch_expr="torch.amax({i0}, dim=1)", keras="layers.GlobalMaxPooling1D()",
    desc="Maximum of a batch-first sequence [L, C] over time → [C].",
))


def _lp_shape(in_shapes, params):
    s = in_shapes[0]
    if len(s) != 3:
        raise ShapeError(f"LPPool2D expects [C, H, W] input, got {s}")
    return _shape.pool_nd(s, 2, {**params, "padding": 0})


reg.register(nn_block(
    "core.lppool2d", "LPPool2D", "Pooling", family="pool",
    params=(P("norm_type", "float", 2.0, lo=1.0, help="p of the p-norm"),
            P("kernel_size", "int", 2, lo=1), P("stride", "int", 2, lo=1)),
    shape=_lp_shape, layout="cl",
    torch="nn.LPPool2d(norm_type={norm_type}, kernel_size={kernel_size}, stride={stride})",
    desc="Power-average pooling: (Σ xᵖ)^(1/p) over windows.",
))
