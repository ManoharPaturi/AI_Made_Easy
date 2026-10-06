"""Padding, cropping, upsampling and pixel (un)shuffle."""
from __future__ import annotations

from ai_made_easy.core.blocks._dsl import P, nn_block
from ai_made_easy.core.codegen import KerasUnsupported
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import ShapeError

reg = get_registry()

_RANK_NAMES = {1: "[C, L]", 2: "[C, H, W]", 3: "[C, D, H, W]"}


def _rank_check(s, rank, name):
    if len(s) != rank + 1:
        raise ShapeError(f"{name} expects {_RANK_NAMES[rank]} input, got {s}")


# ------------------------------------------------------------------ padding

def _pad_shape(rank: int, name: str, reflect: bool = False):
    def fn(in_shapes, params):
        s = in_shapes[0]
        _rank_check(s, rank, name)
        p = int(params["padding"])
        if reflect and any(p >= d for d in s[1:]):
            raise ShapeError(f"{name}: padding {p} must be smaller than every "
                             f"spatial size {s[1:]}")
        return [s[0], *(d + 2 * p for d in s[1:])]

    return fn


def _keras_pad(rank: int, mode: str):
    def fn(c):
        p = int(c["padding"])
        if mode == "constant":
            return f"layers.ZeroPadding{rank}D({p})({c['i0']})"
        return f"pad_spatial({c['i0']}, padding={p}, mode=\"{mode}\")"

    return fn


_PAD_KINDS = (
    ("zero", "ZeroPad", "nn.ZeroPad{r}d", "constant", "Zero"),
    ("reflection", "ReflectionPad", "nn.ReflectionPad{r}d", "reflect", "Mirror (reflect)"),
    ("replication", "ReplicationPad", "nn.ReplicationPad{r}d", "edge", "Edge-replicate"),
    ("circular", "CircularPad", "nn.CircularPad{r}d", "wrap", "Circular (wrap-around)"),
)

for _rank in (1, 2, 3):
    for _key, _label, _torch, _kmode, _desc in _PAD_KINDS:
        _name = f"{_label}{_rank}D"
        reg.register(nn_block(
            f"core.{_key}_pad{_rank}d", _name, "Resizing", family="resize",
            params=(P("padding", "int", 1, lo=0, help="Padding on every side"),),
            shape=_pad_shape(_rank, _name, reflect=_key != "zero"), layout="cl",
            torch=_torch.format(r=_rank) + "(padding={padding})",
            keras_expr=_keras_pad(_rank, _kmode),
            keras_helpers=() if _key == "zero" else ("pad_spatial",),
            desc=f"{_desc} padding of every spatial side.",
        ))


# ----------------------------------------------------------------- cropping

def _crop_shape(rank: int):
    def fn(in_shapes, params):
        s = in_shapes[0]
        _rank_check(s, rank, f"Crop{rank}D")
        c = int(params["crop"])
        out = [d - 2 * c for d in s[1:]]
        if any(d <= 0 for d in out):
            raise ShapeError(f"cropping {c} from each side of {s[1:]} leaves nothing")
        return [s[0], *out]

    return fn


def _torch_crop(rank: int):
    def fn(c):
        k = int(c["crop"])
        sl = ", ".join([f"{k}:-{k}" if k else ":"] * rank)
        return f"{c['i0']}[:, :, {sl}]"

    return fn


for _rank in (1, 2, 3):
    reg.register(nn_block(
        f"core.crop{_rank}d", f"Crop{_rank}D", "Resizing", family="resize",
        params=(P("crop", "int", 1, lo=0, help="Pixels removed from every side"),),
        shape=_crop_shape(_rank), layout="cl",
        torch_expr=_torch_crop(_rank), keras=f"layers.Cropping{_rank}D({{crop}})",
        desc="Symmetric cropping of every spatial side.",
    ))


# --------------------------------------------------------------- upsampling

_UP_MODES = {1: ("nearest", "linear"), 2: ("nearest", "bilinear", "bicubic"),
             3: ("nearest", "trilinear")}


def _up_shape(rank: int):
    def fn(in_shapes, params):
        s = in_shapes[0]
        _rank_check(s, rank, f"Upsample{rank}D")
        f = int(params["scale_factor"])
        return [s[0], *(d * f for d in s[1:])]

    return fn


def _torch_up(c):
    mode = c["mode"]
    align = ", align_corners=False" if mode != "nearest" else ""
    return f"nn.Upsample(scale_factor={c['scale_factor']}, mode=\"{mode}\"{align})"


def _keras_up(rank: int):
    def fn(c):
        f, mode = int(c["scale_factor"]), c["mode"]
        if rank == 2:
            interp = {"nearest": "nearest", "bilinear": "bilinear",
                      "bicubic": "bicubic"}[mode]
            return (f"layers.UpSampling2D(size={f}, interpolation=\"{interp}\")"
                    f"({c['i0']})")
        if mode != "nearest":
            raise KerasUnsupported(f"{mode} interpolation in {rank}-D")
        return f"layers.UpSampling{rank}D(size={f})({c['i0']})"

    return fn


for _rank in (1, 2, 3):
    reg.register(nn_block(
        f"core.upsample{_rank}d", f"Upsample{_rank}D", "Resizing", family="resize",
        params=(P("scale_factor", "int", 2, lo=1),
                P("mode", "enum", "nearest", options=_UP_MODES[_rank])),
        shape=_up_shape(_rank), layout="cl", torch=_torch_up, keras_expr=_keras_up(_rank),
        desc=f"Upsamples a {_RANK_NAMES[_rank]} tensor by an integer factor.",
    ))


# ------------------------------------------------------------ pixel shuffle

def _shuffle_shape(in_shapes, params):
    s = in_shapes[0]
    _rank_check(s, 2, "PixelShuffle")
    r = int(params["upscale_factor"])
    if s[0] % (r * r):
        raise ShapeError(f"PixelShuffle: channels {s[0]} must be divisible by "
                         f"upscale_factor² = {r * r}")
    return [s[0] // (r * r), s[1] * r, s[2] * r]


def _unshuffle_shape(in_shapes, params):
    s = in_shapes[0]
    _rank_check(s, 2, "PixelUnshuffle")
    r = int(params["downscale_factor"])
    if s[1] % r or s[2] % r:
        raise ShapeError(f"PixelUnshuffle: height/width {s[1:]} must be divisible "
                         f"by downscale_factor {r}")
    return [s[0] * r * r, s[1] // r, s[2] // r]


reg.register(nn_block(
    "core.pixel_shuffle", "PixelShuffle", "Resizing", family="resize",
    params=(P("upscale_factor", "int", 2, lo=1),),
    shape=_shuffle_shape, layout="cl",
    torch="nn.PixelShuffle(upscale_factor={upscale_factor})",
    desc="Rearranges [C·r², H, W] into [C, H·r, W·r] (sub-pixel convolution).",
))
reg.register(nn_block(
    "core.pixel_unshuffle", "PixelUnshuffle", "Resizing", family="resize",
    params=(P("downscale_factor", "int", 2, lo=1),),
    shape=_unshuffle_shape, layout="cl",
    torch="nn.PixelUnshuffle(downscale_factor={downscale_factor})",
    desc="Rearranges [C, H·r, W·r] into [C·r², H, W].",
))
