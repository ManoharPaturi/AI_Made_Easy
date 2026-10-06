"""Convolution layers (channels-first IR: [C, L], [C, H, W], [C, D, H, W])."""
from __future__ import annotations

from ai_made_easy.core.blocks import _shape
from ai_made_easy.core.blocks._dsl import P, nn_block
from ai_made_easy.core.codegen import KerasUnsupported
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import ShapeError

reg = get_registry()

_RANK_NAMES = {1: "[C, L]", 2: "[C, H, W]", 3: "[C, D, H, W]"}


def _conv_params(transpose: bool, groups: bool = True) -> tuple:
    params = (
        P("out_channels", "int", 16, lo=1, help="Number of output feature maps"),
        P("kernel_size", "int", 3, lo=1, help="Square/cubic kernel size"),
        P("stride", "int", 1, lo=1),
        P("padding", "int", 0 if transpose else 1, lo=0,
          help="Implicit zero padding on every side"),
        P("dilation", "int", 1, lo=1, help="Spacing between kernel elements"),
    )
    if transpose:
        params += (P("output_padding", "int", 0, lo=0,
                     help="Extra size added to one side of the output"),)
    if groups:
        params += (P("groups", "int", 1, lo=1,
                     help="Blocked connections; in/out channels must divide by it"),)
    return params + (P("bias", "bool", True),)


def _conv_shape(rank: int, transpose: bool):
    def fn(in_shapes, params):
        s = in_shapes[0]
        if len(s) != rank + 1:
            raise ShapeError(f"Conv{rank}D expects {_RANK_NAMES[rank]} input, got {s}")
        groups = int(params.get("groups", 1))
        cin, cout = s[0], int(params["out_channels"])
        if cin % groups:
            raise ShapeError(f"input channels {cin} are not divisible by groups={groups}")
        if cout % groups:
            raise ShapeError(f"out_channels {cout} is not divisible by groups={groups}")
        if transpose:
            op = int(params.get("output_padding", 0))
            if op >= max(int(params["stride"]), int(params["dilation"])):
                raise ShapeError(
                    f"output_padding {op} must be smaller than stride or dilation")
        return _shape.conv_nd(s, rank, params, transpose)

    return fn


def _conv_param_count(rank: int):
    def fn(in_shapes, params):
        cin, cout = in_shapes[0][0], int(params["out_channels"])
        groups = int(params.get("groups", 1))
        k = int(params["kernel_size"]) ** rank
        return cin // groups * cout * k + (cout if params.get("bias", True) else 0)

    return fn


def _torch_conv(rank: int, transpose: bool):
    cls = f"nn.ConvTranspose{rank}d" if transpose else f"nn.Conv{rank}d"

    def fn(c):
        args = [f"in_channels={c['in_channels']}", f"out_channels={c['out_channels']}",
                f"kernel_size={c['kernel_size']}", f"stride={c['stride']}",
                f"padding={c['padding']}"]
        if transpose:
            args.append(f"output_padding={c['output_padding']}")
        if int(c.get("groups", 1)) != 1:
            args.append(f"groups={c['groups']}")
        if int(c["dilation"]) != 1:
            args.append(f"dilation={c['dilation']}")
        if not c.get("bias", True):
            args.append("bias=False")
        return f"{cls}({', '.join(args)})"

    return fn


def keras_padding(c: dict, rank: int) -> tuple[str, str]:
    """(padding mode, input expr) reproducing torch's symmetric zero padding."""
    k, s, p, d = (int(c["kernel_size"]), int(c["stride"]), int(c["padding"]),
                  int(c.get("dilation", 1)))
    if p == 0:
        return "valid", c["i0"]
    if s == 1 and 2 * p == d * (k - 1):
        return "same", c["i0"]
    return "valid", f"layers.ZeroPadding{rank}D({p})({c['i0']})"


def _keras_conv(rank: int, transpose: bool):
    def fn(c):
        k, s, d = int(c["kernel_size"]), int(c["stride"]), int(c["dilation"])
        bias = "" if c.get("bias", True) else ", use_bias=False"
        if transpose:
            if int(c.get("groups", 1)) != 1:
                raise KerasUnsupported("grouped transposed convolution")
            if s > 1 and d > 1:
                raise KerasUnsupported("stride > 1 together with dilation > 1")
            dil = f", dilation_rate={d}" if d != 1 else ""
            layer = (f"layers.Conv{rank}DTranspose(filters={c['out_channels']}, "
                     f"kernel_size={k}, strides={s}, padding=\"valid\", "
                     f"output_padding={c['output_padding']}{dil}{bias})")
            expr = f"{layer}({c['i0']})"
            p = int(c["padding"])
            return f"layers.Cropping{rank}D({p})({expr})" if p else expr
        if s > 1 and d > 1:
            raise KerasUnsupported("stride > 1 together with dilation > 1")
        mode, inp = keras_padding(c, rank)
        groups = int(c.get("groups", 1))
        g = f", groups={groups}" if groups != 1 else ""
        dil = f", dilation_rate={d}" if d != 1 else ""
        return (f"layers.Conv{rank}D(filters={c['out_channels']}, kernel_size={k}, "
                f"strides={s}, padding=\"{mode}\"{dil}{g}{bias})({inp})")

    return fn


_DESCS = {
    False: "{r}-D convolution over a channels-first {shape} tensor.",
    True: "{r}-D transposed convolution (learned upsampling) over {shape}.",
}

for _rank in (1, 2, 3):
    for _transpose in (False, True):
        _tid = f"core.conv_transpose{_rank}d" if _transpose else f"core.conv{_rank}d"
        _name = f"ConvTranspose{_rank}D" if _transpose else f"Conv{_rank}D"
        reg.register(nn_block(
            _tid, _name, "Convolution", family="conv",
            params=_conv_params(_transpose), shape=_conv_shape(_rank, _transpose),
            param_fn=_conv_param_count(_rank), layout="cl",
            torch=_torch_conv(_rank, _transpose), keras_expr=_keras_conv(_rank, _transpose),
            desc=_DESCS[_transpose].format(r=_rank, shape=_RANK_NAMES[_rank]),
        ))


# ---------------------------------------------------- depthwise / separable

def _dw_params() -> tuple:
    return (
        P("depth_multiplier", "int", 1, lo=1, help="Output maps per input channel"),
        P("kernel_size", "int", 3, lo=1),
        P("stride", "int", 1, lo=1),
        P("padding", "int", 1, lo=0),
        P("dilation", "int", 1, lo=1),
        P("bias", "bool", True),
    )


def _dw_shape(rank: int):
    def fn(in_shapes, params):
        s = in_shapes[0]
        if len(s) != rank + 1:
            raise ShapeError(f"DepthwiseConv{rank}D expects {_RANK_NAMES[rank]} input, got {s}")
        out = _shape.conv_nd(s, rank, {**params, "out_channels": 1})
        return [s[0] * int(params["depth_multiplier"]), *out[1:]]

    return fn


def _torch_dw(rank: int):
    def fn(c):
        cin = c["in_channels"]
        bias = "" if c["bias"] else ", bias=False"
        return (f"nn.Conv{rank}d({cin}, {cin * int(c['depth_multiplier'])}, "
                f"kernel_size={c['kernel_size']}, stride={c['stride']}, "
                f"padding={c['padding']}, dilation={c['dilation']}, groups={cin}{bias})")

    return fn


def _keras_dw(rank: int):
    def fn(c):
        if int(c["stride"]) > 1 and int(c["dilation"]) > 1:
            raise KerasUnsupported("stride > 1 together with dilation > 1")
        mode, inp = keras_padding(c, rank)
        bias = "" if c["bias"] else ", use_bias=False"
        return (f"layers.DepthwiseConv{rank}D(kernel_size={c['kernel_size']}, "
                f"strides={c['stride']}, padding=\"{mode}\", "
                f"depth_multiplier={c['depth_multiplier']}, "
                f"dilation_rate={c['dilation']}{bias})({inp})")

    return fn


def _sep_params() -> tuple:
    return (P("out_channels", "int", 32, lo=1),) + _dw_params()


def _sep_shape(rank: int):
    dw = _dw_shape(rank)

    def fn(in_shapes, params):
        out = dw(in_shapes, params)
        return [int(params["out_channels"]), *out[1:]]

    return fn


def _torch_sep(rank: int):
    def fn(c):
        cin = c["in_channels"]
        mid = cin * int(c["depth_multiplier"])
        bias = "" if c["bias"] else ", bias=False"
        return (f"nn.Sequential(nn.Conv{rank}d({cin}, {mid}, kernel_size={c['kernel_size']}, "
                f"stride={c['stride']}, padding={c['padding']}, dilation={c['dilation']}, "
                f"groups={cin}, bias=False), nn.Conv{rank}d({mid}, {c['out_channels']}, "
                f"kernel_size=1{bias}))")

    return fn


def _keras_sep(rank: int):
    def fn(c):
        if int(c["stride"]) > 1 and int(c["dilation"]) > 1:
            raise KerasUnsupported("stride > 1 together with dilation > 1")
        mode, inp = keras_padding(c, rank)
        bias = "" if c["bias"] else ", use_bias=False"
        return (f"layers.SeparableConv{rank}D(filters={c['out_channels']}, "
                f"kernel_size={c['kernel_size']}, strides={c['stride']}, "
                f"padding=\"{mode}\", depth_multiplier={c['depth_multiplier']}, "
                f"dilation_rate={c['dilation']}{bias})({inp})")

    return fn


def _dw_count(rank):
    def fn(s, p):
        cin, m, k = s[0][0], int(p["depth_multiplier"]), int(p["kernel_size"]) ** rank
        return cin * m * k + (cin * m if p["bias"] else 0)
    return fn


def _sep_count(rank):
    def fn(s, p):
        cin, m, k = s[0][0], int(p["depth_multiplier"]), int(p["kernel_size"]) ** rank
        out = int(p["out_channels"])
        return cin * m * k + cin * m * out + (out if p["bias"] else 0)
    return fn


for _rank in (1, 2):
    reg.register(nn_block(
        f"core.depthwise_conv{_rank}d", f"DepthwiseConv{_rank}D", "Convolution",
        family="conv", params=_dw_params(), shape=_dw_shape(_rank), layout="cl",
        param_fn=_dw_count(_rank), torch=_torch_dw(_rank), keras_expr=_keras_dw(_rank),
        desc=f"Per-channel {_rank}-D convolution (groups = input channels).",
    ))
    reg.register(nn_block(
        f"core.separable_conv{_rank}d", f"SeparableConv{_rank}D", "Convolution",
        family="conv", params=_sep_params(), shape=_sep_shape(_rank), layout="cl",
        param_fn=_sep_count(_rank), torch=_torch_sep(_rank), keras_expr=_keras_sep(_rank),
        desc=f"Depthwise {_rank}-D convolution followed by a 1×1 pointwise convolution.",
    ))
