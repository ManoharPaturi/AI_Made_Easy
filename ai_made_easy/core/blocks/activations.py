"""Activation blocks. Shape passes through; rank must be >= 1."""
from __future__ import annotations

from ai_made_easy.core.registry import get_registry
from ai_made_easy.core import summary
from ai_made_easy.core.spec import BlockDefinition, ParamSpec, PortSpec
from ai_made_easy.core.blocks._palette import family_color

reg = get_registry()

_ACT_COLOR = family_color("activation")


def _passthrough(in_shapes, params):
    (s,) = in_shapes
    return list(s)


def _act(type_id: str, name: str, pt_layer: str, keras_layer: str = "",
         params: tuple = (), param_fn=None) -> BlockDefinition:
    return BlockDefinition(
        type_id=type_id,
        display_name=name,
        category="Activations",
        color=_ACT_COLOR,
        params=params,
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        shape_fn=_passthrough,
        param_fn=param_fn,
        pytorch_layer=pt_layer,
        keras_layer=keras_layer,
    )


_dim_param = (ParamSpec(name="dim", type="int", default=-1, minimum=-4,
                        help="IR axis (no batch dim); -1 = last"),)

reg.register(_act("core.relu", "ReLU", "nn.ReLU()", "layers.ReLU()"))
reg.register(_act("core.leaky_relu", "LeakyReLU",
                  "nn.LeakyReLU(negative_slope={negative_slope})",
                  "layers.LeakyReLU(negative_slope={negative_slope})",
                  params=(ParamSpec(name="negative_slope", type="float", default=0.01, minimum=0.0),)))
reg.register(_act("core.prelu", "PReLU", "nn.PReLU()", "layers.PReLU()",
                  param_fn=summary.prelu_params))
reg.register(_act("core.elu", "ELU", "nn.ELU(alpha={alpha})",
                  "layers.ELU(negative_slope={alpha})",
                  params=(ParamSpec(name="alpha", type="float", default=1.0, minimum=0.0),)))
reg.register(_act("core.selu", "SELU", "nn.SELU()", "layers.SELU()"))
reg.register(_act("core.gelu", "GELU", "nn.GELU()", 'layers.Activation("gelu")'))
reg.register(_act("core.silu", "SiLU", "nn.SiLU()", 'layers.Activation("silu")'))
reg.register(_act("core.mish", "Mish", "nn.Mish()", 'layers.Activation("mish")'))
reg.register(_act("core.tanh", "Tanh", "nn.Tanh()", 'layers.Activation("tanh")'))
reg.register(_act("core.sigmoid", "Sigmoid", "nn.Sigmoid()",
                  'layers.Activation("sigmoid")'))
reg.register(_act("core.softmax", "Softmax", "nn.Softmax(dim={torch_dim})",
                  "layers.Softmax(axis={keras_axis})", params=_dim_param))
reg.register(_act("core.log_softmax", "LogSoftmax", "nn.LogSoftmax(dim={torch_dim})"))
reg.register(_act("core.softplus", "Softplus", "nn.Softplus()",
                  'layers.Activation("softplus")'))
reg.register(_act("core.hardswish", "Hardswish", "nn.Hardswish()",
                  'layers.Activation("hard_swish")'))


def _glu_shape(in_shapes, params):
    (s,) = in_shapes
    if s[-1] % 2 != 0:
        raise ShapeError(f"GLU needs an even last dim, got {s}")
    return [*s[:-1], s[-1] // 2]


reg.register(
    BlockDefinition(
        type_id="core.glu",
        display_name="GLU",
        category="Activations",
        color=_ACT_COLOR,
        params=_dim_param,
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        shape_fn=_glu_shape,
        pytorch_layer="nn.GLU(dim={torch_dim})",
    )
)
