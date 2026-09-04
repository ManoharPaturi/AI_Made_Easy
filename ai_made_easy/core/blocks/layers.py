"""Core layer blocks: linear, convolutions, pooling, recurrence, embedding.

Fragments follow the BlockDefinition contract: ``pytorch_layer`` is the
__init__ constructor, ``pytorch_expr`` the forward expression (defaults to
``self.{self_var}({i0})``), keras equivalents inline in the functional API.
"""
from __future__ import annotations

from ai_made_easy.core.blocks import _shape
from ai_made_easy.core import summary
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

_LAYER_COLOR = family_color("model")


def _linear_shape(in_shapes, params):
    (s,) = in_shapes
    if len(s) != 1:
        raise ShapeError(
            f"Dense expects a flat input [F] (got {s}); insert a Flatten block"
        )
    return [int(params["units"])]


reg.register(
    BlockDefinition(
        type_id="core.dense",
        display_name="Dense / Linear",
        category="Layers",
        color=_LAYER_COLOR,
        params=(
            ParamSpec(name="units", type="int", default=128, minimum=1),
            ParamSpec(name="bias", type="bool", default=True, help="Learnable bias"),
        ),
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        shape_fn=_linear_shape,
        param_fn=summary.linear_params,
        pytorch_layer="nn.Linear(in_features={in_features}, out_features={units}, bias={bias})",
        keras_layer="layers.Dense(units={units}, use_bias={bias})",
    )
)

reg.register(
    BlockDefinition(
        type_id="core.flatten",
        display_name="Flatten",
        category="Layers",
        color=_LAYER_COLOR,
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        shape_fn=lambda s, p: [shape_volume(s[0])],
        pytorch_layer="nn.Flatten()",
        keras_layer="layers.Flatten()",
    )
)

def _embedding_shape(in_shapes, params):
    (s,) = in_shapes
    if len(s) not in (1, 2):
        raise ShapeError("Embedding expects index input [L] or [B, L]")
    return [s[0], int(params["embedding_dim"])]


reg.register(
    BlockDefinition(
        type_id="core.embedding",
        display_name="Embedding",
        category="Layers",
        color=_LAYER_COLOR,
        params=(
            ParamSpec(name="num_embeddings", type="int", default=1000, minimum=1,
                      help="Vocabulary size"),
            ParamSpec(name="embedding_dim", type="int", default=64, minimum=1),
        ),
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        shape_fn=_embedding_shape,
        param_fn=summary.embedding_params,
        pytorch_layer="nn.Embedding(num_embeddings={num_embeddings}, embedding_dim={embedding_dim})",
        keras_layer="layers.Embedding(input_dim={num_embeddings}, output_dim={embedding_dim})",
    )
)


# --------------------------------------------------------------------- convs

def _conv(rank: int, transpose: bool, type_id: str, name: str) -> BlockDefinition:
    params = (
        ParamSpec(name="out_channels", type="int", default=16, minimum=1),
        ParamSpec(name="kernel_size", type="int", default=3, minimum=1,
                  help="Square kernel; must fit the input size + 2*padding"),
        ParamSpec(name="stride", type="int", default=1, minimum=1,
                  help="Slide per step (>= 1)"),
        ParamSpec(name="padding", type="int", default=1 if not transpose else 0, minimum=0,
                  help="Zero-pad added on each side"),
        ParamSpec(name="dilation", type="int", default=1, minimum=1,
                  help="1 = normal; >1 spreads the kernel over a wider area"),
    )
    if transpose:
        params = params + (
            ParamSpec(name="output_padding", type="int", default=0, minimum=0),
        )
    torch_cls = f"nn.ConvTranspose{rank}d" if transpose else f"nn.Conv{rank}d"
    keras_cls = f"layers.Conv{rank}DTranspose" if transpose else f"layers.Conv{rank}D"
    pad = "(" + ", ".join(["{padding}"] * rank) + ",)"
    return BlockDefinition(
        type_id=type_id,
        display_name=name,
        category="Layers",
        color=_LAYER_COLOR,
        params=params,
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        shape_fn=lambda s, p, r=rank, t=transpose: _shape.conv_nd(s[0], r, p, t),
        param_fn=summary.conv_params(rank),
        pytorch_layer=(
            f"{torch_cls}(in_channels={{in_channels}}, out_channels={{out_channels}}, "
            f"kernel_size={{kernel_size}}, stride={{stride}}, padding={{padding}}, "
            f"dilation={{dilation}}"
            + (", output_padding={output_padding})" if transpose else ")")
        ),
        keras_layer=(
            f"{keras_cls}(filters={{out_channels}}, kernel_size={{kernel_size}}, "
            f"strides={{stride}}, padding={pad}, dilation_rate={{dilation}}"
            + (", output_padding={output_padding})" if transpose else ")")
        ),
    )


reg.register(_conv(1, False, "core.conv1d", "Conv1D"))
reg.register(_conv(2, False, "core.conv2d", "Conv2D"))
reg.register(_conv(3, False, "core.conv3d", "Conv3D"))
reg.register(_conv(2, True, "core.conv_transpose2d", "ConvTranspose2D"))
reg.register(_conv(3, True, "core.conv_transpose3d", "ConvTranspose3D"))


# --------------------------------------------------------------------- pools

def _pool(type_id: str, name: str, rank: int, keras_cls: str) -> BlockDefinition:
    torch_cls = f"nn.MaxPool{rank}d" if "Max" in name else f"nn.AvgPool{rank}d"
    return BlockDefinition(
        type_id=type_id,
        display_name=name,
        category="Layers",
        color=_LAYER_COLOR,
        params=(
            ParamSpec(name="kernel_size", type="int", default=2, minimum=1,
                      help="Window size; must fit the input size + 2*padding"),
            ParamSpec(name="stride", type="int", default=2, minimum=1),
            ParamSpec(name="padding", type="int", default=0, minimum=0),
        ),
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        shape_fn=lambda s, p, r=rank: _shape.pool_nd(s[0], r, p),
        pytorch_layer=(
            f"{torch_cls}(kernel_size={{kernel_size}}, stride={{stride}}, padding={{padding}})"
        ),
        keras_layer=(
            f"{keras_cls}(pool_size={{kernel_size}}, strides={{stride}}, padding='valid')"
        ),
    )


reg.register(_pool("core.maxpool1d", "MaxPool1D", 1, "layers.MaxPooling1D"))
reg.register(_pool("core.maxpool2d", "MaxPool2D", 2, "layers.MaxPooling2D"))
reg.register(_pool("core.maxpool3d", "MaxPool3D", 3, "layers.MaxPooling3D"))
reg.register(_pool("core.avgpool1d", "AvgPool1D", 1, "layers.AveragePooling1D"))
reg.register(_pool("core.avgpool2d", "AvgPool2D", 2, "layers.AveragePooling2D"))


reg.register(
    BlockDefinition(
        type_id="core.adaptive_avgpool2d",
        display_name="AdaptiveAvgPool2D",
        category="Layers",
        color=_LAYER_COLOR,
        params=(
            ParamSpec(name="output_height", type="int", default=1, minimum=1),
            ParamSpec(name="output_width", type="int", default=1, minimum=1),
        ),
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        shape_fn=lambda s, p: [s[0][0], int(p["output_height"]), int(p["output_width"])],
        pytorch_layer="nn.AdaptiveAvgPool2d(output_size=({output_height}, {output_width}))",
    )
)


def _global_pool(type_id, name, torch_expr, keras_layer, ranks):
    def shape_fn(in_shapes, params):
        (s,) = in_shapes
        if len(s) not in ranks:
            raise ShapeError(f"{name} expects {' or '.join(str(r) for r in ranks)}-rank input, got {s}")
        return [s[0]] if len(s) == 3 else [s[-1]]

    return BlockDefinition(
        type_id=type_id,
        display_name=name,
        category="Layers",
        color=_LAYER_COLOR,
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        shape_fn=shape_fn,
        pytorch_expr=torch_expr,
        keras_layer=keras_layer,
    )


reg.register(_global_pool(
    "core.global_avgpool2d", "GlobalAvgPool (Image)",
    "torch.mean({i0}, dim=(2, 3))", "layers.GlobalAveragePooling2D()", ranks=(3,),
))
reg.register(_global_pool(
    "core.global_maxpool2d", "GlobalMaxPool (Image)",
    "torch.amax({i0}, dim=(2, 3))", "layers.GlobalMaxPooling2D()", ranks=(3,),
))
reg.register(_global_pool(
    "core.mean_over_time", "MeanOverTime (Sequence)",
    "torch.mean({i0}, dim=1)", "layers.GlobalAveragePooling1D()", ranks=(2,),
))
reg.register(_global_pool(
    "core.max_over_time", "MaxOverTime (Sequence)",
    "torch.amax({i0}, dim=1)", "layers.GlobalMaxPooling1D()", ranks=(2,),
))


# ---------------------------------------------------------------- recurrence

def _recurrent(type_id: str, name: str, torch_cls: str, keras_cls: str) -> BlockDefinition:
    def shape_fn(in_shapes, params):
        (s,) = in_shapes
        if len(s) != 2:
            raise ShapeError(f"{name} expects sequence input [L, C], got {s}")
        dirs = 2 if params["bidirectional"] else 1
        return [s[0], int(params["hidden_size"]) * dirs]

    return BlockDefinition(
        type_id=type_id,
        display_name=name,
        category="Layers",
        color=_LAYER_COLOR,
        params=(
            ParamSpec(name="hidden_size", type="int", default=64, minimum=1),
            ParamSpec(name="num_layers", type="int", default=1, minimum=1),
            ParamSpec(name="bias", type="bool", default=True),
            ParamSpec(name="bidirectional", type="bool", default=False),
        ),
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        shape_fn=shape_fn,
        param_fn=summary.lstm_params if torch_cls == "nn.LSTM" else summary.gru_params,
        pytorch_layer=(
            f"{torch_cls}(input_size={{input_size}}, hidden_size={{hidden_size}}, "
            f"num_layers={{num_layers}}, bias={{bias}}, batch_first=True, "
            f"bidirectional={{bidirectional}})"
        ),
        pytorch_expr="self.{self_var}({i0})[0]",
        keras_layer=(
            f"{keras_cls}(units={{hidden_size}}, return_sequences=True, "
            f"use_bias={{bias}}, go_backwards={{bidirectional}})"
        ),
    )


reg.register(_recurrent("core.lstm", "LSTM", "nn.LSTM", "layers.LSTM"))
reg.register(_recurrent("core.gru", "GRU", "nn.GRU", "layers.GRU"))
