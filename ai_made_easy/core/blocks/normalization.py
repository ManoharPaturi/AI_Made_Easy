"""Normalization + dropout blocks."""
from __future__ import annotations

from ai_made_easy.core.registry import get_registry
from ai_made_easy.core import summary
from ai_made_easy.core.spec import (
    BlockDefinition,
    ParamSpec,
    PortSpec,
    ShapeError,
    require_rank,
)
from ai_made_easy.core.blocks._palette import family_color

reg = get_registry()

_NORM_COLOR = family_color("normalization")


def _passthrough(in_shapes, params):
    (s,) = in_shapes
    return list(s)


def _rank_pass(rank: int, name: str):
    def shape_fn(in_shapes, params):
        require_rank(in_shapes[0], rank, name)
        return list(in_shapes[0])

    return shape_fn


def _batch_norm1d_shape(in_shapes, params):
    (s,) = in_shapes
    if len(s) not in (1, 2):
        raise ShapeError("BatchNorm1D expects [F] or [L, C] input, got {s}")
    return list(s)


reg.register(
    BlockDefinition(
        type_id="core.batch_norm1d",
        display_name="BatchNorm1D",
        category="Normalization",
        color=_NORM_COLOR,
        params=(ParamSpec(name="epsilon", type="float", default=1e-5, minimum=1e-12),
                ParamSpec(name="momentum", type="float", default=0.1, minimum=0.0, maximum=1.0)),
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        shape_fn=_batch_norm1d_shape,
        param_fn=summary.batch_norm_params,
        pytorch_layer="nn.BatchNorm1d(num_features={num_features}, eps={epsilon}, momentum={momentum})",
        keras_layer="layers.BatchNormalization(epsilon={epsilon}, momentum={momentum})",
    )
)

reg.register(
    BlockDefinition(
        type_id="core.batch_norm2d",
        display_name="BatchNorm2D",
        category="Normalization",
        color=_NORM_COLOR,
        params=(ParamSpec(name="epsilon", type="float", default=1e-5, minimum=1e-12),
                ParamSpec(name="momentum", type="float", default=0.1, minimum=0.0, maximum=1.0)),
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        shape_fn=_rank_pass(3, "BatchNorm2D"),
        param_fn=summary.batch_norm_params,
        pytorch_layer="nn.BatchNorm2d(num_features={in_channels}, eps={epsilon}, momentum={momentum})",
        keras_layer="layers.BatchNormalization(epsilon={epsilon}, momentum={momentum})",
    )
)

reg.register(
    BlockDefinition(
        type_id="core.layer_norm",
        display_name="LayerNorm",
        category="Normalization",
        color=_NORM_COLOR,
        params=(ParamSpec(name="epsilon", type="float", default=1e-5, minimum=1e-12),),
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        shape_fn=_passthrough,
        param_fn=summary.layer_norm_params,
        pytorch_layer="nn.LayerNorm(normalized_shape={normalized_shape}, eps={epsilon})",
        keras_layer="layers.LayerNormalization(axis=None, epsilon={epsilon})",
    )
)

reg.register(
    BlockDefinition(
        type_id="core.group_norm",
        display_name="GroupNorm",
        category="Normalization",
        color=_NORM_COLOR,
        params=(ParamSpec(name="num_groups", type="int", default=8, minimum=1),
                ParamSpec(name="epsilon", type="float", default=1e-5, minimum=1e-12)),
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        shape_fn=_rank_pass(3, "GroupNorm"),
        param_fn=summary.group_norm_params,
        pytorch_layer="nn.GroupNorm(num_groups={num_groups}, num_channels={in_channels}, eps={epsilon})",
        keras_layer="layers.GroupNormalization(groups={num_groups}, axis=-1, epsilon={epsilon})",
    )
)

reg.register(
    BlockDefinition(
        type_id="core.dropout",
        display_name="Dropout",
        category="Normalization",
        color=_NORM_COLOR,
        params=(ParamSpec(name="p", type="float", default=0.5, minimum=0.0, maximum=1.0),),
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        shape_fn=_passthrough,
        pytorch_layer="nn.Dropout(p={p})",
        keras_layer="layers.Dropout(rate={p})",
    )
)

reg.register(
    BlockDefinition(
        type_id="core.dropout2d",
        display_name="Dropout2D",
        category="Normalization",
        color=_NORM_COLOR,
        params=(ParamSpec(name="p", type="float", default=0.5, minimum=0.0, maximum=1.0),),
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        shape_fn=_rank_pass(3, "Dropout2D"),
        pytorch_layer="nn.Dropout2d(p={p})",
        keras_layer="layers.SpatialDropout2D(rate={p})",
    )
)
