"""Attention blocks (v1: self-attention over a single input sequence)."""
from __future__ import annotations

from ai_made_easy.core.registry import get_registry
from ai_made_easy.core import summary
from ai_made_easy.core.spec import BlockDefinition, ParamSpec, PortSpec, ShapeError
from ai_made_easy.core.blocks._palette import family_color

reg = get_registry()

_ATTN_COLOR = family_color("model")


def _seq_passthrough(in_shapes, params):
    (s,) = in_shapes
    if len(s) != 2:
        raise ShapeError(f"expected sequence input [L, C], got {s}")
    if s[1] % int(params["nhead"]) != 0:
        raise ShapeError(
            f"input channels C={s[1]} must be divisible by nhead={params['nhead']}"
        )
    return list(s)


def _mha_shape(in_shapes, params):
    (s,) = in_shapes
    if len(s) != 2:
        raise ShapeError(f"MultiheadAttention expects [L, C] input, got {s}")
    embed = int(params["embed_dim"]) if params.get("embed_dim") else s[1]
    if embed != s[1]:
        raise ShapeError(
            f"embed_dim={embed} must equal the input channels C={s[1]} "
            f"(or set 0 for auto)"
        )
    if embed % int(params["num_heads"]) != 0:
        raise ShapeError(
            f"embed_dim {embed} must be divisible by num_heads {params['num_heads']}"
        )
    return list(s)


def _mha_keras(ctx):
    embed = ctx["in_shapes"][0][-1]
    key_dim = embed // int(ctx["num_heads"])
    return (
        f"layers.MultiHeadAttention(num_heads={ctx['num_heads']}, "
        f"key_dim={key_dim})({ctx['i0']}, {ctx['i0']})"
    )


def _mha_checks(p: dict) -> list[tuple[str, str]]:
    """embed_dim (when set explicitly) must divide evenly by num_heads."""
    embed, heads = p.get("embed_dim", 0), p.get("num_heads", 4)
    if embed and heads and embed % heads:
        return [("error",
                 f"embed_dim {embed} is not divisible by num_heads {heads} "
                 f"— pick a width that splits into {heads} equal parts "
                 "(or set embed_dim = 0 for auto)")]
    return []


reg.register(
    BlockDefinition(
        type_id="core.multihead_attention",
        display_name="MultiheadAttention",
        category="Attention",
        color=_ATTN_COLOR,
        checks_fn=_mha_checks,
        params=(
            ParamSpec(name="embed_dim", type="int", default=0, minimum=0,
                      help="0 = auto (use input channels)"),
            ParamSpec(name="num_heads", type="int", default=4, minimum=1),
            ParamSpec(name="dropout", type="float", default=0.0, minimum=0.0, maximum=1.0),
        ),
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        shape_fn=_mha_shape,
        param_fn=summary.mha_params,
        pytorch_layer=(
            "nn.MultiheadAttention(embed_dim={in_channels}, num_heads={num_heads}, "
            "dropout={dropout}, batch_first=True)"
        ),
        pytorch_expr="self.{self_var}({i0}, {i0}, {i0})[0]",
        keras_expr=_mha_keras,
    )
)

reg.register(
    BlockDefinition(
        type_id="core.transformer_encoder",
        display_name="TransformerEncoderLayer",
        category="Attention",
        color=_ATTN_COLOR,
        params=(
            ParamSpec(name="nhead", type="int", default=4, minimum=1),
            ParamSpec(name="dim_feedforward", type="int", default=256, minimum=1),
            ParamSpec(name="dropout", type="float", default=0.1,
                      minimum=0.0, maximum=1.0),
        ),
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        shape_fn=_seq_passthrough,
        param_fn=summary.transformer_encoder_params,
        pytorch_layer=(
            "nn.TransformerEncoderLayer(d_model={in_channels}, nhead={nhead}, "
            "dim_feedforward={dim_feedforward}, dropout={dropout}, batch_first=True)"
        ),
        keras_layer=(
            "layers.TransformerEncoder(intermediate_dim={dim_feedforward}, "
            "num_heads={nhead})"
        ),
    )
)
