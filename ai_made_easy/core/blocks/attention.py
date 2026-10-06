"""Attention and transformer blocks over batch-first sequences [L, D]."""
from __future__ import annotations

from ai_made_easy.core.blocks._dsl import P, nn_block
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import ShapeError

reg = get_registry()


def _seq(s, name):
    if len(s) != 2:
        raise ShapeError(f"{name} expects a batch-first sequence [L, D], got {s}")


def _heads_divide(d: int, heads: int, name: str) -> None:
    if d % heads:
        raise ShapeError(f"{name}: model width D={d} is not divisible by "
                         f"{heads} heads")


# ------------------------------------------------------- multi-head attention

def _mha_shape(in_shapes, params):
    s = in_shapes[0]
    _seq(s, "MultiheadAttention")
    embed = int(params.get("embed_dim") or 0) or s[1]
    if embed != s[1]:
        raise ShapeError(f"embed_dim={embed} must equal the input width D={s[1]} "
                         "(set 0 to infer it)")
    _heads_divide(embed, int(params["num_heads"]), "MultiheadAttention")
    return list(s)


def _mha_checks(p):
    embed, heads = int(p.get("embed_dim") or 0), int(p.get("num_heads", 1))
    if embed and heads and embed % heads:
        return [("error", f"embed_dim {embed} is not divisible by num_heads {heads}")]
    return []


def _keras_mha(c):
    d = c["input_shape"][-1]
    drop = f", dropout={c['dropout']}" if float(c["dropout"]) else ""
    return (f"layers.MultiHeadAttention(num_heads={c['num_heads']}, "
            f"key_dim={d // int(c['num_heads'])}{drop})({c['i0']}, {c['i0']})")


reg.register(nn_block(
    "core.multihead_attention", "MultiheadAttention", "Attention", family="attention",
    params=(P("embed_dim", "int", 0, lo=0, help="0 = input width D"),
            P("num_heads", "int", 4, lo=1),
            P("dropout", "float", 0.0, lo=0.0, hi=1.0)),
    shape=_mha_shape, layout="ir", checks=_mha_checks,
    param_fn=lambda s, p: 4 * s[0][-1] * s[0][-1] + 4 * s[0][-1],
    torch=("nn.MultiheadAttention(embed_dim={input_size}, num_heads={num_heads}, "
           "dropout={dropout}, batch_first=True)"),
    torch_expr="self.{self_var}({i0}, {i0}, {i0}, need_weights=False)[0]",
    keras_expr=_keras_mha,
    desc="Scaled dot-product self-attention with multiple heads.",
))


def _cross_shape(in_shapes, params):
    q, kv = in_shapes
    _seq(q, "Cross Attention (query)")
    _seq(kv, "Cross Attention (key/value)")
    if q[1] != kv[1]:
        raise ShapeError(f"query width {q[1]} must equal key/value width {kv[1]}")
    _heads_divide(q[1], int(params["num_heads"]), "Cross Attention")
    return list(q)


def _keras_cross(c):
    d = c["input_shape"][-1]
    return (f"layers.MultiHeadAttention(num_heads={c['num_heads']}, "
            f"key_dim={d // int(c['num_heads'])})({c['i0']}, {c['i1']})")


reg.register(nn_block(
    "core.cross_attention", "Cross Attention", "Attention", family="attention",
    params=(P("num_heads", "int", 4, lo=1), P("dropout", "float", 0.0, lo=0.0, hi=1.0)),
    inputs=("query", "memory"), shape=_cross_shape, layout="ir",
    param_fn=lambda s, p: 4 * s[0][-1] * s[0][-1] + 4 * s[0][-1],
    torch=("nn.MultiheadAttention(embed_dim={input_size}, num_heads={num_heads}, "
           "dropout={dropout}, batch_first=True)"),
    torch_expr="self.{self_var}({i0}, {i1}, {i1}, need_weights=False)[0]",
    keras_expr=_keras_cross,
    desc="Attention from a query sequence to a separate key/value sequence.",
))


# -------------------------------------------------------------- transformers

_TF_PARAMS = (
    P("nhead", "int", 4, lo=1, help="Attention heads; must divide the width D"),
    P("dim_feedforward", "int", 256, lo=1),
    P("dropout", "float", 0.1, lo=0.0, hi=1.0),
    P("activation", "enum", "relu", options=("relu", "gelu")),
    P("norm_first", "bool", False, help="Pre-LN (True) or post-LN (False)"),
)


def _encoder_shape(in_shapes, params):
    s = in_shapes[0]
    _seq(s, "TransformerEncoderLayer")
    _heads_divide(s[1], int(params["nhead"]), "TransformerEncoderLayer")
    return list(s)


def _tf_count(in_shapes, params):
    d, f = in_shapes[0][-1], int(params["dim_feedforward"])
    return (4 * d * d + 4 * d) + (d * f + f + f * d + d) + 4 * d


def _torch_tf(cls: str):
    def fn(c):
        args = [f"d_model={c['input_size']}", f"nhead={c['nhead']}",
                f"dim_feedforward={c['dim_feedforward']}", f"dropout={c['dropout']}"]
        if c["activation"] != "relu":
            args.append(f"activation=\"{c['activation']}\"")
        args.append("batch_first=True")
        if c.get("norm_first"):
            args.append("norm_first=True")
        return f"nn.{cls}({', '.join(args)})"

    return fn


def _keras_encoder(c):
    return (f"transformer_encoder({c['i0']}, num_heads={c['nhead']}, "
            f"ff_dim={c['dim_feedforward']}, dropout={c['dropout']}, "
            f"activation=\"{c['activation']}\", norm_first={bool(c['norm_first'])})")


reg.register(nn_block(
    "core.transformer_encoder", "TransformerEncoderLayer", "Attention",
    family="attention", params=_TF_PARAMS, shape=_encoder_shape, layout="ir",
    param_fn=_tf_count, torch=_torch_tf("TransformerEncoderLayer"),
    keras_expr=_keras_encoder, keras_helpers=("transformer_encoder",),
    desc="Self-attention + feed-forward block with residuals and LayerNorm.",
))


def _decoder_shape(in_shapes, params):
    tgt, mem = in_shapes
    _seq(tgt, "TransformerDecoderLayer (target)")
    _seq(mem, "TransformerDecoderLayer (memory)")
    if tgt[1] != mem[1]:
        raise ShapeError(f"target width {tgt[1]} must equal memory width {mem[1]}")
    _heads_divide(tgt[1], int(params["nhead"]), "TransformerDecoderLayer")
    return list(tgt)


def _keras_decoder(c):
    return (f"transformer_decoder({c['i0']}, {c['i1']}, num_heads={c['nhead']}, "
            f"ff_dim={c['dim_feedforward']}, dropout={c['dropout']}, "
            f"activation=\"{c['activation']}\")")


def _torch_decoder_expr(c):
    return f"self.{c['self_var']}({c['i0']}, {c['i1']})"


reg.register(nn_block(
    "core.transformer_decoder", "TransformerDecoderLayer", "Attention",
    family="attention", params=_TF_PARAMS[:4], inputs=("target", "memory"),
    shape=_decoder_shape, layout="ir",
    param_fn=lambda s, p: _tf_count(s, p) + 4 * s[0][-1] ** 2 + 6 * s[0][-1],
    torch=_torch_tf("TransformerDecoderLayer"), torch_expr=_torch_decoder_expr,
    keras_expr=_keras_decoder, keras_helpers=("transformer_decoder",),
    desc="Self-attention, cross-attention to memory, and feed-forward.",
))


# ------------------------------------------------------- channel attention

def _se_shape(in_shapes, params):
    s = in_shapes[0]
    if len(s) not in (2, 3, 4):
        raise ShapeError(f"Squeeze-Excite expects [C, L], [C, H, W] or [C, D, H, W], got {s}")
    return list(s)


reg.register(nn_block(
    "core.squeeze_excite", "Squeeze-Excite", "Attention", family="attention",
    params=(P("reduction", "int", 16, lo=1, help="Channel reduction ratio"),),
    shape=_se_shape, layout="cl",
    param_fn=lambda s, p: (lambda c, h: c * h + h + h * c + c)(
        s[0][0], max(s[0][0] // int(p["reduction"]), 1)),
    torch="SqueezeExcite(channels={in_channels}, reduction={reduction})",
    torch_helpers=("SqueezeExcite",),
    keras_expr="squeeze_excite({i0}, reduction={reduction})",
    keras_helpers=("squeeze_excite",),
    desc="Channel attention: global pool → bottleneck MLP → per-channel gate.",
))
