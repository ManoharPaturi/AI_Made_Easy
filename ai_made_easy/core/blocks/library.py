"""More library layers (3.5): the torch.nn and Keras 3 layers the catalog still lacked.

Padding with a constant, channel shuffle, cosine similarity and pairwise distance,
EmbeddingBag, Unfold / Fold, fractional max pooling, LP pooling in 1-D and 3-D,
Softmax2d, Unflatten, feature alpha dropout, the full encoder-decoder nn.Transformer,
Keras's ConvLSTM2D and additive (Bahdanau) attention, and spectral-normalized Dense and
Conv2D layers. Every one is checked by the codegen harness (designed shape = real
shape) and, where it has weights, against PyTorch's parameter count. Helpers use
``nn.functional`` (not ``F``) because the generated file imports ``F`` only when a layer
line uses it.
"""
from __future__ import annotations

import math

from ai_made_easy.core.blocks import _shape
from ai_made_easy.core.blocks._dsl import P, ShapeError, nn_block
from ai_made_easy.core.codegen.helpers import KERAS_HELPERS, PYTORCH_HELPERS
from ai_made_easy.core.registry import get_registry

reg = get_registry()
_BEFORE = {b.type_id for b in reg.all()}
_RANKS = {1: "[C, L]", 2: "[C, H, W]", 3: "[C, D, H, W]"}

TORCH = {
    "ConvLSTM2d": '''\
class ConvLSTM2d(nn.Module):
    """Convolutional LSTM (Shi et al., 2015): an LSTM whose gates are convolutions, over
    a sequence of feature maps [B, T, C, H, W] -> the last state [B, hidden, H, W] (or
    every step's)."""

    def __init__(self, in_channels: int, hidden: int, kernel_size: int = 3,
                 return_sequences: bool = False) -> None:
        super().__init__()
        self.hidden, self.return_sequences = hidden, return_sequences
        self.gates = nn.Conv2d(in_channels + hidden, 4 * hidden, kernel_size,
                               padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _, h, w = x.shape
        hx = x.new_zeros(b, self.hidden, h, w)
        cx = torch.zeros_like(hx)
        steps = []
        for i in range(t):
            gi, gf, gg, go = self.gates(torch.cat([x[:, i], hx], dim=1)).chunk(4, dim=1)
            cx = torch.sigmoid(gf) * cx + torch.sigmoid(gi) * torch.tanh(gg)
            hx = torch.sigmoid(go) * torch.tanh(cx)
            steps.append(hx)
        return torch.stack(steps, dim=1) if self.return_sequences else hx
''',
    "AdditiveAttention": '''\
class AdditiveAttention(nn.Module):
    """Additive (Bahdanau-style) attention, as Keras's AdditiveAttention: scores
    sum(scale * tanh(q + k)) over the features, softmax over the keys, then the values."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        bound = (6 / dim) ** 0.5   # Glorot-uniform, as Keras
        self.scale = nn.Parameter(torch.empty(dim).uniform_(-bound, bound))

    def forward(self, query: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        scores = (self.scale * torch.tanh(query.unsqueeze(2) + value.unsqueeze(1))).sum(-1)
        return scores.softmax(dim=-1) @ value
''',
}

KERAS = {
    "pad_constant": '''\
def pad_constant(x, padding: int, value: float):
    """Pad every spatial axis with a constant (channels-last)."""
    widths = [[0, 0]] + [[padding, padding]] * (len(x.shape) - 2) + [[0, 0]]
    return ops.pad(x, widths, constant_values=value)
''',
    "channel_shuffle": '''\
def channel_shuffle(x, groups: int):
    """ShuffleNet's channel shuffle (channels-last): interleave the channel groups."""
    spatial, channels = tuple(x.shape[1:-1]), x.shape[-1]
    x = ops.reshape(x, (-1, *spatial, groups, channels // groups))
    x = ops.swapaxes(x, -1, -2)
    return ops.reshape(x, (-1, *spatial, channels))
''',
}

PYTORCH_HELPERS.update(TORCH)
KERAS_HELPERS.update(KERAS)


def _need(s, rank: int, name: str) -> None:
    if len(s) != rank + 1:
        raise ShapeError(f"{name} expects {_RANKS[rank]} input, got {s}")


# ------------------------------------------------------------------ padding

def _constant_pad_shape(rank: int):
    def fn(in_shapes, params):
        s = in_shapes[0]
        _need(s, rank, f"ConstantPad{rank}D")
        p = int(params["padding"])
        return [s[0], *(d + 2 * p for d in s[1:])]

    return fn


for _rank in (1, 2, 3):
    reg.register(nn_block(
        f"core.constant_pad{_rank}d", f"ConstantPad{_rank}D", "Resizing", family="resize",
        params=(P("padding", "int", 1, lo=0, help="Padding on every side"),
                P("value", "float", 0.0, lo=-1e6, hi=1e6, help="The value padded in")),
        shape=_constant_pad_shape(_rank), layout="cl",
        torch=f"nn.ConstantPad{_rank}d(padding={{padding}}, value={{value}})",
        keras_expr=lambda c: (f"pad_constant({c['i0']}, padding={int(c['padding'])}, "
                              f"value={float(c['value'])})"),
        keras_helpers=("pad_constant",),
        desc="Pads every spatial side with a constant value (zero padding is value 0).",
    ))


# ------------------------------------------------------------------ channel shuffle

def _shuffle_shape(in_shapes, params):
    s = in_shapes[0]
    g = int(params["groups"])
    if len(s) < 2:
        raise ShapeError(f"Channel Shuffle expects [C, ...] feature maps, got {s}")
    if s[0] % g:
        raise ShapeError(f"Channel Shuffle: {s[0]} channels do not split into {g} groups")
    return list(s)


reg.register(nn_block(
    "core.channel_shuffle", "Channel Shuffle", "Resizing", family="resize",
    params=(P("groups", "int", 2, lo=1, help="Channel groups to interleave"),),
    shape=_shuffle_shape, layout="cl",
    torch="nn.ChannelShuffle({groups})",
    keras_expr=lambda c: f"channel_shuffle({c['i0']}, groups={int(c['groups'])})",
    keras_helpers=("channel_shuffle",),
    desc="ShuffleNet's channel shuffle: after a grouped convolution, interleave the "
         "groups so the next one mixes them ([C, ...] → same shape).",
))


# ------------------------------------------------------------------ similarities

def _pair_shape(name: str):
    def fn(in_shapes, params):
        a, b = in_shapes
        if len(a) != 1 or a != b:
            raise ShapeError(f"{name} expects two feature vectors of the same size [F]; "
                             f"got {a} and {b}")
        return [1]

    return fn


reg.register(nn_block(
    "core.cosine_similarity", "Cosine Similarity", "Merge", family="merge",
    params=(P("eps", "float", 1e-8, lo=0.0),),
    inputs=("in1", "in2"), shape=_pair_shape("Cosine Similarity"), layout="ir",
    torch_expr="F.cosine_similarity({i0}, {i1}, dim=-1, eps={eps}).unsqueeze(-1)",
    keras_expr="layers.Dot(axes=-1, normalize=True)([{i0}, {i1}])",
    desc="Cosine of the angle between two feature vectors ([F], [F] → [1], in −1..1).",
))

reg.register(nn_block(
    "core.pairwise_distance", "Pairwise Distance", "Merge", family="merge",
    params=(P("p", "float", 2.0, lo=1.0, help="The p of the p-norm (2 = Euclidean)"),),
    inputs=("in1", "in2"), shape=_pair_shape("Pairwise Distance"), layout="ir",
    torch_expr="F.pairwise_distance({i0}, {i1}, p={p}).unsqueeze(-1)",
    keras_expr=lambda c: (f"ops.expand_dims(ops.power(ops.sum(ops.power(ops.abs("
                          f"{c['i0']} - {c['i1']} + 1e-6), {float(c['p'])}), axis=-1), "
                          f"1.0 / {float(c['p'])}), -1)"),
    desc="The p-norm distance between two feature vectors ([F], [F] → [1]).",
))


# ------------------------------------------------------------------ embedding bag

def _bag_shape(in_shapes, params):
    s = in_shapes[0]
    if len(s) != 1:
        raise ShapeError(f"Embedding Bag expects token ids [L], got {s}")
    return [int(params["embedding_dim"])]


_BAG_REDUCE = {"mean": "ops.mean", "sum": "ops.sum", "max": "ops.max"}

reg.register(nn_block(
    "core.embedding_bag", "Embedding Bag", "Embedding", family="model",
    params=(P("num_embeddings", "int", 1000, lo=1, help="Vocabulary size"),
            P("embedding_dim", "int", 64, lo=1),
            P("mode", "enum", "mean", options=("mean", "sum", "max"))),
    shape=_bag_shape, layout="ir", input_dtype="int",
    param_fn=lambda s, p: int(p["num_embeddings"]) * int(p["embedding_dim"]),
    torch=('nn.EmbeddingBag({num_embeddings}, {embedding_dim}, mode="{mode}")'),
    keras_expr=lambda c: (f"{_BAG_REDUCE[c['mode']]}(layers.Embedding("
                          f"{int(c['num_embeddings'])}, {int(c['embedding_dim'])})"
                          f"({c['i0']}), axis=1)"),
    desc="Embeds a bag of token ids and pools them in one step (mean, sum or max): "
         "[L] → [D]. Cheaper than Embedding then a pooling for bag-of-words models.",
))


# ------------------------------------------------------------------ unfold / fold

def _unfold_shape(in_shapes, params):
    s = in_shapes[0]
    _need(s, 2, "Unfold")
    k, st, pad = int(params["kernel_size"]), int(params["stride"]), int(params["padding"])
    out = [_shape.conv_out(d, k, st, pad, 1) for d in s[1:]]
    if min(out) <= 0:
        raise ShapeError(f"Unfold: a {k}×{k} window does not fit {s[1:]}")
    return [s[0] * k * k, out[0] * out[1]]


reg.register(nn_block(
    "core.unfold", "Unfold (sliding patches)", "Tensor Ops", family="tensor",
    params=(P("kernel_size", "int", 3, lo=1), P("stride", "int", 1, lo=1),
            P("padding", "int", 0, lo=0)),
    shape=_unfold_shape, layout="ir",
    torch="nn.Unfold(kernel_size={kernel_size}, stride={stride}, padding={padding})",
    desc="Every sliding k×k window of a feature map as a column: [C, H, W] → "
         "[C·k·k, windows] (im2col).",
))


def _fold_shape(in_shapes, params):
    s = in_shapes[0]
    if len(s) != 2:
        raise ShapeError(f"Fold expects columns [C·k·k, windows], got {s}")
    k, st, pad = int(params["kernel_size"]), int(params["stride"]), int(params["padding"])
    h, w = int(params["height"]), int(params["width"])
    if s[0] % (k * k):
        raise ShapeError(f"Fold: {s[0]} rows are not a multiple of k·k = {k * k}")
    windows = _shape.conv_out(h, k, st, pad, 1) * _shape.conv_out(w, k, st, pad, 1)
    if windows != s[1]:
        raise ShapeError(f"Fold: a {h}×{w} output has {windows} windows, the input {s[1]}")
    return [s[0] // (k * k), h, w]


reg.register(nn_block(
    "core.fold", "Fold (patches → map)", "Tensor Ops", family="tensor",
    params=(P("height", "int", 8, lo=1), P("width", "int", 8, lo=1),
            P("kernel_size", "int", 3, lo=1), P("stride", "int", 1, lo=1),
            P("padding", "int", 0, lo=0)),
    shape=_fold_shape, layout="ir",
    torch=("nn.Fold(output_size=({height}, {width}), kernel_size={kernel_size}, "
           "stride={stride}, padding={padding})"),
    desc="The inverse of Unfold: sums sliding-window columns back into a feature map "
         "[C·k·k, windows] → [C, H, W] (overlaps add up).",
))


# ------------------------------------------------------------------ pooling

def _fractional_shape(rank: int):
    def fn(in_shapes, params):
        s = in_shapes[0]
        _need(s, rank, f"FractionalMaxPool{rank}D")
        r = float(params["output_ratio"])
        k = int(params["kernel_size"])
        out = [int(d * r) for d in s[1:]]
        if min(out) < 1 or any(o + k - 1 > d for o, d in zip(out, s[1:], strict=True)):
            raise ShapeError(f"FractionalMaxPool{rank}D: ratio {r} with a {k}-wide window "
                             f"does not fit {s[1:]}")
        return [s[0], *out]

    return fn


for _rank in (2, 3):
    reg.register(nn_block(
        f"core.fractional_maxpool{_rank}d", f"FractionalMaxPool{_rank}D", "Pooling",
        family="pool",
        params=(P("kernel_size", "int", 2, lo=1),
                P("output_ratio", "float", 0.5, lo=0.05, hi=0.99,
                  help="Output size as a fraction of the input (any, not only 1/2)")),
        shape=_fractional_shape(_rank), layout="cl",
        torch=(f"nn.FractionalMaxPool{_rank}d(kernel_size={{kernel_size}}, "
               "output_ratio={output_ratio})"),
        desc="Max pooling by a fractional factor, over randomly placed windows during "
             "training (Graham, 2014): gentler downsampling than ×½.",
    ))


def _lp_shape(rank: int):
    def fn(in_shapes, params):
        s = in_shapes[0]
        _need(s, rank, f"LPPool{rank}D")
        return _shape.pool_nd(s, rank, {**params, "padding": 0})

    return fn


for _rank in (1, 3):
    reg.register(nn_block(
        f"core.lppool{_rank}d", f"LPPool{_rank}D", "Pooling", family="pool",
        params=(P("norm_type", "float", 2.0, lo=1.0, help="p of the p-norm"),
                P("kernel_size", "int", 2, lo=1), P("stride", "int", 2, lo=1)),
        shape=_lp_shape(_rank), layout="cl",
        torch=(f"nn.LPPool{_rank}d(norm_type={{norm_type}}, kernel_size={{kernel_size}}, "
               "stride={stride})"),
        desc="Power-average pooling: (Σ xᵖ)^(1/p) over windows (p = 1: sum, p → ∞: max).",
    ))


# ------------------------------------------------------------------ softmax2d, unflatten

def _softmax2d_shape(in_shapes, params):
    s = in_shapes[0]
    _need(s, 2, "Softmax2D")
    return list(s)


reg.register(nn_block(
    "core.softmax2d", "Softmax2D", "Activations", family="activation",
    shape=_softmax2d_shape, layout="cl",
    torch="nn.Softmax2d()", keras="layers.Softmax(axis=-1)",
    desc="Softmax over the channels at every pixel ([C, H, W] → same): per-pixel class "
         "probabilities for segmentation.",
))


def _unflatten_shape(in_shapes, params):
    s = list(in_shapes[0])
    dim = int(params["dim"])
    sizes = _shape.parse_int_list(params["sizes"])
    if not -len(s) <= dim < len(s):
        raise ShapeError(f"Unflatten: dim {dim} is outside a {len(s)}-axis input")
    if math.prod(sizes) != s[dim]:
        raise ShapeError(f"Unflatten: {sizes} has {math.prod(sizes)} elements, axis {dim} "
                         f"has {s[dim]}")
    dim %= len(s)
    return s[:dim] + sizes + s[dim + 1:]


reg.register(nn_block(
    "core.unflatten", "Unflatten", "Tensor Ops", family="tensor",
    params=(P("dim", "int", 0, lo=-8, hi=8, help="The axis to split (0 = the first sample axis)"),
            P("sizes", "str", "4, 8", help="Its new sizes; their product must match")),
    shape=_unflatten_shape, layout="ir",
    torch_expr=lambda c: (f"{c['i0']}.unflatten({int(c['dim']) % c['in_rank'] + 1}, "
                          f"({', '.join(str(v) for v in _shape.parse_int_list(c['sizes']))},))"),
    keras_expr=lambda c: (f"layers.Reshape({tuple(_unflatten_shape([c['input_shape']], c))})"
                          f"({c['i0']})"),
    desc="Splits one axis into several, e.g. [64] → [4, 16] (the inverse of Flatten on "
         "that axis).",
))


# ------------------------------------------------------------------ dropout

reg.register(nn_block(
    "core.feature_alpha_dropout", "Feature Alpha Dropout", "Regularization",
    family="regularization",
    params=(P("p", "float", 0.1, lo=0.0, hi=0.99),),
    shape=lambda in_shapes, params: list(in_shapes[0]), layout="any",
    torch="nn.FeatureAlphaDropout(p={p})",
    desc="Alpha dropout of whole channels: keeps SELU networks self-normalizing while "
         "dropping feature maps (training only).",
))


# ------------------------------------------------------------------ the full Transformer

def _transformer_shape(in_shapes, params):
    src, tgt = in_shapes
    if len(src) != 2 or len(tgt) != 2 or src[1] != tgt[1]:
        raise ShapeError(f"Transformer expects a source [L, E] and a target [T, E] with "
                         f"the same E; got {src} and {tgt}")
    if src[1] % int(params["nhead"]):
        raise ShapeError(f"Transformer: d_model {src[1]} is not divisible by nhead "
                         f"{params['nhead']}")
    return list(tgt)


def _transformer_params(in_shapes, params) -> int:
    e, f = in_shapes[0][1], int(params["dim_feedforward"])
    attn = 4 * e * e + 4 * e
    ff = 2 * e * f + f + e
    enc = attn + ff + 4 * e
    dec = 2 * attn + ff + 6 * e
    return (int(params["num_encoder_layers"]) * enc + int(params["num_decoder_layers"]) * dec
            + 4 * e)   # + the encoder's and the decoder's final LayerNorm


def _keras_transformer(c) -> str:
    x = c["i0"]
    for _ in range(int(c["num_encoder_layers"])):
        x = (f"transformer_encoder({x}, num_heads={int(c['nhead'])}, "
             f"ff_dim={int(c['dim_feedforward'])}, dropout={float(c['dropout'])})")
    memory = f"layers.LayerNormalization(epsilon=1e-5)({x})"
    y = c["i1"]
    for _ in range(int(c["num_decoder_layers"])):
        y = (f"transformer_decoder({y}, {memory}, num_heads={int(c['nhead'])}, "
             f"ff_dim={int(c['dim_feedforward'])}, dropout={float(c['dropout'])})")
    return f"layers.LayerNormalization(epsilon=1e-5)({y})"


reg.register(nn_block(
    "core.transformer", "Transformer (encoder-decoder)", "Attention", family="attention",
    params=(P("nhead", "int", 4, lo=1), P("num_encoder_layers", "int", 2, lo=1),
            P("num_decoder_layers", "int", 2, lo=1), P("dim_feedforward", "int", 256, lo=1),
            P("dropout", "float", 0.1, lo=0.0, hi=0.9),
            P("causal", "bool", True, help="The target sees only earlier target steps")),
    inputs=("src", "tgt"), shape=_transformer_shape, param_fn=_transformer_params,
    layout="ir",
    torch=("nn.Transformer(d_model={input_size}, nhead={nhead}, "
           "num_encoder_layers={num_encoder_layers}, num_decoder_layers={num_decoder_layers}, "
           "dim_feedforward={dim_feedforward}, dropout={dropout}, batch_first=True)"),
    torch_expr=lambda c: (
        f"self.{c['self_var']}({c['i0']}, {c['i1']}"
        + (f", tgt_mask=nn.Transformer.generate_square_subsequent_mask({c['i1']}.size(1), "
           f"device={c['i1']}.device), tgt_is_causal=True" if c["causal"] else "") + ")"),
    keras_expr=_keras_transformer, keras_helpers=("transformer_encoder", "transformer_decoder"),
    desc="The original encoder-decoder Transformer (Vaswani et al., 2017), as nn.Transformer: "
         "the source [L, E] is encoded, the target [T, E] attends to it → [T, E]. For "
         "translation and other sequence-to-sequence tasks.",
))


# ------------------------------------------------------------------ ConvLSTM2D

def _convlstm_shape(in_shapes, params):
    s = in_shapes[0]
    if len(s) != 4:
        raise ShapeError(f"ConvLSTM2D expects a sequence of feature maps [T, C, H, W], "
                         f"got {s}")
    k = int(params["kernel_size"])
    if k % 2 == 0:
        raise ShapeError("ConvLSTM2D: kernel_size must be odd (the maps keep their size)")
    t, _, h, w = s
    hidden = int(params["filters"])
    return [t, hidden, h, w] if params["return_sequences"] else [hidden, h, w]


def _convlstm_params(in_shapes, params) -> int:
    c, hidden, k = in_shapes[0][1], int(params["filters"]), int(params["kernel_size"])
    return (c + hidden) * 4 * hidden * k * k + 4 * hidden


def _keras_convlstm(c) -> str:
    seq = bool(c["return_sequences"])
    x = f"layers.Permute((1, 3, 4, 2))({c['i0']})"   # [T, C, H, W] -> [T, H, W, C]
    y = (f"layers.ConvLSTM2D({int(c['filters'])}, {int(c['kernel_size'])}, padding=\"same\", "
         f"return_sequences={seq})({x})")
    return (f"layers.Permute((1, 4, 2, 3))({y})" if seq    # back to [T, hidden, H, W]
            else f"layers.Permute((3, 1, 2))({y})")       # back to [hidden, H, W]


reg.register(nn_block(
    "core.conv_lstm2d", "ConvLSTM2D", "Recurrent", family="recurrent",
    params=(P("filters", "int", 16, lo=1, help="Hidden channels"),
            P("kernel_size", "int", 3, lo=1, help="Odd: the maps keep their size"),
            P("return_sequences", "bool", False, help="Every step's state, not the last")),
    shape=_convlstm_shape, param_fn=_convlstm_params, layout="ir",
    torch=lambda c: (f"ConvLSTM2d({c['in_shapes'][0][1]}, {int(c['filters'])}, "
                     f"kernel_size={int(c['kernel_size'])}, "
                     f"return_sequences={bool(c['return_sequences'])})"),
    torch_helpers=("ConvLSTM2d",),
    keras_expr=_keras_convlstm,
    desc="An LSTM over a sequence of feature maps whose gates are convolutions (Shi et "
         "al., 2015), as Keras's ConvLSTM2D: [T, C, H, W] → [hidden, H, W]. For video "
         "frames, radar and weather maps.",
))


# ------------------------------------------------------------------ additive attention

def _additive_shape(in_shapes, params):
    q, v = in_shapes
    if len(q) != 2 or len(v) != 2 or q[1] != v[1]:
        raise ShapeError(f"Additive Attention expects a query [Tq, D] and values [Tv, D] "
                         f"with the same D; got {q} and {v}")
    return list(q)


reg.register(nn_block(
    "core.additive_attention", "Additive Attention", "Attention", family="attention",
    inputs=("query", "value"), shape=_additive_shape, layout="ir",
    param_fn=lambda s, p: s[0][1],
    torch=lambda c: f"AdditiveAttention({c['in_shapes'][0][1]})",
    torch_expr="self.{self_var}({i0}, {i1})", torch_helpers=("AdditiveAttention",),
    keras_expr="layers.AdditiveAttention()([{i0}, {i1}])",
    desc="Additive (Bahdanau, 2015) attention, as Keras's AdditiveAttention: each query "
         "step scores every value with a learned tanh comparison ([Tq, D], [Tv, D] → "
         "[Tq, D]).",
))


# ------------------------------------------------------------------ spectral norm

def _sn_dense_shape(in_shapes, params):
    s = in_shapes[0]
    return [*s[:-1], int(params["units"])]


reg.register(nn_block(
    "core.spectral_norm_dense", "Spectral-Norm Dense", "Linear", family="model",
    params=(P("units", "int", 128, lo=1), P("bias", "bool", True)),
    shape=_sn_dense_shape, layout="ir",
    param_fn=lambda s, p: s[0][-1] * int(p["units"]) + (int(p["units"]) if p["bias"] else 0),
    torch=("nn.utils.parametrizations.spectral_norm(nn.Linear({input_size}, {units}, "
           "bias={bias}))"),
    keras="layers.SpectralNormalization(layers.Dense({units}, use_bias={bias}))",
    desc="A Dense layer whose weight is divided by its largest singular value (Miyato et "
         "al., 2018): keeps a GAN discriminator or critic Lipschitz-smooth.",
))


def _sn_conv_shape(in_shapes, params):
    s = in_shapes[0]
    _need(s, 2, "Spectral-Norm Conv2D")
    return _shape.conv_nd(s, 2, {**params, "dilation": 1})


reg.register(nn_block(
    "core.spectral_norm_conv2d", "Spectral-Norm Conv2D", "Convolution", family="conv",
    params=(P("out_channels", "int", 32, lo=1), P("kernel_size", "int", 3, lo=1),
            P("stride", "int", 1, lo=1), P("padding", "int", 1, lo=0)),
    shape=_sn_conv_shape, layout="cl",
    param_fn=lambda s, p: s[0][0] * int(p["out_channels"]) * int(p["kernel_size"]) ** 2
    + int(p["out_channels"]),
    torch=("nn.utils.parametrizations.spectral_norm(nn.Conv2d({in_channels}, {out_channels}, "
           "{kernel_size}, stride={stride}, padding={padding}))"),
    keras_expr=lambda c: (
        (f"layers.SpectralNormalization(layers.Conv2D({int(c['out_channels'])}, "
         f"{int(c['kernel_size'])}, strides={int(c['stride'])}, padding=\"valid\"))("
         f"layers.ZeroPadding2D({int(c['padding'])})({c['i0']}))")
        if int(c["padding"]) else
        (f"layers.SpectralNormalization(layers.Conv2D({int(c['out_channels'])}, "
         f"{int(c['kernel_size'])}, strides={int(c['stride'])}))({c['i0']})")),
    desc="A Conv2D whose kernel is divided by its largest singular value (Miyato et al., "
         "2018), the usual layer of a spectrally normalized GAN discriminator.",
))


LIBRARY_IDS = tuple(sorted({b.type_id for b in reg.all()} - _BEFORE))
