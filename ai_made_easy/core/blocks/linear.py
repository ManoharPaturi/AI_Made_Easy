"""Linear layers and embeddings."""
from __future__ import annotations

from ai_made_easy.core.blocks._dsl import P, nn_block
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import ShapeError

reg = get_registry()


def _dense_shape(in_shapes, params):
    s = in_shapes[0]
    return [*s[:-1], int(params["units"])]


def _dense_params(in_shapes, params):
    units = int(params["units"])
    return in_shapes[0][-1] * units + (units if params["bias"] else 0)


reg.register(nn_block(
    "core.dense", "Dense / Linear", "Linear", family="model",
    params=(P("units", "int", 128, lo=1, help="Output features"),
            P("bias", "bool", True, help="Learnable additive bias")),
    shape=_dense_shape, param_fn=_dense_params, layout="ir",
    torch="nn.Linear(in_features={input_size}, out_features={units}, bias={bias})",
    keras="layers.Dense(units={units}, use_bias={bias})",
    desc="Fully connected layer applied to the last axis: y = xWᵀ + b.",
))


def _bilinear_shape(in_shapes, params):
    a, b = in_shapes
    if len(a) != 1 or len(b) != 1:
        raise ShapeError(f"Bilinear expects two flat inputs [F1], [F2]; got {a} and {b}")
    return [int(params["units"])]


reg.register(nn_block(
    "core.bilinear", "Bilinear", "Linear", family="model",
    params=(P("units", "int", 64, lo=1), P("bias", "bool", True)),
    inputs=("in1", "in2"), shape=_bilinear_shape, layout="ir",
    param_fn=lambda s, p: s[0][0] * s[1][0] * int(p["units"]) + (int(p["units"]) if p["bias"] else 0),
    torch=("nn.Bilinear(in1_features={in_features}, in2_features={in2_features}, "
           "out_features={units}, bias={bias})"),
    torch_expr="self.{self_var}({i0}, {i1})",
    desc="Bilinear interaction of two vectors: y = x₁ᵀ A x₂ + b.",
))

reg.register(nn_block(
    "core.identity", "Identity", "Linear", family="model",
    shape=lambda s, p: list(s[0]),
    torch="nn.Identity()", keras="layers.Identity()",
    desc="Pass-through placeholder layer.",
))


# ------------------------------------------------------------- embeddings

def _embedding_shape(in_shapes, params):
    s = in_shapes[0]
    if len(s) not in (1, 2):
        raise ShapeError(f"Embedding expects integer indices [L] or [N, L], got {s}")
    return [*s, int(params["embedding_dim"])]


def _embedding_torch(ctx):
    pad = int(ctx.get("padding_idx", -1))
    pad_arg = f", padding_idx={pad}" if pad >= 0 else ""
    return (f"nn.Embedding(num_embeddings={ctx['num_embeddings']}, "
            f"embedding_dim={ctx['embedding_dim']}{pad_arg})")


def _embedding_checks(p):
    pad = int(p.get("padding_idx", -1))
    if pad >= int(p.get("num_embeddings", 1)):
        return [("error", f"padding_idx {pad} must be smaller than num_embeddings "
                          f"{p.get('num_embeddings')}")]
    return []


reg.register(nn_block(
    "core.embedding", "Embedding", "Embedding", family="model",
    params=(P("num_embeddings", "int", 1000, lo=1, help="Vocabulary size"),
            P("embedding_dim", "int", 64, lo=1),
            P("padding_idx", "int", -1, lo=-1, help="-1 = no padding index")),
    shape=_embedding_shape, layout="ir", input_dtype="int", checks=_embedding_checks,
    param_fn=lambda s, p: int(p["num_embeddings"]) * int(p["embedding_dim"]),
    torch=_embedding_torch,
    keras="layers.Embedding(input_dim={num_embeddings}, output_dim={embedding_dim})",
    desc="Lookup table mapping integer indices to dense vectors.",
))


def _seq_even(in_shapes, params):
    s = in_shapes[0]
    if len(s) != 2:
        raise ShapeError(f"positional encoding expects a sequence [L, D], got {s}")
    if s[1] % 2:
        raise ShapeError(f"sinusoidal encoding needs an even feature size D, got {s[1]}")
    return list(s)


def _seq_any(in_shapes, params):
    s = in_shapes[0]
    if len(s) != 2:
        raise ShapeError(f"positional embedding expects a sequence [L, D], got {s}")
    return list(s)


reg.register(nn_block(
    "core.positional_encoding", "Positional Encoding", "Embedding", family="model",
    shape=_seq_even, layout="ir",
    torch="PositionalEncoding(length={seq_len}, dim={features})",
    torch_helpers=("PositionalEncoding",),
    keras_expr="PositionalEncoding()({i0})", keras_helpers=("positional_encoding",),
    desc="Adds fixed sinusoidal position signals to a [L, D] sequence.",
))

reg.register(nn_block(
    "core.learned_positional", "Learned Positional Embedding", "Embedding", family="model",
    shape=_seq_any, layout="ir",
    param_fn=lambda s, p: s[0][0] * s[0][1],
    torch="LearnedPositionalEmbedding(length={seq_len}, dim={features})",
    torch_helpers=("LearnedPositionalEmbedding",),
    keras_expr="LearnedPositionalEmbedding()({i0})",
    keras_helpers=("learned_positional_embedding",),
    desc="Adds trainable per-position vectors to a [L, D] sequence.",
))
