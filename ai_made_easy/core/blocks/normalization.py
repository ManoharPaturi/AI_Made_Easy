"""Normalization layers and regularization (dropout, noise)."""
from __future__ import annotations

from ai_made_easy.core.blocks._dsl import P, nn_block, passthrough, rank_exact, same_rank
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import ShapeError, shape_volume

reg = get_registry()

_EPS = P("epsilon", "float", 1e-5, lo=1e-12, hi=1.0)
_MOMENTUM = P("momentum", "float", 0.1, lo=0.0, hi=1.0,
              help="PyTorch convention: weight of the new batch statistic")
_AFFINE = P("affine", "bool", True, help="Learnable scale and shift")


def _keras_bn(c):
    momentum = round(1.0 - float(c["momentum"]), 6)
    affine = "" if c.get("affine", True) else ", center=False, scale=False"
    return (f"layers.BatchNormalization(axis=-1, momentum={momentum}, "
            f"epsilon={c['epsilon']}{affine})({c['i0']})")


def _torch_bn(rank: int):
    def fn(c):
        affine = "" if c.get("affine", True) else ", affine=False"
        return (f"nn.BatchNorm{rank}d(num_features={c['num_features']}, eps={c['epsilon']}, "
                f"momentum={c['momentum']}{affine})")
    return fn


def _bn_count(in_shapes, params):
    return 2 * in_shapes[0][0] if params.get("affine", True) else 0


for _rank, _ranks, _shape_name in ((1, (1, 2), "[F] or [C, L]"), (2, (3,), "[C, H, W]"),
                                   (3, (4,), "[C, D, H, W]")):
    reg.register(nn_block(
        f"core.batch_norm{_rank}d", f"BatchNorm{_rank}D", "Normalization",
        family="normalization", params=(_EPS, _MOMENTUM, _AFFINE),
        shape=same_rank(*_ranks, name=f"BatchNorm{_rank}D", layout="cl"), layout="cl",
        param_fn=_bn_count, torch=_torch_bn(_rank), keras_expr=_keras_bn,
        desc=f"Batch normalization over channel axis of {_shape_name}.",
    ))


# ----------------------------------------------------------------- layer norm

def _ln_torch(c):
    shape = c["normalized_shape"] if c["normalized_dims"] == "all" else f"[{c['features']}]"
    affine = "" if c.get("affine", True) else ", elementwise_affine=False"
    return f"nn.LayerNorm(normalized_shape={shape}, eps={c['epsilon']}{affine})"


def _ln_keras(c):
    axis = (str(list(range(1, c["in_rank"] + 1))) if c["normalized_dims"] == "all"
            and c["in_rank"] > 1 else "-1")
    affine = "" if c.get("affine", True) else ", center=False, scale=False"
    return f"layers.LayerNormalization(axis={axis}, epsilon={c['epsilon']}{affine})({c['i0']})"


def _ln_count(in_shapes, params):
    if not params.get("affine", True):
        return 0
    s = in_shapes[0]
    return 2 * (shape_volume(s) if params["normalized_dims"] == "all" else s[-1])


reg.register(nn_block(
    "core.layer_norm", "LayerNorm", "Normalization", family="normalization",
    params=(P("normalized_dims", "enum", "last", options=("last", "all"),
              help="Normalize the last axis (transformers) or the whole sample"),
            _EPS, _AFFINE),
    shape=passthrough, layout="ir", param_fn=_ln_count,
    torch=_ln_torch, keras_expr=_ln_keras,
    desc="Per-sample normalization over the last axis or every axis.",
))


def _rms_shape(in_shapes, params):
    return list(in_shapes[0])


reg.register(nn_block(
    "core.rms_norm", "RMSNorm", "Normalization", family="normalization",
    params=(P("epsilon", "float", 1e-6, lo=1e-12, hi=1.0),),
    shape=_rms_shape, layout="ir", param_fn=lambda s, p: s[0][-1],
    torch="nn.RMSNorm(normalized_shape={features}, eps={epsilon})",
    keras_expr="layers.RMSNormalization(axis=-1, epsilon={epsilon})({i0})",
    desc="Root-mean-square normalization over the last axis (LLaMA-style).",
))


# --------------------------------------------------------------- group norm

def _group_shape(in_shapes, params):
    s = in_shapes[0]
    if len(s) < 2:
        raise ShapeError(f"GroupNorm expects a channels-first tensor [C, ...], got {s}")
    groups = int(params["num_groups"])
    if s[0] % groups:
        raise ShapeError(f"GroupNorm: channels {s[0]} are not divisible by "
                         f"num_groups={groups}")
    return list(s)


reg.register(nn_block(
    "core.group_norm", "GroupNorm", "Normalization", family="normalization",
    params=(P("num_groups", "int", 8, lo=1, help="Must divide the channel count"),
            _EPS, _AFFINE),
    shape=_group_shape, layout="cl",
    param_fn=lambda s, p: 2 * s[0][0] if p.get("affine", True) else 0,
    torch=lambda c: (f"nn.GroupNorm(num_groups={c['num_groups']}, "
                     f"num_channels={c['in_channels']}, eps={c['epsilon']}"
                     + ("" if c.get("affine", True) else ", affine=False") + ")"),
    keras_expr=lambda c: (f"layers.GroupNormalization(groups={c['num_groups']}, axis=-1, "
                          f"epsilon={c['epsilon']}"
                          + ("" if c.get("affine", True) else ", center=False, scale=False")
                          + f")({c['i0']})"),
    desc="Normalizes groups of channels per sample (batch-size independent).",
))

for _rank, _shape_name in ((1, "[C, L]"), (2, "[C, H, W]"), (3, "[C, D, H, W]")):
    reg.register(nn_block(
        f"core.instance_norm{_rank}d", f"InstanceNorm{_rank}D", "Normalization",
        family="normalization",
        params=(_EPS, P("affine", "bool", False, help="Learnable scale and shift")),
        shape=rank_exact(_rank + 1, f"InstanceNorm{_rank}D"), layout="cl",
        param_fn=lambda s, p: 2 * s[0][0] if p.get("affine") else 0,
        torch=lambda c, r=_rank: (f"nn.InstanceNorm{r}d(num_features={c['in_channels']}, "
                                  f"eps={c['epsilon']}"
                                  + (", affine=True" if c.get("affine") else "") + ")"),
        keras_expr=lambda c: (f"layers.GroupNormalization(groups={c['in_channels']}, axis=-1, "
                              f"epsilon={c['epsilon']}"
                              + ("" if c.get("affine") else ", center=False, scale=False")
                              + f")({c['i0']})"),
        desc=f"Per-sample, per-channel normalization of {_shape_name}.",
    ))

reg.register(nn_block(
    "core.local_response_norm", "LocalResponseNorm", "Normalization",
    family="normalization",
    params=(P("size", "int", 5, lo=1, help="Neighbouring channels"),
            P("alpha", "float", 1e-4, lo=0.0), P("beta", "float", 0.75, lo=0.0),
            P("k", "float", 1.0, lo=0.0)),
    shape=same_rank(2, 3, 4, name="LocalResponseNorm"), layout="cl",
    torch="nn.LocalResponseNorm(size={size}, alpha={alpha}, beta={beta}, k={k})",
    desc="AlexNet-style normalization across neighbouring channels (PyTorch).",
))


# ------------------------------------------------------------ regularization

_P = P("p", "float", 0.5, lo=0.0, hi=1.0, help="Drop probability")

reg.register(nn_block(
    "core.dropout", "Dropout", "Regularization", family="regularization",
    params=(_P,), shape=passthrough,
    torch="nn.Dropout(p={p})", keras="layers.Dropout(rate={p})",
    desc="Randomly zeroes elements during training.",
))

for _rank, _shape_name in ((1, "[C, L]"), (2, "[C, H, W]"), (3, "[C, D, H, W]")):
    reg.register(nn_block(
        f"core.dropout{_rank}d", f"Dropout{_rank}D", "Regularization",
        family="regularization", params=(_P,),
        shape=rank_exact(_rank + 1, f"Dropout{_rank}D"), layout="cl",
        torch=f"nn.Dropout{_rank}d(p={{p}})", keras=f"layers.SpatialDropout{_rank}D(rate={{p}})",
        desc=f"Drops whole channels of {_shape_name} during training.",
    ))

reg.register(nn_block(
    "core.alpha_dropout", "AlphaDropout", "Regularization", family="regularization",
    params=(_P,), shape=passthrough,
    torch="nn.AlphaDropout(p={p})", keras="layers.AlphaDropout(rate={p})",
    desc="Dropout that preserves SELU self-normalization.",
))

reg.register(nn_block(
    "core.gaussian_noise", "GaussianNoise", "Regularization", family="regularization",
    params=(P("stddev", "float", 0.1, lo=0.0),), shape=passthrough,
    torch="GaussianNoise(stddev={stddev})", torch_helpers=("GaussianNoise",),
    keras="layers.GaussianNoise(stddev={stddev})",
    desc="Adds zero-mean Gaussian noise during training.",
))

reg.register(nn_block(
    "core.gaussian_dropout", "GaussianDropout", "Regularization", family="regularization",
    params=(_P,), shape=passthrough,
    torch="GaussianDropout(p={p})", torch_helpers=("GaussianDropout",),
    keras="layers.GaussianDropout(rate={p})",
    desc="Multiplicative 1-centred Gaussian noise during training.",
))
