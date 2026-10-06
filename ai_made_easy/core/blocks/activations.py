"""Activation functions (shape-preserving unless noted)."""
from __future__ import annotations

from ai_made_easy.core.blocks._dsl import P, norm_axis, nn_block, passthrough
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import ShapeError

reg = get_registry()


def _act(type_id, name, torch, keras="", *, params=(), desc="", keras_expr="",
         layout="any", param_fn=None, shape=passthrough):
    return nn_block(type_id, name, "Activations", family="activation", params=params,
                    shape=shape, torch=torch, keras=keras, keras_expr=keras_expr,
                    layout=layout, desc=desc, param_fn=param_fn)


def _kact(name: str, **kwargs) -> str:
    """layers.Activation over keras.activations.<name>, with kwargs if any."""
    if not kwargs:
        return f'layers.Activation("{name}")'
    args = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    return f"layers.Activation(lambda t: keras.activations.{name}(t, {args}))"


_SLOPE = P("negative_slope", "float", 0.01, lo=0.0)
_ALPHA = P("alpha", "float", 1.0, lo=0.0)
_LAMBD = P("lambd", "float", 0.5, lo=0.0, help="Shrinkage threshold λ")

_SIMPLE = [
    ("core.relu", "ReLU", "nn.ReLU()", "layers.ReLU()", "max(0, x)."),
    ("core.relu6", "ReLU6", "nn.ReLU6()", "layers.ReLU(max_value=6.0)", "min(max(0, x), 6)."),
    ("core.selu", "SELU", "nn.SELU()", 'layers.Activation("selu")',
     "Scaled ELU for self-normalizing networks."),
    ("core.silu", "SiLU / Swish", "nn.SiLU()", 'layers.Activation("silu")', "x · sigmoid(x)."),
    ("core.mish", "Mish", "nn.Mish()", 'layers.Activation("mish")', "x · tanh(softplus(x))."),
    ("core.tanh", "Tanh", "nn.Tanh()", 'layers.Activation("tanh")', "Hyperbolic tangent."),
    ("core.sigmoid", "Sigmoid", "nn.Sigmoid()", 'layers.Activation("sigmoid")',
     "Logistic function 1 / (1 + e⁻ˣ)."),
    ("core.hardsigmoid", "HardSigmoid", "nn.Hardsigmoid()", 'layers.Activation("hard_sigmoid")',
     "Piecewise-linear sigmoid: relu6(x + 3) / 6."),
    ("core.hardswish", "HardSwish", "nn.Hardswish()", 'layers.Activation("hard_swish")',
     "x · relu6(x + 3) / 6."),
    ("core.hardtanh", "HardTanh", "nn.Hardtanh()", 'layers.Activation("hard_tanh")',
     "Clamp to [-1, 1]."),
    ("core.softsign", "Softsign", "nn.Softsign()", 'layers.Activation("softsign")',
     "x / (1 + |x|)."),
    ("core.logsigmoid", "LogSigmoid", "nn.LogSigmoid()", 'layers.Activation("log_sigmoid")',
     "log(sigmoid(x))."),
    ("core.tanhshrink", "Tanhshrink", "nn.Tanhshrink()", 'layers.Activation("tanh_shrink")',
     "x − tanh(x)."),
]
for _tid, _name, _t, _k, _d in _SIMPLE:
    reg.register(_act(_tid, _name, _t, _k, desc=_d))

reg.register(_act("core.leaky_relu", "LeakyReLU",
                  "nn.LeakyReLU(negative_slope={negative_slope})",
                  "layers.LeakyReLU(negative_slope={negative_slope})",
                  params=(_SLOPE,), desc="max(0, x) + slope · min(0, x)."))
reg.register(_act("core.prelu", "PReLU", "nn.PReLU()",
                  keras_expr=lambda c: (f"layers.PReLU(shared_axes="
                                        f"{list(range(1, c['in_rank'] + 1))})({c['i0']})"),
                  desc="LeakyReLU with one learnable slope.", param_fn=lambda s, p: 1))
reg.register(_act("core.elu", "ELU", "nn.ELU(alpha={alpha})", "layers.ELU(alpha={alpha})",
                  params=(_ALPHA,), desc="x if x > 0 else α(eˣ − 1)."))
reg.register(_act("core.celu", "CELU", "nn.CELU(alpha={alpha})",
                  keras_expr=lambda c: f"{_kact('celu', alpha=c['alpha'])}({c['i0']})",
                  params=(_ALPHA,), desc="Continuously differentiable ELU."))
reg.register(_act("core.gelu", "GELU", "nn.GELU(approximate=\"{approximate}\")",
                  keras_expr=lambda c: (
                      f"{_kact('gelu', approximate=c['approximate'] == 'tanh')}({c['i0']})"
                      if c["approximate"] == "tanh" else f'layers.Activation("gelu")({c["i0"]})'),
                  params=(P("approximate", "enum", "none", options=("none", "tanh")),),
                  desc="Gaussian error linear unit."))
reg.register(_act("core.softplus", "Softplus", "nn.Softplus(beta={beta})",
                  keras_expr=lambda c: (
                      f'layers.Activation("softplus")({c["i0"]})' if float(c["beta"]) == 1.0
                      else f"layers.Activation(lambda t: keras.activations.softplus("
                           f"t * {c['beta']}) / {c['beta']})({c['i0']})"),
                  params=(P("beta", "float", 1.0, lo=1e-6),),
                  desc="Smooth ReLU: log(1 + e^(βx)) / β."))
reg.register(_act("core.hardshrink", "Hardshrink", "nn.Hardshrink(lambd={lambd})",
                  keras_expr=lambda c: f"{_kact('hard_shrink', threshold=c['lambd'])}({c['i0']})",
                  params=(_LAMBD,), desc="x if |x| > λ else 0."))
reg.register(_act("core.softshrink", "Softshrink", "nn.Softshrink(lambd={lambd})",
                  keras_expr=lambda c: f"{_kact('soft_shrink', threshold=c['lambd'])}({c['i0']})",
                  params=(_LAMBD,), desc="Soft thresholding by λ."))
reg.register(_act("core.threshold", "Threshold", "nn.Threshold(threshold={threshold}, value={value})",
                  keras_expr=lambda c: (
                      _kact("threshold", threshold=c["threshold"], default_value=c["value"])
                      + f"({c['i0']})"),
                  params=(P("threshold", "float", 0.0, lo=-1e9),
                          P("value", "float", 0.0, lo=-1e9)),
                  desc="x if x > threshold else value."))
reg.register(_act("core.rrelu", "RReLU", "nn.RReLU(lower={lower}, upper={upper})",
                  params=(P("lower", "float", 1 / 8, lo=0.0), P("upper", "float", 1 / 3, lo=0.0)),
                  desc="Leaky ReLU with a random slope during training (PyTorch)."))


# --------------------------------------------------------- axis-addressed

_DIM = P("dim", "int", -1, lo=-4, hi=3, help="Axis without the batch dim; -1 = last")


def _axis_shape(name):
    def fn(in_shapes, params):
        s = in_shapes[0]
        norm_axis(int(params["dim"]), len(s), name)
        return list(s)
    return fn


reg.register(_act("core.softmax", "Softmax", "nn.Softmax(dim={torch_dim})",
                  "layers.Softmax(axis={keras_axis})", params=(_DIM,),
                  shape=_axis_shape("Softmax"), desc="Normalizes an axis to probabilities."))
reg.register(_act("core.log_softmax", "LogSoftmax", "nn.LogSoftmax(dim={torch_dim})",
                  keras_expr=lambda c: (f"layers.Activation(lambda t: keras.activations"
                                        f".log_softmax(t, axis={c['keras_axis']}))({c['i0']})"),
                  params=(_DIM,), shape=_axis_shape("LogSoftmax"),
                  desc="log(softmax(x)) — pair with NLLLoss."))
reg.register(_act("core.softmin", "Softmin", "nn.Softmin(dim={torch_dim})",
                  keras_expr=lambda c: f"layers.Softmax(axis={c['keras_axis']})(-{c['i0']})",
                  params=(_DIM,), shape=_axis_shape("Softmin"),
                  desc="softmax(−x) along an axis."))


def _glu_shape(in_shapes, params):
    s = in_shapes[0]
    axis = norm_axis(int(params["dim"]), len(s), "GLU")
    if s[axis] % 2:
        raise ShapeError(f"GLU halves axis {params['dim']}: size {s[axis]} must be even")
    out = list(s)
    out[axis] //= 2
    return out


reg.register(_act("core.glu", "GLU", "nn.GLU(dim={torch_dim})",
                  keras_expr=lambda c: (f"layers.Activation(lambda t: keras.activations"
                                        f".glu(t, axis={c['keras_axis']}))({c['i0']})"),
                  params=(_DIM,), shape=_glu_shape,
                  desc="Gated linear unit: a ⊗ σ(b) over halves of an axis."))
