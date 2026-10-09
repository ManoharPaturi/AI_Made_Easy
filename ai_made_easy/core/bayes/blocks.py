"""Bayesian deep-learning layers (MC dropout, Bayes-by-Backprop, mixture density head),
the last-layer Laplace approximation and calibration / conformal evaluation blocks."""
from __future__ import annotations

from dataclasses import replace

from ai_made_easy.core.blocks._dsl import P, ShapeError, nn_block
from ai_made_easy.core.palette import family_color
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import BlockDefinition, PortSpec

CATEGORY = "Bayesian Deep Learning"
STOCHASTIC = ("bayes.mc_dropout", "bayes.linear", "bayes.conv2d")
KL_LAYERS = ("bayes.linear", "bayes.conv2d")
CALIBRATION = ("eval.ece", "eval.temperature_scaling", "eval.conformal")
SAMPLES = P("samples", "int", 30, lo=1, hi=1000,
            help="Forward passes averaged when predicting (the largest value in the design "
                 "is used)")


def _same(in_shapes, p):
    return list(in_shapes[0])


def _linear_shape(in_shapes, p):
    s = in_shapes[0]
    return [*s[:-1], int(p["units"])]


def _linear_params(in_shapes, p):
    f = in_shapes[0][-1]
    return 2 * (f * int(p["units"]) + int(p["units"]))


def _conv_shape(in_shapes, p):
    s = in_shapes[0]
    if len(s) != 3:
        raise ShapeError(f"Bayes Conv2D reads images [C, H, W], got {s}")
    k, st, pad = int(p["kernel_size"]), int(p["stride"]), int(p["padding"])
    h, w = (s[1] + 2 * pad - k) // st + 1, (s[2] + 2 * pad - k) // st + 1
    if h < 1 or w < 1:
        raise ShapeError("the kernel is larger than the padded input")
    return [int(p["out_channels"]), h, w]


def _conv_params(in_shapes, p):
    c, k, o = in_shapes[0][0], int(p["kernel_size"]), int(p["out_channels"])
    return 2 * (c * o * k * k + o)


def _mdn_shape(in_shapes, p):
    s = in_shapes[0]
    if len(s) != 1:
        raise ShapeError(f"the MDN head reads features [F], got {s}: add Flatten before it")
    return [int(p["components"]) * (1 + 2 * int(p["targets"]))]


def _mdn_params(in_shapes, p):
    out = int(p["components"]) * (1 + 2 * int(p["targets"]))
    return in_shapes[0][0] * out + out


def _layers() -> list[BlockDefinition]:
    mc = nn_block(
        "bayes.mc_dropout", "MC Dropout", CATEGORY, family="regularization",
        params=(P("p", "float", 0.1, lo=0.0, hi=0.95), SAMPLES), shape=_same,
        torch=lambda c: f"MCDropout({float(c['p'])}, {int(c['samples'])})",
        torch_helpers=("MCDropout",),
        desc="Dropout that stays on when predicting: averaging several passes gives a "
             "predictive mean and its uncertainty (Monte Carlo dropout).")
    linear = nn_block(
        "bayes.linear", "Bayes Linear", CATEGORY, family="model",
        params=(P("units", "int", 64, lo=1),
                P("prior_sigma", "float", 1.0, lo=1e-6, help="Prior N(0, sigma^2) on weights"),
                SAMPLES),
        shape=_linear_shape, param_fn=_linear_params,
        torch=lambda c: (f"BayesLinear({c['input_shape'][-1]}, {int(c['units'])}, "
                         f"{float(c['prior_sigma'])}, {int(c['samples'])})"),
        torch_helpers=("BayesLinear",),
        desc="Bayes-by-Backprop: every weight is a Gaussian (mean and spread are learned); "
             "the KL to the prior is added to the loss.")
    conv = nn_block(
        "bayes.conv2d", "Bayes Conv2D", CATEGORY, family="conv",
        params=(P("out_channels", "int", 16, lo=1), P("kernel_size", "int", 3, lo=1),
                P("stride", "int", 1, lo=1), P("padding", "int", 1, lo=0),
                P("prior_sigma", "float", 1.0, lo=1e-6), SAMPLES),
        shape=_conv_shape, param_fn=_conv_params,
        torch=lambda c: (f"BayesConv2d({c['input_shape'][0]}, {int(c['out_channels'])}, "
                         f"{int(c['kernel_size'])}, {int(c['stride'])}, {int(c['padding'])}, "
                         f"{float(c['prior_sigma'])}, {int(c['samples'])})"),
        torch_helpers=("BayesConv2d",),
        desc="Bayes-by-Backprop convolution with Gaussian weights.")
    mdn = nn_block(
        "bayes.mdn", "Mixture Density Head", CATEGORY, family="model",
        params=(P("components", "int", 5, lo=1, hi=100),
                P("targets", "int", 1, lo=1, hi=100, help="Number of target columns")),
        shape=_mdn_shape, param_fn=_mdn_params,
        torch=lambda c: (f"MDNHead({c['input_shape'][0]}, {int(c['components'])}, "
                         f"{int(c['targets'])})"),
        torch_helpers=("MDNHead",),
        desc="Predicts a mixture of Gaussians instead of one value: multi-modal targets and "
             "input-dependent noise (trains with the mixture's negative log-likelihood).")
    out = []
    for defn in (mc, linear, conv, mdn):
        role = "distribution" if defn.type_id == "bayes.mdn" else "tensor"
        out.append(replace(defn, outputs=(PortSpec("out", role=role),),
                           meta={**(defn.meta or {}), "bayes": True}))
    return out


def _config() -> list[BlockDefinition]:
    color = family_color("evaluation")

    def block(type_id, name, params, desc, category="Metrics"):
        return BlockDefinition(type_id=type_id, display_name=name, category=category,
                               color=color, params=params, description=desc,
                               library="PyTorch", meta={"kind": "calibration", "bayes": True})

    return [
        block("bayes.laplace", "Laplace Approximation (last layer)",
              (P("prior_precision", "float", 0.0, lo=0.0,
                 help="0: tuned on the validation set"), P("samples", "int", 30, lo=2, hi=1000)),
              "After training, puts a Gaussian posterior on the last layer's weights "
              "(diagonal Laplace): predictions get uncertainty without retraining.",
              CATEGORY),
        block("eval.ece", "Calibration (ECE)", (P("bins", "int", 15, lo=2, hi=100),),
              "Expected calibration error and a reliability diagram: does 80% confidence mean "
              "80% correct?"),
        block("eval.temperature_scaling", "Temperature Scaling", (),
              "Fits one temperature on the validation set that makes the probabilities "
              "calibrated (Guo et al., 2017); applied when serving."),
        block("eval.conformal", "Conformal Prediction",
              (P("alpha", "float", 0.1, lo=0.001, hi=0.5,
                 help="Miss rate: 0.1 gives 90% prediction sets / intervals"),),
              "Split conformal prediction: sets of classes or intervals that contain the "
              "truth with the chosen probability, calibrated on the validation set."),
    ]


def register_all() -> None:
    reg = get_registry()
    for defn in (*_layers(), *_config()):
        reg.register(defn)
