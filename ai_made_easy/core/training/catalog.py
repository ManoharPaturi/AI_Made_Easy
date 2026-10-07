"""Training components: losses, optimizers, LR schedulers, metrics, trainer.

One table per component kind is the single source of truth: it registers the
canvas blocks *and* drives code generation for PyTorch and Keras 3. Each entry
renders a framework expression from the block's resolved params; ``keras`` is
``None`` when Keras has no faithful equivalent (reported as a clear error).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ai_made_easy.core.palette import family_color
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import BlockDefinition, P

Render = Callable[[dict], str]


@dataclass(frozen=True)
class Component:
    type_id: str
    name: str
    params: tuple = ()
    torch: Render | None = None
    keras: Render | None = None
    desc: str = ""
    meta: dict = field(default_factory=dict)
    checks: Callable | None = None


def _floats(value) -> list[float]:
    return [float(v) for v in str(value).replace(";", ",").split(",") if v.strip()]


def _ints(value) -> list[int]:
    return [int(float(v)) for v in str(value).replace(";", ",").split(",") if v.strip()]


# ===================================================================== losses
# meta: task = multiclass | binary | multilabel | regression | distribution
#       target = int | float ; input = logits | probs | log_probs | raw

def _class_weight_arg(p: dict, torch_kw: str = "weight") -> str:
    weights = _floats(p.get("class_weights", ""))
    return f"{torch_kw}=torch.tensor({weights}), " if weights else ""


def _weights_check(p):
    try:
        _floats(p.get("class_weights", ""))
    except ValueError:
        return [("error", "class_weights must be comma-separated numbers")]
    return []


LOSSES: list[Component] = [
    Component(
        "train.loss_cross_entropy", "Cross-Entropy",
        (P("label_smoothing", "float", 0.0, lo=0.0, hi=1.0),
         P("class_weights", "str", "", help="Optional per-class weights, comma-separated")),
        torch=lambda p: (f"nn.CrossEntropyLoss({_class_weight_arg(p)}"
                         f"label_smoothing={p['label_smoothing']})").replace(", )", ")"),
        keras=lambda p: (f"keras.losses.CategoricalCrossentropy(from_logits=True, "
                         f"label_smoothing={p['label_smoothing']})"
                         if float(p["label_smoothing"]) > 0 else
                         "keras.losses.SparseCategoricalCrossentropy(from_logits=True)"),
        desc="Multi-class classification on logits (softmax applied internally).",
        meta={"task": "multiclass", "target": "int", "input": "logits",
              "keras_one_hot": lambda p: float(p["label_smoothing"]) > 0},
        checks=_weights_check),
    Component(
        "train.loss_nll", "Negative Log-Likelihood",
        (P("class_weights", "str", "", help="Optional per-class weights"),),
        torch=lambda p: f"nn.NLLLoss({_class_weight_arg(p).rstrip(', ')})",
        keras=lambda p: ("lambda y, p: keras.losses.sparse_categorical_crossentropy("
                         "y, ops.exp(p))"),
        desc="Multi-class loss on log-probabilities (pair with a final LogSoftmax).",
        meta={"task": "multiclass", "target": "int", "input": "log_probs"},
        checks=_weights_check),
    Component(
        "train.loss_focal", "Focal Loss",
        (P("gamma", "float", 2.0, lo=0.0, help="Focusing parameter"),
         P("alpha", "float", 0.25, lo=0.0, hi=1.0, help="Balance factor (0 = off)")),
        torch=lambda p: f"FocalLoss(gamma={p['gamma']}, alpha={p['alpha']})",
        keras=lambda p: (f"keras.losses.CategoricalFocalCrossentropy(alpha={p['alpha']}, "
                         f"gamma={p['gamma']}, from_logits=True)"),
        desc="Cross-entropy that down-weights easy examples (class imbalance).",
        meta={"task": "multiclass", "target": "int", "input": "logits",
              "keras_one_hot": lambda p: True, "helper": "FocalLoss"}),
    Component(
        "train.loss_multi_margin", "Multi-Class Hinge",
        (P("margin", "float", 1.0, lo=0.0), P("p", "enum", "1", options=("1", "2"))),
        torch=lambda p: f"nn.MultiMarginLoss(p={p['p']}, margin={p['margin']})",
        desc="Multi-class hinge (SVM-style) loss on scores.",
        meta={"task": "multiclass", "target": "int", "input": "logits"}),
    Component(
        "train.loss_bce_logits", "Binary Cross-Entropy (logits)",
        (P("pos_weight", "float", 1.0, lo=0.0, help="Weight of positive examples"),),
        torch=lambda p: ("nn.BCEWithLogitsLoss()" if float(p["pos_weight"]) == 1.0 else
                         f"nn.BCEWithLogitsLoss(pos_weight=torch.tensor([{p['pos_weight']}]))"),
        keras=lambda p: "keras.losses.BinaryCrossentropy(from_logits=True)",
        desc="Binary (1 unit) or multi-label (K units) classification on logits.",
        meta={"task": "binary", "target": "float", "input": "logits"}),
    Component(
        "train.loss_bce", "Binary Cross-Entropy (probabilities)", (),
        torch=lambda p: "nn.BCELoss()",
        keras=lambda p: "keras.losses.BinaryCrossentropy()",
        desc="Binary / multi-label loss on probabilities (pair with a final Sigmoid).",
        meta={"task": "binary", "target": "float", "input": "probs"}),
    Component(
        "train.loss_multilabel_soft_margin", "Multi-Label Soft Margin", (),
        torch=lambda p: "nn.MultiLabelSoftMarginLoss()",
        keras=lambda p: "keras.losses.BinaryCrossentropy(from_logits=True)",
        desc="Multi-label classification on logits with multi-hot targets.",
        meta={"task": "multilabel", "target": "float", "input": "logits"}),
    Component(
        "train.loss_mse", "Mean Squared Error", (),
        torch=lambda p: "nn.MSELoss()", keras=lambda p: "keras.losses.MeanSquaredError()",
        desc="Regression: mean of squared errors.",
        meta={"task": "regression", "target": "float", "input": "raw"}),
    Component(
        "train.loss_l1", "Mean Absolute Error", (),
        torch=lambda p: "nn.L1Loss()", keras=lambda p: "keras.losses.MeanAbsoluteError()",
        desc="Regression: mean of absolute errors (robust to outliers).",
        meta={"task": "regression", "target": "float", "input": "raw"}),
    Component(
        "train.loss_smooth_l1", "Smooth L1",
        (P("beta", "float", 1.0, lo=1e-6, help="Transition point between L2 and L1"),),
        torch=lambda p: f"nn.SmoothL1Loss(beta={p['beta']})",
        keras=lambda p: (f"lambda y, p: keras.losses.huber(y, p, delta={p['beta']}) "
                         f"/ {p['beta']}"),
        desc="Quadratic near zero, linear beyond beta.",
        meta={"task": "regression", "target": "float", "input": "raw"}),
    Component(
        "train.loss_huber", "Huber",
        (P("delta", "float", 1.0, lo=1e-6),),
        torch=lambda p: f"nn.HuberLoss(delta={p['delta']})",
        keras=lambda p: f"keras.losses.Huber(delta={p['delta']})",
        desc="Huber loss: L2 for small errors, L1 for large ones.",
        meta={"task": "regression", "target": "float", "input": "raw"}),
    Component(
        "train.loss_log_cosh", "Log-Cosh", (),
        torch=lambda p: "lambda pred, target: torch.mean(torch.log(torch.cosh(pred - target)))",
        keras=lambda p: "keras.losses.LogCosh()",
        desc="Smooth regression loss: log(cosh(error)).",
        meta={"task": "regression", "target": "float", "input": "raw"}),
    Component(
        "train.loss_poisson", "Poisson NLL", (),
        torch=lambda p: "nn.PoissonNLLLoss(log_input=True)",
        keras=lambda p: "lambda y, p: keras.losses.poisson(y, ops.exp(p))",
        desc="Count regression; the model outputs log-rates.",
        meta={"task": "regression", "target": "float", "input": "raw"}),
    Component(
        "train.loss_kl_div", "KL Divergence", (),
        torch=lambda p: 'nn.KLDivLoss(reduction="batchmean")',
        keras=lambda p: "lambda y, p: keras.losses.kl_divergence(y, ops.exp(p))",
        desc="Distribution matching; model outputs log-probabilities, targets probabilities.",
        meta={"task": "distribution", "target": "float", "input": "log_probs"}),
]


# ================================================================= optimizers

def _wd(p) -> str:
    wd = float(p.get("weight_decay", 0.0))
    return f", weight_decay={wd}" if wd else ""


def _kwd(p) -> str:
    wd = float(p.get("weight_decay", 0.0))
    return f", weight_decay={wd}" if wd else ""


_LR = lambda default: P("lr", "float", default, lo=0.0, help="Learning rate")  # noqa: E731
_B1 = P("beta1", "float", 0.9, lo=0.0, hi=0.999999)
_B2 = P("beta2", "float", 0.999, lo=0.0, hi=0.999999)
_EPS = lambda d: P("eps", "float", d, lo=0.0)  # noqa: E731
_WD = lambda d=0.0: P("weight_decay", "float", d, lo=0.0)  # noqa: E731


def _sgd_checks(p):
    if p.get("nesterov") and float(p.get("momentum", 0)) <= 0:
        return [("error", "Nesterov momentum requires momentum > 0")]
    return []


OPTIMIZERS: list[Component] = [
    Component(
        "train.sgd", "SGD",
        (_LR(1e-2), P("momentum", "float", 0.9, lo=0.0, hi=0.999999),
         P("nesterov", "bool", False), _WD()),
        torch=lambda p: (f"torch.optim.SGD(model.parameters(), lr={p['lr']}, "
                         f"momentum={p['momentum']}, nesterov={p['nesterov']}{_wd(p)})"),
        keras=lambda p: (f"keras.optimizers.SGD(learning_rate={p['lr']}, "
                         f"momentum={p['momentum']}, nesterov={p['nesterov']}{_kwd(p)})"),
        desc="Stochastic gradient descent with optional (Nesterov) momentum.",
        checks=_sgd_checks),
    Component(
        "train.adam", "Adam",
        (_LR(1e-3), _B1, _B2, _EPS(1e-8), _WD(), P("amsgrad", "bool", False)),
        torch=lambda p: (f"torch.optim.Adam(model.parameters(), lr={p['lr']}, "
                         f"betas=({p['beta1']}, {p['beta2']}), eps={p['eps']}{_wd(p)}"
                         + (", amsgrad=True" if p.get("amsgrad") else "") + ")"),
        keras=lambda p: (f"keras.optimizers.Adam(learning_rate={p['lr']}, beta_1={p['beta1']}, "
                         f"beta_2={p['beta2']}, epsilon={p['eps']}"
                         + (", amsgrad=True" if p.get("amsgrad") else "") + f"{_kwd(p)})"),
        desc="Adaptive moment estimation."),
    Component(
        "train.adamw", "AdamW",
        (_LR(1e-3), _B1, _B2, _EPS(1e-8), _WD(1e-2), P("amsgrad", "bool", False)),
        torch=lambda p: (f"torch.optim.AdamW(model.parameters(), lr={p['lr']}, "
                         f"betas=({p['beta1']}, {p['beta2']}), eps={p['eps']}, "
                         f"weight_decay={p['weight_decay']}"
                         + (", amsgrad=True" if p.get("amsgrad") else "") + ")"),
        keras=lambda p: (f"keras.optimizers.AdamW(learning_rate={p['lr']}, "
                         f"weight_decay={p['weight_decay']}, beta_1={p['beta1']}, "
                         f"beta_2={p['beta2']}, epsilon={p['eps']}"
                         + (", amsgrad=True" if p.get("amsgrad") else "") + ")"),
        desc="Adam with decoupled weight decay."),
    Component(
        "train.nadam", "NAdam",
        (_LR(2e-3), _B1, _B2, _EPS(1e-8), _WD()),
        torch=lambda p: (f"torch.optim.NAdam(model.parameters(), lr={p['lr']}, "
                         f"betas=({p['beta1']}, {p['beta2']}), eps={p['eps']}{_wd(p)})"),
        keras=lambda p: (f"keras.optimizers.Nadam(learning_rate={p['lr']}, beta_1={p['beta1']}, "
                         f"beta_2={p['beta2']}, epsilon={p['eps']}{_kwd(p)})"),
        desc="Adam with Nesterov momentum."),
    Component(
        "train.radam", "RAdam",
        (_LR(1e-3), _B1, _B2, _EPS(1e-8), _WD()),
        torch=lambda p: (f"torch.optim.RAdam(model.parameters(), lr={p['lr']}, "
                         f"betas=({p['beta1']}, {p['beta2']}), eps={p['eps']}{_wd(p)})"),
        desc="Rectified Adam (variance-adaptive warmup)."),
    Component(
        "train.adamax", "Adamax",
        (_LR(2e-3), _B1, _B2, _EPS(1e-8), _WD()),
        torch=lambda p: (f"torch.optim.Adamax(model.parameters(), lr={p['lr']}, "
                         f"betas=({p['beta1']}, {p['beta2']}), eps={p['eps']}{_wd(p)})"),
        keras=lambda p: (f"keras.optimizers.Adamax(learning_rate={p['lr']}, beta_1={p['beta1']}, "
                         f"beta_2={p['beta2']}, epsilon={p['eps']}{_kwd(p)})"),
        desc="Adam variant based on the infinity norm."),
    Component(
        "train.adagrad", "Adagrad",
        (_LR(1e-2), _EPS(1e-10), _WD()),
        torch=lambda p: (f"torch.optim.Adagrad(model.parameters(), lr={p['lr']}, "
                         f"eps={p['eps']}{_wd(p)})"),
        keras=lambda p: (f"keras.optimizers.Adagrad(learning_rate={p['lr']}, "
                         f"epsilon={p['eps']}{_kwd(p)})"),
        desc="Per-parameter learning rates from accumulated squared gradients."),
    Component(
        "train.adadelta", "Adadelta",
        (_LR(1.0), P("rho", "float", 0.9, lo=0.0, hi=1.0), _EPS(1e-6), _WD()),
        torch=lambda p: (f"torch.optim.Adadelta(model.parameters(), lr={p['lr']}, "
                         f"rho={p['rho']}, eps={p['eps']}{_wd(p)})"),
        keras=lambda p: (f"keras.optimizers.Adadelta(learning_rate={p['lr']}, rho={p['rho']}, "
                         f"epsilon={p['eps']}{_kwd(p)})"),
        desc="Adagrad extension with a moving window of gradient updates."),
    Component(
        "train.rmsprop", "RMSprop",
        (_LR(1e-2), P("alpha", "float", 0.99, lo=0.0, hi=1.0), _EPS(1e-8),
         P("momentum", "float", 0.0, lo=0.0, hi=0.999999), P("centered", "bool", False), _WD()),
        torch=lambda p: (f"torch.optim.RMSprop(model.parameters(), lr={p['lr']}, "
                         f"alpha={p['alpha']}, eps={p['eps']}, momentum={p['momentum']}, "
                         f"centered={p['centered']}{_wd(p)})"),
        keras=lambda p: (f"keras.optimizers.RMSprop(learning_rate={p['lr']}, rho={p['alpha']}, "
                         f"momentum={p['momentum']}, epsilon={p['eps']}, "
                         f"centered={p['centered']}{_kwd(p)})"),
        desc="Divides the gradient by a running RMS of recent magnitudes."),
    Component(
        "train.adafactor", "Adafactor",
        (_LR(1e-2), _WD()),
        torch=lambda p: f"torch.optim.Adafactor(model.parameters(), lr={p['lr']}{_wd(p)})",
        keras=lambda p: f"keras.optimizers.Adafactor(learning_rate={p['lr']}{_kwd(p)})",
        desc="Memory-efficient adaptive optimizer with factored second moments."),
    Component(
        "train.asgd", "ASGD",
        (_LR(1e-2), P("lambd", "float", 1e-4, lo=0.0), P("alpha", "float", 0.75, lo=0.0),
         P("t0", "float", 1e6, lo=0.0), _WD()),
        torch=lambda p: (f"torch.optim.ASGD(model.parameters(), lr={p['lr']}, "
                         f"lambd={p['lambd']}, alpha={p['alpha']}, t0={p['t0']}{_wd(p)})"),
        desc="Averaged stochastic gradient descent."),
    Component(
        "train.rprop", "Rprop",
        (_LR(1e-2), P("eta_minus", "float", 0.5, lo=0.0, hi=1.0),
         P("eta_plus", "float", 1.2, lo=1.0)),
        torch=lambda p: (f"torch.optim.Rprop(model.parameters(), lr={p['lr']}, "
                         f"etas=({p['eta_minus']}, {p['eta_plus']}))"),
        desc="Resilient backpropagation (sign-based, full-batch friendly)."),
]


# ================================================================= schedulers
# meta: step = epoch | batch | metric

def _milestones_check(p):
    try:
        ms = _ints(p.get("milestones", ""))
    except ValueError:
        return [("error", "milestones must be comma-separated integers")]
    if ms != sorted(ms) or len(set(ms)) != len(ms):
        return [("error", "milestones must be strictly increasing")]
    return []


def _cyclic_check(p):
    if float(p["base_lr"]) >= float(p["max_lr"]):
        return [("error", "base_lr must be smaller than max_lr")]
    return []


def _keras_closed_form(expr: str) -> Render:
    return lambda p: f"lambda epoch, lr: {expr.format(**p)}"


SCHEDULERS: list[Component] = [
    Component(
        "train.step_lr", "Step LR",
        (P("step_size", "int", 10, lo=1, help="Epochs between decays"),
         P("gamma", "float", 0.1, lo=0.0, hi=1.0)),
        torch=lambda p: (f"torch.optim.lr_scheduler.StepLR(optimizer, step_size={p['step_size']}, "
                         f"gamma={p['gamma']})"),
        keras=_keras_closed_form("LR0 * {gamma} ** (epoch // {step_size})"),
        desc="Multiply the LR by gamma every step_size epochs.", meta={"step": "epoch"}),
    Component(
        "train.multistep_lr", "Multi-Step LR",
        (P("milestones", "str", "30, 60", help="Epochs at which to decay"),
         P("gamma", "float", 0.1, lo=0.0, hi=1.0)),
        torch=lambda p: (f"torch.optim.lr_scheduler.MultiStepLR(optimizer, "
                         f"milestones={_ints(p['milestones'])}, gamma={p['gamma']})"),
        keras=lambda p: (f"lambda epoch, lr: LR0 * {p['gamma']} ** "
                         f"sum(epoch >= m for m in {_ints(p['milestones'])})"),
        desc="Decay by gamma at the given epochs.", meta={"step": "epoch"},
        checks=_milestones_check),
    Component(
        "train.exponential_lr", "Exponential LR",
        (P("gamma", "float", 0.95, lo=0.0, hi=1.0),),
        torch=lambda p: f"torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma={p['gamma']})",
        keras=_keras_closed_form("LR0 * {gamma} ** epoch"),
        desc="Multiply the LR by gamma every epoch.", meta={"step": "epoch"}),
    Component(
        "train.cosine_annealing_lr", "Cosine Annealing",
        (P("T_max", "int", 50, lo=1, help="Epochs per half cosine"),
         P("eta_min", "float", 1e-6, lo=0.0)),
        torch=lambda p: (f"torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, "
                         f"T_max={p['T_max']}, eta_min={p['eta_min']})"),
        keras=_keras_closed_form(
            "{eta_min} + (LR0 - {eta_min}) * (1 + math.cos(math.pi * min(epoch, {T_max}) "
            "/ {T_max})) / 2"),
        desc="Cosine decay from the initial LR to eta_min.", meta={"step": "epoch"}),
    Component(
        "train.warm_restarts_lr", "Cosine Warm Restarts",
        (P("T_0", "int", 10, lo=1), P("T_mult", "int", 1, lo=1),
         P("eta_min", "float", 0.0, lo=0.0)),
        torch=lambda p: (f"torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, "
                         f"T_0={p['T_0']}, T_mult={p['T_mult']}, eta_min={p['eta_min']})"),
        desc="SGDR: cosine cycles that restart, optionally growing.", meta={"step": "epoch"}),
    Component(
        "train.warmup_cosine_lr", "Warmup + Cosine",
        (P("warmup_epochs", "int", 5, lo=0), P("eta_min", "float", 1e-6, lo=0.0)),
        torch=lambda p: (
            "torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=["
            f"torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, "
            f"total_iters={max(int(p['warmup_epochs']), 1)}), "
            f"torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, "
            f"T_max=max(EPOCHS - {p['warmup_epochs']}, 1), eta_min={p['eta_min']})], "
            f"milestones=[{max(int(p['warmup_epochs']), 1)}])"),
        keras=lambda p: (
            f"lambda epoch, lr: LR0 * (epoch + 1) / {max(int(p['warmup_epochs']), 1)} "
            f"if epoch < {p['warmup_epochs']} else {p['eta_min']} + (LR0 - {p['eta_min']}) * "
            f"(1 + math.cos(math.pi * (epoch - {p['warmup_epochs']}) / "
            f"max(EPOCHS - {p['warmup_epochs']}, 1))) / 2"),
        desc="Linear warmup, then cosine decay over the remaining epochs.",
        meta={"step": "epoch"}),
    Component(
        "train.linear_lr", "Linear LR",
        (P("start_factor", "float", 0.1, lo=1e-8, hi=1.0),
         P("end_factor", "float", 1.0, lo=0.0, hi=1.0),
         P("total_iters", "int", 5, lo=1, help="Epochs of the linear ramp")),
        torch=lambda p: (f"torch.optim.lr_scheduler.LinearLR(optimizer, start_factor="
                         f"{p['start_factor']}, end_factor={p['end_factor']}, "
                         f"total_iters={p['total_iters']})"),
        keras=_keras_closed_form(
            "LR0 * ({start_factor} + ({end_factor} - {start_factor}) * "
            "min(epoch, {total_iters}) / {total_iters})"),
        desc="Linearly ramp the LR factor (warmup or decay).", meta={"step": "epoch"}),
    Component(
        "train.constant_lr", "Constant LR (warm phase)",
        (P("factor", "float", 0.333, lo=0.0, hi=1.0), P("total_iters", "int", 5, lo=1)),
        torch=lambda p: (f"torch.optim.lr_scheduler.ConstantLR(optimizer, factor={p['factor']}, "
                         f"total_iters={p['total_iters']})"),
        keras=_keras_closed_form("LR0 * ({factor} if epoch < {total_iters} else 1.0)"),
        desc="Scale the LR by a constant factor for the first epochs.", meta={"step": "epoch"}),
    Component(
        "train.polynomial_lr", "Polynomial LR",
        (P("total_iters", "int", 50, lo=1), P("power", "float", 1.0, lo=0.0)),
        torch=lambda p: (f"torch.optim.lr_scheduler.PolynomialLR(optimizer, "
                         f"total_iters={p['total_iters']}, power={p['power']})"),
        keras=_keras_closed_form(
            "LR0 * (1 - min(epoch, {total_iters}) / {total_iters}) ** {power}"),
        desc="Polynomial decay to zero over total_iters epochs.", meta={"step": "epoch"}),
    Component(
        "train.plateau_lr", "Reduce on Plateau",
        (P("factor", "float", 0.5, lo=0.0, hi=0.999), P("patience", "int", 5, lo=0),
         P("min_lr", "float", 0.0, lo=0.0)),
        torch=lambda p: (f"torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=\"min\", "
                         f"factor={p['factor']}, patience={p['patience']}, min_lr={p['min_lr']})"),
        keras=lambda p: (f"keras.callbacks.ReduceLROnPlateau(monitor=\"val_loss\", "
                         f"factor={p['factor']}, patience={p['patience']}, min_lr={p['min_lr']})"),
        desc="Reduce the LR when validation loss stops improving.", meta={"step": "metric"}),
    Component(
        "train.one_cycle_lr", "One-Cycle",
        (P("max_lr", "float", 1e-2, lo=1e-8), P("pct_start", "float", 0.3, lo=0.0, hi=1.0),
         P("anneal_strategy", "enum", "cos", options=("cos", "linear"))),
        torch=lambda p: (f"torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr={p['max_lr']}, "
                         f"total_steps=EPOCHS * len(train_loader), pct_start={p['pct_start']}, "
                         f"anneal_strategy=\"{p['anneal_strategy']}\")"),
        desc="Super-convergence schedule stepped every batch.", meta={"step": "batch"}),
    Component(
        "train.cyclic_lr", "Cyclic LR",
        (P("base_lr", "float", 1e-4, lo=0.0), P("max_lr", "float", 1e-2, lo=0.0),
         P("step_size_up", "int", 2000, lo=1, help="Batches per half cycle"),
         P("mode", "enum", "triangular", options=("triangular", "triangular2", "exp_range"))),
        torch=lambda p: (f"torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr={p['base_lr']}, "
                         f"max_lr={p['max_lr']}, step_size_up={p['step_size_up']}, "
                         f"mode=\"{p['mode']}\")"),
        desc="Cycle the LR between two bounds every batch.", meta={"step": "batch"},
        checks=_cyclic_check),
]


# ===================================================================== metrics
# meta: tasks = set of task kinds the metric applies to; key = printed name

METRICS: list[Component] = [
    Component("eval.accuracy", "Accuracy", desc="Fraction of correct predictions.",
              meta={"tasks": {"multiclass", "binary", "multilabel"}, "key": "accuracy"}),
    Component("eval.balanced_accuracy", "Balanced Accuracy",
              desc="Mean per-class recall (robust to imbalance).",
              meta={"tasks": {"multiclass", "binary"}, "key": "balanced_accuracy"}),
    Component("eval.top_k_accuracy", "Top-k Accuracy", (P("k", "int", 5, lo=1),),
              desc="Target among the k highest scores.",
              meta={"tasks": {"multiclass"}, "key": "top_k_accuracy"}),
    Component("eval.precision", "Precision",
              (P("average", "enum", "macro", options=("macro", "micro", "weighted")),),
              desc="TP / (TP + FP).", meta={"tasks": {"multiclass", "binary"}, "key": "precision"}),
    Component("eval.recall", "Recall",
              (P("average", "enum", "macro", options=("macro", "micro", "weighted")),),
              desc="TP / (TP + FN).", meta={"tasks": {"multiclass", "binary"}, "key": "recall"}),
    Component("eval.f1", "F1 Score",
              (P("average", "enum", "macro", options=("macro", "micro", "weighted")),),
              desc="Harmonic mean of precision and recall.",
              meta={"tasks": {"multiclass", "binary"}, "key": "f1"}),
    Component("eval.mcc", "Matthews Correlation",
              desc="Correlation between predictions and targets (−1..1).",
              meta={"tasks": {"multiclass", "binary"}, "key": "mcc"}),
    Component("eval.cohen_kappa", "Cohen's Kappa", desc="Agreement beyond chance.",
              meta={"tasks": {"multiclass", "binary"}, "key": "kappa"}),
    Component("eval.roc_auc", "ROC-AUC", desc="Area under the ROC curve (one-vs-rest).",
              meta={"tasks": {"multiclass", "binary", "multilabel"}, "key": "roc_auc"}),
    Component("eval.average_precision", "Average Precision (PR-AUC)",
              desc="Area under the precision-recall curve.",
              meta={"tasks": {"binary", "multilabel", "multiclass"}, "key": "avg_precision"}),
    Component("eval.log_loss", "Log Loss", desc="Cross-entropy of predicted probabilities.",
              meta={"tasks": {"multiclass", "binary"}, "key": "log_loss"}),
    Component("eval.confusion_matrix", "Confusion Matrix",
              (P("labels", "str", "", help="Optional class names, comma-separated"),),
              desc="Counts of actual vs predicted classes (saved to confusion.json).",
              meta={"tasks": {"multiclass", "binary"}, "key": "confusion"}),
    Component("eval.mse", "MSE", desc="Mean squared error.",
              meta={"tasks": {"regression"}, "key": "mse"}),
    Component("eval.rmse", "RMSE", desc="Root mean squared error.",
              meta={"tasks": {"regression"}, "key": "rmse"}),
    Component("eval.mae", "MAE", desc="Mean absolute error.",
              meta={"tasks": {"regression"}, "key": "mae"}),
    Component("eval.median_ae", "Median Absolute Error", desc="Median of absolute errors.",
              meta={"tasks": {"regression"}, "key": "median_ae"}),
    Component("eval.mape", "MAPE", desc="Mean absolute percentage error.",
              meta={"tasks": {"regression"}, "key": "mape"}),
    Component("eval.r2", "R² Score", desc="Coefficient of determination.",
              meta={"tasks": {"regression"}, "key": "r2"}),
    Component("eval.explained_variance", "Explained Variance", desc="1 − Var(err) / Var(y).",
              meta={"tasks": {"regression"}, "key": "explained_variance"}),
]


# ===================================================================== others

TRAINER = Component(
    "train.trainer", "Trainer",
    (P("epochs", "int", 10, lo=1), P("batch_size", "int", 32, lo=1),
     P("device", "enum", "auto", options=("auto", "cpu", "cuda", "mps")),
     P("seed", "int", 42, lo=0),
     P("early_stopping_patience", "int", 0, lo=0, help="Epochs without improvement; 0 = off"),
     P("grad_clip_norm", "float", 0.0, lo=0.0, help="Max gradient norm; 0 = off"),
     P("accumulation_steps", "int", 1, lo=1, help="Batches per optimizer step"),
     P("mixed_precision", "bool", False, help="Automatic mixed precision on CUDA/MPS")),
    desc="Training loop: epochs, batching, device, early stopping, clipping, AMP.")

KFOLD = Component(
    "train.kfold", "K-Fold Cross-Validation",
    (P("k", "int", 5, lo=2, hi=20), P("stratified", "bool", True), P("seed", "int", 42, lo=0)),
    desc="Retrain on k folds and report mean ± std (tabular/array datasets).")

PREDICT = Component(
    "eval.predict", "Prediction Preview",
    (P("n_samples", "int", 5, lo=1, hi=50),
     P("show_probabilities", "bool", True, help="Print the top-3 class probabilities")),
    desc="Print predictions for test examples after training.")


def _losses_by_id() -> dict[str, Component]:
    return {c.type_id: c for c in LOSSES}


LOSS_IDS = tuple(c.type_id for c in LOSSES)
OPTIMIZER_IDS = tuple(c.type_id for c in OPTIMIZERS)
SCHEDULER_IDS = tuple(c.type_id for c in SCHEDULERS)
METRIC_IDS = tuple(c.type_id for c in METRICS)
COMPONENTS: dict[str, Component] = {
    c.type_id: c for c in (*LOSSES, *OPTIMIZERS, *SCHEDULERS, *METRICS, TRAINER, KFOLD, PREDICT)
}

_CATEGORY = {
    **{t: ("Loss", "training") for t in LOSS_IDS},
    **{t: ("Optimizer", "training") for t in OPTIMIZER_IDS},
    **{t: ("Scheduler", "training") for t in SCHEDULER_IDS},
    **{t: ("Metrics", "evaluation") for t in METRIC_IDS},
    "train.trainer": ("Training", "training"),
    "train.kfold": ("Training", "training"),
    "eval.predict": ("Metrics", "evaluation"),
}


def _libraries(c: Component) -> str:
    if c.type_id in METRIC_IDS or c.type_id in ("train.trainer", "train.kfold", "eval.predict"):
        return "PyTorch · Keras"
    return " · ".join(lib for lib, ok in (("PyTorch", c.torch), ("Keras", c.keras)) if ok)


def register_all() -> None:
    reg = get_registry()
    for type_id, comp in COMPONENTS.items():
        category, family = _CATEGORY[type_id]
        reg.register(BlockDefinition(
            type_id=type_id, display_name=comp.name, category=category,
            color=family_color(family), params=tuple(comp.params),
            checks_fn=comp.checks, description=comp.desc, library=_libraries(comp),
            meta={"kind": category.lower(), **{k: v for k, v in comp.meta.items()
                                               if not callable(v)}},
        ))


def render(comp_id: str, params: dict, framework: str) -> str | None:
    comp = COMPONENTS[comp_id]
    fn = comp.torch if framework == "pytorch" else comp.keras
    return fn(params) if fn else None


def resolved(comp_id: str, params: dict | None) -> dict[str, Any]:
    comp = COMPONENTS[comp_id]
    out = {p.name: p.default for p in comp.params}
    out.update(params or {})
    return out
