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


# Metric learning (3.3): the model outputs an embedding (its last layer's width is the
# embedding size) and the loss learns how classes sit in that space. Each one also scores
# embeddings against its class centres, so accuracy and the other metrics still work.
_CLASSES = P("classes", "int", 10, lo=2, help="Number of classes in the data")
_METRIC_META = {"task": "multiclass", "target": "int", "input": "embedding", "metric": True}


def _margin(kind: str) -> Render:
    return lambda p: (f"MarginSoftmaxLoss(EMBED_DIM, NUM_CLASSES, scale={p['scale']}, "
                      f"margin={p['margin']}, kind=\"{kind}\")")


LOSSES += [
    Component(
        "train.loss_arcface", "ArcFace",
        (_CLASSES, P("scale", "float", 30.0, lo=1.0, help="Logit scale s"),
         P("margin", "float", 0.5, lo=0.0, hi=1.5, help="Additive angular margin (radians)")),
        torch=_margin("arcface"),
        desc="Additive angular margin loss (Deng et al., 2019): embeddings of a class "
             "gather around a learned centre with a margin between classes. Face and "
             "product recognition, verification. The trained network outputs embeddings.",
        meta={**_METRIC_META, "torch_helper": "MarginSoftmaxLoss"}),
    Component(
        "train.loss_cosface", "CosFace",
        (_CLASSES, P("scale", "float", 30.0, lo=1.0, help="Logit scale s"),
         P("margin", "float", 0.35, lo=0.0, hi=1.0, help="Additive cosine margin")),
        torch=_margin("cosface"),
        desc="Large margin cosine loss (Wang et al., 2018): like ArcFace with the margin "
             "on the cosine; a little easier to train. The network outputs embeddings.",
        meta={**_METRIC_META, "torch_helper": "MarginSoftmaxLoss"}),
    Component(
        "train.loss_triplet", "Triplet (batch-hard)",
        (_CLASSES, P("margin", "float", 0.3, lo=0.0, help="Distance margin")),
        torch=lambda p: f"TripletLoss(EMBED_DIM, NUM_CLASSES, margin={p['margin']})",
        desc="For each example, the farthest same-class and the nearest other-class "
             "example in the batch (Hermans et al., 2017): same-class embeddings end up "
             "closer by a margin. Use batches with several examples per class.",
        meta={**_METRIC_META, "torch_helper": "TripletLoss"}),
    Component(
        "train.loss_supcon", "Supervised Contrastive",
        (_CLASSES, P("temperature", "float", 0.1, lo=1e-3, hi=1.0)),
        torch=lambda p: f"SupConLoss(EMBED_DIM, NUM_CLASSES, temperature={p['temperature']})",
        desc="InfoNCE / NT-Xent with labels (Khosla et al., 2020): every same-class pair "
             "in the batch is a positive, everything else a negative.",
        meta={**_METRIC_META, "torch_helper": "SupConLoss"}),
]
LOSSES += [
    Component(
        "train.loss_nt_xent", "NT-Xent (SimCLR)",
        (_CLASSES, P("temperature", "float", 0.5, lo=1e-3, hi=2.0)),
        torch=lambda p: f"NTXentLoss(EMBED_DIM, NUM_CLASSES, temperature={p['temperature']})",
        desc="Self-supervised contrastive learning (Chen et al., 2020): every training image "
             "is augmented twice and the network learns to match the two views among all "
             "the others in the batch. Labels are never used to learn, only to measure the "
             "embedding (accuracy of the nearest class average). Images, with augmentation.",
        meta={**_METRIC_META, "torch_helper": "NTXentLoss", "views": 2}),
]
# Pair losses (3.4): pairs come from the labels in each batch, so they work with any dataset
# that has classes, with no separate pairs file.
LOSSES += [
    Component(
        "train.loss_cosine_embedding", "Cosine Embedding (pairs)",
        (_CLASSES, P("margin", "float", 0.2, lo=-1.0, hi=1.0,
                     help="Different-class pairs are pushed below this cosine")),
        torch=lambda p: f"CosineEmbeddingPairsLoss(EMBED_DIM, NUM_CLASSES, margin={p['margin']})",
        desc="Every pair of examples in the batch: same class pulls the cosine towards 1, "
             "different classes push it below the margin (torch's CosineEmbeddingLoss on "
             "pairs made from the labels). Verification and duplicate detection.",
        meta={**_METRIC_META, "torch_helper": "CosineEmbeddingPairsLoss"}),
    Component(
        "train.loss_margin_ranking", "Margin Ranking (pairs)",
        (_CLASSES, P("margin", "float", 0.2, lo=0.0, hi=2.0,
                     help="How much more similar a same-class example must be")),
        torch=lambda p: f"MarginRankingPairsLoss(EMBED_DIM, NUM_CLASSES, margin={p['margin']})",
        desc="Ranks a same-class example above an other-class one, by a margin of cosine "
             "similarity, for every such pair around each example in the batch (torch's "
             "MarginRankingLoss on pairs made from the labels). Retrieval and search.",
        meta={**_METRIC_META, "torch_helper": "MarginRankingPairsLoss"}),
]
METRIC_LOSS_IDS = tuple(c.type_id for c in LOSSES if c.meta.get("metric"))
PAIR_LOSS_IDS = ("train.loss_cosine_embedding", "train.loss_margin_ranking")


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
    Component(
        "train.lion", "Lion",
        (_LR(1e-4), _B1, P("beta2", "float", 0.99, lo=0.0, hi=0.999999), _WD(1e-2)),
        torch=lambda p: (f"Lion(model.parameters(), lr={p['lr']}, "
                         f"betas=({p['beta1']}, {p['beta2']}), "
                         f"weight_decay={p['weight_decay']})"),
        keras=lambda p: (f"keras.optimizers.Lion(learning_rate={p['lr']}, beta_1={p['beta1']}, "
                         f"beta_2={p['beta2']}{_kwd(p)})"),
        desc="Evolved sign momentum (Chen et al., 2023): one state per weight, half Adam's "
             "memory. Use a learning rate about 10x smaller than AdamW's.",
        meta={"torch_helper": "Lion"}),
    Component(
        "train.lamb", "LAMB",
        (_LR(1e-3), _B1, _B2, _EPS(1e-6), _WD(1e-2)),
        torch=lambda p: (f"Lamb(model.parameters(), lr={p['lr']}, "
                         f"betas=({p['beta1']}, {p['beta2']}), eps={p['eps']}, "
                         f"weight_decay={p['weight_decay']})"),
        keras=lambda p: (f"keras.optimizers.Lamb(learning_rate={p['lr']}, beta_1={p['beta1']}, "
                         f"beta_2={p['beta2']}, epsilon={p['eps']}{_kwd(p)})"),
        desc="Adam with a per-layer trust ratio (You et al., 2020): stays stable at very "
             "large batch sizes.",
        meta={"torch_helper": "Lamb"}),
    Component(
        "train.lookahead_adamw", "Lookahead (AdamW)",
        (_LR(1e-3), _B1, _B2, _WD(1e-2), P("k", "int", 5, lo=1, help="Fast steps per sync"),
         P("alpha", "float", 0.5, lo=0.0, hi=1.0, help="How far the slow weights move")),
        torch=lambda p: (f"LookaheadAdamW(model.parameters(), lr={p['lr']}, "
                         f"betas=({p['beta1']}, {p['beta2']}), "
                         f"weight_decay={p['weight_decay']}, k={int(p['k'])}, "
                         f"alpha={p['alpha']})"),
        desc="AdamW whose weights are pulled toward a slow moving copy every k steps "
             "(Zhang et al., 2019): less sensitive to the learning rate.",
        meta={"torch_helper": "LookaheadAdamW"}),
    Component(
        "train.sam", "SAM (Sharpness-Aware)",
        (P("base", "enum", "sgd", options=("sgd", "adamw"), help="The optimizer SAM wraps"),
         _LR(0.05), P("rho", "float", 0.05, lo=0.0, help="Neighbourhood size"),
         P("momentum", "float", 0.9, lo=0.0, hi=1.0, help="SGD momentum"), _WD(5e-4),
         P("adaptive", "bool", False, help="ASAM: scale the step by each weight's size")),
        torch=lambda p: (f"SAM(model.parameters(), base=\"{p['base']}\", rho={p['rho']}, "
                         f"adaptive={bool(p['adaptive'])}, lr={p['lr']}, "
                         f"momentum={p['momentum']}, weight_decay={p['weight_decay']})"),
        desc="Sharpness-Aware Minimization (Foret et al., 2021): each step first moves to "
             "the worst weights nearby, then descends from there, which finds flatter minima "
             "that generalize better. Two forward/backward passes per step; runs in full "
             "precision.",
        meta={"torch_helper": "SAM", "sam": True}),
    Component(
        "train.schedule_free_adamw", "Schedule-Free AdamW",
        (_LR(2.5e-3), _B1, _B2, _EPS(1e-8), _WD(0.0),
         P("warmup_steps", "int", 0, lo=0, help="Linear warm-up steps")),
        torch=lambda p: (f"ScheduleFreeAdamW(model.parameters(), lr={p['lr']}, "
                         f"betas=({p['beta1']}, {p['beta2']}), eps={p['eps']}, "
                         f"weight_decay={p['weight_decay']}, "
                         f"warmup_steps={int(p['warmup_steps'])})"),
        desc="AdamW without a learning-rate schedule (Defazio et al., 2024): it averages "
             "its iterates instead, and evaluates the average. Leave out the scheduler.",
        meta={"torch_helper": "ScheduleFreeAdamW", "schedule_free": True}),
]

_PROTOTYPES = '''\
class ClassPrototypes(nn.Module):
    """Running mean embedding of each class; scores embeddings by cosine similarity."""

    def __init__(self, embed_dim: int, num_classes: int, momentum: float = 0.9) -> None:
        super().__init__()
        self.momentum = momentum
        self.register_buffer("prototypes", torch.zeros(num_classes, embed_dim))
        self.register_buffer("seen", torch.zeros(num_classes))

    @torch.no_grad()
    def update(self, emb: torch.Tensor, target: torch.Tensor) -> None:
        emb = nn.functional.normalize(emb.detach().float(), dim=1)
        for c in target.unique():
            mean = emb[target == c].mean(0)
            old = self.prototypes[c]
            self.prototypes[c] = mean if self.seen[c] == 0 else (
                self.momentum * old + (1 - self.momentum) * mean)
            self.seen[c] = 1

    def scores(self, emb: torch.Tensor) -> torch.Tensor:
        protos = nn.functional.normalize(self.prototypes, dim=1)
        emb = nn.functional.normalize(emb.to(protos.device).float(), dim=1)
        return 10.0 * emb @ protos.t()
'''

# PyTorch optimizers and losses that are not in torch: emitted into the training script
TRAIN_HELPERS = {
    "MarginSoftmaxLoss": '''\
class MarginSoftmaxLoss(nn.Module):
    """ArcFace (Deng et al., 2019) / CosFace (Wang et al., 2018): softmax over the scaled
    cosine to learned class centres, with a margin on the true class."""

    def __init__(self, embed_dim: int, num_classes: int, scale: float = 30.0,
                 margin: float = 0.5, kind: str = "arcface") -> None:
        super().__init__()
        self.centres = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.01)
        self.scale, self.margin, self.kind = scale, margin, kind

    def cosine(self, emb: torch.Tensor) -> torch.Tensor:
        centres = nn.functional.normalize(self.centres, dim=1)
        return nn.functional.normalize(emb.to(centres.device).float(), dim=1) @ centres.t()

    def scores(self, emb: torch.Tensor) -> torch.Tensor:
        return self.scale * self.cosine(emb)

    def forward(self, emb: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cos = self.cosine(emb)
        target = target.long().reshape(-1)
        true = nn.functional.one_hot(target, cos.shape[1]).bool()
        if self.kind == "arcface":
            angle = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
            shifted = torch.cos(angle + self.margin)
        else:
            shifted = cos - self.margin
        logits = torch.where(true, shifted, cos) * self.scale
        return nn.functional.cross_entropy(logits, target)
''',
    "TripletLoss": _PROTOTYPES + '''

class TripletLoss(ClassPrototypes):
    """Batch-hard triplet loss (Hermans et al., 2017) on L2-normalised embeddings."""

    def __init__(self, embed_dim: int, num_classes: int, margin: float = 0.3) -> None:
        super().__init__(embed_dim, num_classes)
        self.margin = margin

    def forward(self, emb: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.long().reshape(-1)
        e = nn.functional.normalize(emb.float(), dim=1)
        dist = torch.cdist(e, e)
        same = target[:, None] == target[None, :]
        hardest_pos = (dist * same).max(dim=1).values
        hardest_neg = dist.masked_fill(same, float("inf")).min(dim=1).values
        has_neg = torch.isfinite(hardest_neg)
        if self.training:
            self.update(e, target)
        if not has_neg.any():
            return (dist * 0).sum()
        return torch.relu(hardest_pos - hardest_neg + self.margin)[has_neg].mean()
''',
    "NTXentLoss": _PROTOTYPES + '''

class NTXentLoss(ClassPrototypes):
    """NT-Xent (SimCLR, Chen et al., 2020). A paired batch holds two views of each image,
    the first views then the second; each view's positive is its twin. Unpaired batches
    (the test set) score same-class examples as positives instead."""

    def __init__(self, embed_dim: int, num_classes: int, temperature: float = 0.5) -> None:
        super().__init__(embed_dim, num_classes)
        self.temperature = temperature
        self.paired = True

    def forward(self, emb: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.long().reshape(-1)
        e = nn.functional.normalize(emb.float(), dim=1)
        if self.training:
            self.update(e, target)      # class averages: only to measure the embedding
        n = len(e)
        eye = torch.eye(n, dtype=torch.bool, device=e.device)
        sim = (e @ e.t() / self.temperature).masked_fill(eye, float("-inf"))
        if self.paired and n % 2 == 0 and n > 1:
            twin = (torch.arange(n, device=e.device) + n // 2) % n
            return nn.functional.cross_entropy(sim, twin)
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        positive = (target[:, None] == target[None, :]) & ~eye
        count = positive.sum(1)
        anchors = count > 0
        if not anchors.any():
            return (sim.masked_fill(eye, 0) * 0).sum()
        return -(log_prob.masked_fill(~positive, 0).sum(1)[anchors] / count[anchors]).mean()
''',
    "SupConLoss": _PROTOTYPES + '''

class SupConLoss(ClassPrototypes):
    """Supervised contrastive loss (Khosla et al., 2020): InfoNCE where every same-class
    example in the batch is a positive."""

    def __init__(self, embed_dim: int, num_classes: int, temperature: float = 0.1) -> None:
        super().__init__(embed_dim, num_classes)
        self.temperature = temperature

    def forward(self, emb: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.long().reshape(-1)
        e = nn.functional.normalize(emb.float(), dim=1)
        sim = e @ e.t() / self.temperature
        eye = torch.eye(len(e), dtype=torch.bool, device=e.device)
        sim = sim.masked_fill(eye, float("-inf"))
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        positive = (target[:, None] == target[None, :]) & ~eye
        count = positive.sum(1)
        if self.training:
            self.update(e, target)
        anchors = count > 0
        if not anchors.any():
            return (sim.masked_fill(eye, 0) * 0).sum()
        mean_pos = log_prob.masked_fill(~positive, 0).sum(1)[anchors] / count[anchors]
        return -mean_pos.mean()
''',
    "CosineEmbeddingPairsLoss": _PROTOTYPES + '''

class CosineEmbeddingPairsLoss(ClassPrototypes):
    """CosineEmbeddingLoss over every pair in the batch: 1 - cos for same-class pairs,
    max(0, cos - margin) for the others. Each kind is averaged on its own, so a batch with
    few same-class pairs still learns from them. Labelled pairs use their own labels."""

    def __init__(self, embed_dim: int, num_classes: int, margin: float = 0.2) -> None:
        super().__init__(embed_dim, num_classes)
        self.margin = margin

    def scores(self, emb: torch.Tensor) -> torch.Tensor:
        if emb.ndim == 3:   # labelled pairs: a logit that says "same"
            cos = nn.functional.cosine_similarity(emb[:, 0].float(), emb[:, 1].float(), dim=1)
            return 10.0 * (cos - (1 + self.margin) / 2)[:, None]
        return super().scores(emb)

    def forward(self, emb: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.long().reshape(-1)
        if emb.ndim == 3:   # labelled pairs [batch, 2, embedding]: 1 = same, 0 = different
            cos = nn.functional.cosine_similarity(emb[:, 0].float(), emb[:, 1].float(), dim=1)
            return torch.where(target > 0, 1 - cos, torch.relu(cos - self.margin)).mean()
        e = nn.functional.normalize(emb.float(), dim=1)
        if self.training:
            self.update(e, target)
        cos = e @ e.t()
        eye = torch.eye(len(e), dtype=torch.bool, device=e.device)
        same = (target[:, None] == target[None, :]) & ~eye
        other = target[:, None] != target[None, :]
        loss = (cos * 0).sum()
        if same.any():
            loss = loss + (1 - cos[same]).mean()
        if other.any():
            loss = loss + torch.relu(cos[other] - self.margin).mean()
        return loss
''',
    "MarginRankingPairsLoss": _PROTOTYPES + '''

class MarginRankingPairsLoss(ClassPrototypes):
    """MarginRankingLoss on cosine similarity: around each example, every same-class
    neighbour should be more similar than every other-class one by the margin. On labelled
    pairs, it ranks the items' scores the way the labels say."""

    def __init__(self, embed_dim: int, num_classes: int, margin: float = 0.2) -> None:
        super().__init__(embed_dim, num_classes)
        self.margin = margin

    def scores(self, emb: torch.Tensor) -> torch.Tensor:
        if emb.ndim == 3:   # labelled pairs: a logit that says "a ranks higher"
            return (emb[:, 0, :1] - emb[:, 1, :1]).float()
        return super().scores(emb)

    def forward(self, emb: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.long().reshape(-1)
        if emb.ndim == 3:   # labelled pairs: each item's score; 1 = a ranks above b
            sign = target.float() * 2 - 1
            return nn.functional.margin_ranking_loss(
                emb[:, 0, 0].float(), emb[:, 1, 0].float(), sign, margin=self.margin)
        e = nn.functional.normalize(emb.float(), dim=1)
        if self.training:
            self.update(e, target)
        cos = e @ e.t()
        eye = torch.eye(len(e), dtype=torch.bool, device=e.device)
        same = (target[:, None] == target[None, :]) & ~eye
        other = target[:, None] != target[None, :]
        # [anchor, positive, negative]: the positive should win by the margin
        gap = torch.relu(self.margin - cos[:, :, None] + cos[:, None, :])
        valid = same[:, :, None] & other[:, None, :]
        if not valid.any():
            return (cos * 0).sum()
        return gap[valid].mean()
''',
    "ScheduleFreeAdamW": '''\
class ScheduleFreeAdamW(torch.optim.Optimizer):
    """Schedule-free AdamW (Defazio et al., 2024). Training runs at y, an interpolation of
    the fast iterate z and the running average x; eval() switches the weights to x and
    train() back to y."""

    def __init__(self, params, lr: float = 2.5e-3, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0, warmup_steps: int = 0) -> None:
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                      warmup_steps=warmup_steps, k=0, weight_sum=0.0,
                                      training=True))

    @torch.no_grad()
    def eval(self) -> None:
        for group in self.param_groups:
            if group["training"]:
                beta1 = group["betas"][0]
                for p in group["params"]:
                    if "z" in self.state[p]:
                        p.lerp_(self.state[p]["z"], 1 - 1 / beta1)    # y -> x
                group["training"] = False

    @torch.no_grad()
    def train(self) -> None:
        for group in self.param_groups:
            if not group["training"]:
                beta1 = group["betas"][0]
                for p in group["params"]:
                    if "z" in self.state[p]:
                        p.lerp_(self.state[p]["z"], 1 - beta1)        # x -> y
                group["training"] = True

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            k = group["k"]
            warm = min(1.0, (k + 1) / group["warmup_steps"]) if group["warmup_steps"] else 1.0
            lr = group["lr"] * warm * (1 - beta2 ** (k + 1)) ** 0.5
            group["weight_sum"] += lr ** 2
            c = lr ** 2 / group["weight_sum"] if group["weight_sum"] else 0.0
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "z" not in state:
                    state["z"], state["v"] = p.detach().clone(), torch.zeros_like(p)
                z, v = state["z"], state["v"]
                v.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                update = p.grad / v.sqrt().add_(group["eps"])
                if group["weight_decay"]:
                    update.add_(p, alpha=group["weight_decay"])
                p.lerp_(z, 1 - 1 / beta1)        # y -> x
                z.sub_(update, alpha=lr)
                p.lerp_(z, c)                    # the average moves toward z
                p.lerp_(z, 1 - beta1)            # x -> y
            group["k"] = k + 1
        return loss
''',
    "SAM": '''\
class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (Foret et al., 2021) around SGD or AdamW: first_step()
    climbs to the worst weights within rho, second_step() restores them and steps with
    the gradient taken there."""

    def __init__(self, params, base: str = "sgd", rho: float = 0.05, adaptive: bool = False,
                 lr: float = 0.05, momentum: float = 0.9, weight_decay: float = 0.0) -> None:
        super().__init__(params, dict(rho=rho, adaptive=adaptive, lr=lr))
        if base == "adamw":
            self.base_optimizer = torch.optim.AdamW(self.param_groups, lr=lr,
                                                    weight_decay=weight_decay)
        else:
            self.base_optimizer = torch.optim.SGD(self.param_groups, lr=lr, momentum=momentum,
                                                  weight_decay=weight_decay)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    def _grad_norm(self) -> torch.Tensor:
        norms = [((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2)
                 for group in self.param_groups for p in group["params"] if p.grad is not None]
        return torch.norm(torch.stack(norms), p=2) if norms else torch.tensor(0.0)

    @torch.no_grad()
    def first_step(self) -> None:
        norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e)
                self.state[p]["e_w"] = e

    @torch.no_grad()
    def second_step(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if "e_w" in self.state[p]:
                    p.sub_(self.state[p].pop("e_w"))
        self.base_optimizer.step()

    def step(self, closure=None):
        raise RuntimeError("SAM steps with first_step() and second_step()")
''',
    "Lion": '''\
class Lion(torch.optim.Optimizer):
    """Lion (Chen et al., 2023): steps by the sign of an interpolated momentum."""

    def __init__(self, params, lr: float = 1e-4, betas=(0.9, 0.99),
                 weight_decay: float = 0.0) -> None:
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["exp_avg"] = torch.zeros_like(p)
                m = state["exp_avg"]
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(torch.sign(m * beta1 + p.grad * (1 - beta1)), alpha=-group["lr"])
                m.mul_(beta2).add_(p.grad, alpha=1 - beta2)
        return loss
''',
    "Lamb": '''\
class Lamb(torch.optim.Optimizer):
    """LAMB (You et al., 2020): Adam scaled per layer by ||w|| / ||update||."""

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-6,
                 weight_decay: float = 0.01) -> None:
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                t, m, v = state["step"], state["exp_avg"], state["exp_avg_sq"]
                m.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                update = (m / (1 - beta1 ** t)) / ((v / (1 - beta2 ** t)).sqrt() + group["eps"])
                update.add_(p, alpha=group["weight_decay"])
                w_norm, u_norm = p.norm(), update.norm()
                trust = torch.where((w_norm > 0) & (u_norm > 0), w_norm / u_norm,
                                    torch.ones_like(w_norm))
                p.add_(update * trust, alpha=-group["lr"])
        return loss
''',
    "LookaheadAdamW": '''\
class LookaheadAdamW(torch.optim.AdamW):
    """Lookahead (Zhang et al., 2019) around AdamW: every k steps the slow weights move
    alpha of the way to the fast ones, and the fast weights restart from there."""

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999),
                 weight_decay: float = 0.01, k: int = 5, alpha: float = 0.5) -> None:
        super().__init__(params, lr=lr, betas=betas, weight_decay=weight_decay)
        self.k, self.alpha, self.fast_steps = k, alpha, 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        self.fast_steps += 1
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "slow" not in state:
                    state["slow"] = p.detach().clone()
                elif self.fast_steps % self.k == 0:
                    state["slow"].add_(p - state["slow"], alpha=self.alpha)
                    p.copy_(state["slow"])
        return loss
''',
}


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
     P("mixed_precision", "bool", False, help="Automatic mixed precision on CUDA/MPS"),
     P("deterministic", "bool", False,
       help="Deterministic kernels for bit-reproducible runs (slower)"),
     P("ema_decay", "float", 0.0, lo=0.0, hi=0.99999,
       help="Evaluate and keep an exponential moving average of the weights "
            "(e.g. 0.999); 0 = off"),
     P("swa_start", "int", 0, lo=0,
       help="Stochastic weight averaging: average the weights of every epoch from this "
            "one on (PyTorch); 0 = off")),
    desc="Training loop: epochs, batching, device, early stopping, clipping, AMP, "
         "weight averaging (EMA or SWA).",
    checks=lambda p: ([("error", "use one kind of weight averaging: EMA or SWA")]
                      if float(p.get("ema_decay") or 0) and int(p.get("swa_start") or 0)
                      else []) + ([("warning", "SWA starts after the last epoch, so it never "
                                               "averages anything")]
                                  if int(p.get("swa_start") or 0) > int(p.get("epochs") or 1)
                                  else []))

KFOLD = Component(
    "train.kfold", "K-Fold Cross-Validation",
    (P("k", "int", 5, lo=2, hi=20), P("stratified", "bool", True), P("seed", "int", 42, lo=0),
     P("refit", "bool", False, help="Then train the final model on the training split as "
                                    "usual (stacking ensembles use its out-of-fold "
                                    "predictions)")),
    desc="Retrain on k folds and report mean ± std (tabular/array datasets); the "
         "out-of-fold predictions are saved for stacking.")

PREDICT = Component(
    "eval.predict", "Prediction Preview",
    (P("n_samples", "int", 5, lo=1, hi=50),
     P("show_probabilities", "bool", True, help="Print the top-3 class probabilities")),
    desc="Print predictions for test examples after training.")


EXPLAIN = Component(
    "eval.permutation_importance", "Feature Importance",
    (P("n_repeats", "int", 5, lo=1, hi=50, help="Shuffles per feature (more = steadier)"),
     P("max_samples", "int", 2000, lo=50, help="Test rows used (a sample keeps it fast)")),
    desc="After training, shuffles each input feature of the test set in turn and "
         "measures how much the loss gets worse: the features the model relies on most. "
         "Tables and feature vectors, PyTorch.")


EXPLAIN_IG = Component(
    "eval.integrated_gradients", "Integrated Gradients",
    (P("steps", "int", 32, lo=4, hi=256, help="Points on the path from the baseline"),
     P("baseline", "enum", "mean", options=("mean", "zeros"),
       help="Where the path starts: the average row, or all zeros"),
     P("max_samples", "int", 500, lo=10, help="Test rows explained")),
    desc="After training, integrates the gradient of the predicted class (or the output) "
         "along a path from a baseline to each test row (Sundararajan et al., 2017): how "
         "much each feature moved the prediction. Tables and feature vectors, PyTorch.")


EXPLAIN_SHAP = Component(
    "eval.shap_values", "SHAP Values",
    (P("background", "int", 100, lo=10, help="Training rows the explanation compares with"),
     P("max_samples", "int", 300, lo=10, help="Test rows explained")),
    desc="After training, SHAP values (expected gradients, Lundberg & Lee, 2017): each "
         "feature's share of the gap between a prediction and the average prediction over "
         "background rows. Tables and feature vectors, PyTorch. "
         "pip install 'ai-made-easy[explain]'.",
    meta={"requires": ("shap",), "extra": "explain"})


EXPLAIN_GRADCAM = Component(
    "eval.grad_cam", "Grad-CAM Heatmaps",
    (P("n_samples", "int", 8, lo=1, hi=12, help="Test images explained (up to 12, "
                                                       "the most a run shows)"),
     P("opacity", "float", 0.5, lo=0.1, hi=0.9, help="How strongly the heatmap is drawn")),
    desc="After training, Grad-CAM (Selvaraju et al., 2017) on the last convolution: a "
         "heatmap over each of a few test images showing where the model looked for its "
         "prediction, saved with the run and shown in both apps. Image classifiers with "
         "a Conv2d layer, PyTorch.")


EXPLAIN_ATTENTION = Component(
    "eval.attention_maps", "Attention Maps",
    (P("n_samples", "int", 6, lo=1, hi=12, help="Test samples explained (up to 12, "
                                                      "the most a run shows)"),
     P("opacity", "float", 0.5, lo=0.1, hi=0.9, help="How strongly the map is drawn over "
                                                      "an image")),
    desc="After training, the attention weights of the model's last attention layer on a "
         "few test samples, saved with the run and shown in both apps: for a vision "
         "transformer, how much each image patch is attended to, drawn over the image; for "
         "text and sequences, the attention matrix (row: a position, columns: where it "
         "looked). Models with attention or transformer blocks, PyTorch.")


EXPLAIN_SALIENCY = Component(
    "eval.audio_saliency", "Audio Saliency",
    (P("n_samples", "int", 6, lo=1, hi=12, help="Test clips explained (up to 12, "
                                                      "the most a run shows)"),
     P("opacity", "float", 0.5, lo=0.1, hi=0.9, help="How strongly the map is drawn")),
    desc="After training, gradient saliency (Simonyan et al., 2014) for a few test clips: "
         "how much each time-frequency point of the spectrogram (or each moment of a raw "
         "waveform) moves the predicted class, drawn over the input and saved with the run. "
         "Audio classifiers, PyTorch.")

PICTURE_EXPLAIN_IDS = ("eval.grad_cam", "eval.attention_maps", "eval.audio_saliency")


def _losses_by_id() -> dict[str, Component]:
    return {c.type_id: c for c in LOSSES}


LOSS_IDS = tuple(c.type_id for c in LOSSES)
OPTIMIZER_IDS = tuple(c.type_id for c in OPTIMIZERS)
SCHEDULER_IDS = tuple(c.type_id for c in SCHEDULERS)
METRIC_IDS = tuple(c.type_id for c in METRICS)
COMPONENTS: dict[str, Component] = {
    c.type_id: c for c in (*LOSSES, *OPTIMIZERS, *SCHEDULERS, *METRICS, TRAINER, KFOLD, PREDICT,
                           EXPLAIN, EXPLAIN_IG, EXPLAIN_SHAP, EXPLAIN_GRADCAM,
                           EXPLAIN_ATTENTION, EXPLAIN_SALIENCY)
}

_CATEGORY = {
    **{t: ("Loss", "training") for t in LOSS_IDS},
    **{t: ("Optimizer", "training") for t in OPTIMIZER_IDS},
    **{t: ("Scheduler", "training") for t in SCHEDULER_IDS},
    **{t: ("Metrics", "evaluation") for t in METRIC_IDS},
    "train.trainer": ("Training", "training"),
    "train.kfold": ("Training", "training"),
    "eval.predict": ("Metrics", "evaluation"),
    "eval.permutation_importance": ("Metrics", "evaluation"),
    "eval.integrated_gradients": ("Metrics", "evaluation"),
    "eval.shap_values": ("Metrics", "evaluation"),
    "eval.grad_cam": ("Metrics", "evaluation"),
    "eval.attention_maps": ("Metrics", "evaluation"),
    "eval.audio_saliency": ("Metrics", "evaluation"),
}
EXPLAIN_IDS = ("eval.permutation_importance", "eval.integrated_gradients", "eval.shap_values")
EXPLAIN_METHODS = {"eval.permutation_importance": "permutation",
                   "eval.integrated_gradients": "integrated_gradients",
                   "eval.shap_values": "shap"}


def _libraries(c: Component) -> str:
    if c.type_id in (*EXPLAIN_IDS, *PICTURE_EXPLAIN_IDS):
        return "PyTorch"
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
            requires=tuple(comp.meta.get("requires", ())), extra=comp.meta.get("extra", ""),
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
