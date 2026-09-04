"""Training config blocks: optimizers, losses, schedulers, trainer.

Config blocks: params editable on canvas; the Phase 2 training-script
generator collects them into a runnable ``train.py``.
"""
from __future__ import annotations

from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import BlockDefinition, ParamSpec
from ai_made_easy.core.blocks._palette import family_color

reg = get_registry()

_TRAIN_COLOR = family_color("training")


def _cfg(**kwargs) -> None:
    kwargs.setdefault("category", "Training")
    kwargs.setdefault("color", _TRAIN_COLOR)
    reg.register(BlockDefinition(**kwargs))


_lr = ParamSpec(name="lr", type="float", default=1e-3, minimum=0.0)

_cfg(
    type_id="train.sgd",
    display_name="SGD",
    params=(
        _lr,
        ParamSpec(name="momentum", type="float", default=0.0, minimum=0.0),
        ParamSpec(name="nesterov", type="bool", default=False),
    ),
)

_cfg(
    type_id="train.adam",
    display_name="Adam",
    params=(
        _lr,
        ParamSpec(name="beta1", type="float", default=0.9, minimum=0.0, maximum=1.0),
        ParamSpec(name="beta2", type="float", default=0.999, minimum=0.0, maximum=1.0),
        ParamSpec(name="eps", type="float", default=1e-8, minimum=1e-12),
    ),
)

_cfg(
    type_id="train.adamw",
    display_name="AdamW",
    params=(
        _lr,
        ParamSpec(name="weight_decay", type="float", default=1e-2, minimum=0.0),
        ParamSpec(name="beta1", type="float", default=0.9, minimum=0.0, maximum=1.0),
        ParamSpec(name="beta2", type="float", default=0.999, minimum=0.0, maximum=1.0),
    ),
)

_cfg(type_id="train.adagrad", display_name="Adagrad",
     params=(_lr, ParamSpec(name="eps", type="float", default=1e-10, minimum=1e-12)))
_cfg(type_id="train.rmsprop", display_name="RMSprop",
     params=(_lr, ParamSpec(name="alpha", type="float", default=0.99, minimum=0.0, maximum=1.0),
             ParamSpec(name="eps", type="float", default=1e-8, minimum=1e-12)))
_cfg(type_id="train.adadelta", display_name="Adadelta",
     params=(_lr, ParamSpec(name="rho", type="float", default=0.9, minimum=0.0, maximum=1.0)))
_cfg(type_id="train.adamax", display_name="Adamax",
     params=(_lr, ParamSpec(name="beta1", type="float", default=0.9, minimum=0.0, maximum=1.0),
             ParamSpec(name="beta2", type="float", default=0.999, minimum=0.0, maximum=1.0)))
_cfg(type_id="train.nadam", display_name="NAdam",
     params=(_lr, ParamSpec(name="beta1", type="float", default=0.9, minimum=0.0, maximum=1.0),
             ParamSpec(name="beta2", type="float", default=0.999, minimum=0.0, maximum=1.0)))
_cfg(type_id="train.radam", display_name="RAdam",
     params=(_lr, ParamSpec(name="beta1", type="float", default=0.9, minimum=0.0, maximum=1.0),
             ParamSpec(name="beta2", type="float", default=0.999, minimum=0.0, maximum=1.0)))

_cfg(type_id="train.loss_cross_entropy", display_name="Loss: CrossEntropy",
     params=(ParamSpec(name="label_smoothing", type="float", default=0.0,
                       minimum=0.0, maximum=1.0),))
_cfg(type_id="train.loss_mse", display_name="Loss: MSE", params=())
_cfg(type_id="train.loss_bce_logits", display_name="Loss: BCEWithLogits", params=())
_cfg(type_id="train.loss_l1", display_name="Loss: L1 / MAE", params=())
_cfg(type_id="train.loss_smooth_l1", display_name="Loss: SmoothL1 / Huber",
     params=(ParamSpec(name="beta", type="float", default=1.0, minimum=0.0),))
_cfg(type_id="train.loss_poisson", display_name="Loss: PoissonNLL", params=())
_cfg(type_id="train.loss_kl_div", display_name="Loss: KLDiv", params=())
_cfg(type_id="train.loss_focal", display_name="Loss: Focal (CE)",
     params=(ParamSpec(name="gamma", type="float", default=2.0, minimum=0.0),
             ParamSpec(name="alpha", type="float", default=0.25, minimum=0.0, maximum=1.0,
                       help="Class weighting (0 = off)"),))

_cfg(
    type_id="train.step_lr",
    display_name="Scheduler: StepLR",
    params=(
        ParamSpec(name="step_size", type="int", default=10, minimum=1),
        ParamSpec(name="gamma", type="float", default=0.1, minimum=0.0, maximum=10.0),
    ),
)
_cfg(
    type_id="train.cosine_annealing_lr",
    display_name="Scheduler: CosineAnnealing",
    params=(
        ParamSpec(name="T_max", type="int", default=50, minimum=1),
        ParamSpec(name="eta_min", type="float", default=1e-6, minimum=0.0),
    ),
)
_cfg(
    type_id="train.one_cycle_lr",
    display_name="Scheduler: OneCycle",
    params=(
        ParamSpec(name="max_lr", type="float", default=1e-2, minimum=1e-8),
        ParamSpec(name="pct_start", type="float", default=0.3,
                  minimum=0.0, maximum=1.0),
    ),
)
_cfg(
    type_id="train.multistep_lr",
    display_name="Scheduler: MultiStep",
    params=(ParamSpec(name="milestones", type="str", default="30, 60",
                      help="Comma-separated epoch numbers"),
            ParamSpec(name="gamma", type="float", default=0.1, minimum=0.0, maximum=10.0)),
)
_cfg(
    type_id="train.exponential_lr",
    display_name="Scheduler: Exponential",
    params=(ParamSpec(name="gamma", type="float", default=0.95, minimum=0.0, maximum=10.0),),
)
_cfg(
    type_id="train.warm_restarts_lr",
    display_name="Scheduler: CosineWarmRestarts",
    params=(ParamSpec(name="T_0", type="int", default=10, minimum=1),
            ParamSpec(name="T_mult", type="int", default=1, minimum=1)),
)
_cfg(
    type_id="train.plateau_lr",
    display_name="Scheduler: ReduceLROnPlateau",
    params=(ParamSpec(name="factor", type="float", default=0.5, minimum=0.0, maximum=1.0),
            ParamSpec(name="patience", type="int", default=5, minimum=1)),
)
_cfg(
    type_id="train.linear_lr",
    display_name="Scheduler: Linear (warmup)",
    params=(ParamSpec(name="start_factor", type="float", default=0.1, minimum=0.0, maximum=1.0),
            ParamSpec(name="total_iters", type="int", default=5, minimum=1)),
)

_cfg(
    type_id="train.trainer",
    display_name="Trainer",
    params=(
        ParamSpec(name="epochs", type="int", default=10, minimum=1),
        ParamSpec(name="batch_size", type="int", default=32, minimum=1),
        ParamSpec(name="device", type="enum", default="auto",
                  options=("auto", "cpu", "mps", "cuda")),
        ParamSpec(name="seed", type="int", default=42, minimum=0),
        ParamSpec(name="early_stopping_patience", type="int", default=0, minimum=0,
                  help="0 = disabled"),
    ),
)

# audit addition (Orange3 Test-and-Score): honest results via k-fold CV
reg.register(BlockDefinition(
    type_id="train.kfold",
    display_name="K-Fold Cross-Validation",
    category="Training",
    color=_TRAIN_COLOR,
    params=(
        ParamSpec(name="k", type="int", default=5, minimum=2, maximum=10,
                  help="How many times to retrain on a different split"),
        ParamSpec(name="stratified", type="bool", default=True,
                  help="Keep the class mix even in every fold"),
        ParamSpec(name="seed", type="int", default=42, minimum=0),
    ),
))
