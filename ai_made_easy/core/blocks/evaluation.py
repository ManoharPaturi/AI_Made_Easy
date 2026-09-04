"""Evaluation metric blocks (config; Phase 2 wires them into train.py)."""
from __future__ import annotations

from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import BlockDefinition, ParamSpec
from ai_made_easy.core.blocks._palette import family_color

reg = get_registry()

_EVAL_COLOR = family_color("evaluation")


def _cfg(**kwargs) -> None:
    kwargs.setdefault("category", "Evaluation")
    kwargs.setdefault("color", _EVAL_COLOR)
    reg.register(BlockDefinition(**kwargs))


_avg = ParamSpec(name="average", type="enum", default="macro",
                 options=("macro", "micro", "weighted", "binary"))

_cfg(type_id="eval.accuracy", display_name="Accuracy", params=())
_cfg(type_id="eval.precision", display_name="Precision", params=(_avg,))
_cfg(type_id="eval.recall", display_name="Recall", params=(_avg,))
_cfg(type_id="eval.f1", display_name="F1 Score", params=(_avg,))
_cfg(
    type_id="eval.confusion_matrix",
    display_name="Confusion Matrix",
    params=(ParamSpec(name="labels", type="str", default="",
                      help="Comma-separated class ids; empty = infer"),),
)
_cfg(type_id="eval.mse", display_name="Metric: MSE", params=())
_cfg(type_id="eval.mae", display_name="Metric: MAE", params=())
# audit addition (Orange3 ROC Analysis): teaches thresholds + imbalanced data
_cfg(type_id="eval.roc_auc", display_name="ROC-AUC", params=())

# audit follow-up (Orange3 Predictions): use the trained model on new data
_cfg(
    type_id="eval.predict",
    display_name="Predict (use your model)",
    params=(
        ParamSpec(name="n_samples", type="int", default=5, minimum=1, maximum=20,
                  help="How many fresh examples to predict after training"),
        ParamSpec(name="show_probabilities", type="bool", default=True,
                  help="Show the model's confidence for each class"),
    ),
)
