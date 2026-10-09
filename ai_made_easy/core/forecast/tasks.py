"""The forecasting task and how a design is recognised as one."""
from __future__ import annotations

from ai_made_easy.core.forecast.blocks import FORECAST_DATA, FORECAST_METRICS, FORECAST_MODELS
from ai_made_easy.core.tasks import Task, register_task, register_task_resolver

register_task(Task(
    "forecasting", "Time-series forecasting",
    "Predict the next steps of one or many time series, optionally with uncertainty.",
    target="future values [H]", output_role="distribution", modalities=("timeseries",),
    trainer_kind="forecasting", serving="forecast",
    default_metrics=("eval.mase", "eval.smape"),
    meta={"loss_tasks": (), "losses": ["train.loss_mse", "train.loss_l1", "train.loss_huber",
                                       "train.loss_smooth_l1"],
          "metrics": list(FORECAST_METRICS), "default_loss": None,
          "default_optimizer": "Adam (lr = 1e-3)"}))


def model_of(graph):  # noqa: ANN001, ANN201
    return next((n for n in graph.nodes.values() if n.type_id in FORECAST_MODELS), None)


def dataset_of(graph):  # noqa: ANN001, ANN201
    return next((n for n in graph.nodes.values() if n.type_id in FORECAST_DATA), None)


def forecast_task(graph) -> str | None:  # noqa: ANN001
    return "forecasting" if model_of(graph) is not None or dataset_of(graph) is not None \
        else None


register_task_resolver(forecast_task)
