"""The state-space forecasting task."""
from __future__ import annotations

from ai_made_easy.core.tasks import Task, register_task, register_task_resolver

register_task(Task(
    "state_space_forecasting", "State-space forecasting",
    "Decompose one series into trend, seasonality and cycles with the Kalman filter and "
    "forecast it with intervals.",
    target="the next values of the series", output_role="distribution",
    modalities=("timeseries",), family="ssm", trainer_kind="forecasting",
    serving="forecast"))


def ssm_task(graph) -> str | None:  # noqa: ANN001
    return "state_space_forecasting" if any(n.type_id == "ssm.model"
                                            for n in graph.nodes.values()) else None


register_task_resolver(ssm_task)
