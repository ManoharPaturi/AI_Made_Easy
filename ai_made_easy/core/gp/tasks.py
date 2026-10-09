"""The Gaussian-process task."""
from __future__ import annotations

from ai_made_easy.core.tasks import Task, register_task, register_task_resolver

register_task(Task(
    "gp_regression", "Gaussian-process regression / classification",
    "Smooth functions of a few inputs with calibrated uncertainty, from a kernel.",
    target="target column", output_role="distribution", modalities=("tabular",), family="gp",
    trainer_kind="bayesian", serving="predictive"))


def gp_task(graph) -> str | None:  # noqa: ANN001
    return "gp_regression" if any(n.type_id == "gp.model" for n in graph.nodes.values()) \
        else None


register_task_resolver(gp_task)
