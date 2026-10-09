"""The density-estimation task (normalizing flows) and its resolver."""
from __future__ import annotations

from ai_made_easy.core.flows.blocks import DENSITY_DATA, FLOW_LAYERS
from ai_made_easy.core.tasks import Task, register_task, register_task_resolver

register_task(Task(
    "density_estimation", "Density estimation (normalizing flow)",
    "Learn the probability density of the data with an invertible network: exact "
    "log-likelihoods for any point and new samples drawn from the model.",
    target="the data itself", output_role="latent", modalities=("tabular",),
    trainer_kind="flow", serving="density", default_metrics=(),
    meta={"loss_tasks": (), "losses": [], "metrics": [], "default_loss": None,
          "default_optimizer": "Adam (lr = 1e-3)"}))


def is_flow(graph) -> bool:  # noqa: ANN001
    return any(n.type_id in FLOW_LAYERS for n in graph.nodes.values())


def dataset_of(graph):  # noqa: ANN001, ANN201
    """The flow's dataset: generated 2-D points or a CSV table."""
    return next((n for n in graph.nodes.values()
                 if n.type_id in DENSITY_DATA or n.type_id == "data.csv"), None)


def flow_task(graph) -> str | None:  # noqa: ANN001
    return "density_estimation" if is_flow(graph) else None


register_task_resolver(flow_task)
