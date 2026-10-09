"""The recommendation task and its resolver."""
from __future__ import annotations

from ai_made_easy.core.recsys.blocks import MODELS, REC_DATA
from ai_made_easy.core.tasks import Task, register_task, register_task_resolver

register_task(Task(
    "recommendation", "Recommendation",
    "Learn which items each user will like from past interactions; rank items per user.",
    target="interactions or ratings", output_role="tensor", modalities=("interactions",),
    trainer_kind="recommendation", serving="ranking",
    meta={"loss_tasks": (), "losses": [], "metrics": [], "default_loss": None,
          "default_optimizer": "Adam (lr = 1e-2)"}))


def dataset_of(graph):  # noqa: ANN001, ANN201
    return next((n for n in graph.nodes.values() if n.type_id in REC_DATA), None)


def recsys_task(graph) -> str | None:  # noqa: ANN001
    types = {n.type_id for n in graph.nodes.values()}
    return "recommendation" if types & set(MODELS) or dataset_of(graph) is not None else None


register_task_resolver(recsys_task)
