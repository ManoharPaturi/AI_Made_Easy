"""The Bayesian modeling task and the design's dataset."""
from __future__ import annotations

from ai_made_easy.core.ppl.blocks import BY_ID, PPL_DATA
from ai_made_easy.core.tasks import Task, register_task, register_task_resolver

TABLE_DATA = ("data.csv", *PPL_DATA)

register_task(Task(
    "bayesian_modeling", "Bayesian modeling",
    "Priors and a likelihood for the data: posterior distributions with uncertainty.",
    target="posterior over parameters", output_role="distribution", modalities=("tabular",),
    family="ppl", trainer_kind="bayesian", serving="predictive"))


def dataset_of(graph):  # noqa: ANN001, ANN201
    return next((n for n in graph.nodes.values() if n.type_id in TABLE_DATA), None)


def ppl_task(graph) -> str | None:  # noqa: ANN001
    types = {n.type_id for n in graph.nodes.values()}
    if types & (set(BY_ID) | {"ppl.deterministic", "ppl.sampler"}):
        return "bayesian_modeling"
    return None


register_task_resolver(ppl_task)
