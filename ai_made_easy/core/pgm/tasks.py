"""Graphical-model tasks and the design's dataset."""
from __future__ import annotations

from ai_made_easy.core.pgm.blocks import PGM_DATA, VARIABLES
from ai_made_easy.core.tasks import Task, register_task, register_task_resolver

TABLE_DATA = ("data.csv", *PGM_DATA)

register_task(Task(
    "probabilistic_inference", "Probabilistic inference",
    "Model how variables depend on each other and answer questions under evidence.",
    target="posterior distributions", output_role="distribution", modalities=("tabular",),
    family="pgm", trainer_kind="bayesian", serving="posteriors"))
register_task(Task(
    "regime_detection", "Regime detection (HMM)",
    "Find the hidden states behind a sequence and when it switches between them.",
    target="hidden state per step", output_role="distribution", modalities=("timeseries",),
    family="pgm", trainer_kind="bayesian", serving="states"))


def dataset_of(graph):  # noqa: ANN001, ANN201
    return next((n for n in graph.nodes.values() if n.type_id in TABLE_DATA), None)


def pgm_task(graph) -> str | None:  # noqa: ANN001
    types = {n.type_id for n in graph.nodes.values()}
    if "pgm.hmm" in types:
        return "regime_detection"
    if types & set(VARIABLES):
        return "probabilistic_inference"
    return None


register_task_resolver(pgm_task)
