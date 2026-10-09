"""Graph tasks (node classification, graph classification, link prediction) and the
resolver that recognises them."""
from __future__ import annotations

from ai_made_easy.core.gnn.blocks import GLOBAL_POOLS, GRAPH_BLOCKS, GRAPH_DATA
from ai_made_easy.core.tasks import Task, register_task, register_task_resolver

_COMMON = {"loss_tasks": (), "losses": ["train.loss_cross_entropy"],
           "metrics": ["eval.accuracy", "eval.f1"],
           "default_loss": None, "default_optimizer": "Adam (lr = 1e-2, weight decay 5e-4)"}

register_task(Task(
    "node_classification", "Node classification",
    "Predict a class for every node of a graph from its features and its neighbours.",
    target="node class", output_role="logits", classification=True, modalities=("graph",),
    trainer_kind="graph", serving="node_labels", meta=dict(_COMMON)))
register_task(Task(
    "graph_classification", "Graph classification",
    "Predict a class for a whole graph (a molecule, a protein, a network).",
    target="graph class", output_role="logits", classification=True, modalities=("graph",),
    trainer_kind="graph", serving="label_probs", meta=dict(_COMMON)))
register_task(Task(
    "link_prediction", "Link prediction",
    "Embed the nodes and score node pairs: which missing edges exist?",
    target="edges", output_role="latent", modalities=("graph",), trainer_kind="graph",
    serving="link_scores", meta={**_COMMON, "losses": [],
                                 "metrics": ["eval.roc_auc", "eval.average_precision"]}))


def _types(graph) -> set[str]:  # noqa: ANN001
    return {n.type_id for n in graph.nodes.values()}


def dataset_of(graph):  # noqa: ANN001, ANN201
    return next((n for n in graph.nodes.values() if n.type_id in GRAPH_DATA), None)


def graph_kind(graph) -> str | None:  # noqa: ANN001
    """node_classification | graph_classification | link_prediction (None: not a graph
    design)."""
    types = _types(graph)
    data = dataset_of(graph)
    if not types & set(GRAPH_BLOCKS) and "graph.link_decoder" not in types and data is None:
        return None
    if "graph.link_decoder" in types:
        return "link_prediction"
    if types & set(GLOBAL_POOLS) or (data is not None and GRAPH_DATA[data.type_id] == "graphs"):
        return "graph_classification"
    return "node_classification"


register_task_resolver(graph_kind)
