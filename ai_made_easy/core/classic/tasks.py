"""Classic ML tasks: what a scikit-learn / boosting pipeline does, from its estimator.

Classifiers and regressors share the neural tasks (multiclass, binary, regression);
clustering and anomaly detection are classic-only. Binary vs multi-class comes from the
data where it is known without reading files (the wizard's recipe, a bundled demo
dataset, generated data); a table file counts as multi-class until the wizard says
otherwise.
"""
from __future__ import annotations

from ai_made_easy.core.classic import catalog as ml
from ai_made_easy.core.tasks import Task, register_task, register_task_resolver
from ai_made_easy.core.training.data_catalog import SKLEARN

register_task(Task(
    "clustering", "Clustering",
    "Group similar rows without labels; a label column, if any, only scores the groups.",
    target="none (labels optional, for scoring)", output_role="tensor",
    modalities=("tabular", "text"), family="classic", serving="clusters",
    default_metrics=("eval.silhouette",),
    meta={"losses": (), "metrics": tuple(m[0] for m in ml.CLUSTER_METRICS)}))
register_task(Task(
    "anomaly_detection", "Anomaly detection",
    "Flag rows that do not look like the rest; labels, if any, are used only to evaluate.",
    target="none (0 / 1 labels optional, for evaluation)", output_role="tensor",
    modalities=("tabular",), family="classic", serving="anomalies",
    meta={"losses": (), "metrics": ()}))

_BY_KIND = {ml.REGRESSION: "regression", ml.CLUSTERING: "clustering",
            ml.ANOMALY: "anomaly_detection"}


def _two_classes(graph) -> bool:  # noqa: ANN001
    recipe = (graph.meta or {}).get("recipe") or {}
    if recipe.get("task") in ("binary", "multiclass"):
        return recipe["task"] == "binary"
    for node in graph.nodes.values():
        params = node.params
        if node.type_id == "data.sklearn":
            return SKLEARN.get(str(params.get("dataset")), ("", 0, 0))[2] == 2
        if node.type_id == "data.synthetic":
            return str(params.get("kind")) in ("moons", "circles") \
                or int(params.get("n_classes", 3) or 3) == 2
    return False


def classic_task(graph) -> str | None:  # noqa: ANN001
    est = next((n for n in graph.nodes.values() if n.type_id in ml.ESTIMATOR_IDS), None)
    if est is None:
        return None
    kind = ml.BY_ID[est.type_id].task
    if kind in _BY_KIND:
        return _BY_KIND[kind]
    return "binary" if _two_classes(graph) else "multiclass"


register_task_resolver(classic_task)
