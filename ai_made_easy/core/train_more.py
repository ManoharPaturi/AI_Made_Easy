"""Train more (3.8): continue a finished network for more epochs.

The new run starts from the finished run's best weights (its fitted preprocessing comes
from the same data, rebuilt identically) and trains for the epochs asked for, as a
tracked run of its own whose note names the run it continues. The optimizer and any
schedule start afresh: Adam's moments are not saved with the weights.
"""
from __future__ import annotations

import copy

from ai_made_easy.core.graph import Graph


class TrainMoreError(ValueError):
    """A run that can't be continued."""


def _with_epochs(graph: dict, epochs: int) -> dict:
    graph = copy.deepcopy(graph)
    trainers = [n for n in graph.get("nodes", []) if n.get("type") == "train.trainer"]
    if not trainers:
        raise TrainMoreError("the run's design has no Trainer block")
    for node in trainers:
        node.setdefault("params", {})["epochs"] = int(epochs)
    return graph


def start(mgr, run_id: str, epochs: int, lr_scale: float = 1.0) -> str:  # noqa: ANN001
    from ai_made_easy.core.pipelines import stages as S

    record = mgr.history.get(run_id)
    if record.status != "finished":
        raise TrainMoreError(f"run {run_id} is {record.status}: only a finished run continues")
    if not 1 <= int(epochs) <= 10_000:
        raise TrainMoreError("epochs must be between 1 and 10000")
    if not 0 < float(lr_scale) <= 10:
        raise TrainMoreError("the learning-rate scale must be above 0 and at most 10")
    source = mgr.history.path(run_id)
    try:
        S.script_of(source)
    except FileNotFoundError as exc:
        raise TrainMoreError("Train more continues PyTorch networks; retrain other models "
                             "with more epochs instead") from exc
    if not any(source.glob("*_best.pt")):
        raise TrainMoreError(f"run {run_id} saved no weights to continue from")
    graph = Graph.from_dict(_with_epochs(record.graph, epochs))
    params = {"freeze": "none", "unfreeze_after": 0, "lr_scale": float(lr_scale),
              "epochs": int(epochs)}
    new_id = mgr.start_stage(graph, lambda wd: S.finetune(wd, source, params),
                             kind="continue", project=record.project,
                             tags=["train-more"])
    mgr.history.update(new_id, note=f"Continues {run_id} ({record.name}) for {int(epochs)} "
                                    "more epochs")
    return new_id
