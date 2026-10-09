"""Task registry: what a model is trained to do, and everything that follows from it.

A :class:`Task` ties together the target format, the role of the model's
output, the losses and metrics that apply, default metrics, the trainer kind
and the serving schema. Training specs, lints, the data workspace, deployment
and the (future) task wizard all read tasks from here instead of hard-coding
task names. New families register their own tasks (detection, segmentation,
forecasting, graph tasks, ...) with :func:`register_task`.

Pure Python, Qt-free.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# how a model is trained; the training template is chosen by this
TRAINER_KINDS = ("supervised", "detection", "segmentation", "forecasting", "speech",
                 "adversarial", "diffusion", "vae", "language_model", "seq2seq",
                 "flow", "self_supervised", "rl", "bayesian")


@dataclass(frozen=True)
class Task:
    id: str
    label: str
    description: str = ""
    family: str = "neural"
    target: str = ""               # target format, e.g. "class index", "float vector"
    output_role: str = "logits"    # role of the model's output port (core.spec.ROLES)
    classification: bool = False
    modalities: tuple[str, ...] = ()   # data modalities the task supports ("" = all)
    trainer_kind: str = "supervised"
    serving: str = "values"        # serving response schema id
    default_metrics: tuple[str, ...] = ()
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------ catalogs
    def losses(self) -> list[str]:
        """Loss block ids that train this task."""
        if "losses" in self.meta:
            return list(self.meta["losses"])
        from ai_made_easy.core.training import catalog as cat

        return [i for i in cat.LOSS_IDS
                if cat.COMPONENTS[i].meta.get("task") in self.loss_tasks]

    def metrics(self) -> list[str]:
        """Metric block ids that evaluate this task."""
        if "metrics" in self.meta:
            return list(self.meta["metrics"])
        from ai_made_easy.core.training import catalog as cat

        return [i for i in cat.METRIC_IDS if self.id in cat.COMPONENTS[i].meta.get("tasks", ())]

    @property
    def loss_tasks(self) -> tuple[str, ...]:
        """Loss ``meta['task']`` values that resolve to this task."""
        return tuple(self.meta.get("loss_tasks", (self.id,)))

    def to_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "description": self.description,
                "family": self.family, "target": self.target, "output_role": self.output_role,
                "classification": self.classification, "modalities": list(self.modalities),
                "trainer_kind": self.trainer_kind, "serving": self.serving,
                "default_metrics": list(self.default_metrics), "losses": self.losses(),
                "metrics": self.metrics()}


_TASKS: dict[str, Task] = {}


def register_task(task: Task) -> Task:
    if task.trainer_kind not in TRAINER_KINDS:
        raise ValueError(f"{task.id}: unknown trainer kind {task.trainer_kind!r}")
    _TASKS[task.id] = task
    return task


def get_task(task_id: str) -> Task:
    try:
        return _TASKS[task_id]
    except KeyError:
        raise KeyError(f"unknown task {task_id!r}; one of {sorted(_TASKS)}") from None


def all_tasks(family: str | None = None) -> list[Task]:
    return [t for t in _TASKS.values() if family is None or t.family == family]


def resolve_task(loss_task: str, num_outputs: int = 1) -> Task:
    """The task a loss trains, refined by the model's output size.

    A binary loss on several outputs is multi-label classification.
    """
    if loss_task == "binary" and num_outputs > 1:
        return _TASKS["multilabel"]
    for task in _TASKS.values():
        if loss_task == task.id or loss_task in task.loss_tasks:
            return task
    raise KeyError(f"no task trains with loss task {loss_task!r}")


# resolvers that recognise a task from the design's blocks before the loss-based rule
# (e.g. a detector head means object detection); each returns a task id or None
TaskResolver = Callable[[object], "str | None"]
_RESOLVERS: list[TaskResolver] = []


def register_task_resolver(resolver: TaskResolver) -> None:
    if resolver not in _RESOLVERS:
        _RESOLVERS.append(resolver)


def task_of(graph) -> Task | None:  # noqa: ANN001 — core.graph.Graph
    """The task of a neural design (task blocks first, then its loss block), or None."""
    from ai_made_easy.core.training import catalog as cat

    for resolver in _RESOLVERS:
        found = resolver(graph)
        if found:
            return _TASKS[found]
    loss = next((n for n in graph.nodes.values() if n.type_id in cat.LOSS_IDS), None)
    if loss is None:
        return None
    outputs = 1
    try:
        chain = graph.model_nodes()
        outputs = int(graph.infer_shapes()[chain[-1].instance_id][0]) if chain else 1
    except Exception:  # noqa: BLE001 — incomplete designs: assume one output
        pass
    return resolve_task(cat.COMPONENTS[loss.type_id].meta["task"], outputs)


# ------------------------------------------------------------- built-in tasks

register_task(Task(
    "multiclass", "Multi-class classification",
    "Assign each sample exactly one of N classes.",
    target="class index", output_role="logits", classification=True, serving="label_probs",
    default_metrics=("eval.accuracy",)))
register_task(Task(
    "binary", "Binary classification",
    "Decide between two classes from one output (logit).",
    target="0 / 1", output_role="logits", classification=True, serving="label_probs",
    default_metrics=("eval.accuracy", "eval.roc_auc")))
register_task(Task(
    "multilabel", "Multi-label classification",
    "Any number of N labels can apply to a sample (one logit per label).",
    target="multi-hot vector", output_role="logits", classification=True,
    serving="multi_label", default_metrics=("eval.accuracy",),
    meta={"loss_tasks": ("multilabel",)}))
register_task(Task(
    "regression", "Regression",
    "Predict one or more continuous values.",
    target="float vector", output_role="tensor", serving="values",
    default_metrics=("eval.mae", "eval.r2")))
register_task(Task(
    "distribution", "Distribution matching",
    "Match a target probability distribution (e.g. KL-divergence training).",
    target="probability vector", output_role="probs", serving="values"))
