"""Vision tasks and how a design is recognised as one.

A task-head block (detector, segmenter, U-Net, ...) decides the task. Without
one, a design that reads a vision dataset (COCO, YOLO, VOC, mask folder,
synthetic shapes) and outputs a class map [C, H, W] is semantic segmentation.
"""
from __future__ import annotations

from ai_made_easy.core.tasks import Task, register_task, register_task_resolver
from ai_made_easy.core.vision.blocks import TASK_BLOCKS, VISION_DATA, VISION_LOSSES

_BOX_METRICS = ("eval.map", "eval.map50")
# detectors compute their own losses; the optimizer default is AdamW at a low learning rate
_BOX_DEFAULTS = {"default_loss": None, "default_optimizer": "AdamW (lr = 1e-4)"}

register_task(Task(
    "detection", "Object detection",
    "Find every object: a box, a class and a confidence per object.",
    target="boxes + labels", output_role="boxes", modalities=("image",),
    trainer_kind="detection", serving="boxes", default_metrics=_BOX_METRICS,
    meta={"loss_tasks": (), "losses": [], "metrics": list(_BOX_METRICS), **_BOX_DEFAULTS}))
register_task(Task(
    "instance_segmentation", "Instance segmentation",
    "Find every object and outline it with a pixel mask.",
    target="boxes + labels + masks", output_role="masks", modalities=("image",),
    trainer_kind="detection", serving="instances",
    default_metrics=("eval.map", "eval.mask_map"),
    meta={"loss_tasks": (), "losses": [],
          "metrics": [*_BOX_METRICS, "eval.mask_map"], **_BOX_DEFAULTS}))
register_task(Task(
    "keypoints", "Keypoint detection",
    "Find objects and their keypoints (joints, corners, landmarks).",
    target="boxes + labels + keypoints", output_role="keypoints", modalities=("image",),
    trainer_kind="detection", serving="keypoints", default_metrics=("eval.map", "eval.pck"),
    meta={"loss_tasks": (), "losses": [], "metrics": [*_BOX_METRICS, "eval.pck"],
          **_BOX_DEFAULTS}))
register_task(Task(
    "semantic_segmentation", "Semantic segmentation",
    "Give every pixel a class (background, road, tumour, ...).",
    target="class mask [H, W]", output_role="masks", classification=False,
    modalities=("image",), trainer_kind="segmentation", serving="mask",
    default_metrics=("eval.miou", "eval.pixel_accuracy"),
    meta={"loss_tasks": (), "losses": ["train.loss_cross_entropy", *VISION_LOSSES],
          "metrics": ["eval.miou", "eval.dice", "eval.pixel_accuracy"],
          "default_loss": "pixel cross-entropy", "default_optimizer": "AdamW (lr = 1e-3)"}))

VISION_TASKS = ("detection", "instance_segmentation", "keypoints", "semantic_segmentation")


def head_of(graph):  # noqa: ANN001, ANN201 — Graph -> NodeInstance | None
    return next((n for n in graph.nodes.values() if n.type_id in TASK_BLOCKS), None)


def dataset_of(graph):  # noqa: ANN001, ANN201
    return next((n for n in graph.nodes.values() if n.type_id in VISION_DATA), None)


def vision_task(graph) -> str | None:  # noqa: ANN001
    head = head_of(graph)
    if head is not None:
        return TASK_BLOCKS[head.type_id]
    if dataset_of(graph) is None and not any(n.type_id in VISION_LOSSES
                                             for n in graph.nodes.values()):
        return None
    try:
        chain = graph.model_nodes()
        out = graph.infer_shapes()[chain[-1].instance_id]
    except Exception:  # noqa: BLE001 — incomplete design
        return None
    return "semantic_segmentation" if len(out) == 3 else None


register_task_resolver(vision_task)
