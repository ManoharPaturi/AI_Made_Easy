"""Design rules for vision tasks (registered as lints; Quick Fixes in core.fixes).

Messages say what is wrong, why it matters and the fix; "set <param> to <value>"
phrasing is turned into a one-click fix on the flagged block.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from ai_made_easy.core.lints import LintContext, _issue, _name, register_rule
from ai_made_easy.core.training import catalog as cat
from ai_made_easy.core.vision.blocks import (
    SSD_SIDES,
    SYNTHETIC_CLASSES,
    TASK_BLOCKS,
    VISION_LOSSES,
    VISION_METRICS,
)
from ai_made_easy.core.vision.tasks import VISION_TASKS, dataset_of, head_of, vision_task

DIRECT_HEADS = {"vision.detector", "vision.instance_segmenter", "vision.keypoint_detector",
                "vision.hf_detector", "vision.segmenter", "vision.hf_segmenter"}
BOX_HEADS = {"vision.detector", "vision.instance_segmenter", "vision.keypoint_detector",
             "vision.hf_detector"}
ANCHOR_ARCHS = ("fasterrcnn", "maskrcnn", "keypointrcnn", "retinanet")
# what each dataset format can provide
PROVIDES = {
    "data.coco": {"detection", "instance_segmentation", "keypoints", "semantic_segmentation"},
    "data.yolo": {"detection", "instance_segmentation", "keypoints", "semantic_segmentation"},
    "data.voc": {"detection", "semantic_segmentation"},
    "data.mask_folder": {"semantic_segmentation"},
    "data.synthetic_shapes": set(VISION_TASKS),
}
UNSUPPORTED_AUGMENT = {"prep.auto_augment", "prep.rand_augment", "prep.trivial_augment",
                       "prep.augmix", "prep.mixup_cutmix", "prep.random_erasing"}


@lru_cache(maxsize=32)
def _coco_info(path: str, _mtime: float) -> tuple[int, int]:
    data = json.loads(Path(path).read_text())
    cats = data.get("categories", [])
    return len(cats), max((len(c.get("keypoints") or []) for c in cats), default=0)


def dataset_info(node) -> dict:  # noqa: ANN001
    """Classes / keypoints a vision dataset block declares (None when unknown)."""
    p = dict(node.resolved_params())
    info: dict = {"classes": None, "keypoints": None, "background": False}
    if node.type_id == "data.synthetic_shapes":
        info.update(classes=len(SYNTHETIC_CLASSES), keypoints=3)
    elif node.type_id == "data.mask_folder":
        info.update(classes=int(p["num_classes"]), background=True)
    elif node.type_id in ("data.yolo", "data.voc"):
        names = str(p.get("classes") or "")
        if names and not names.lower().endswith((".yaml", ".yml", ".txt")):
            info["classes"] = len([n for n in names.split(",") if n.strip()])
    elif node.type_id == "data.coco":
        path = Path(str(p["annotations"])).expanduser()
        if path.is_file():
            try:
                info["classes"], info["keypoints"] = _coco_info(str(path),
                                                                path.stat().st_mtime)
            except (OSError, ValueError):
                pass
    return info


def vision_rules(ctx: LintContext) -> list:
    task = vision_task(ctx.graph)
    if task is None:
        return []
    out: list = []
    graph = ctx.graph
    heads = [n for n in graph.nodes.values() if n.type_id in TASK_BLOCKS]
    head = head_of(graph)
    if len(heads) > 1:
        out.append(_issue("error", "a design trains one vision task: keep one detector / "
                                   "segmenter block", heads[1].instance_id))
    if head is not None and head.type_id in DIRECT_HEADS:
        prods = ctx.producers(head.instance_id)
        cons = ctx.consumers(head.instance_id)
        if (prods and prods[0].type_id != "core.input") or \
                (cons and cons[0].type_id != "core.output"):
            out.append(_issue(
                "error", f"{_name(head)} takes the image straight from the Input and feeds the "
                         "Output: it resizes and normalizes images itself. Remove the blocks "
                         "in between (use augmentation blocks for image transforms).",
                head.instance_id))
    out += _dataset_rules(ctx, task, head)
    if head is not None:
        out += _head_rules(ctx, head)
    out += _pipeline_rules(ctx, task)
    return out


def _dataset_rules(ctx: LintContext, task: str, head) -> list:  # noqa: ANN001
    data = dataset_of(ctx.graph)
    if data is None:
        return [] if head is None else [_issue(
            "warning", "no dataset block: training uses generated shapes (Synthetic Shapes); "
                       "add a COCO, YOLO, VOC or mask-folder dataset for your images",
            head.instance_id)]
    out = []
    if task not in PROVIDES[data.type_id]:
        what = {"keypoints": "keypoint annotations", "instance_segmentation": "instance masks",
                "detection": "boxes", "semantic_segmentation": "pixel masks"}[task]
        out.append(_issue("error", f"{_name(data)} has no {what}: use a dataset that "
                                   f"provides them for {task.replace('_', ' ')}",
                          data.instance_id))
        return out
    info = dataset_info(data)
    n = info["classes"]
    if n is None:
        return out
    if task == "semantic_segmentation":
        want = n if info["background"] else n + 1
        node = head or ctx.last_compute()
        have = (int(head.resolved_params()["num_classes"]) if head is not None
                else (ctx.shapes.get(node.instance_id) or [0])[0] if node else 0)
        if node is not None and have != want:
            bg = "" if info["background"] else f" ({n} classes + background)"
            fix = " set num_classes to " + str(want) if head is not None else \
                f" make the model output {want} channels"
            out.append(_issue("warning", f"{_name(data)} has {want} pixel classes{bg} but the "
                                         f"model predicts {have}:{fix}", node.instance_id))
    elif head is not None:
        have = int(head.resolved_params()["num_classes"])
        if have != n:
            out.append(_issue("warning", f"{_name(data)} has {n} object classes but "
                                         f"{_name(head)} predicts {have} (background is added "
                                         f"automatically): set num_classes to {n}",
                              head.instance_id))
        if task == "keypoints" and info["keypoints"]:
            k = int(head.resolved_params().get("num_keypoints", 0))
            if k != info["keypoints"]:
                out.append(_issue("warning", f"{_name(data)} annotates {info['keypoints']} "
                                             f"keypoints per object: set num_keypoints to "
                                             f"{info['keypoints']}", head.instance_id))
    return out


def _head_rules(ctx: LintContext, head) -> list:  # noqa: ANN001
    p = dict(head.resolved_params())
    out = []
    shape = ctx.shapes.get(ctx.chain[0].instance_id) or []
    arch = str(p.get("arch", ""))
    if head.type_id in BOX_HEADS and arch.startswith(ANCHOR_ARCHS) and len(shape) == 3:
        short = int(p.get("min_size") or 0) or min(shape[1:])
        if int(p.get("min_size") or 0) > 2 * min(shape[1:]):
            out.append(_issue(
                "warning", f"{_name(head)} enlarges your {min(shape[1:])} px images to "
                           f"{p['min_size']} px: slower training without extra detail. "
                           "Set min_size to 0 to use the Input size", head.instance_id))
        elif short < 128:
            out.append(_issue(
                "warning", f"{_name(head)}'s default anchors span 32–512 px but images are "
                           f"{short} px: most anchors fall outside the image. Use images of "
                           "at least 256 px or an anchor-free detector (FCOS, DETR)",
                head.instance_id))
    if head.type_id == "vision.detector" and arch in SSD_SIDES and p.get("weights") == "coco" \
            and int(p["num_classes"]) != 90:
        out.append(_issue("error", f"{arch} COCO weights cannot change the number of classes: "
                                   "use weights imagenet to train your own classes",
                          head.instance_id))
    if head.type_id == "vision.hf_segmenter" and len(shape) == 3 and (shape[1] % 32 or
                                                                      shape[2] % 32):
        out.append(_issue("warning", f"SegFormer downsamples by 32: use an Input height and "
                                     f"width that are multiples of 32 (got {shape[1]}×"
                                     f"{shape[2]}) to keep edges aligned", head.instance_id))
    return out


def _pipeline_rules(ctx: LintContext, task: str) -> list:
    out = []
    box = task != "semantic_segmentation"
    for node in ctx.graph.nodes.values():
        tid = node.type_id
        if tid == "prep.normalize":
            out.append(_issue("warning", "vision models normalize images themselves: remove "
                                         "Normalize (images are fed in the 0–1 range)",
                              node.instance_id))
        elif tid in UNSUPPORTED_AUGMENT:
            out.append(_issue("warning", f"{_name(node)} cannot move boxes, masks or keypoints "
                                         "with the image and is skipped for vision tasks",
                              node.instance_id))
        elif box and (tid in VISION_LOSSES or tid in cat.LOSS_IDS):
            out.append(_issue("warning", f"{_name(node)} is ignored: detectors compute their "
                                         "own losses", node.instance_id))
        elif tid in VISION_METRICS and task not in VISION_METRICS[tid]["tasks"]:
            out.append(_issue("warning", f"{_name(node)} does not apply to "
                                         f"{task.replace('_', ' ')} and is skipped",
                              node.instance_id))
        elif tid in cat.METRIC_IDS:
            out.append(_issue("warning", f"{_name(node)} is a classification / regression "
                                         f"metric and is skipped for {task.replace('_', ' ')}",
                              node.instance_id))
        elif tid == "train.loss_cross_entropy" or tid in VISION_LOSSES:
            continue
        elif tid in cat.LOSS_IDS:
            out.append(_issue("warning", f"{_name(node)} does not train segmentation: use Cross "
                                         "Entropy, Dice, Tversky, Lovász or Pixel Focal loss",
                              node.instance_id))
    return out


def model_licenses(ctx: LintContext) -> list:
    """Pretrained weights whose license restricts use (e.g. non-commercial)."""
    from ai_made_easy.core.vision.blocks import TIMM_MODELS

    out = []
    for node in ctx.nodes_of("vision.timm_backbone"):
        p = dict(node.resolved_params())
        license_ = str(TIMM_MODELS.get(p["model"], {}).get("license", "")).lower()
        if p.get("weights") == "imagenet" and "-nc" in license_:
            out.append(_issue("warning", f"{p['model']} weights are licensed {license_}: "
                                         "non-commercial use only. Pick another model to ship "
                                         "a commercial product", node.instance_id))
    return out


register_rule(vision_rules)
register_rule(model_licenses)
