"""Augmentation preview for vision datasets: boxes, masks and keypoints drawn on every view.

Runs the training script's own pipeline (always-applied transforms, box / mask /
keypoint-aware augmentation, resize to the model input) on a few images, in a
subprocess of the training environment, and saves overlays for the original,
the evaluation view and several random training views.
"""
from __future__ import annotations

from pathlib import Path

from ai_made_easy.core.data.augment import MARKER, PreviewError, register_previewer, run_preview
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.vision import runtime

_SCRIPT = r'''
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import tv_tensors
from torchvision.transforms import v2

cfg = json.loads(Path(sys.argv[1]).read_text())
torch.manual_seed(cfg["seed"])
random.seed(cfg["seed"])
''' + runtime.DATA_CODE + runtime.ENCODE_CODE + r'''
out = Path(cfg["out"])
out.mkdir(parents=True, exist_ok=True)
ns = {"v2": v2, "torch": torch}
always = [eval(e, ns) for e in cfg["always"]]
train = [eval(e, ns) for e in cfg["train"]]
size = cfg["input_size"]
finish = [v2.Resize(size, antialias=True)] if size else []
eval_tf = v2.Compose([*always, *finish])
train_tf = v2.Compose([*always, *train, *finish])
records, names = load_records(cfg["dataset"], keypoints=True)
picked = [r for r in records if r["boxes"] or r.get("semantic") is not None][: cfg["images"]]
if not picked:
    picked = records[: cfg["images"]]


def views(rec):
    arr = load_image(rec)
    h, w = arr.shape[:2]
    image = tv_tensors.Image(torch.from_numpy(np.array(arr, copy=True)).permute(2, 0, 1))
    sample = {"boxes": tv_tensors.BoundingBoxes(
        torch.tensor(rec["boxes"], dtype=torch.float32).reshape(-1, 4), format="XYXY",
        canvas_size=(h, w)), "labels": torch.tensor(rec["labels"], dtype=torch.int64)}
    if cfg["semantic"]:
        sample["semantic"] = tv_tensors.Mask(torch.from_numpy(semantic_mask(rec, h, w)))
    elif rec["boxes"]:
        sample["masks"] = tv_tensors.Mask(torch.from_numpy(instance_masks(rec, h, w)))
    if rec.get("keypoints"):
        kp = np.stack([np.asarray(k, np.float32) for k in rec["keypoints"]])
        sample["keypoints"] = tv_tensors.KeyPoints(torch.from_numpy(kp[..., :2]),
                                                   canvas_size=(h, w))
    return image, sample


def draw(image, sample, path):
    arr = image.permute(1, 2, 0).numpy().astype(np.uint8)
    mask = None
    if "semantic" in sample:
        mask = sample["semantic"].numpy()
    elif "masks" in sample and len(sample["masks"]):
        mask = np.zeros(arr.shape[:2], np.int64)
        for m, c in zip(sample["masks"].numpy(), sample["labels"].numpy()):
            mask[m > 0] = c + 1
    true = {"boxes": sample["boxes"].numpy(), "labels": sample["labels"].numpy()}
    if "keypoints" in sample:
        true["keypoints"] = sample["keypoints"].numpy()
    overlay(arr, None, true if not cfg["semantic"] else None, names, mask).save(path)
    return str(path)


result = []
for i, rec in enumerate(picked):
    image, sample = views(rec)
    item = {"source": rec["image"] if isinstance(rec["image"], str) else f"generated #{i}",
            "size": [int(image.shape[2]), int(image.shape[1])]}
    item["original"] = draw(image, sample, out / f"{i}_original.png")
    e_img, e_sample = eval_tf(image, sample)
    item["eval"] = draw(e_img, e_sample, out / f"{i}_eval.png")
    item["eval_shape"] = list(e_img.shape)
    item["train"] = []
    for k in range(cfg["variants"]):
        t_img, t_sample = train_tf(image, sample)
        item["train"].append(draw(t_img, t_sample, out / f"{i}_train_{k}.png"))
    result.append(item)
print("''' + MARKER + r'''" + json.dumps({"images": result}))
'''


def vision_preview(graph: Graph, out_dir, *, base=None, python=None, images: int = 4,  # noqa: ANN001
                   variants: int = 6, seed: int = 0, timeout: float = 180) -> dict | None:
    from ai_made_easy.core.vision.profile import _resolved
    from ai_made_easy.core.vision.tasks import dataset_of, vision_task
    from ai_made_easy.core.vision.template import _augmentations

    data = dataset_of(graph)
    if data is None:
        return None
    dataset = {"block": data.type_id, **_resolved(dict(data.resolved_params()), base)}
    for key in ("images_dir", "annotations", "labels_dir", "root", "masks_dir"):
        if dataset.get(key) and not Path(dataset[key]).exists():
            raise PreviewError(f"{key.replace('_', ' ')} {dataset[key]} was not found")
    notes: list[str] = []
    always, augment = _augmentations(graph, notes)
    try:
        shape = graph.infer_shapes()[graph.model_nodes()[0].instance_id]
        size = [shape[1], shape[2]] if len(shape) == 3 else None
    except Exception:  # noqa: BLE001 — incomplete designs keep the image size
        size = None
    task = vision_task(graph)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"dataset": dataset, "out": str(out), "seed": seed, "variants": variants,
           "images": images, "always": always, "train": augment, "input_size": size,
           "semantic": task == "semantic_segmentation"}
    result = run_preview(_SCRIPT, cfg, out, python, timeout)
    result.update(always=always, train_transforms=augment, skipped=notes)
    return result


register_previewer(vision_preview)
