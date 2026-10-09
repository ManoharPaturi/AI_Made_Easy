"""Data workspace profiles for vision datasets (COCO, YOLO, VOC, mask folders, shapes).

Reads annotations with the same readers the training script uses
(``core.vision.runtime``) and reports what matters for detection and
segmentation: objects per class, object sizes (COCO small / medium / large),
empty images, mask class shares, missing files and boxes outside the image.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ai_made_easy.core.data.health import ClassCount, Finding
from ai_made_easy.core.data.lints import LOCAL_BLOCKS
from ai_made_easy.core.data.profile import DataProfile, register_profiler, resolve_path
from ai_made_easy.core.vision.blocks import VISION_DATA

PATH_PARAMS = ("images_dir", "annotations", "labels_dir", "root", "masks_dir")
SMALL, MEDIUM = 32 ** 2, 96 ** 2   # COCO area thresholds (pixels)
MAX_MASKS = 300                     # semantic masks read for class shares
MAX_SIZES = 400                     # image headers read for sizes


def _resolved(params: dict, base) -> dict:
    out = dict(params)
    for key in PATH_PARAMS:
        if out.get(key):
            out[key] = str(resolve_path(str(out[key]), base))
    if str(out.get("classes", "")).lower().endswith((".yaml", ".yml", ".txt")):
        out["classes"] = str(resolve_path(str(out["classes"]), base))
    return out


def profile_vision(type_id: str, params: dict, base=None) -> DataProfile:
    from ai_made_easy.core.vision.runtime import namespace

    ns = namespace()
    p = {"block": type_id, **_resolved(params, base)}
    for key in PATH_PARAMS:
        if p.get(key) and not Path(p[key]).exists():
            return DataProfile(type_id, "annotations", source=p[key],
                               error=f"{key.replace('_', ' ')} {p[key]} does not exist")
    records, names = ns["load_records"](p, keypoints=True)
    profile = DataProfile(type_id, "annotations", rows=len(records),
                          source=str(p.get("annotations") or p.get("labels_dir") or
                                     p.get("root") or p.get("masks_dir") or "generated"))
    if not records:
        profile.error = "no images found"
        return profile
    findings: list[Finding] = []
    has_boxes = any(r["boxes"] for r in records)
    semantic = type_id == "data.mask_folder" or (
        type_id == "data.voc" and any(r.get("semantic") for r in records))
    missing = [r["image"] for r in records
               if not isinstance(r["image"], np.ndarray) and not Path(r["image"]).exists()]
    if missing:
        findings.append(Finding("warning", f"{len(missing)} annotated image(s) are missing "
                                           f"on disk (e.g. {Path(missing[0]).name})",
                                "fix images_dir or remove those annotations"))
    details: list[str] = []
    objects = _box_stats(records, names, profile, findings, details) if has_boxes else 0
    if semantic:
        _mask_stats(ns, records, names, params, profile, findings, details)
    kinds = ["boxes"] if has_boxes else []
    if any(r.get("polygons") and any(r["polygons"]) for r in records) or \
            any(r.get("masks") is not None for r in records):
        kinds.append("instance masks")
    if any(r.get("keypoints") for r in records):
        kinds.append("keypoints")
    if semantic or type_id == "data.synthetic_shapes":
        kinds.append("pixel masks")
    sizes = _image_sizes(records)
    if sizes:
        hs, ws = zip(*sizes, strict=True)
        details.insert(0, f"image size {min(ws)}–{max(ws)} × {min(hs)}–{max(hs)} px")
    profile.summary = (f"{len(records):,} images · {len(names)} classes"
                       + (f" · {objects:,} objects" if objects else "")
                       + (f" · {', '.join(kinds)}" if kinds else ""))
    profile.task = ("detection" if has_boxes else "segmentation")
    profile.details = details
    profile.findings = findings
    return profile


def _box_stats(records, names, profile, findings, details) -> int:  # noqa: ANN001
    """Object statistics; returns the number of objects."""
    counts = np.zeros(max(len(names), 1), int)
    areas, outside, empty = [], 0, 0
    for r in records:
        labels = r["labels"]
        if not labels:
            empty += 1
        for c in labels:
            if 0 <= c < len(counts):
                counts[c] += 1
        size = r.get("size") or (None, None)
        h, w = size if size[0] else (None, None)
        if h is None and isinstance(r["image"], np.ndarray):
            h, w = r["image"].shape[:2]
        for x0, y0, x1, y1 in r["boxes"]:
            areas.append(max(x1 - x0, 0) * max(y1 - y0, 0))
            if h and (x1 > w * 1.02 or y1 > h * 1.02 or x0 < -0.02 * w or y0 < -0.02 * h):
                outside += 1
    total = int(counts.sum())
    profile.classes = [ClassCount(n, int(c)) for n, c in zip(names, counts, strict=False)]
    if areas:
        a = np.asarray(areas)
        small, medium = float((a < SMALL).mean()), float(((a >= SMALL) & (a < MEDIUM)).mean())
        details.append(f"{total:,} objects · {total / len(records):.1f} per image · "
                       f"small {small:.0%} · medium {medium:.0%} · large "
                       f"{1 - small - medium:.0%} (COCO sizes)")
        if small > 0.5:
            findings.append(Finding(
                "warning", f"{small:.0%} of objects are smaller than 32×32 px",
                "use a larger Input size / min_size, or an FPN detector"))
    if empty:
        share = empty / len(records)
        details.append(f"{empty:,} images without objects ({share:.0%})")
        if share > 0.3:
            findings.append(Finding("warning", f"{share:.0%} of images have no objects",
                                    "a few negatives help; many slow training down"))
    if outside:
        findings.append(Finding(
            "warning", f"{outside} box(es) lie outside their image",
            "YOLO labels must be normalized 0–1; COCO bbox is x, y, width, height"))
    present = counts[counts > 0]
    if len(present) > 1 and present.max() / present.min() >= 10:
        rare = names[int(np.argmin(np.where(counts > 0, counts, counts.max() + 1)))]
        findings.append(Finding(
            "warning", f"class imbalance: {present.max() / present.min():.0f}× more objects in "
                       f"the largest class than in '{rare}'",
            "collect more examples of rare classes", [rare]))
    unused = [n for n, c in zip(names, counts, strict=False) if c == 0]
    if unused and total:
        findings.append(Finding("info", f"{len(unused)} class(es) have no objects "
                                        f"(e.g. '{unused[0]}')", "", unused[:5]))
    return total


def _mask_stats(ns, records, names, params, profile, findings, details) -> None:  # noqa: ANN001
    classes = int(params.get("num_classes") or len(names) + 1)
    ignore = int(params.get("ignore_index", 255))
    pixels = np.zeros(256, np.int64)
    for r in records[:MAX_MASKS]:
        if not r.get("semantic"):
            continue
        h, w = (r["image"].shape[:2] if isinstance(r["image"], np.ndarray) else (None, None))
        mask = ns["semantic_mask"](r, h, w)
        pixels += np.bincount(mask.reshape(-1).clip(0, 255), minlength=256)
    scored = pixels.copy()
    scored[ignore] = 0
    total = max(int(scored.sum()), 1)
    too_big = [v for v in np.flatnonzero(scored) if v >= classes]
    if too_big:
        findings.append(Finding(
            "warning", f"masks contain class value {int(max(too_big))} but num_classes is "
                       f"{classes}", f"set num_classes to {int(max(too_big)) + 1} or remap "
                                     "the mask values"))
    labels = names if len(names) >= classes else ["background", *names]
    profile.classes = [ClassCount(labels[c] if c < len(labels) else str(c),
                                  int(scored[c])) for c in range(classes)]
    share = ", ".join(f"{labels[c] if c < len(labels) else c} {scored[c] / total:.0%}"
                      for c in range(min(classes, 8)))
    details.append(f"pixel share: {share}")
    fg = 1 - scored[0] / total
    if fg < 0.05:
        findings.append(Finding("warning", f"only {fg:.1%} of pixels are foreground",
                                "add Dice or Lovász loss to counter the imbalance"))


def _image_sizes(records: list) -> list[tuple[int, int]]:
    from PIL import Image

    out = []
    for r in records[:MAX_SIZES]:
        if isinstance(r["image"], np.ndarray):
            out.append(r["image"].shape[:2])
        elif r.get("size") and r["size"][0]:
            out.append(tuple(int(v) for v in r["size"]))
        elif Path(r["image"]).exists():
            w, h = Image.open(r["image"]).size
            out.append((h, w))
    return out


def register() -> None:
    for type_id in VISION_DATA:
        register_profiler(type_id, lambda params, base, _t=type_id: profile_vision(_t, params,
                                                                                    base))
        LOCAL_BLOCKS.add(type_id)
