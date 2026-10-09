"""Runtime code shared by generated vision scripts and the app.

Each string below is plain, self-contained Python (numpy, PIL; torch only where
noted). The training-script template embeds them verbatim; the app executes
the same source (``namespace()``) to profile datasets and in tests, so the
data a run trains on and the data the designer reports on are identical.
"""
from __future__ import annotations

DATA_CODE = r'''
# ======================================================================== vision data
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def _images_in(folder) -> list:
    folder = Path(str(folder)).expanduser()
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def _record(image, boxes=(), labels=(), polygons=None, keypoints=None, semantic=None,
            masks=None) -> dict:
    return {"image": image, "boxes": [list(map(float, b)) for b in boxes],
            "labels": [int(v) for v in labels], "polygons": polygons, "keypoints": keypoints,
            "semantic": semantic, "masks": masks}


def read_coco(images_dir, annotations) -> tuple:
    """COCO JSON -> records; categories become contiguous 0-based classes."""
    data = json.loads(Path(str(annotations)).expanduser().read_text())
    cats = sorted(data.get("categories", []), key=lambda c: c["id"])
    index = {c["id"]: i for i, c in enumerate(cats)}
    names = [str(c.get("name", c["id"])) for c in cats]
    n_kp = max((len(c.get("keypoints") or []) for c in cats), default=0)
    by_image: dict = {}
    for a in data.get("annotations", []):
        if not a.get("iscrowd"):
            by_image.setdefault(a["image_id"], []).append(a)
    root = Path(str(images_dir)).expanduser()
    records = []
    for img in data.get("images", []):
        boxes, labels, polys, kps = [], [], [], []
        for a in by_image.get(img["id"], []):
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0 or a["category_id"] not in index:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(index[a["category_id"]])
            polys.append(a.get("segmentation") or [])
            k = a.get("keypoints") or []
            kps.append(np.asarray(k, np.float32).reshape(-1, 3) if k
                       else np.zeros((n_kp, 3), np.float32))
        rec = _record(str(root / img["file_name"]), boxes, labels, polys,
                      kps if n_kp else None)
        rec["size"] = (img.get("height"), img.get("width"))
        records.append(rec)
    return records, names


def _class_names(spec: str, count: int = 0) -> list:
    spec = str(spec or "").strip()
    if spec.lower().endswith((".yaml", ".yml")):
        text = Path(spec).expanduser().read_text()
        try:
            import yaml

            names = yaml.safe_load(text).get("names", [])
            return [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
        except ImportError:
            pass
        names, in_names = [], False
        for line in text.splitlines():
            if line.strip().startswith("names:"):
                rest = line.split(":", 1)[1].strip()
                if rest.startswith("["):
                    return [n.strip().strip("'\"") for n in rest.strip("[]").split(",")
                            if n.strip()]
                in_names = True
            elif in_names and line.startswith((" ", "\t", "-")):
                item = line.strip().lstrip("-").strip()
                names.append(item.split(":", 1)[1].strip().strip("'\"") if ":" in item
                             else item.strip("'\""))
            elif in_names and line.strip():
                break
        return names
    if spec.lower().endswith(".txt"):
        return [ln.strip() for ln in Path(spec).expanduser().read_text().splitlines()
                if ln.strip()]
    names = [n.strip() for n in spec.split(",") if n.strip()]
    return names or [f"class_{i}" for i in range(count)]


def read_yolo(images_dir, labels_dir, classes="", keypoints: bool = False) -> tuple:
    """YOLO txt labels: boxes (cx cy w h), segmentation polygons or pose keypoints."""
    images_dir = Path(str(images_dir)).expanduser()
    labels_dir = Path(str(labels_dir)).expanduser()
    records, top = [], -1
    for path in _images_in(images_dir):
        label = labels_dir / path.relative_to(images_dir).with_suffix(".txt")
        w, h = Image.open(path).size
        boxes, labels, polys, kps = [], [], [], []
        lines = label.read_text().splitlines() if label.exists() else []
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls, vals = int(float(parts[0])), [float(v) for v in parts[1:]]
            top = max(top, cls)
            if keypoints and len(vals) > 4 and (len(vals) - 4) % 3 == 0:
                cx, cy, bw, bh = vals[:4]
                pts = np.asarray(vals[4:], np.float32).reshape(-1, 3) * [w, h, 1]
                kps.append(pts)
                polys.append([])
            elif len(vals) > 4 and len(vals) % 2 == 0:
                xy = np.asarray(vals, np.float32).reshape(-1, 2) * [w, h]
                (x0, y0), (x1, y1) = xy.min(0), xy.max(0)
                cx, cy, bw, bh = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h
                polys.append([xy.reshape(-1).tolist()])
            else:
                cx, cy, bw, bh = vals[:4]
                polys.append([])
            boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w,
                          (cy + bh / 2) * h])
            labels.append(cls)
        rec = _record(str(path), boxes, labels, polys, kps if keypoints else None)
        rec["size"] = (h, w)
        records.append(rec)
    return records, _class_names(classes, top + 1)


def read_voc(root, classes="") -> tuple:
    """Pascal VOC: Annotations/*.xml boxes, SegmentationClass/*.png masks when present."""
    import xml.etree.ElementTree as ET

    root = Path(str(root)).expanduser()
    parsed, seen = [], []
    for xml_path in sorted((root / "Annotations").glob("*.xml")):
        tree = ET.parse(xml_path).getroot()
        objects = []
        for obj in tree.findall("object"):
            name = obj.findtext("name", "").strip()
            box = obj.find("bndbox")
            coords = [float(box.findtext(k, "0")) for k in ("xmin", "ymin", "xmax", "ymax")]
            objects.append((name, [coords[0] - 1, coords[1] - 1, coords[2], coords[3]]))
            if name not in seen:
                seen.append(name)
        parsed.append((tree.findtext("filename", xml_path.stem + ".jpg"), xml_path.stem,
                       objects))
    names = _class_names(classes) or sorted(seen)
    index = {n: i for i, n in enumerate(names)}
    records = []
    for filename, stem, objects in parsed:
        keep = [(index[n], b) for n, b in objects if n in index]
        mask = root / "SegmentationClass" / f"{stem}.png"
        records.append(_record(str(root / "JPEGImages" / filename), [b for _, b in keep],
                               [c for c, _ in keep],
                               semantic=str(mask) if mask.exists() else None))
    return records, names


def read_mask_folder(images_dir, masks_dir, num_classes, class_names="") -> tuple:
    masks_dir = Path(str(masks_dir)).expanduser()
    by_stem = {p.stem: p for p in masks_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS}
    records = [_record(str(p), semantic=str(by_stem[p.stem]))
               for p in _images_in(images_dir) if p.stem in by_stem]
    names = _class_names(class_names) or ["background"] + [
        f"class_{i}" for i in range(1, int(num_classes))]
    return records, names


SHAPES = ("square", "circle", "triangle")


def synthetic_shapes(n_images: int, size: int, max_objects: int, seed: int) -> tuple:
    """Random squares / circles / triangles with exact boxes, masks and keypoints."""
    rng = np.random.default_rng(seed)
    colors = [(230, 80, 70), (80, 200, 120), (90, 130, 240)]
    records = []
    for _ in range(n_images):
        base = rng.integers(10, 60, size=3)
        img = Image.fromarray((rng.normal(0, 8, (size, size, 3)) + base).clip(0, 255)
                              .astype(np.uint8))
        draw = ImageDraw.Draw(img)
        masks, labels = [], []
        for _ in range(int(rng.integers(1, max_objects + 1))):
            cls = int(rng.integers(0, 3))
            s = int(rng.integers(max(size // 8, 6), max(size // 3, 8)))
            x0, y0 = (int(v) for v in rng.integers(0, size - s, size=2))
            layer = Image.new("L", (size, size), 0)
            ld = ImageDraw.Draw(layer)
            shape = [(x0, y0, x0 + s, y0 + s)]
            if cls == 0:
                draw.rectangle(shape[0], fill=colors[0])
                ld.rectangle(shape[0], fill=1)
            elif cls == 1:
                draw.ellipse(shape[0], fill=colors[1])
                ld.ellipse(shape[0], fill=1)
            else:
                tri = [(x0 + s / 2, y0), (x0, y0 + s), (x0 + s, y0 + s)]
                draw.polygon(tri, fill=colors[2])
                ld.polygon(tri, fill=1)
            m = np.asarray(layer, bool)
            masks = [prev & ~m for prev in masks]
            masks.append(m)
            labels.append(cls)
        keep = [i for i, m in enumerate(masks) if m.sum() >= 30]
        masks = [masks[i] for i in keep]
        labels = [labels[i] for i in keep]
        boxes, kps = [], []
        semantic = np.zeros((size, size), np.uint8)
        for m, cls in zip(masks, labels):
            ys, xs = np.nonzero(m)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
            top, left, right = ys.argmin(), xs.argmin(), xs.argmax()
            kps.append(np.asarray([[xs[top], ys[top], 2], [xs[left], ys[left], 2],
                                   [xs[right], ys[right], 2]], np.float32))
            semantic[m] = cls + 1
        records.append(_record(np.asarray(img), boxes, labels, keypoints=kps,
                               semantic=semantic, masks=np.stack(masks) if masks else
                               np.zeros((0, size, size), bool)))
    return records, list(SHAPES)


def load_records(dataset: dict, keypoints: bool = False) -> tuple:
    """(records, class names) for a dataset block's parameters."""
    block = dataset["block"]
    if block == "data.coco":
        return read_coco(dataset["images_dir"], dataset["annotations"])
    if block == "data.yolo":
        return read_yolo(dataset["images_dir"], dataset["labels_dir"], dataset["classes"],
                         keypoints)
    if block == "data.voc":
        return read_voc(dataset["root"], dataset["classes"])
    if block == "data.mask_folder":
        return read_mask_folder(dataset["images_dir"], dataset["masks_dir"],
                                dataset["num_classes"], dataset["class_names"])
    if block == "data.synthetic_shapes":
        return synthetic_shapes(int(dataset["n_images"]), int(dataset["image_size"]),
                                int(dataset["max_objects"]), int(dataset["seed"]))
    raise ValueError(f"unknown vision dataset {block!r}")


def load_image(rec: dict) -> np.ndarray:
    """HxWx3 uint8 RGB."""
    if isinstance(rec["image"], np.ndarray):
        return rec["image"]
    return np.asarray(Image.open(rec["image"]).convert("RGB"))


def _decode_rle(rle: dict, h: int, w: int) -> np.ndarray:
    counts = rle["counts"]
    if isinstance(counts, str):
        from pycocotools import mask as coco_mask  # compressed RLE

        return coco_mask.decode(rle).astype(bool)
    flat = np.zeros(h * w, bool)
    pos, value = 0, False
    for n in counts:
        flat[pos:pos + n] = value
        pos, value = pos + n, not value
    return flat.reshape(w, h).T


def instance_masks(rec: dict, h: int, w: int) -> np.ndarray:
    """[N, H, W] bool masks (from arrays, polygons, RLE, or filled boxes)."""
    if rec.get("masks") is not None:
        return np.asarray(rec["masks"], bool)
    out = np.zeros((len(rec["boxes"]), h, w), bool)
    for i, box in enumerate(rec["boxes"]):
        seg = (rec.get("polygons") or [[]] * len(rec["boxes"]))[i]
        if isinstance(seg, dict):
            out[i] = _decode_rle(seg, h, w)
        elif seg:
            layer = Image.new("L", (w, h), 0)
            for poly in seg:
                if len(poly) >= 6:
                    ImageDraw.Draw(layer).polygon([float(v) for v in poly], fill=1)
            out[i] = np.asarray(layer, bool)
        else:
            x0, y0, x1, y1 = (int(round(v)) for v in box)
            out[i, max(y0, 0):y1, max(x0, 0):x1] = True
    return out


def semantic_mask(rec: dict, h: int, w: int) -> np.ndarray:
    """[H, W] class indices: 0 = background, object class k -> k + 1."""
    sem = rec.get("semantic")
    if isinstance(sem, np.ndarray):
        return sem.astype(np.int64)
    if sem:
        return np.asarray(Image.open(sem)).astype(np.int64)
    out = np.zeros((h, w), np.int64)
    for m, cls in zip(instance_masks(rec, h, w), rec["labels"]):
        out[m] = cls + 1
    return out


def split_records(records: list, val: float, test: float, seed: int, shuffle: bool) -> tuple:
    order = np.arange(len(records))
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    n_test = int(round(len(order) * test))
    n_val = int(round(len(order) * val))
    if len(order) >= 3:
        n_val, n_test = max(n_val, 1) if val > 0 else 0, max(n_test, 1) if test > 0 else 0
    test_idx = order[:n_test]
    val_idx = order[n_test:n_test + n_val]
    train_idx = order[n_test + n_val:]
    return ([records[i] for i in train_idx], [records[i] for i in val_idx],
            [records[i] for i in test_idx])
'''

METRICS_CODE = r'''
# ======================================================================== vision metrics
def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.asarray(a, np.float64).reshape(-1, 4), np.asarray(b, np.float64).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).clip(0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clip(0).prod(-1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)


def mask_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    a = a.reshape(a.shape[0], int(np.prod(a.shape[1:]))).astype(np.float64)
    b = b.reshape(b.shape[0], int(np.prod(b.shape[1:]))).astype(np.float64)
    inter = a @ b.T
    union = a.sum(1)[:, None] + b.sum(1)[None, :] - inter
    return inter / np.maximum(union, 1e-12)


IOU_THRESHOLDS = np.round(np.arange(0.5, 0.96, 0.05), 2)
RECALL_POINTS = np.linspace(0.0, 1.0, 101)


def average_precision(preds: list, targets: list, kind: str = "boxes",
                      max_dets: int = 100) -> dict:
    """COCO-style AP (101-point interpolation, per class, IoU 0.50:0.95).

    preds: dicts with boxes / scores / labels (and masks); targets: boxes / labels (masks).
    Returns {"map", "map50", "map75", "per_class": {label: AP}}.
    """
    classes = sorted({int(c) for t in targets for c in np.asarray(t["labels"]).tolist()})
    per_thr = np.full((len(IOU_THRESHOLDS), len(classes)), np.nan)
    for ci, cls in enumerate(classes):
        n_gt = 0
        scores, matched = [], [[] for _ in IOU_THRESHOLDS]
        for p, t in zip(preds, targets):
            gl = np.asarray(t["labels"]) == cls
            pl = np.asarray(p["labels"]) == cls
            n_gt += int(gl.sum())
            ps = np.asarray(p["scores"], np.float64)[pl]
            order = np.argsort(-ps, kind="mergesort")[:max_dets]
            ps = ps[order]
            if kind == "masks":
                ious = mask_iou(np.asarray(p["masks"])[pl][order], np.asarray(t["masks"])[gl])
            else:
                ious = box_iou(np.asarray(p["boxes"])[pl][order], np.asarray(t["boxes"])[gl])
            scores.extend(ps.tolist())
            for ti, thr in enumerate(IOU_THRESHOLDS):
                taken = np.zeros(ious.shape[1], bool)
                for d in range(len(ps)):
                    best, best_j = min(thr, 1 - 1e-10), -1
                    for j in range(ious.shape[1]):
                        if not taken[j] and ious[d, j] >= best:
                            best, best_j = ious[d, j], j
                    if best_j >= 0:
                        taken[best_j] = True
                    matched[ti].append(best_j >= 0)
        if n_gt == 0:
            continue
        order = np.argsort(-np.asarray(scores), kind="mergesort")
        for ti in range(len(IOU_THRESHOLDS)):
            tp = np.asarray(matched[ti], bool)[order]
            tps, fps = np.cumsum(tp), np.cumsum(~tp)
            recall = tps / n_gt
            precision = tps / np.maximum(tps + fps, np.spacing(1))
            for k in range(len(precision) - 1, 0, -1):
                precision[k - 1] = max(precision[k - 1], precision[k])
            idx = np.searchsorted(recall, RECALL_POINTS, side="left")
            q = np.zeros(len(RECALL_POINTS))
            ok = idx < len(precision)
            q[ok] = precision[idx[ok]]
            per_thr[ti, ci] = q.mean()
    if not classes or np.all(np.isnan(per_thr)):
        return {"map": 0.0, "map50": 0.0, "map75": 0.0, "per_class": {}}
    return {"map": float(np.nanmean(per_thr)), "map50": float(np.nanmean(per_thr[0])),
            "map75": float(np.nanmean(per_thr[5])),
            "per_class": {c: float(np.nanmean(per_thr[:, i])) for i, c in enumerate(classes)
                          if not np.all(np.isnan(per_thr[:, i]))}}


def keypoint_pck(preds: list, targets: list, threshold: float = 0.2) -> float:
    """Share of visible keypoints within threshold x box diagonal (boxes matched at IoU .5)."""
    hits = total = 0
    for p, t in zip(preds, targets):
        if not len(t["boxes"]) or not len(p["boxes"]):
            total += int((np.asarray(t.get("keypoints", np.zeros((0, 0, 3))))[..., 2] > 0).sum())
            continue
        ious = box_iou(p["boxes"], t["boxes"])
        for j in range(ious.shape[1]):
            gk = np.asarray(t["keypoints"][j])
            vis = gk[:, 2] > 0
            total += int(vis.sum())
            i = int(ious[:, j].argmax())
            if ious[i, j] < 0.5:
                continue
            box = np.asarray(t["boxes"][j])
            diag = float(np.hypot(box[2] - box[0], box[3] - box[1])) or 1.0
            dist = np.hypot(*(np.asarray(p["keypoints"][i])[:, :2] - gk[:, :2]).T)
            hits += int((dist[vis] <= threshold * diag).sum())
    return hits / total if total else 0.0


def segmentation_scores(conf: np.ndarray) -> dict:
    """mIoU / Dice / pixel accuracy from a confusion matrix [true, pred]."""
    tp = np.diag(conf).astype(np.float64)
    gt, pr = conf.sum(1), conf.sum(0)
    present = (gt + pr) > 0
    iou = tp / np.maximum(gt + pr - tp, 1)
    dice = 2 * tp / np.maximum(gt + pr, 1)
    return {"miou": float(iou[present].mean()) if present.any() else 0.0,
            "dice": float(dice[present].mean()) if present.any() else 0.0,
            "pixel_accuracy": float(tp.sum() / max(conf.sum(), 1)),
            "per_class_iou": [float(v) if ok else None for v, ok in zip(iou, present)]}


def confusion(pred: np.ndarray, true: np.ndarray, classes: int, ignore: int = 255) -> np.ndarray:
    keep = (true != ignore) & (true >= 0) & (true < classes)
    return np.bincount(classes * true[keep].astype(np.int64) + pred[keep].astype(np.int64),
                       minlength=classes * classes).reshape(classes, classes)
'''

ENCODE_CODE = r'''
# ======================================================================== encoding
def mask_to_rle(mask: np.ndarray) -> dict:
    """COCO uncompressed RLE (column-major counts, starting with zeros)."""
    flat = np.asarray(mask, bool).T.reshape(-1)
    changes = np.flatnonzero(np.diff(np.concatenate([[0], flat.view(np.uint8), [0]])))
    runs = np.diff(np.concatenate([[0], changes, [flat.size]])).tolist()
    if runs and runs[-1] == 0:
        runs = runs[:-1]
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": runs}


PALETTE = np.asarray([[0, 0, 0], [230, 80, 70], [80, 200, 120], [90, 130, 240],
                      [240, 200, 60], [200, 90, 220], [70, 210, 220], [250, 150, 60],
                      [150, 150, 150], [120, 60, 30]], np.uint8)


def overlay(image: np.ndarray, pred: dict | None = None, true: dict | None = None,
            names: list | None = None, mask: np.ndarray | None = None) -> "Image.Image":
    """Ground truth (green) and predictions (red) drawn on an image; masks are blended."""
    canvas = image.copy()
    if mask is not None:
        color = PALETTE[np.asarray(mask) % len(PALETTE)]
        hit = np.asarray(mask) > 0
        canvas[hit] = (0.45 * color[hit] + 0.55 * canvas[hit]).astype(np.uint8)
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    names = names or []

    def label(c) -> str:
        c = int(c)
        return names[c] if 0 <= c < len(names) else str(c)

    for src, color in ((true, (80, 220, 120)), (pred, (240, 80, 70))):
        if not src:
            continue
        scores = src.get("scores")
        for k, box in enumerate(np.asarray(src.get("boxes", [])).reshape(-1, 4)):
            draw.rectangle([float(v) for v in box], outline=color, width=2)
            text = label(src["labels"][k])
            if scores is not None:
                text += f" {float(scores[k]):.2f}"
            draw.text((float(box[0]) + 2, float(box[1]) + 1), text, fill=color)
        for pts in src.get("keypoints", []) if src.get("keypoints") is not None else []:
            pts = np.asarray(pts)
            for row in pts.reshape(-1, pts.shape[-1]):  # (x, y) or (x, y, visibility)
                x, y = float(row[0]), float(row[1])
                if len(row) < 3 or row[2] > 0:
                    draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=color)
    return img


def decode_image(item) -> "Image.Image":
    """Raw bytes, base64 / data-URL strings or nested HxWxC lists -> RGB image."""
    import base64
    import io

    if isinstance(item, (bytes, bytearray)):
        return Image.open(io.BytesIO(item)).convert("RGB")
    if isinstance(item, str):
        data = item.split(",", 1)[1] if item.startswith("data:") else item
        return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
    arr = np.asarray(item)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, -1)
    if arr.dtype != np.uint8:
        arr = (arr * 255 if arr.max() <= 1.0 else arr).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")
'''


def namespace() -> dict:
    """Execute the runtime code in a fresh namespace (numpy + PIL only)."""
    import json
    from pathlib import Path

    import numpy as np
    from PIL import Image, ImageDraw

    ns: dict = {"np": np, "json": json, "Path": Path, "Image": Image, "ImageDraw": ImageDraw}
    for code in (DATA_CODE, METRICS_CODE, ENCODE_CODE):
        exec(compile(code, "<vision-runtime>", "exec"), ns)  # noqa: S102 — our own source
    return ns
