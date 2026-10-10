"""What a dataset is, from a path: its format, size, target, classes and the tasks it fits.

:func:`detect` looks at a file or folder and returns :class:`DataFacts`. Recipes read the
facts to build a design that fits the data (input width, classes, image size, the dataset
block and its columns); the wizard shows them to the user and preselects a task. With no
path the facts are ``demo``: every recipe then uses its own demo dataset.

Pure Python (pandas for tables, Pillow when it is installed for image sizes).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ai_made_easy.core.data.profile import (
    AUDIO_SUFFIXES, IMAGE_SUFFIXES, MAX_CATEGORIES, TEXT_SUFFIXES, guess_format, profile_frame,
    read_table)

# data kinds a recipe can declare it reads
KINDS = ("demo", "table", "text_table", "image_folder", "text_folder", "audio_folder", "coco",
         "yolo", "voc", "mask_folder", "series", "interactions", "graph", "corpus", "pairs")
TABLE_SUFFIXES = {".csv", ".tsv", ".parquet", ".pq", ".xlsx", ".xls", ".json", ".jsonl"}
SAMPLE_ROWS = 20_000
TARGET_NAMES = ("target", "label", "labels", "class", "y", "outcome", "species", "category",
                "price", "value")
USER_NAMES = ("user", "user_id", "userid", "customer", "customer_id")
ITEM_NAMES = ("item", "item_id", "itemid", "product", "product_id", "movie", "movie_id")
TIME_NAMES = ("date", "time", "timestamp", "datetime", "ds", "month", "day")


@dataclass
class DataFacts:
    kind: str = "demo"
    source: str = ""                     # resolved path ("" for demo data)
    format: str = ""                     # table format (csv, tsv, parquet, ...)
    rows: int = 0                        # rows, images, clips, documents, interactions
    target: str = ""                     # target column (tables)
    features: list[str] = field(default_factory=list)
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    cardinalities: list[int] = field(default_factory=list)   # categories per text column
    text_column: str = ""
    time_column: str = ""
    columns: dict[str, str] = field(default_factory=dict)    # role -> column (user, item, ...)
    classes: list[str] = field(default_factory=list)
    channels: int = 0
    image_size: int = 0                  # typical (median) side length
    missing: bool = False
    tasks: list[str] = field(default_factory=list)           # tasks the data fits, best first
    paths: dict[str, str] = field(default_factory=dict)      # images_dir, labels_dir, ...
    details: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def demo(self) -> bool:
        return self.kind == "demo"

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def task(self) -> str:
        return self.tasks[0] if self.tasks else ""

    def summary(self) -> str:
        if self.demo:
            return "no data yet: each recipe trains on its demo dataset"
        if self.error:
            return self.error
        parts = [self.kind.replace("_", " "), f"{self.rows:,} rows" if self.rows else ""]
        if self.features:
            parts.append(f"{len(self.features)} features")
        if self.classes:
            parts.append(f"{len(self.classes)} classes")
        if self.image_size:
            parts.append(f"~{self.image_size}px, {self.channels} channel(s)")
        return " · ".join(p for p in parts if p)

    def to_dict(self) -> dict:
        return {**asdict(self), "summary": self.summary(), "n_classes": self.n_classes}

    @classmethod
    def from_dict(cls, data: dict | None) -> DataFacts:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in (data or {}).items() if k in known})


def demo_facts() -> DataFacts:
    return DataFacts()


# ===================================================================== detection

def detect(path: str | Path | None, *, target: str = "", task: str = "") -> DataFacts:
    """Facts about the data at ``path`` (a table file or a dataset folder).

    ``target`` names the target column of a table; ``task`` is a hint that settles the
    ambiguous cases (a table with a time column read as forecasting, two text columns
    read as sequence-to-sequence pairs)."""
    if not path:
        return demo_facts()
    p = Path(str(path)).expanduser()
    if not p.exists():
        return DataFacts(kind="table", source=str(p), error=f"{p} does not exist")
    try:
        return _folder(p) if p.is_dir() else _file(p, target, task)
    except Exception as exc:  # noqa: BLE001 — unreadable data is reported, not raised
        return DataFacts(kind="table", source=str(p), error=f"could not read {p.name}: {exc}")


def _files(root: Path, suffixes: set[str], limit: int = 2000) -> list[Path]:
    out = []
    for f in root.rglob("*"):
        if f.is_file() and f.suffix.lower() in suffixes:
            out.append(f)
            if len(out) >= limit:
                break
    return out


def _image_shape(files: list[Path]) -> tuple[int, int]:
    """(median side, channels) of a few images; (0, 0) without Pillow."""
    try:
        from PIL import Image
    except ImportError:
        return 0, 0
    sides, channels = [], []
    for f in files[:40]:
        try:
            with Image.open(f) as im:
                sides.append(min(im.size))
                channels.append(1 if im.mode in ("L", "1", "I", "F", "I;16") else 3)
        except Exception:  # noqa: BLE001, S112 — skip unreadable files
            continue
    if not sides:
        return 0, 0
    sides.sort()
    return sides[len(sides) // 2], max(set(channels), key=channels.count)


def _coco_file(root: Path) -> Path | None:
    for f in sorted(root.glob("*.json")) + sorted(root.glob("annotations/*.json")):
        try:
            if f.stat().st_size > 512 * 2**20:
                continue
            data = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and "images" in data and "annotations" in data:
            return f
    return None


def _first_dir(root: Path, *names: str) -> Path | None:
    return next((root / n for n in names if (root / n).is_dir()), None)


def _folder(root: Path) -> DataFacts:
    coco = _coco_file(root)
    if coco is not None:
        data = json.loads(coco.read_text())
        cats = data.get("categories") or []
        images = _first_dir(root, "images", "train", "img") or root
        keypoints = any(c.get("keypoints") for c in cats)
        masks = any(a.get("segmentation") for a in data.get("annotations", [])[:200])
        tasks = ["detection"] + (["instance_segmentation"] if masks else []) \
            + (["keypoints"] if keypoints else [])
        if keypoints:
            tasks = ["keypoints", *[t for t in tasks if t != "keypoints"]]
        side, ch = _image_shape(_files(images, IMAGE_SUFFIXES, 40))
        return DataFacts(kind="coco", source=str(root), rows=len(data.get("images", [])),
                         classes=[str(c.get("name", c.get("id"))) for c in cats],
                         image_size=side, channels=ch or 3, tasks=tasks,
                         paths={"images_dir": str(images), "annotations": str(coco)},
                         details=[f"COCO annotations {coco.name}: "
                                  f"{len(data.get('annotations', [])):,} objects"])
    labels = _first_dir(root, "labels")
    images = _first_dir(root, "images")
    if labels is not None and images is not None and _files(labels, {".txt"}, 1):
        names = root / "classes.txt"
        classes = [c.strip() for c in names.read_text().splitlines() if c.strip()] \
            if names.is_file() else []
        imgs = _files(images, IMAGE_SUFFIXES)
        side, ch = _image_shape(imgs)
        return DataFacts(kind="yolo", source=str(root), rows=len(imgs), classes=classes,
                         image_size=side, channels=ch or 3,
                         tasks=["detection", "instance_segmentation", "keypoints"],
                         paths={"images_dir": str(images), "labels_dir": str(labels)},
                         details=["YOLO txt labels"],
                         warnings=[] if classes else ["no classes.txt: classes are numbered"])
    voc = next((d for d in [root, *root.glob("*"), *root.glob("*/*")]
                if d.is_dir() and (d / "Annotations").is_dir() and (d / "JPEGImages").is_dir()),
               None)
    if voc is not None:
        imgs = _files(voc / "JPEGImages", IMAGE_SUFFIXES)
        side, ch = _image_shape(imgs)
        return DataFacts(kind="voc", source=str(voc), rows=len(imgs), image_size=side,
                         channels=ch or 3, tasks=["detection"], paths={"root": str(voc)},
                         details=["Pascal VOC XML annotations"])
    masks = _first_dir(root, "masks", "labels", "annotations")
    if images is not None and masks is not None:
        imgs = _files(images, IMAGE_SUFFIXES)
        side, ch = _image_shape(imgs)
        n = _mask_classes(_files(masks, IMAGE_SUFFIXES, 20))
        return DataFacts(kind="mask_folder", source=str(root), rows=len(imgs),
                         classes=[str(i) for i in range(n)] if n else [], image_size=side,
                         channels=ch or 3, tasks=["semantic_segmentation"],
                         paths={"images_dir": str(images), "masks_dir": str(masks)},
                         details=["images with per-pixel class masks"])
    return _class_folders(root)


def _mask_classes(files: list[Path]) -> int:
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return 0
    seen: set[int] = set()
    for f in files:
        with Image.open(f) as im:
            seen |= {int(v) for v in np.unique(np.asarray(im)) if int(v) != 255}
    return max(seen) + 1 if seen else 0


def _class_folders(root: Path) -> DataFacts:
    subdirs = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
    if not subdirs:
        if _files(root, TEXT_SUFFIXES, 1):
            n = len(_files(root, TEXT_SUFFIXES))
            return DataFacts(kind="corpus", source=str(root), rows=n,
                             tasks=["language_modeling"], details=[f"{n} text files"])
        return DataFacts(kind="image_folder", source=str(root),
                         error="expected one sub-folder per class (or COCO / YOLO / VOC / "
                               "images + masks)")
    counts: dict[str, dict[str, int]] = {}
    for d in subdirs:
        for f in d.iterdir():
            if f.is_file():
                suffix = f.suffix.lower()
                kind = ("image" if suffix in IMAGE_SUFFIXES else "audio" if suffix in
                        AUDIO_SUFFIXES else "text" if suffix in TEXT_SUFFIXES else "")
                if kind:
                    counts.setdefault(kind, {}).setdefault(d.name, 0)
                    counts[kind][d.name] += 1
    if not counts:
        return DataFacts(kind="image_folder", source=str(root),
                         error="no images, audio or text files in the class folders")
    modality = max(counts, key=lambda k: sum(counts[k].values()))
    per_class = counts[modality]
    classes = sorted(per_class)
    facts = DataFacts(kind=f"{modality}_folder", source=str(root),
                      rows=sum(per_class.values()), classes=classes)
    facts.details.append("per class: " + ", ".join(f"{c} {per_class[c]:,}"
                                                   for c in classes[:12]))
    if len(classes) < 2:
        facts.warnings.append("only one class folder: add a folder per class")
    small = min(per_class.values())
    if small < 10:
        facts.warnings.append(f"the smallest class has only {small} example(s)")
    if modality == "image":
        files = [f for d in subdirs for f in list(d.iterdir())[:10] if f.suffix.lower()
                 in IMAGE_SUFFIXES]
        facts.image_size, facts.channels = _image_shape(files)
        facts.channels = facts.channels or 3
    facts.tasks = ["keyword_spotting", "multiclass"] if modality == "audio" else \
        ["binary" if len(classes) == 2 else "multiclass", "multiclass"]
    facts.tasks = list(dict.fromkeys(facts.tasks))
    return facts


def _pick(columns: list[str], names: tuple[str, ...]) -> str:
    low = {c.lower(): c for c in columns}
    return next((low[n] for n in names if n in low), "")


def _file(path: Path, target: str, task: str) -> DataFacts:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES and suffix not in TABLE_SUFFIXES:
        size = path.stat().st_size
        return DataFacts(kind="corpus", source=str(path), rows=size,
                         tasks=["language_modeling"], details=[f"{size:,} characters of text"])
    if suffix not in TABLE_SUFFIXES:
        return DataFacts(kind="table", source=str(path),
                         error=f"unsupported file type {suffix or '(none)'}: use a table "
                               "(.csv, .tsv, .parquet, .xlsx, .json) or a folder")
    fmt = guess_format(path)
    frame = read_table(path, fmt, nrows=SAMPLE_ROWS)
    columns = [str(c) for c in frame.columns]
    facts = DataFacts(kind="table", source=str(path), format=fmt, rows=len(frame))
    if len(frame) == SAMPLE_ROWS:
        facts.details.append(f"read the first {SAMPLE_ROWS:,} rows")
    user, item = _pick(columns, USER_NAMES), _pick(columns, ITEM_NAMES)
    if user and item and task in ("", "recommendation"):
        rating = _pick(columns, ("rating", "ratings", "score", "stars"))
        facts.kind, facts.tasks = "interactions", ["recommendation"]
        facts.columns = {"user": user, "item": item, **({"rating": rating} if rating else {})}
        facts.details.append(f"{frame[user].nunique():,} users · {frame[item].nunique():,} items")
        return facts
    source, dest = _pick(columns, ("source", "src", "from")), _pick(columns, ("target", "dst",
                                                                               "to"))
    nodes = next((f for f in sorted(path.parent.glob("*node*")) if f.suffix.lower()
                  in TABLE_SUFFIXES), None)
    if source and dest and nodes is not None and task in ("", "node_classification",
                                                           "link_prediction"):
        facts.kind, facts.tasks = "graph", ["node_classification", "link_prediction"]
        facts.columns = {"source": source, "target": dest}
        facts.paths = {"edges_path": str(path), "nodes_path": str(nodes)}
        facts.details.append(f"{len(frame):,} edges; nodes in {nodes.name}")
        return facts
    texts = [c for c in columns if _is_text(frame[c])]
    if task == "sequence_to_sequence" or (len(texts) >= 2 and len(columns) == 2):
        facts.kind, facts.tasks = "pairs", ["sequence_to_sequence"]
        src, dst = (texts + columns)[:2] if len(texts) < 2 else texts[:2]
        facts.columns = {"source": src, "target": dst}
        return facts
    time_col = _time_column(frame, columns)
    target = target if target in columns else ""
    if time_col and (task in ("forecasting", "state_space_forecasting") or not target):
        numeric = [c for c in columns if c != time_col and str(frame[c].dtype)[:3] in
                   ("int", "flo")]
        if numeric:
            facts.kind, facts.time_column = "series", time_col
            facts.target = target or _pick(numeric, TARGET_NAMES) or numeric[-1]
            facts.columns = {"id": _pick(columns, ("id", "series", "series_id", "store",
                                                   "item_id"))}
            facts.tasks = ["forecasting", "state_space_forecasting"]
            facts.features = [c for c in numeric if c != facts.target]
            return facts
    if texts and (not target or target not in texts):
        facts.kind, facts.text_column = "text_table", texts[0]
        facts.target = target or _pick([c for c in columns if c not in texts],
                                       TARGET_NAMES) or next(c for c in columns
                                                             if c not in texts)
        classes = sorted(frame[facts.target].astype(str).unique())
        facts.classes = classes
        facts.tasks = ["binary" if len(classes) == 2 else "multiclass"]
        return facts
    facts.target = target or _pick(columns, TARGET_NAMES) or columns[-1]
    profile = profile_frame(frame, target=facts.target)
    facts.features = [c.name for c in profile.columns if c.role == "feature"]
    for col in profile.columns:
        if col.role != "feature":
            continue
        if col.kind in ("numeric", "boolean"):
            facts.numeric.append(col.name)
        elif col.kind in ("categorical", "text"):
            facts.categorical.append(col.name)
            facts.cardinalities.append(min(col.unique, MAX_CATEGORIES))
        if col.missing:
            facts.missing = True
    facts.warnings += [f.message for f in profile.findings if f.severity in ("warning",
                                                                             "error")][:6]
    if profile.task == "classification" or profile.classes:
        facts.classes = [c.name for c in profile.classes]
        facts.tasks = ["binary" if len(facts.classes) == 2 else "multiclass", "multiclass"]
    else:
        facts.tasks = ["regression", "gp_regression", "bayesian_modeling"]
    facts.tasks = list(dict.fromkeys(facts.tasks))
    return facts


def _is_text(series) -> bool:  # noqa: ANN001
    """Free text: string values averaging over 30 characters with many distinct values."""
    import pandas as pd

    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False
    values = series.dropna().astype(str)
    return bool(len(values)) and values.str.len().mean() > 30 \
        and values.nunique() > 0.5 * len(values)


def _time_column(frame, columns: list[str]) -> str:  # noqa: ANN001
    import pandas as pd

    for c in columns:
        if str(frame[c].dtype).startswith("datetime"):
            return c
    named = _pick(columns, TIME_NAMES)
    if named:
        try:
            pd.to_datetime(frame[named].head(50))
            return named
        except (ValueError, TypeError):
            return ""
    return ""
