"""Split preview: how many samples of each class land in train / validation / test.

Reproduces the generated PyTorch scripts' splitting exactly (same seed, same
stratification rule, same label order), so the preview is what training will
see before any row filtering (e.g. imputation by dropping rows).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ai_made_easy.core.data.profile import (
    FOLDER_SUFFIXES,
    profile_folder,
    read_table,
    resolve_path,
)

SPLITS = ("train", "val", "test")


@dataclass
class SplitPreview:
    totals: dict[str, int] = field(default_factory=dict)
    per_class: dict[str, dict[str, int]] = field(default_factory=dict)  # class -> split -> n
    method: str = ""
    notes: list[str] = field(default_factory=list)

    def missing_classes(self) -> dict[str, list[str]]:
        """split -> classes with no samples in that split."""
        out: dict[str, list[str]] = {}
        for split in SPLITS:
            if not self.totals.get(split):
                continue
            empty = [c for c, counts in self.per_class.items() if not counts.get(split)]
            if empty:
                out[split] = empty
        return out

    def to_dict(self) -> dict:
        return {"totals": self.totals, "per_class": self.per_class, "method": self.method,
                "notes": self.notes, "missing_classes": self.missing_classes()}


def split_indices(n: int, y, *, val: float, test: float, seed: int, shuffle: bool = True,
                  stratify: bool = False, chronological: bool = False):
    """Same algorithm as ``split_indices`` in the generated data section."""
    import numpy as np

    n_val, n_test = int(n * val), int(n * test)
    if chronological:
        idx = np.arange(n)
        return idx[: n - n_val - n_test], idx[n - n_val - n_test: n - n_test], idx[n - n_test:]
    rng = np.random.default_rng(seed)
    if stratify and y is not None and np.ndim(y) == 1 and len(np.unique(y)) < max(n // 5, 2):
        train, val_i, test_i = [], [], []
        for c in np.unique(y):
            members = np.flatnonzero(y == c)
            rng.shuffle(members)
            cv, ct = int(round(len(members) * val)), int(round(len(members) * test))
            val_i += list(members[:cv])
            test_i += list(members[cv:cv + ct])
            train += list(members[cv + ct:])
        return (rng.permutation(train).astype(int), rng.permutation(val_i).astype(int),
                rng.permutation(test_i).astype(int))
    idx = rng.permutation(n) if shuffle else np.arange(n)
    return idx[n_val + n_test:], idx[:n_val], idx[n_val:n_val + n_test]


def image_split(n: int, *, val: float, test: float, seed: int):
    """The image pipeline's split: one permutation, validation first."""
    import numpy as np

    idx = np.random.default_rng(seed).permutation(n)
    n_val, n_test = int(n * val), int(n * test)
    return idx[n_val + n_test:], idx[:n_val], idx[n_val:n_val + n_test]


def dataset_labels(type_id: str, params: dict, base=None, classification: bool = True):
    """(labels array in the script's sample order, class names) or (None, reason)."""
    import numpy as np

    if type_id in FOLDER_SUFFIXES:
        profile = profile_folder(type_id, params.get("root", ""), params, base)
        if profile.error:
            return None, profile.error
        names = [c.name for c in profile.classes if c.count]
        y = np.concatenate([np.full(c.count, i) for i, c in enumerate(
            c for c in profile.classes if c.count)]) if names else np.zeros(0, int)
        return y, names
    if type_id in ("data.csv", "data.text_csv"):
        path = resolve_path(params.get("path", ""), base)
        column = params.get("target_column" if type_id == "data.csv" else "label_column")
        try:
            frame = read_table(path, str(params.get("format") or "csv"))
        except Exception as exc:  # noqa: BLE001
            return None, f"could not read {Path(path).name}: {exc}"
        if column not in frame.columns:
            return None, f"column '{column}' is not in the table"
        y = frame[column]
        if type_id == "data.text_csv":
            frame = frame.dropna(subset=[c for c in (params.get("text_column"), column)
                                         if c in frame.columns])
            y = frame[column]
        if not classification:
            return np.zeros(len(y)), []
        classes = sorted(y.astype(str).unique())
        return y.astype(str).map({c: i for i, c in enumerate(classes)}).to_numpy(), classes
    if type_id == "data.numpy":
        path = resolve_path(params.get("path", ""), base)
        try:
            with np.load(path) as archive:
                y = archive[str(params.get("y_key", "y"))]
        except Exception as exc:  # noqa: BLE001
            return None, str(exc)
        if not classification or y.ndim != 1:
            return np.zeros(len(y)), []
        classes = [str(c) for c in np.unique(y)]
        return np.searchsorted(np.unique(y), y), classes
    return None, "split preview needs a local dataset (file or folder)"


def split_preview(type_id: str, params: dict, split: dict, *, base=None,
                  task: str = "multiclass", modality: str = "") -> SplitPreview | str:
    """Per-class split counts, or a reason string when the data cannot be read."""
    import numpy as np

    classification = task in ("multiclass", "binary", "multilabel", "classification")
    y, classes = dataset_labels(type_id, params, base, classification)
    if y is None:
        return str(classes)
    val, test = float(split.get("val_fraction", 0.1)), float(split.get("test_fraction", 0.1))
    seed = int(split.get("seed", 42))
    preview = SplitPreview()
    if type_id == "data.image_folder" or modality == "image":
        parts = image_split(len(y), val=val, test=test, seed=seed)
        preview.method = "random permutation (image pipeline)"
        if split.get("stratify"):
            preview.notes.append("image folders are split by one random permutation; "
                                 "stratify applies to tables and arrays")
    else:
        chrono = modality == "timeseries" or type_id == "data.timeseries_csv"
        stratify = bool(split.get("stratify")) and task in ("multiclass", "binary",
                                                             "classification")
        parts = split_indices(len(y), y, val=val, test=test, seed=seed,
                              shuffle=bool(split.get("shuffle", True)), stratify=stratify,
                              chronological=chrono)
        preview.method = ("chronological" if chrono else
                          "stratified" if stratify and len(np.unique(y)) < max(len(y) // 5, 2)
                          else "random" if split.get("shuffle", True) else "in file order")
    for name, idx in zip(SPLITS, parts, strict=True):
        preview.totals[name] = int(len(idx))
    if classification and classes:
        for i, cls in enumerate(classes):
            preview.per_class[cls] = {name: int((y[idx] == i).sum())
                                      for name, idx in zip(SPLITS, parts, strict=True)}
    for split_name, empty in preview.missing_classes().items():
        preview.notes.append(f"{len(empty)} class(es) have no {split_name} samples "
                             f"(e.g. '{empty[0]}')")
    return preview
