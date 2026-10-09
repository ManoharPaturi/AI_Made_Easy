"""Data lints: dataset profile findings in the context of the whole design.

``data_issues(graph, cache)`` turns profile findings into Problems-panel
warnings and adds checks that need the pipeline: missing values without an
Impute block, identifier / leakage columns that are not dropped, a regression
loss on categorical targets, classes too small for the validation split.
Findings the pipeline already handles (dropped columns, class balancing) are
suppressed. Profiles are cached by file signature, so repeated validation is
cheap; ``ProfileCache.pending`` lets callers profile in the background.
"""
from __future__ import annotations

import threading
from pathlib import Path

from ai_made_easy.core.data.profile import (
    FOLDER_SUFFIXES,
    DataProfile,
    profile_dataset,
    resolve_path,
)
from ai_made_easy.core.graph import Graph, ValidationIssue

# dataset blocks whose data is on disk and gets profiled (task families add theirs)
LOCAL_BLOCKS: set[str] = {*FOLDER_SUFFIXES, "data.csv", "data.text_csv", "data.timeseries_csv",
                          "data.numpy", "data.json"}
# parameters naming the file / folder whose changes invalidate a cached profile
PATH_KEYS = ("root", "path", "annotations", "labels_dir", "masks_dir", "images_dir")


def _signature(type_id: str, params: dict, base) -> tuple:
    """Changes whenever the data (or the params that shape the profile) change."""
    key = tuple(sorted((k, str(v)) for k, v in params.items()))
    raw = (params.get("root") if type_id in FOLDER_SUFFIXES else
           next((params[k] for k in PATH_KEYS if params.get(k)), None))
    if raw is None:
        return (type_id, key)
    path = resolve_path(str(raw), base)
    try:
        st = path.stat()
    except OSError:
        return (type_id, key, str(path), None)
    stamp: tuple = (st.st_size, st.st_mtime_ns)
    if path.is_dir():  # adding / removing files touches the folder's mtime
        try:
            stamp += tuple(sorted((d.name, d.stat().st_mtime_ns)
                                  for d in path.iterdir() if d.is_dir()))
        except OSError:
            pass
    return (type_id, key, str(path), stamp)


class ProfileCache:
    """Thread-safe cache of dataset profiles keyed by data signature."""

    def __init__(self, limit: int = 32):
        self._items: dict[tuple, DataProfile] = {}
        self._lock = threading.Lock()
        self._limit = limit

    def key(self, type_id: str, params: dict, base=None) -> tuple:
        return _signature(type_id, params, base)

    def get(self, type_id: str, params: dict, base=None) -> DataProfile | None:
        with self._lock:
            return self._items.get(self.key(type_id, params, base))

    def profile(self, type_id: str, params: dict, base=None) -> DataProfile:
        key = self.key(type_id, params, base)
        with self._lock:
            hit = self._items.get(key)
        if hit is not None:
            return hit
        result = profile_dataset(type_id, params, base)
        with self._lock:
            if len(self._items) >= self._limit:
                self._items.pop(next(iter(self._items)))
            self._items[key] = result
        return result

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def pending(self, graph: Graph, base=None) -> list[tuple[str, dict]]:
        """Local datasets in ``graph`` whose profile is not cached yet."""
        return [(t, p) for _nid, t, p in dataset_nodes(graph)
                if t in LOCAL_BLOCKS and self.get(t, p, base) is None]


def dataset_nodes(graph: Graph) -> list[tuple[str, str, dict]]:
    return [(n.instance_id, n.type_id, dict(n.resolved_params()))
            for n in graph.nodes.values() if n.type_id.startswith("data.")]


def _pipeline(graph: Graph) -> dict:

    steps = {n.type_id: dict(n.resolved_params()) for n in graph.nodes.values()
             if n.type_id.startswith("prep.")}
    from ai_made_easy.core.tasks import task_of as _task_of

    task = _task_of(graph)
    return {"steps": steps, "task": task.id if task else ""}


def task_of(graph: Graph) -> str:
    return _pipeline(graph)["task"]


def lint_profile(profile: DataProfile, node_id: str, graph: Graph) -> list[ValidationIssue]:
    pipe = _pipeline(graph)
    steps, task = pipe["steps"], pipe["task"]
    dropped = {c.strip() for c in str((steps.get("prep.drop_columns") or {}).get("columns", ""))
               .split(",") if c.strip()}
    out: list[ValidationIssue] = []
    for f in profile.findings:
        if f.severity == "info":
            continue
        if f.classes and profile.kind == "table" and set(f.classes) <= dropped:
            continue
        if "class imbalance" in f.message and "prep.class_balance" in steps:
            continue
        out.append(ValidationIssue("warning", f.message + (f" ({f.hint})" if f.hint else ""),
                                   node_id))
    if profile.kind == "table" and profile.type_id == "data.csv":
        gaps = [c.name for c in profile.columns if c.role == "feature" and c.kind == "numeric"
                and c.missing and c.name not in dropped]
        if gaps and "prep.impute" not in steps:
            out.append(ValidationIssue(
                "warning", f"{len(gaps)} numeric column(s) have missing values (e.g. "
                           f"'{gaps[0]}') and there is no Impute Missing Values block — NaN "
                           "values reach the model and the loss becomes NaN", node_id))
        target = next((c for c in profile.columns if c.role == "target"), None)
        if target is not None and task:
            if task == "regression" and target.kind in ("categorical", "text", "boolean"):
                out.append(ValidationIssue(
                    "warning", f"target '{target.name}' is categorical but the loss is a "
                               "regression loss — use a classification loss", node_id))
            if task in ("multiclass", "binary") and target.kind == "numeric" \
                    and target.unique > 50 and str(target.dtype).startswith("float"):
                out.append(ValidationIssue(
                    "warning", f"target '{target.name}' looks continuous ({target.unique:,} "
                               "distinct values) but the loss is a classification loss — "
                               "use a regression loss", node_id))
            if task == "binary" and len(profile.classes) > 2:
                out.append(ValidationIssue(
                    "warning", f"binary loss but the target has {len(profile.classes)} "
                               "classes — use Cross-Entropy", node_id))
    split = steps.get("prep.split") or {"val_fraction": 0.1, "test_fraction": 0.1,
                                        "stratify": True}
    val = float(split.get("val_fraction", 0.1))
    if profile.classes and val > 0 and split.get("stratify") and profile.kind != "folder":
        tiny = [c.name for c in profile.classes if 0 < c.count and round(c.count * val) == 0]
        if tiny:
            out.append(ValidationIssue(
                "warning", f"{len(tiny)} class(es) are too small to appear in the validation "
                           f"split (e.g. '{tiny[0]}')", node_id))
    return out


def data_issues(graph: Graph, cache: ProfileCache | None = None, base=None, *,
                compute: bool = True) -> list[ValidationIssue]:
    """Data warnings for every dataset block (``compute=False``: cached profiles only)."""
    cache = cache or ProfileCache()
    out: list[ValidationIssue] = []
    for node_id, type_id, params in dataset_nodes(graph):
        if type_id not in LOCAL_BLOCKS:
            continue
        profile = (cache.profile(type_id, params, base) if compute
                   else cache.get(type_id, params, base))
        if profile is None:
            continue
        if profile.error and not profile.findings:
            out.append(ValidationIssue("warning", profile.error, node_id))
            continue
        out += lint_profile(profile, node_id, graph)
    return out


def project_base(project_path: str | Path | None) -> Path | None:
    """Relative dataset paths resolve against the project file's folder."""
    return Path(project_path).expanduser().parent if project_path else None
