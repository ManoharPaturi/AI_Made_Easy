"""Project-file migrations: upgrade older graph JSON to the current schema.

Each step rewrites a project dict from schema N to N + 1, preserving the
behaviour the design had when it was saved.
"""
from __future__ import annotations

import copy

CURRENT_SCHEMA = 3


def _v1_to_v2(data: dict) -> dict:
    """V1 → V2: LayerNorm defaulted to normalizing every axis; adaptive pooling
    used separate height/width; Input had no dtype."""
    for node in data.get("nodes", []):
        params = node.setdefault("params", {})
        if node.get("type") == "core.layer_norm":
            params.setdefault("normalized_dims", "all")
        if node.get("type") == "core.adaptive_avgpool2d":
            h = params.pop("output_height", None)
            w = params.pop("output_width", None)
            if "output_size" not in params and (h or w):
                params["output_size"] = int(h or w)
        if node.get("type") == "core.input":
            params.setdefault("dtype", "float32")
    return data


def _v2_to_v3(data: dict) -> dict:
    """V2 → V3: nothing to rewrite. Port roles and multi-input ports live in the block
    registry, so every V2 design keeps its meaning; saved V3 files also record the
    design's family and task (see ``stamp``), which readers may use without loading
    the registry and which are recomputed on every save."""
    return data


_STEPS = {1: _v1_to_v2, 2: _v2_to_v3}


def migrate(data: dict) -> dict:
    """Return ``data`` upgraded to CURRENT_SCHEMA (input is not modified)."""
    version = int(data.get("schema_version", 1) or 1)
    if version > CURRENT_SCHEMA:
        raise ValueError(f"this project was saved by a newer version (schema {version}); "
                         "please update AI Made Easy")
    out = copy.deepcopy(data)
    while version < CURRENT_SCHEMA:
        out = _STEPS[version](out)
        version += 1
    out["schema_version"] = CURRENT_SCHEMA
    return out


def stamp(data: dict) -> dict:
    """A copy of a project ready to save: the current schema plus ``family`` and
    ``task`` (ids, or None for a design with no task yet)."""
    from ai_made_easy.core.families import family_of
    from ai_made_easy.core.graph import Graph
    from ai_made_easy.core.tasks import task_of

    out = migrate(data)
    out["family"] = family_of(out).id
    try:
        task = task_of(Graph.from_dict(out))
    except Exception:  # noqa: BLE001 — incomplete designs have no task yet
        task = None
    out["task"] = task.id if task else None
    return out


def kind_of(data: dict) -> dict:
    """``{"family": label, "task": label or None}`` of a project file: read from the V3
    descriptors when present, worked out from the design for older files."""
    from ai_made_easy.core.families import get_family
    from ai_made_easy.core.tasks import get_task

    if "family" not in data or not data.get("task"):
        data = stamp(data)   # older files, or a task this version can now work out
    try:
        family = get_family(data["family"]).label
    except (KeyError, ValueError):
        family = str(data["family"])
    try:
        task = get_task(data["task"]).label if data.get("task") else None
    except (KeyError, ValueError):
        task = str(data["task"])
    return {"family": family, "task": task}
