"""Project-file migrations: upgrade older graph JSON to the current schema.

Each step rewrites a project dict from schema N to N + 1, preserving the
behaviour the design had when it was saved.
"""
from __future__ import annotations

import copy

CURRENT_SCHEMA = 2


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


_STEPS = {1: _v1_to_v2}


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
