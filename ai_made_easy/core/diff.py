"""What changed between two designs (3.3): added and removed blocks, changed
parameters and rewired connections. Used to compare a run's design with the canvas.

Blocks are matched by id, so a design restored from a run and then edited compares
block by block. Positions, notes and groups are ignored. Pure Python.
"""
from __future__ import annotations

from typing import Any


def _nodes(design: dict) -> dict[str, dict]:
    return {n["id"]: n for n in design.get("nodes", [])}


def _edges(design: dict) -> set[tuple[str, str]]:
    return {(e["from"], e["to"]) for e in design.get("edges", [])}


def diff(before: dict, after: dict) -> dict[str, Any]:
    """``after`` compared with ``before``."""
    a, b = _nodes(before), _nodes(after)
    added = [{"id": i, "type": b[i]["type"]} for i in b if i not in a]
    removed = [{"id": i, "type": a[i]["type"]} for i in a if i not in b]
    changed, retyped = [], []
    for i in a.keys() & b.keys():
        if a[i]["type"] != b[i]["type"]:
            retyped.append({"id": i, "from": a[i]["type"], "to": b[i]["type"]})
            continue
        pa, pb = a[i].get("params") or {}, b[i].get("params") or {}
        for key in sorted(pa.keys() | pb.keys()):
            if pa.get(key) != pb.get(key):
                changed.append({"id": i, "type": b[i]["type"], "param": key,
                                "from": pa.get(key), "to": pb.get(key)})
    ea, eb = _edges(before), _edges(after)
    wired = [{"from": f, "to": t} for f, t in sorted(eb - ea)]
    unwired = [{"from": f, "to": t} for f, t in sorted(ea - eb)]
    changed.sort(key=lambda c: (c["id"], c["param"]))
    out = {"added": sorted(added, key=lambda x: x["id"]),
           "removed": sorted(removed, key=lambda x: x["id"]),
           "retyped": sorted(retyped, key=lambda x: x["id"]),
           "changed": changed, "wired": wired, "unwired": unwired}
    out["same"] = not any(out.values())
    out["summary"] = summary(out)
    return out


def summary(d: dict) -> str:
    if d.get("same"):
        return "The designs are the same."
    parts = []
    for key, word in (("added", "added"), ("removed", "removed"), ("retyped", "replaced"),
                      ("changed", "parameter change"), ("wired", "new wire"),
                      ("unwired", "removed wire")):
        n = len(d.get(key, []))
        if n:
            plural = "s" if n != 1 and word.endswith(("change", "wire")) else ""
            noun = f"{n} block{'s' if n != 1 else ''} {word}" if key in (
                "added", "removed", "retyped") else f"{n} {word}{plural}"
            parts.append(noun)
    return ", ".join(parts) + "."
