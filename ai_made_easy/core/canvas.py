"""Canvas conveniences shared by the desktop and web UIs (3.3): copy / paste of blocks,
sticky notes and groups.

A **clipboard fragment** is plain design JSON (``{"nodes": [...], "edges": [...]}``) with
a marker, so a copied selection can be pasted into another design, another window or the
other UI, and a whole saved design can be pasted too. Pasting renames ids that clash and
keeps only the wires between pasted blocks.

**Notes** and **groups** live in the design's ``meta`` (``meta.notes``,
``meta.groups``); code generation ignores them:

- note: ``{"id", "text", "position": [x, y], "size": [w, h]}``
- group: ``{"id", "title", "nodes": [block ids], "color"}``

Pure Python, Qt-free. ``web/src/canvas.ts`` mirrors these functions.
"""
from __future__ import annotations

import copy
import json
import re
from typing import Any

MARKER = "aime_clipboard"
PASTE_OFFSET = (40.0, 40.0)
GROUP_COLORS = ("#91A7FF", "#63E6BE", "#FFC078", "#F783AC", "#B197FC", "#99E9F2")


def _ends(edge: dict) -> tuple[str, str, str, str]:
    src, src_port = edge["from"].rsplit("/", 1)
    dst, dst_port = edge["to"].rsplit("/", 1)
    return src, src_port, dst, dst_port


def copy_fragment(design: dict, ids: list[str] | set[str]) -> dict:
    """The selected blocks, their parameters and the wires between them."""
    chosen = set(ids)
    nodes = [copy.deepcopy(n) for n in design.get("nodes", []) if n["id"] in chosen]
    edges = [dict(e) for e in design.get("edges", [])
             if _ends(e)[0] in chosen and _ends(e)[2] in chosen]
    groups = [copy.deepcopy(g) for g in (design.get("meta") or {}).get("groups", [])
              if g.get("nodes") and set(g["nodes"]) <= chosen]
    return {MARKER: 1, "nodes": nodes, "edges": edges, "groups": groups}


def to_text(fragment: dict) -> str:
    return json.dumps(fragment, indent=1)


def parse(text: str) -> dict | None:
    """A clipboard fragment or a whole design from clipboard text, else None."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("nodes"), list):
        return None
    if not all(isinstance(n, dict) and "id" in n and "type" in n for n in data["nodes"]):
        return None
    edges = data.get("edges") or []
    if not all(isinstance(e, dict) and "/" in str(e.get("from", "")) and "/" in
               str(e.get("to", "")) for e in edges):
        return None
    groups = data.get("groups", (data.get("meta") or {}).get("groups", []))
    return {MARKER: 1, "nodes": data["nodes"], "edges": edges,
            "groups": groups if isinstance(groups, list) else []}


def _fresh(base: str, taken: set[str]) -> str:
    stem = re.sub(r"_\d+$", "", base) or "block"
    if base not in taken:
        return base
    i = 2
    while f"{stem}_{i}" in taken:
        i += 1
    return f"{stem}_{i}"


def paste_fragment(design: dict, fragment: dict,
                   offset: tuple[float, float] = PASTE_OFFSET,
                   at: tuple[float, float] | None = None) -> tuple[dict, list[str]]:
    """``design`` with ``fragment`` added; returns (new design, the pasted ids).

    Ids that already exist get a ``_2``, ``_3``, ... suffix. Blocks move by ``offset``,
    or so the fragment's top-left corner lands on ``at``.
    """
    out = copy.deepcopy(design)
    out.setdefault("nodes", [])
    out.setdefault("edges", [])
    taken = {n["id"] for n in out["nodes"]}
    positions = [n.get("position") or [0.0, 0.0] for n in fragment["nodes"]]
    if at is not None and positions:
        dx = at[0] - min(p[0] for p in positions)
        dy = at[1] - min(p[1] for p in positions)
    else:
        dx, dy = offset
    renamed: dict[str, str] = {}
    for node in fragment["nodes"]:
        new_id = _fresh(node["id"], taken)
        taken.add(new_id)
        renamed[node["id"]] = new_id
        pos = node.get("position") or [0.0, 0.0]
        out["nodes"].append({"id": new_id, "type": node["type"],
                             "params": copy.deepcopy(node.get("params") or {}),
                             "position": [float(pos[0]) + dx, float(pos[1]) + dy]})
    for edge in fragment.get("edges", []):
        src, src_port, dst, dst_port = _ends(edge)
        if src in renamed and dst in renamed:
            out["edges"].append({"from": f"{renamed[src]}/{src_port}",
                                 "to": f"{renamed[dst]}/{dst_port}"})
    meta = out.setdefault("meta", {})
    for group in fragment.get("groups", []):
        members = [renamed[n] for n in group.get("nodes", []) if n in renamed]
        if members:
            add_group(meta, members, group.get("title", "Group"), group.get("color"))
    return out, list(renamed.values())


def duplicate(design: dict, ids: list[str] | set[str]) -> tuple[dict, list[str]]:
    return paste_fragment(design, copy_fragment(design, ids))


# ------------------------------------------------------------------ notes and groups

def _next_id(items: list[dict], prefix: str) -> str:
    taken = {i.get("id") for i in items}
    n = 1
    while f"{prefix}{n}" in taken:
        n += 1
    return f"{prefix}{n}"


def add_note(meta: dict, text: str, position: tuple[float, float],
             size: tuple[float, float] = (220.0, 120.0)) -> str:
    notes = meta.setdefault("notes", [])
    nid = _next_id(notes, "note")
    notes.append({"id": nid, "text": text, "position": [float(position[0]), float(position[1])],
                  "size": [float(size[0]), float(size[1])]})
    return nid


def add_group(meta: dict, node_ids: list[str], title: str = "Group",
              color: str | None = None) -> str:
    groups = meta.setdefault("groups", [])
    gid = _next_id(groups, "group")
    groups.append({"id": gid, "title": title or "Group", "nodes": list(dict.fromkeys(node_ids)),
                   "color": color or GROUP_COLORS[len(groups) % len(GROUP_COLORS)]})
    return gid


def clean(meta: dict | None, node_ids: set[str] | list[str]) -> dict:
    """Notes and groups that are well formed; groups lose deleted blocks and empty
    groups disappear. Returns a new meta."""
    out = dict(meta or {})
    alive = set(node_ids)
    notes = [n for n in out.get("notes", []) if isinstance(n, dict) and "position" in n]
    groups = []
    for g in out.get("groups", []):
        if not isinstance(g, dict):
            continue
        members = [n for n in g.get("nodes", []) if n in alive]
        if members:
            groups.append({**g, "nodes": members})
    for key, value in (("notes", notes), ("groups", groups)):
        if value:
            out[key] = value
        else:
            out.pop(key, None)
    return out


def group_bounds(group: dict, positions: dict[str, tuple[float, float]],
                 node_size: tuple[float, float] = (230.0, 60.0),
                 pad: float = 24.0) -> tuple[float, float, float, float] | None:
    """(x, y, width, height) of a frame around a group's blocks."""
    pts = [positions[n] for n in group.get("nodes", []) if n in positions]
    if not pts:
        return None
    x0 = min(p[0] for p in pts) - pad
    y0 = min(p[1] for p in pts) - pad - 22.0      # room for the title
    x1 = max(p[0] for p in pts) + node_size[0] + pad
    y1 = max(p[1] for p in pts) + node_size[1] + pad
    return x0, y0, x1 - x0, y1 - y0


def strip(design: dict) -> dict:
    """The design without notes and groups (what code generation sees)."""
    out = copy.deepcopy(design)
    meta: dict[str, Any] = out.get("meta") or {}
    meta.pop("notes", None)
    meta.pop("groups", None)
    return out
