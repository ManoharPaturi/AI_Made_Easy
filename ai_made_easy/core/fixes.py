"""One-click fixes: turn known validation issues into graph edits.

Pure functions over the IR (Qt-free, unit-tested). ``fix_for_issue`` mirrors
core/suggestions.py's pattern table: each recognised issue maps to a
transform that returns a NEW Graph — the UI reloads it onto the canvas.
"""
from __future__ import annotations

import copy
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ai_made_easy.core.graph import Graph, ValidationIssue

import uuid


def _fresh_id(base: str, graph: "Graph") -> str:
    while base in graph.nodes:
        base = f"{base}_{uuid.uuid4().hex[:3]}"
    return base


def _clamp_params(graph: "Graph", node_id: str) -> "Graph | None":
    node = graph.nodes[node_id]
    resolved = node.resolved_params()
    changed = False
    for spec in node.definition().params:
        if spec.type not in ("int", "float"):
            continue
        v = resolved.get(spec.name)
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if spec.minimum is not None and v < float(spec.minimum):
            node.params[spec.name] = spec.minimum
            changed = True
        elif spec.maximum is not None and v > float(spec.maximum):
            node.params[spec.name] = spec.maximum
            changed = True
    return graph if changed else None


def fix_for_issue(graph: "Graph", issue: "ValidationIssue"):
    """Return (button_label, description, fixed_graph) or None."""
    msg, nid = issue.message, issue.node_id
    g = copy.deepcopy(graph)
    node = g.nodes.get(nid) if nid else None

    # 💡 rank mismatch on a Dense block -> insert Flatten between
    if node is not None and "Flatten" in msg and node.type_id == "core.dense":
        in_edge = next((e for e in g.edges
                        if e.target_id == nid), None)
        if in_edge is not None:
            src = g.nodes[in_edge.source_id]
            flat_id = _fresh_id("flatten", g)
            from ai_made_easy.core.graph import Edge, NodeInstance
            mid_x = (src.position[0] + node.position[0]) / 2
            mid_y = (src.position[1] + node.position[1]) / 2
            g.add_node(NodeInstance(flat_id, "core.flatten",
                                    {}, (mid_x, mid_y)))
            g.edges.remove(in_edge)
            g.add_edge(Edge(in_edge.source_id, in_edge.source_port,
                            flat_id, "in"))
            g.add_edge(Edge(flat_id, "out", nid, in_edge.target_port))
            return ("＋ Flatten", "insert a Flatten block between "
                    f"{src.definition().display_name} and Dense", g)

    # out-of-range parameter -> clamp to the nearest bound
    if node is not None and re.search(r"too (small|big)", msg):
        fixed = _clamp_params(g, nid)
        if fixed is not None:
            return ("Clamp", "move the out-of-range value back inside "
                    "its allowed range", fixed)

    # MultiheadAttention width mismatch -> switch to auto
    if node is not None and node.type_id == "core.multihead_attention" and (
            "divisible" in msg or ("channels" in msg and "embed" in msg)):
        node.params["embed_dim"] = 0
        return ("Use auto", "set embed_dim = 0 so it matches the input "
                "automatically", g)

    # Text Splitter overlap >= chunk -> halve it
    if node is not None and node.type_id == "llm.text_splitter" \
            and "chunk_overlap" in msg:
        chunk = node.resolved_params().get("chunk_size", 500)
        node.params["chunk_overlap"] = max(chunk // 4, 0)
        return ("Halve", "set chunk_overlap to a safe fraction of "
                "chunk_size", g)

    # dataset vs Input shape mismatch -> reshape the Input to the dataset
    m = re.search(r"has (\d+) features but the Input block", msg)
    if m:
        inputs = [n for n in g.nodes.values() if n.type_id == "core.input"]
        if inputs:
            inputs[0].params["shape"] = m.group(1)
            return ("Match", "set the Input block's shape to the dataset's "
                    f"{m.group(1)} features", g)

    # disconnected input with an obvious nearest predecessor -> wire it
    if node is not None and "is not connected" in msg:
        candidates = [n for n in g.nodes.values()
                      if n.instance_id != nid
                      and n.definition().outputs
                      and n.type_id not in ("core.output",)]
        nearby = [n for n in candidates
                  if abs(n.position[1] - node.position[1]) < 120
                  and n.position[0] < node.position[0] + 80]
        if nearby:
            nearest = max(nearby, key=lambda n: n.position[0])
            from ai_made_easy.core.graph import Edge
            defn = node.definition()
            port = next((p for p in defn.inputs if p.dtype == "tensor"), None)
            out = nearest.definition().outputs[0]
            g.add_edge(Edge(nearest.instance_id, out.name,
                            nid, port.name if port else "in"))
            return ("Wire it", f"connect {nearest.definition().display_name} "
                    "into this block", g)

    # off-path block -> offer deletion
    if issue.severity == "warning" and "not connected to your model" in msg \
            and node is not None:
        g.edges = [e for e in g.edges
                   if e.source_id != nid and e.target_id != nid]
        del g.nodes[nid]
        return ("Remove", "delete the disconnected block from the canvas", g)

    return None
