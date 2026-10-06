"""Fix hints appended to common validation failures.

Pure functions over the IR — ``add_tips`` post-processes validation issues so
every consumer (canvas, CLI, MCP agents) shows how to resolve a problem, not
just that it exists.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ai_made_easy.core.graph import Graph, ValidationIssue

HINT = "Fix:"


def _source_names(graph: "Graph", node_id: str) -> list[str]:
    names = []
    for e in graph.edges:
        if e.target_id == node_id and e.source_id in graph.nodes:
            names.append(graph.nodes[e.source_id].definition().display_name)
    return names


def _tip_for(graph: "Graph", issue: "ValidationIssue") -> str | None:
    msg = issue.message
    nid = issue.node_id
    node = graph.nodes.get(nid) if nid else None
    type_id = node.type_id if node else ""

    if "is not connected" in msg and "input" in msg:
        return "connect an upstream output port to this input."
    if "not reachable" in msg:
        return "connect Input → … → Output into one path."
    if type_id.startswith("core.conv") and re.search(r"expects \[C", msg):
        src = _source_names(graph, nid)
        return (f"the input from {src[0]} has the wrong rank; "
                if src else "") + "check the Input shape or add a Reshape."
    if "divisible by" in msg and "groups" in msg:
        return "choose groups that divide both channel counts."
    if "heads" in msg and "divisible" in msg:
        return "pick a head count that divides the model width (e.g. 8 heads for 256)."
    if "embed_dim" in msg and "must equal" in msg:
        return "set embed_dim to 0 to infer it from the input."
    if "too large for" in msg and "kernel" in msg:
        return "reduce kernel_size or stride, or add padding."
    if "cycle" in msg:
        return "remove the connection that points back upstream."
    return None


def add_tips(issues: list, graph: "Graph") -> list:
    """Return new issues with a fix hint appended where one is known."""
    from ai_made_easy.core.graph import ValidationIssue

    out = []
    for issue in issues:
        tip = _tip_for(graph, issue)
        if tip and HINT not in issue.message:
            issue = ValidationIssue(issue.severity, f"{issue.message} {HINT} {tip}",
                                    issue.node_id)
        out.append(issue)
    return out
