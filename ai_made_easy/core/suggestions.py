"""Kid-friendly fix tips for common validation failures.

Pure functions over the IR — ``add_tips`` post-processes validation issues
so every consumer (canvas, CLI, MCP agents) can show a 💡 hint that tells
a learner *how* to fix the problem, not just that it exists.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ai_made_easy.core.graph import Graph, ValidationIssue


def _source_names(graph: "Graph", node_id: str) -> list[str]:
    """Display names of the blocks feeding ``node_id``."""
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
        return "drag a wire from the previous block's output dot into this block's input dot"
    if "not reachable" in msg:
        return "connect your blocks into one chain: Input → layers → Output"
    if re.search(r"expected rank 1|needs rank 1|\[F\]", msg) and type_id == "core.dense":
        src = _source_names(graph, nid)
        if src:
            return (f"add a Flatten block between {src[0]} and "
                    f"{node.definition().display_name} — it turns the image "
                    "into one long list of numbers")
        return "add a Flatten block before this one — it turns an image into a list of numbers"
    if "channels" in msg and type_id.startswith("core.conv"):
        return "open this block and match 'in_channels' to the picture coming in (or check the block before it)"
    if "channels" in msg and "Multihead" in (node.definition().display_name if node else ""):
        return "set 'embed_dim' to 0 (= auto) and it will match the input for you"
    if "% num_heads" in msg or "divisible" in msg:
        return "pick a number of heads that divides the model width evenly (e.g. 8 heads for width 256)"
    if "features but" in msg and "Input" in msg:
        return "open the Input block and change its shape so the total size matches the dataset"
    if "cycle" in msg:
        return "a wire looped back on itself — remove the wire that points backwards"
    return None


def add_tips(issues: list, graph: "Graph") -> list:
    """Return new issues with 💡 tips appended where a fix is known."""
    from ai_made_easy.core.graph import ValidationIssue

    out = []
    for issue in issues:
        tip = _tip_for(graph, issue)
        if tip and "💡" not in issue.message:
            issue = ValidationIssue(issue.severity,
                                    f"{issue.message}  💡 {tip}",
                                    issue.node_id)
        out.append(issue)
    return out
