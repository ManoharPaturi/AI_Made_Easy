"""Canvas feature package — the ONLY place that imports OdenGraphQt.

Public surface: CanvasArea, CanvasController, node_type_for,
NODE_TYPE_TO_BLOCK (kept source-compatible for tests/scripts).
"""
from ai_made_easy.ui.canvas.adapter import CanvasController
from ai_made_easy.ui.canvas.area import CanvasArea
from ai_made_easy.ui.canvas.node_factory import (
    BLOCK_TO_NODE_TYPE,
    NODE_TYPE_TO_BLOCK,
    node_type_for,
)

__all__ = [
    "CanvasArea",
    "CanvasController",
    "NODE_TYPE_TO_BLOCK",
    "BLOCK_TO_NODE_TYPE",
    "node_type_for",
]
