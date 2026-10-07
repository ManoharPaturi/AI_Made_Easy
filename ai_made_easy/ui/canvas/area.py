"""CanvasArea: the embeddable canvas page (OdenGraphQt viewer host)."""
from __future__ import annotations

from PySide6 import QtWidgets

from ai_made_easy.ui.canvas.adapter import CanvasController


class CanvasArea(QtWidgets.QWidget):
    """Hosts the node-graph viewer; exposes the adapter to the rest of the UI."""

    def __init__(self, adapter: CanvasController, parent=None):
        super().__init__(parent)
        self.adapter = adapter
        self.node_graph = adapter.node_graph
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(adapter.widget)
