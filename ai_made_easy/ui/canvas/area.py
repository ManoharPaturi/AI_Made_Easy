"""CanvasArea: the embeddable canvas page (OdenGraphQt viewer host) and its minimap."""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ai_made_easy.ui.canvas.adapter import CanvasController
from ai_made_easy.ui.canvas.minimap import Minimap


class CanvasArea(QtWidgets.QWidget):
    """Hosts the node-graph viewer; exposes the adapter to the rest of the UI."""

    MARGIN = 12

    def __init__(self, adapter: CanvasController, parent=None):
        super().__init__(parent)
        self.adapter = adapter
        self.node_graph = adapter.node_graph
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(adapter.widget)
        self.setAccessibleName("Design canvas")
        self.minimap = Minimap(adapter.node_graph.viewer(), self)
        self.minimap.hide()

    def set_minimap(self, visible: bool) -> None:
        self.minimap.setVisible(visible)
        if visible:
            self._place_minimap()
            self.minimap.raise_()

    def set_theme(self, tokens: dict) -> None:
        self.minimap.set_theme(tokens)

    def _place_minimap(self) -> None:
        size = self.minimap.size()
        self.minimap.move(QtCore.QPoint(self.width() - size.width() - self.MARGIN,
                                        self.height() - size.height() - self.MARGIN))

    def resizeEvent(self, event) -> None:  # noqa: N802, ANN001 — Qt API
        super().resizeEvent(event)
        self._place_minimap()
