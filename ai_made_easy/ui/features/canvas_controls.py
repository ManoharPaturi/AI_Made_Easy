"""CanvasControls: floating zoom/fit/snapshot cluster pinned bottom-right
of the canvas (Langflow CanvasControls logic). Talks only to the adapter.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class CanvasControls(QtWidgets.QFrame):
    snapshot_clicked = QtCore.Signal()
    code_toggle_clicked = QtCore.Signal()

    def __init__(self, canvas_widget: QtWidgets.QWidget, adapter, parent=None):
        super().__init__(canvas_widget)
        self.setObjectName("canvasControls")
        self._adapter = adapter
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        for label, tip, handler in (
            ("＋", "Zoom in", lambda: self._adapter.zoom(+1)),
            ("－", "Zoom out", lambda: self._adapter.zoom(-1)),
            ("⛶", "Fit graph in view", self._adapter.center_view),
            ("⌨️", "Show the Python code for these blocks "
                   "(blocks ↔ python)", lambda: self.code_toggle_clicked.emit()),
            ("📸", "Save canvas as PNG", lambda: self.snapshot_clicked.emit()),
        ):
            btn = QtWidgets.QPushButton(label)
            btn.setToolTip(tip)
            btn.setFixedWidth(34)
            btn.clicked.connect(handler)
            layout.addWidget(btn)

        canvas_widget.installEventFilter(self)
        self._reposition(canvas_widget.size())
        self.show()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 (Qt name)
        if watched is self.parent() and event.type() == QtCore.QEvent.Type.Resize:
            self._reposition(event.size())
        return super().eventFilter(watched, event)

    def _reposition(self, size: QtCore.QSize) -> None:
        self.adjustSize()
        self.move(size.width() - self.width() - 16,
                  size.height() - self.height() - 16)
