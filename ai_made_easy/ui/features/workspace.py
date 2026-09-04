"""Workspace: the ONE page — a fixed card layout (Langflow/n8n canvas page).

Blocks card left, canvas center with the runs console beneath it, inspector
right — resizable via splitters but locked in place (no floating or
rearrangeable docks): a structure a learner can rely on. Each card is a
rounded panel; the panels' own tab bars label their content. Splitter sizes
persist through the shell's QSettings group.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets


def _card(object_name: str, body: QtWidgets.QWidget) -> QtWidgets.QFrame:
    card = QtWidgets.QFrame()
    card.setObjectName(object_name)
    card.setProperty("card", True)  # QSS selector: QFrame[card="true"]
    layout = QtWidgets.QVBoxLayout(card)
    layout.setContentsMargins(6, 6, 6, 6)
    layout.addWidget(body, 1)
    return card


class Workspace(QtWidgets.QWidget):
    """Single-page composition of every feature area."""

    def __init__(self, ctx, parent=None):  # noqa: ANN001 (AppContext)
        super().__init__(parent)
        self.setObjectName("workspacePage")

        # canvas card with a hidden ⌨️ code pane under it (blocks↔python)
        canvas_host = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        canvas_host.addWidget(ctx.canvas_area)
        self.code_pane = QtWidgets.QPlainTextEdit()
        self.code_pane.setReadOnly(True)
        self.code_pane.setObjectName("sideCode")
        self.code_pane.setVisible(False)
        font = self.code_pane.font()
        font.setFamily("Menlo")
        font.setPointSize(11)
        self.code_pane.setFont(font)
        canvas_host.addWidget(self.code_pane)
        canvas_host.setStretchFactor(0, 1)
        canvas_host.setChildrenCollapsible(False)
        canvas_host.setSizes([520, 180])
        self.code_pane_toggled = False

        self._v_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self._v_split.addWidget(_card("panel.canvas", canvas_host))
        self._v_split.addWidget(_card("panel.runconsole", ctx.runconsole))
        self._v_split.setStretchFactor(0, 1)
        self._v_split.setStretchFactor(1, 0)
        self._v_split.setSizes([640, 240])

        self._h_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self._h_split.addWidget(_card("panel.blocks", ctx.palette))
        self._h_split.addWidget(self._v_split)
        self._h_split.addWidget(_card("panel.inspector", ctx.inspector))
        self._h_split.setStretchFactor(0, 0)
        self._h_split.setStretchFactor(1, 1)
        self._h_split.setStretchFactor(2, 0)
        self._h_split.setSizes([280, 900, 330])

        self._v_split.setChildrenCollapsible(False)
        self._h_split.setChildrenCollapsible(False)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.addWidget(self._h_split)

    # -------------------------------------------------- blocks ↔ python
    def toggle_code_pane(self) -> bool:
        """⌨️ show/hide the live code next to the blocks. Returns state."""
        self.code_pane_toggled = not self.code_pane_toggled
        self.code_pane.setVisible(self.code_pane_toggled)
        return self.code_pane_toggled

    def set_side_code(self, code: str) -> None:
        """Live-synced from the preview renderer; cheap when hidden."""
        if self.code_pane_toggled and self.code_pane.toPlainText() != code:
            self.code_pane.setPlainText(code)

    # ------------------------------------------------------- persistence

    def save_state(self, settings: QtCore.QSettings) -> None:
        settings.setValue("hsplit", self._h_split.sizes())
        settings.setValue("vsplit", self._v_split.sizes())

    def restore_state(self, settings: QtCore.QSettings) -> None:
        h = settings.value("hsplit")
        v = settings.value("vsplit")
        if h and len(h) == 3:
            self._h_split.setSizes([int(x) for x in h])
        if v and len(v) == 2:
            self._v_split.setSizes([int(x) for x in v])
