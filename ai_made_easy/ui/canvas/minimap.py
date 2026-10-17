"""Minimap (3.3): a small overview of the whole design in the canvas corner.

A second ``QGraphicsView`` of the same scene, scaled to fit every item. The visible part of
the main canvas is outlined; clicking or dragging in the minimap moves the main view there.
It refreshes on a light timer only while shown.
"""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets


class Minimap(QtWidgets.QGraphicsView):
    SIZE = QtCore.QSize(220, 150)

    def __init__(self, viewer: QtWidgets.QGraphicsView, parent: QtWidgets.QWidget | None = None):
        super().__init__(viewer.scene(), parent)
        self._viewer = viewer
        self._accent = QtGui.QColor("#4c8dff")
        self._dragging = False
        self.setObjectName("canvasMinimap")
        self.setAccessibleName("Minimap")
        self.setToolTip("Minimap: click or drag to move around the design")
        self.setFixedSize(self.SIZE)
        self.setInteractive(False)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setStyleSheet("QGraphicsView#canvasMinimap { border: 1px solid rgba(127,127,127,90);"
                           " border-radius: 6px; }")
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self.refresh)
        self._last_bounds = QtCore.QRectF()

    def set_theme(self, tokens: dict) -> None:
        self._accent = QtGui.QColor(tokens.get("ACCENT", "#4c8dff"))
        bg = tokens.get("PANEL", "#1e2126")
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(bg)))
        self.viewport().update()

    # ------------------------------------------------------------------ geometry

    def content_rect(self) -> QtCore.QRectF:
        """Everything on the canvas plus the part of it that is in view."""
        items = self.scene().itemsBoundingRect()
        visible = self.visible_rect()
        rect = items.united(visible) if not items.isEmpty() else visible
        return rect.adjusted(-30, -30, 30, 30)

    def visible_rect(self) -> QtCore.QRectF:
        return self._viewer.mapToScene(self._viewer.viewport().rect()).boundingRect()

    def refresh(self) -> None:
        bounds = self.content_rect()
        if bounds != self._last_bounds:
            self._last_bounds = bounds
            self.fitInView(bounds, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.viewport().update()

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # noqa: N802 — Qt API
        super().showEvent(event)
        self._last_bounds = QtCore.QRectF()
        self.refresh()
        self._timer.start()

    def hideEvent(self, event: QtGui.QHideEvent) -> None:  # noqa: N802 — Qt API
        self._timer.stop()
        super().hideEvent(event)

    # ------------------------------------------------------------------ painting

    def drawBackground(self, painter: QtGui.QPainter, rect: QtCore.QRectF) -> None:  # noqa: N802
        painter.fillRect(rect, self.backgroundBrush())

    def drawForeground(self, painter: QtGui.QPainter, rect: QtCore.QRectF) -> None:  # noqa: N802
        visible = self.visible_rect()
        fill = QtGui.QColor(self._accent)
        fill.setAlpha(36)
        pen = QtGui.QPen(self._accent)
        pen.setCosmetic(True)
        pen.setWidthF(1.5)
        painter.setPen(pen)
        painter.setBrush(fill)
        painter.drawRect(visible)

    # ------------------------------------------------------------------ navigation

    def centre_main_on(self, scene_pos: QtCore.QPointF) -> None:
        self._viewer.centerOn(scene_pos)
        # OdenGraphQt pans by moving its scene rect rather than with scroll bars
        scene_range = getattr(self._viewer, "_scene_range", None)
        if isinstance(scene_range, QtCore.QRectF):
            scene_range.moveCenter(scene_pos)
            self._viewer.setSceneRect(scene_range)
        self.viewport().update()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._dragging = True
            self.centre_main_on(self.mapToScene(event.position().toPoint()))
        event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if self._dragging:
            self.centre_main_on(self.mapToScene(event.position().toPoint()))
        event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        self._dragging = False
        event.accept()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:  # noqa: N802 — no zooming here
        event.accept()
