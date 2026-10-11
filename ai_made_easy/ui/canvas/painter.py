"""Node, port and wire rendering for the canvas.

Nodes render as cards: a header band in the block's family colour carrying
the block name, and a body line with the live output shape (model blocks) or
the key setting (configuration blocks). Errors and warnings outline the card;
selection uses the theme accent. Only this package touches OdenGraphQt
internals (enforced by tests/test_structure.py).
"""
from __future__ import annotations

import weakref

from OdenGraphQt.qgraphics.node_base import NodeItem
from OdenGraphQt.qgraphics.pipe import PipeItem
from OdenGraphQt.qgraphics.port import PortItem
from PySide6 import QtCore, QtGui

NODE_WIDTH = 196.0
HEADER_H = 26.0
ROW_H = 22.0
INLINE_HEAD = 20.0     # a pipeline stage's design drawn inside it: kind line, then rows
INLINE_ROW = 16.0
TEXT_COLOR = (20, 22, 26, 255)   # header text on the light family colours

_T: dict = {
    "NODE_BODY": "#23262d", "NODE_BORDER": "#3a3f48", "NODE_TEXT": "#e4e6ea",
    "NODE_SUBTEXT": "#9aa0a9", "ACCENT": "#4c8dff", "ERROR": "#f85149",
    "WARNING": "#e3a008", "WIRE": "#5d6470",
}


def set_theme(tokens: dict) -> None:
    """Adopt theme tokens for every subsequently painted item."""
    _T.update({k: tokens[k] for k in _T if k in tokens})


def install_flat_node_style() -> None:
    """Install the card renderer (idempotent)."""
    if getattr(NodeItem, "_aime_flat", False):
        return
    NodeItem._aime_flat = True
    NodeItem.paint = _flat_paint
    NodeItem.set_proxy_mode = _no_proxy
    NodeItem.auto_switch_mode = lambda self: None
    NodeItem._draw_node_horizontal = _layout_node
    PortItem.paint = _port_paint
    PipeItem.paint = _flow_pipe_paint


def _no_proxy(self, mode) -> None:  # noqa: ANN001
    self._proxy_mode = False


# ------------------------------------------------------------------- layout

def _rows(item) -> int:  # noqa: ANN001
    ins = [p for p in item.inputs if p.isVisible()]
    outs = [p for p in item.outputs if p.isVisible()]
    return max(len(ins), len(outs), 1)


def _layout_node(self) -> None:  # noqa: ANN001
    """Fixed-width card geometry; ports centred in body rows."""
    for text in list(self._input_items.values()) + list(self._output_items.values()):
        text.setVisible(False)
    self._text_item.setVisible(False)
    self._icon_item.setVisible(False)
    rows = _rows(self)
    self._width = NODE_WIDTH
    self._height = HEADER_H + 14.0 + rows * ROW_H
    inline = getattr(self, "_aime_inline", None)
    if inline:
        self._height += INLINE_HEAD + len(inline) * INLINE_ROW + 6.0
    body_top = HEADER_H + 7.0
    for ports, x_of in ((self.inputs, lambda p: -p.boundingRect().width() / 2),
                        (self.outputs, lambda p: self._width - p.boundingRect().width() / 2)):
        visible = [p for p in ports if p.isVisible()]
        offset = (rows - len(visible)) * ROW_H / 2
        for i, port in enumerate(visible):
            centre = body_top + offset + i * ROW_H + ROW_H / 2
            port.setPos(x_of(port), centre - port.boundingRect().height() / 2)
    self._tooltip_disable(self.disabled)
    self.align_widgets(v_offset=HEADER_H)
    self.update()


# ------------------------------------------------------------------- painting

def _font(size: float, bold: bool = False) -> QtGui.QFont:
    font = QtGui.QFont()
    font.setPointSizeF(size)
    font.setBold(bold)
    return font


def _flat_paint(self, painter, option, widget) -> None:  # noqa: ANN001
    painter.save()
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)
    rect = QtCore.QRectF(0.0, 0.0, self._width, self._height)
    radius = 7.0
    card = QtGui.QPainterPath()
    card.addRoundedRect(rect, radius, radius)
    family = QtGui.QColor(*self.color[:3])
    if self.disabled:
        family = family.darker(160)

    painter.fillPath(card, QtGui.QColor(_T["NODE_BODY"]))
    # header band
    painter.save()
    painter.setClipPath(card)
    painter.fillRect(QtCore.QRectF(0, 0, rect.width(), HEADER_H), family)
    painter.restore()

    # outline: error > warning > selected > default
    badge = getattr(self, "_aime_badge", None)
    if badge == "error":
        pen = QtGui.QPen(QtGui.QColor(_T["ERROR"]), 2.0)
    elif badge == "warning":
        pen = QtGui.QPen(QtGui.QColor(_T["WARNING"]), 1.6)
    elif self.selected:
        pen = QtGui.QPen(QtGui.QColor(_T["ACCENT"]), 2.0)
    else:
        pen = QtGui.QPen(QtGui.QColor(_T["NODE_BORDER"]), 1.0)
    if self.selected and badge:
        painter.strokePath(card, QtGui.QPen(QtGui.QColor(_T["ACCENT"]), 4.0))
    painter.strokePath(card, pen)

    # title
    painter.setFont(_font(10.5, bold=True))
    painter.setPen(QtGui.QColor(*TEXT_COLOR[:3]))
    title_rect = QtCore.QRectF(10, 0, rect.width() - (34 if badge else 20), HEADER_H)
    metrics = QtGui.QFontMetricsF(painter.font())
    painter.drawText(title_rect, QtCore.Qt.AlignmentFlag.AlignVCenter,
                     metrics.elidedText(self.name, QtCore.Qt.TextElideMode.ElideRight,
                                        title_rect.width()))
    if badge:
        _paint_badge(painter, rect, badge)

    # body
    labels = getattr(self, "_aime_port_labels", None) or []
    painter.setFont(_font(9.5))
    body_left = 12.0 + (34.0 if labels else 0.0)
    subtitle = getattr(self, "_aime_subtitle", "") or ""
    if labels:
        painter.setPen(QtGui.QColor(_T["NODE_SUBTEXT"]))
        visible = [p for p in self.inputs if p.isVisible()]
        for port, label in zip(visible, labels, strict=False):
            y = port.y() + port.boundingRect().height() / 2
            painter.drawText(QtCore.QRectF(10, y - 8, 60, 16),
                             QtCore.Qt.AlignmentFlag.AlignVCenter, label)
    if subtitle:
        painter.setPen(QtGui.QColor(_T["NODE_TEXT"] if not badge else _T["NODE_SUBTEXT"]))
        card_h = HEADER_H + 14.0 + _rows(self) * ROW_H      # without an inline design
        body = QtCore.QRectF(body_left, HEADER_H + 4, rect.width() - body_left - 12,
                             card_h - HEADER_H - 8)
        fm = QtGui.QFontMetricsF(painter.font())
        painter.drawText(body, QtCore.Qt.AlignmentFlag.AlignVCenter,
                         fm.elidedText(subtitle, QtCore.Qt.TextElideMode.ElideRight,
                                       body.width()))

    inline = getattr(self, "_aime_inline", None)
    if inline:
        _paint_inline(painter, rect, self, inline)

    progress = getattr(self, "_aime_progress", None)
    if progress:
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(_T["ACCENT"]))
        painter.drawRoundedRect(QtCore.QRectF(6, rect.height() - 5,
                                              (rect.width() - 12) * float(progress), 3), 1.5, 1.5)
    painter.restore()


def _paint_inline(painter, rect, item, rows: list) -> None:  # noqa: ANN001
    """The stage's design under its body: the kind, then one row per block."""
    top = HEADER_H + 14.0 + _rows(item) * ROW_H
    pen = QtGui.QPen(QtGui.QColor(_T["NODE_BORDER"]), 1.0)
    pen.setStyle(QtCore.Qt.PenStyle.DashLine)
    painter.setPen(pen)
    painter.drawLine(QtCore.QPointF(6, top), QtCore.QPointF(rect.width() - 6, top))
    painter.setFont(_font(8.5))
    painter.setPen(QtGui.QColor(_T["NODE_SUBTEXT"]))
    fm = QtGui.QFontMetricsF(painter.font())
    title = getattr(item, "_aime_inline_title", "") or ""
    painter.drawText(QtCore.QRectF(10, top + 2, rect.width() - 20, INLINE_HEAD - 4),
                     QtCore.Qt.AlignmentFlag.AlignVCenter,
                     fm.elidedText(title, QtCore.Qt.TextElideMode.ElideRight, rect.width() - 20))
    y = top + INLINE_HEAD
    for name, color, summary, model in rows:
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(color))
        painter.drawRoundedRect(QtCore.QRectF(10, y + 4, 8, 8), 2, 2)
        painter.setPen(QtGui.QColor(_T["NODE_TEXT"] if model else _T["NODE_SUBTEXT"]))
        right = min(fm.horizontalAdvance(summary) + 4, (rect.width() - 34) * 0.5) \
            if summary else 0
        painter.drawText(QtCore.QRectF(24, y, rect.width() - 34 - right, INLINE_ROW),
                         QtCore.Qt.AlignmentFlag.AlignVCenter,
                         fm.elidedText(name, QtCore.Qt.TextElideMode.ElideRight,
                                       rect.width() - 34 - right))
        if summary:
            painter.setPen(QtGui.QColor(_T["NODE_SUBTEXT"]))
            painter.drawText(QtCore.QRectF(rect.width() - 10 - right, y, right, INLINE_ROW),
                             QtCore.Qt.AlignmentFlag.AlignVCenter
                             | QtCore.Qt.AlignmentFlag.AlignRight,
                             fm.elidedText(summary, QtCore.Qt.TextElideMode.ElideRight, right))
        y += INLINE_ROW


def _paint_badge(painter, rect, badge: str) -> None:  # noqa: ANN001
    color = QtGui.QColor(_T["ERROR"] if badge == "error" else _T["WARNING"])
    r = 7.0
    centre = QtCore.QPointF(rect.right() - r - 7, HEADER_H / 2)
    painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 230), 1.2))
    painter.setBrush(color)
    painter.drawEllipse(centre, r, r)
    painter.setFont(_font(8.5, bold=True))
    painter.setPen(QtGui.QColor("#ffffff"))
    painter.drawText(QtCore.QRectF(centre.x() - r, centre.y() - r, 2 * r, 2 * r),
                     QtCore.Qt.AlignmentFlag.AlignCenter, "!")


def _port_paint(self, painter, option, widget) -> None:  # noqa: ANN001
    painter.save()
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    r = 5.5 if not self._hovered else 6.5
    centre = self.boundingRect().center()
    family = QtGui.QColor(*self.color[:3])
    painter.setPen(QtGui.QPen(family, 2.0))
    painter.setBrush(family if self.connected_pipes else QtGui.QColor(_T["NODE_BODY"]))
    painter.drawEllipse(centre, r, r)
    painter.restore()


# --------------------------------------------------------------- wire flow
# While a run is active a dashed overlay crawls along every wire. Pipes
# register weakly so one shared timer can repaint exactly them.

_flow = {"on": False, "phase": 0.0}
_pipes: "weakref.WeakSet[PipeItem]" = weakref.WeakSet()
_orig_pipe_paint = PipeItem.paint


class WireFlowAnimator(QtCore.QObject):
    def __init__(self, parent=None):  # noqa: ANN001
        super().__init__(parent)
        self._timer = QtCore.QTimer(self, interval=80, timeout=self._tick)

    def _tick(self) -> None:
        _flow["phase"] = (_flow["phase"] + 1.6) % 14.0
        for pipe in list(_pipes):
            pipe.update()

    def set_running(self, on: bool) -> None:
        _flow["on"] = on
        if on:
            self._timer.start()
        else:
            self._timer.stop()
            for pipe in list(_pipes):
                pipe.update()


def _flow_pipe_paint(self, painter, option, widget) -> None:  # noqa: ANN001
    _pipes.add(self)
    painter.save()
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    color = QtGui.QColor(_T["ACCENT"]) if self.isSelected() else QtGui.QColor(_T["WIRE"])
    painter.setPen(QtGui.QPen(color, 2.0, QtCore.Qt.PenStyle.SolidLine,
                              QtCore.Qt.PenCapStyle.RoundCap))
    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    painter.drawPath(self.path())
    if _flow["on"]:
        pen = QtGui.QPen(QtGui.QColor(_T["ACCENT"]), 2.4)
        pen.setStyle(QtCore.Qt.PenStyle.DashLine)
        pen.setDashPattern([4, 10])
        pen.setDashOffset(-_flow["phase"])
        painter.setPen(pen)
        painter.drawPath(self.path())
    painter.restore()
