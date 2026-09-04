"""Flat block painter — the reference look.

Blocks render as solid, generously-rounded family-coloured rectangles with a
bold dark-ink label centred in the block (no header strip, no border, no
icon, no port captions). Ports are the only chrome: small dots in a deeper
shade of the family colour; wires inherit that shade automatically.

Only this package is allowed to touch OdenGraphQt internals (enforced by
tests/test_structure.py), so the painter patch lives here.
"""
from __future__ import annotations

from PySide6 import QtCore, QtGui

from OdenGraphQt.qgraphics.node_base import NodeItem
from OdenGraphQt.qgraphics.port import PortItem

INK = QtGui.QColor(26, 30, 38)          # label + selection ink on pastel fills
SELECTED_RING = QtGui.QColor(26, 30, 38)  # dark ink reads on the dusty paper
DISABLED_FILL = QtGui.QColor(120, 129, 146)

TEXT_COLOR = (26, 30, 38, 255)          # rgba for NodeObject.text_color


def install_flat_node_style() -> None:
    """Replace NodeItem/PortItem painting with the flat renderer (idempotent)."""
    if getattr(NodeItem, "_aime_flat", False):
        return
    NodeItem._aime_flat = True
    NodeItem.paint = _flat_paint
    NodeItem.set_proxy_mode = _flat_set_proxy_mode
    NodeItem.auto_switch_mode = _flat_auto_switch_mode
    PortItem.paint = _flat_port_paint
    _orig_init = NodeItem.__init__

    def _flat_init(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        _orig_init(self, *args, **kwargs)
        # Proxy mode hides labels at low zoom; its paint()-time flipping races
        # Qt's device cache and labels then vanish nondeterministically. Our
        # graphs are small — labels always render, like the reference design.
        self._proxy_mode_threshold = 0

    NodeItem.__init__ = _flat_init


def _flat_auto_switch_mode(self) -> None:  # noqa: ANN001
    """Low-zoom switch measured from the view transform.

    The stock version maps the node rect through ``mapToGlobal``, which
    returns degenerate values on offscreen/headless renders (width 0 always
    wins proxy mode and hides every label). Width = node width * view scale
    is deterministic everywhere.
    """
    viewer = self.viewer()
    if viewer is None:
        return
    scale = viewer.transform().m11()
    self.set_proxy_mode(self._width * scale < self._proxy_mode_threshold)


def _flat_set_proxy_mode(self, mode) -> None:  # noqa: ANN001
    """Low-zoom mode, flat-style: label may drop, icon/port captions never.

    The stock version re-shows the icon and port-name captions; ours keep the
    flat look (ports carry no captions, blocks carry no icon).
    """
    if mode is self._proxy_mode:
        return
    self._proxy_mode = mode
    self._x_item.proxy_mode = mode
    for w in self._widgets.values():
        w.widget().setVisible(not mode)
    self._text_item.setVisible(not mode)
    self._icon_item.setVisible(False)


def _flat_paint(self, painter, option, widget) -> None:  # noqa: ANN001
    self.auto_switch_mode()
    painter.save()
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    rect = self.boundingRect().adjusted(1.0, 1.0, -1.0, -1.0)
    radius = rect.height() * 0.30
    path = QtGui.QPainterPath()
    path.addRoundedRect(rect, radius, radius)
    # NB: disabled/selected are Qt bool properties here, not methods.
    fill = DISABLED_FILL if self.disabled else QtGui.QColor(*self.color)
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.fillPath(path, fill)
    # crisp defining edge so pastel blocks sit firmly on the dusty paper
    viewer = self.viewer()
    edge = QtGui.QPen(fill.darker(165), 1.2)
    edge.setCosmetic(viewer is not None and viewer.get_zoom() < 0.0)
    painter.strokePath(path, edge)
    if self.selected:
        pen = QtGui.QPen(SELECTED_RING, 2.4)
        pen.setCosmetic(viewer is not None and viewer.get_zoom() < 0.0)
        painter.strokePath(path, pen)
    _paint_badge(self, painter, rect)
    # live run progress (Trainer block): a strip filling along the bottom
    progress = getattr(self, "_aime_progress", None)
    if progress:
        bar = 5.0
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(fill.darker(135))
        painter.drawRoundedRect(
            QtCore.QRectF(rect.left() + 4, rect.bottom() - bar - 3,
                          max((rect.width() - 8) * float(progress), bar),
                          bar), 2.5, 2.5)
    painter.restore()
    _center_label(self)


def _paint_badge(node, painter, rect) -> None:  # noqa: ANN001
    """Guardrail badge: red ✖ for errors, amber ! for warnings."""
    badge = getattr(node, "_aime_badge", None)
    if badge is None:
        return
    color = (QtGui.QColor(229, 72, 77) if badge == "error"
             else QtGui.QColor(255, 171, 45))
    r = 9.0
    center = QtCore.QPointF(rect.right() - r - 3.0, rect.top() + r + 3.0)
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.setBrush(color)
    painter.drawEllipse(center, r, r)
    font = QtGui.QFont()
    font.setPointSizeF(8.5)
    font.setBold(True)
    painter.setFont(font)
    painter.setPen(QtGui.QColor(255, 255, 255))
    painter.drawText(QtCore.QRectF(center.x() - r, center.y() - r, r * 2, r * 2),
                     QtCore.Qt.AlignmentFlag.AlignCenter,
                     "✕" if badge == "error" else "!")


def _flat_port_paint(self, painter, option, widget) -> None:  # noqa: ANN001
    """Port dots always in the family shade.

    The stock painter switches to library-wide ACTIVE/HOVER blues once a
    pipe is connected — which breaks the per-family colour story on the
    paper canvas.
    """
    painter.save()
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    rect_w = self._width / 1.8
    rect_h = self._height / 1.8
    rect = QtCore.QRectF(
        self.boundingRect().center().x() - rect_w / 2,
        self.boundingRect().center().y() - rect_h / 2,
        rect_w, rect_h)
    fill = QtGui.QColor(*self.color)
    ring = INK if self._hovered else fill.darker(150)
    painter.setPen(QtGui.QPen(ring, 1.8))
    painter.setBrush(fill)
    painter.drawEllipse(rect)
    if self.connected_pipes:
        w, h = rect.width() / 2.5, rect.height() / 2.5
        inner = QtCore.QRectF(rect.center().x() - w / 2, rect.center().y() - h / 2,
                              w, h)
        painter.setPen(QtGui.QPen(ring, 1.6))
        painter.setBrush(ring)
        painter.drawEllipse(inner)
    painter.restore()


def _center_label(node) -> None:  # noqa: ANN001
    text = node._text_item
    if not getattr(node, "_aime_label", False):
        node._aime_label = True
        node._icon_item.setVisible(False)
        font = QtGui.QFont()
        font.setPointSizeF(11.0)
        font.setBold(True)
        text.setFont(font)
    text.setDefaultTextColor(INK)
    rect = node.boundingRect()
    br = text.boundingRect()
    text.setPos((rect.width() - br.width()) / 2.0,
                (rect.height() - br.height()) / 2.0)
