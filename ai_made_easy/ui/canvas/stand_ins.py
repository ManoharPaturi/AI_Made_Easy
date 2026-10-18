"""Stand-in wires for folded groups on the desktop canvas (3.4).

A folded group hides its blocks, and OdenGraphQt draws a wire only between two visible
ports, so the wires into and out of the group would vanish. Each such wire gets a dashed
stand-in drawn from the card's edge (left for wires coming in, right for wires going out)
to the port outside, or to another folded card. Stand-ins are only drawn: the real wires
are kept, so the design and its code don't change.
"""
from __future__ import annotations

from collections.abc import Callable

from PySide6 import QtCore, QtGui, QtWidgets


class StandIn(QtWidgets.QGraphicsPathItem):
    def __init__(self, pipe, start, end) -> None:  # noqa: ANN001
        super().__init__()
        self.pipe = pipe
        self.start, self.end = start, end   # a port item, or (card, "in" | "out")
        self.setZValue(-1)
        self.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self._ends: tuple | None = None

    @staticmethod
    def _point(end) -> QtCore.QPointF:  # noqa: ANN001
        if isinstance(end, tuple):
            card, side = end
            rect = card.view.sceneBoundingRect()
            return QtCore.QPointF(rect.left() if side == "in" else rect.right(),
                                  rect.center().y())
        return end.mapToScene(end.boundingRect().center())

    def paint(self, painter, option, widget=None) -> None:  # noqa: ANN001
        from ai_made_easy.ui.canvas.painter import _T   # the theme's wire colour, as now

        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QPen(QtGui.QColor(_T["WIRE"]), 2.0, QtCore.Qt.PenStyle.DashLine,
                                  QtCore.Qt.PenCapStyle.RoundCap))
        painter.drawPath(self.path())
        painter.restore()

    def gone(self) -> bool:
        cards = [e[0] for e in (self.start, self.end) if isinstance(e, tuple)]
        return self.pipe.scene() is None or any(
            c.view.scene() is None or not c.folded for c in cards)

    def refresh(self) -> None:
        a, b = self._point(self.start), self._point(self.end)
        ends = (a.x(), a.y(), b.x(), b.y())
        if ends == self._ends:   # an unchanged path is not redrawn (scene.changed loops)
            return
        self._ends = ends
        dx = max(40.0, abs(b.x() - a.x()) / 2)
        path = QtGui.QPainterPath(a)
        path.cubicTo(a.x() + dx, a.y(), b.x() - dx, b.y(), b.x(), b.y())
        self.setPath(path)


class StandInWires:
    def __init__(self, scene: QtWidgets.QGraphicsScene, folded: Callable[[], list],
                 node: Callable) -> None:
        self._scene = scene
        self._folded = folded   # the folded group nodes
        self._node = node       # node id -> node
        self.items: list[StandIn] = []
        scene.changed.connect(self._follow)

    def clear(self) -> None:
        for item in self.items:
            if item.scene() is not None:
                item.scene().removeItem(item)
        self.items = []

    def rebuild(self) -> None:
        """Draw a stand-in for every wire that crosses a folded group's border."""
        self.clear()
        folded_groups = self._folded()
        card_of = {nid: g for g in folded_groups for nid in g.folded["members"]}
        pipes = {}
        for group in folded_groups:
            for nid in group.folded["members"]:
                node = self._node(nid)
                if node is None:
                    continue
                for port in [*node.input_ports(), *node.output_ports()]:
                    for pipe in port.view.connected_pipes:
                        pipes[id(pipe)] = pipe
        for pipe in pipes.values():
            out_port, in_port = pipe.output_port, pipe.input_port
            if out_port is None or in_port is None:
                continue
            src, dst = card_of.get(out_port.node.id), card_of.get(in_port.node.id)
            if src is dst:   # inside one group: hidden with it
                continue
            item = StandIn(pipe, (src, "out") if src else out_port,
                           (dst, "in") if dst else in_port)
            item.refresh()
            self._scene.addItem(item)
            self.items.append(item)

    def _follow(self, *_args) -> None:
        if any(item.gone() for item in self.items):
            self.rebuild()   # a wire or a card went away
            return
        for item in self.items:
            item.refresh()
