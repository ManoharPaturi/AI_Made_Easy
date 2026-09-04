"""CanvasArea: the canvas page — OdenGraphQt view + floating controls.

This is where OdenGraphQt-supplied widgets (palette, properties bin) are
instantiated for the rest of the UI, so features never import OdenGraphQt.
Both widgets are re-skinned here: their library painters read the SYSTEM
palette (dark charcoal under macOS dark mode) and ignore the app theme, so
we install a flat delegate that draws mini block previews in family colours,
and force a theme QPalette so any palette-driven bits follow suit.
"""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from OdenGraphQt import NodesPaletteWidget, PropertiesBinWidget
from OdenGraphQt.custom_widgets.nodes_palette import NodesGridView

from ai_made_easy.core.registry import get_registry
from ai_made_easy.ui.canvas.adapter import CanvasController
from ai_made_easy.ui.canvas.painter import INK


def apply_theme_palette(widget: QtWidgets.QWidget) -> None:
    """Pin classroom palette roles so system dark mode can't leak in."""
    from ai_made_easy.ui.theme import DEFAULT_THEME, THEMES

    t = THEMES[DEFAULT_THEME]
    pal = QtGui.QPalette()
    pal.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(t["PANEL"]))
    pal.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(t["TEXT"]))
    pal.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(t["TEXT"]))
    pal.setColor(QtGui.QPalette.ColorRole.Midlight, QtGui.QColor(t["BORDER"]))
    pal.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(t["SURFACE"]))
    pal.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(t["PANEL"]))
    pal.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(t["ACCENT"]))
    pal.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor("#ffffff"))
    widget.setPalette(pal)
    for child in widget.findChildren(QtWidgets.QWidget):
        child.setPalette(pal)


class FlatPaletteDelegate(QtWidgets.QStyledItemDelegate):
    """Palette entries as mini block previews — same flat family look as canvas."""

    CELL = QtCore.QSize(154, 48)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._colors = {b.display_name: b.color for b in get_registry().all()}

    def sizeHint(self, option, index) -> QtCore.QSize:  # noqa: ANN001
        return self.CELL

    def paint(self, painter, option, index) -> None:  # noqa: ANN001
        model = index.model().sourceModel()
        item = model.item(index.row(), index.column())
        text = item.text()
        rect = QtCore.QRectF(option.rect.adjusted(3, 3, -3, -3))
        fam = QtGui.QColor(self._colors.get(text, "#FFE8CC"))
        selected = bool(option.state
                        & QtWidgets.QStyle.StateFlag.State_Selected)

        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        path = QtGui.QPainterPath()
        path.addRoundedRect(rect, 14, 14)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.fillPath(path, fam)
        painter.strokePath(path, QtGui.QPen(fam.darker(165), 1.2))
        if selected:
            painter.strokePath(path, QtGui.QPen(INK, 2.2))
        # port nubs, like the canvas blocks
        nub = fam.darker(135)
        painter.setBrush(nub)
        cy = rect.center().y()
        painter.drawEllipse(QtCore.QPointF(rect.left() + 1, cy), 3.2, 3.2)
        painter.drawEllipse(QtCore.QPointF(rect.right() - 1, cy), 3.2, 3.2)
        # bold ink label, elided to fit
        font = QtGui.QFont()
        font.setPointSizeF(9.5)
        font.setBold(True)
        painter.setFont(font)
        shown = QtGui.QFontMetrics(font).elidedText(
            text, QtCore.Qt.TextElideMode.ElideRight, rect.width() - 14)
        painter.setPen(INK)
        painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, shown)
        painter.restore()


class CanvasArea(QtWidgets.QWidget):
    """Embeddable canvas page; exposes the adapter and OdenGraphQt widgets."""

    def __init__(self, adapter: CanvasController, parent=None):
        super().__init__(parent)
        self.adapter = adapter
        self.node_graph = adapter.node_graph

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(adapter.widget)

    def make_palette_widget(self) -> "NodesPaletteWidget":
        palette = NodesPaletteWidget(node_graph=self.node_graph)
        apply_theme_palette(palette)
        for view in palette.findChildren(NodesGridView):
            view.setItemDelegate(FlatPaletteDelegate(view))
            model = view.model().sourceModel()
            for row in range(model.rowCount()):
                model.item(row).setSizeHint(FlatPaletteDelegate.CELL)
        return palette

    def make_properties_widget(self) -> "PropertiesBinWidget":
        from ai_made_easy.ui.canvas.prop_widgets_patch import (
            install_prop_widget_patches,
        )

        install_prop_widget_patches()
        prop_bin = PropertiesBinWidget(node_graph=self.node_graph)
        apply_theme_palette(prop_bin)
        self.node_graph.add_properties_bin(prop_bin)
        return prop_bin
