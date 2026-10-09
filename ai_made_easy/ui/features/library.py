"""Block Library: searchable, sectioned tree of every registered block.

Drag a block onto the canvas or press Enter / double-click to place it at the
viewport centre. Grouping mirrors how practitioners think about a pipeline:
data → preprocessing → layers → models → training, plus classic ML and LLMs.
"""
from __future__ import annotations

from typing import Callable

from PySide6 import QtCore, QtGui, QtWidgets

from ai_made_easy.core.spec import missing_requirements
from ai_made_easy.core.registry import get_registry

SECTIONS: list[tuple[str, list[str]]] = [
    ("Data", ["Input / Output", "Data"]),
    ("Preprocessing", ["Preprocessing", "Image Transforms", "Augmentation",
                       "Text Processing", "Audio Processing"]),
    ("Layers", ["Linear", "Convolution", "Pooling", "Resizing", "Recurrent", "Attention",
                "Sequence Models", "Embedding", "Activations", "Normalization", "Regularization",
                "Merge", "Tensor Ops", "Audio Front-End", "Generative", "Bayesian Deep Learning",
                "Graph Neural Networks"]),
    ("Models", ["Vision Tasks", "Forecasting", "Generative Models", "Tabular Models",
                "Recommenders", "Reinforcement Learning", "Pretrained Models",
                "Architectures"]),
    ("Training", ["Loss", "Optimizer", "Scheduler", "Training", "Generative Training",
                  "Metrics"]),
    ("Probabilistic", ["Graphical Models", "Probabilistic Programs", "Gaussian Processes",
                       "Normalizing Flows", "State-Space Models"]),
    ("Classic ML", ["Linear Models", "Support Vector Machines", "Neighbors & Bayes",
                    "Trees & Ensembles", "Gradient Boosting", "Neural (sklearn)", "Clustering",
                    "Anomaly Detection", "Feature Engineering", "Text Features",
                    "Model Selection"]),
    ("LLM", ["LLM"]),
    ("Custom", ["Custom"]),
]

ROLE_TYPE = QtCore.Qt.ItemDataRole.UserRole


def _swatch(color: str) -> QtGui.QIcon:
    pm = QtGui.QPixmap(24, 24)
    pm.setDevicePixelRatio(2.0)
    pm.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pm)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.setBrush(QtGui.QColor(color))
    painter.drawRoundedRect(QtCore.QRectF(2, 3, 8, 6), 2, 2)
    painter.end()
    return QtGui.QIcon(pm)


class _Tree(QtWidgets.QTreeWidget):
    """Tree whose block rows can be dragged onto the canvas."""

    def __init__(self, mime_factory: Callable[[str], QtCore.QMimeData], parent=None):
        super().__init__(parent)
        self._mime_factory = mime_factory
        self.setHeaderHidden(True)
        self.setIndentation(14)
        self.setDragEnabled(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragOnly)
        self.setUniformRowHeights(True)
        self.setAnimated(True)
        self.setExpandsOnDoubleClick(True)

    def mimeData(self, items):  # noqa: N802 (Qt name)
        type_id = items[0].data(0, ROLE_TYPE) if items else None
        return self._mime_factory(type_id) if type_id else QtCore.QMimeData()

    def mimeTypes(self):  # noqa: N802 (Qt name)
        return ["OdenGraphQt/nodes"]


class BlockLibrary(QtWidgets.QWidget):
    """Search box + sectioned tree; emits ``place_requested(type_id)``."""

    place_requested = QtCore.Signal(str)
    block_hovered = QtCore.Signal(str)

    def __init__(self, mime_factory: Callable[[str], QtCore.QMimeData], parent=None):
        super().__init__(parent)
        self.setObjectName("dockBody")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 4)
        layout.setSpacing(6)
        self.search = QtWidgets.QLineEdit()
        self.search.setObjectName("searchField")
        self.search.setPlaceholderText("Search blocks")
        self.search.setClearButtonEnabled(True)
        from ai_made_easy.ui import icons

        self.search.addAction(icons.icon("search", size=16),
                              QtWidgets.QLineEdit.ActionPosition.LeadingPosition)
        layout.addWidget(self.search)
        self.tree = _Tree(mime_factory)
        layout.addWidget(self.tree, 1)
        self.count = QtWidgets.QLabel()
        self.count.setObjectName("blockMeta")
        layout.addWidget(self.count)

        self.search.textChanged.connect(self._filter)
        self.search.returnPressed.connect(self._place_first_visible)
        self.tree.itemActivated.connect(self._activate)
        self.rebuild()

    # ------------------------------------------------------------- build
    def rebuild(self) -> None:
        self.tree.clear()
        by_cat = get_registry().by_category()
        known = {c for _, cats in SECTIONS for c in cats}
        extra = sorted(set(by_cat) - known)
        sections = SECTIONS + ([("Other", extra)] if extra else [])
        bold = QtGui.QFont()
        bold.setBold(True)
        total = 0
        for title, categories in sections:
            present = [c for c in categories if by_cat.get(c)]
            if not present:
                continue
            section = QtWidgets.QTreeWidgetItem([title])
            section.setFont(0, bold)
            section.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
            self.tree.addTopLevelItem(section)
            for cat in present:
                blocks = sorted(by_cat[cat], key=lambda b: b.display_name.lower())
                node = QtWidgets.QTreeWidgetItem([f"{cat}"])
                node.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
                node.setData(0, QtCore.Qt.ItemDataRole.ToolTipRole, f"{len(blocks)} blocks")
                section.addChild(node)
                for block in blocks:
                    item = QtWidgets.QTreeWidgetItem([block.display_name])
                    item.setData(0, ROLE_TYPE, block.type_id)
                    item.setIcon(0, _swatch(block.color))
                    tip = [f"<b>{block.display_name}</b>"]
                    if block.description:
                        tip.append(block.description)
                    if block.library:
                        tip.append(f"<span style='color:gray'>{block.library}</span>")
                    missing = missing_requirements(block)
                    if missing:  # usable, but its generated code needs these packages
                        extra = f"pip install 'ai-made-easy[{block.extra}]'" if block.extra \
                            else "pip install " + " ".join(missing)
                        tip.append(f"<span style='color:#e3a008'>Requires {', '.join(missing)}"
                                   f" — {extra}</span>")
                        item.setText(0, f"{block.display_name}  ⤓")
                    item.setToolTip(0, "<br>".join(tip))
                    item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled
                                  | QtCore.Qt.ItemFlag.ItemIsSelectable
                                  | QtCore.Qt.ItemFlag.ItemIsDragEnabled)
                    node.addChild(item)
                    total += 1
            section.setExpanded(title in ("Data", "Layers"))
        self.count.setText(f"{total} blocks")
        self._total = total

    # ------------------------------------------------------------- search
    def _filter(self, text: str) -> None:
        query = text.strip().lower()
        shown = 0
        for i in range(self.tree.topLevelItemCount()):
            section = self.tree.topLevelItem(i)
            section_hit = False
            for j in range(section.childCount()):
                cat = section.child(j)
                cat_hit = False
                cat_match = query and query in cat.text(0).lower()
                for k in range(cat.childCount()):
                    item = cat.child(k)
                    type_id = item.data(0, ROLE_TYPE)
                    hit = (not query or cat_match or query in item.text(0).lower()
                           or query in type_id.lower()
                           or query in (item.toolTip(0) or "").lower())
                    item.setHidden(not hit)
                    cat_hit |= hit
                    shown += int(hit)
                cat.setHidden(not cat_hit)
                cat.setExpanded(bool(query) and cat_hit)
                section_hit |= cat_hit
            section.setHidden(not section_hit)
            if query:
                section.setExpanded(section_hit)
        self.count.setText(f"{shown} of {self._total} blocks" if query else f"{self._total} blocks")

    def _visible_blocks(self) -> list[QtWidgets.QTreeWidgetItem]:
        found = []
        it = QtWidgets.QTreeWidgetItemIterator(self.tree)
        while it.value():
            item = it.value()
            if item.data(0, ROLE_TYPE) and not item.isHidden() and not item.parent().isHidden():
                found.append(item)
            it += 1
        return found

    def _place_first_visible(self) -> None:
        current = self.tree.currentItem()
        if current is not None and current.data(0, ROLE_TYPE) and not current.isHidden():
            self.place_requested.emit(current.data(0, ROLE_TYPE))
            return
        blocks = self._visible_blocks()
        if blocks:
            self.place_requested.emit(blocks[0].data(0, ROLE_TYPE))

    def _activate(self, item: QtWidgets.QTreeWidgetItem) -> None:
        type_id = item.data(0, ROLE_TYPE)
        if type_id:
            self.place_requested.emit(type_id)

    def focus_search(self) -> None:
        self.search.setFocus()
        self.search.selectAll()
