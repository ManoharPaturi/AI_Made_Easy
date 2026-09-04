"""PaletteFeature: search-first block palette (left dock).

Search box + ranked results (⏎ places at viewport center) above the
category palette; pretty tab names; the "Notes" tab exposes the Backdrop
node (Langflow NoteNode / Orange annotations logic).
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ai_made_easy.core.registry import get_registry

# one friendly emoji per family — the same colour story the blocks tell
_TAB_EMOJI = {
    "Data": "📥", "Preprocessing": "🧹", "Layers": "🧱", "Activations": "⚡",
    "Attention": "🤖", "Normalization": "📏", "Tensor Ops": "🔀",
    "Training": "🎯", "Evaluation": "📊", "Architectures": "🏰",
    "LLM": "💬", "Custom": "⭐", "Notes": "📝",
}


def prettify_palette_tabs(palette: QtWidgets.QWidget) -> None:
    """aim.foo -> '⚡ Activations'; the OdenGraphQt built-ins tab becomes '📝 Notes'."""
    for tab_bar in palette.findChildren(QtWidgets.QTabBar):
        for i in range(tab_bar.count()):
            label = tab_bar.tabText(i)
            if label.startswith("OdenGraphQt"):
                tab_bar.setTabText(i, f"{_TAB_EMOJI['Notes']} Notes")
            elif label.startswith("aim."):
                pretty = label[4:].replace("tensorops", "tensor ops")
                shown = "LLM" if pretty == "llm" else pretty.replace("_", " ").title()
                emoji = _TAB_EMOJI.get(shown, "")
                tab_bar.setTabText(i, f"{emoji} {shown}" if emoji else shown)
    for tab_widget in palette.findChildren(QtWidgets.QTabWidget):
        for i in range(tab_widget.count()):
            if "Layers" in tab_widget.tabText(i):
                tab_widget.setCurrentIndex(i)
                break


class PaletteFeature(QtWidgets.QWidget):
    place_requested = QtCore.Signal(str)  # block type_id

    def __init__(self, palette_widget: QtWidgets.QWidget, parent=None):
        super().__init__(parent)
        prettify_palette_tabs(palette_widget)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        from ai_made_easy.ui.features.missions import MissionsPanel

        self.missions = MissionsPanel()
        layout.addWidget(self.missions)

        self.box = QtWidgets.QLineEdit()
        self.box.setPlaceholderText("Search blocks…  (⏎ places on canvas)")
        self.box.setClearButtonEnabled(True)
        layout.addWidget(self.box)

        self.results = QtWidgets.QListWidget()
        self.results.setMaximumHeight(220)
        self.results.setVisible(False)
        layout.addWidget(self.results)

        layout.addWidget(palette_widget, stretch=1)

        self.box.textChanged.connect(self._search)
        self.box.returnPressed.connect(self._place_first)
        self.results.itemDoubleClicked.connect(
            lambda item: self.place_requested.emit(
                item.data(QtCore.Qt.ItemDataRole.UserRole)))

    def _search(self, text: str) -> None:
        self.results.clear()
        hits = get_registry().search(text)
        for block in hits:
            item = QtWidgets.QListWidgetItem(
                f"{block.display_name}  ·  {block.category}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, block.type_id)
            self.results.addItem(item)
        self.results.setVisible(bool(hits) and bool(text.strip()))
        if hits:
            self.results.setCurrentRow(0)

    def _place_first(self) -> None:
        if not self.results.count():
            return
        item = self.results.currentItem()
        if item is not None:
            type_id = item.data(QtCore.Qt.ItemDataRole.UserRole)
            self.box.clear()
            self.results.clear()
            self.results.setVisible(False)
            self.place_requested.emit(type_id)
