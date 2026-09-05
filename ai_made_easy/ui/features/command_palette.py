"""CommandPalette: ⌘K quick actions (the Linear/Raycast move, kid-sized).

One floating card under the header: type a few letters, get fuzzy-matched
actions (with their shortcuts) and blocks (⏎ places one at viewport
centre, exactly like the palette search). Esc closes; ↑↓ walk; ⏎ runs.
"""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from ai_made_easy.core.registry import get_registry

_ROLE_KIND = QtCore.Qt.ItemDataRole.UserRole       # "action" | "block"
_ROLE_PAYLOAD = QtCore.Qt.ItemDataRole.UserRole + 1  # QAction | type_id
_ROLE_SECTION = QtCore.Qt.ItemDataRole.UserRole + 2  # True = header row


def fuzzy_score(text: str, query: str) -> int | None:
    """Subsequence match with bonuses for prefixes and runs.

    Bigger is better; None = no match. An empty query matches everything
    at score 0 so the palette opens as a browsable menu.
    """
    text, query = text.lower(), query.lower()
    if not query:
        return 0
    if query in text:                      # substring beats subsequence
        return 1000 - text.index(query)
    score, ti, run = 0, 0, 0
    for ch in query:
        found = text.find(ch, ti)
        if found < 0:
            return None
        score += 10
        if found == 0:
            score += 30                    # starts-with bonus
        if found == ti:
            run += 1
            score += 8 * run               # consecutive-run bonus
        else:
            run = 0
        ti = found + 1
    return score


class CommandPalette(QtWidgets.QDialog):
    """Frameless fuzzy launcher; built once by the Workbench."""

    place_requested = QtCore.Signal(str)   # block type_id

    def __init__(self, actions: dict, parent=None):
        super().__init__(parent)
        self.setObjectName("cmdPalette")
        self.setWindowFlags(QtCore.Qt.WindowType.Popup)
        self.setModal(True)
        self.setMinimumWidth(560)
        self.setMaximumHeight(420)
        self._actions = dict(actions)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 8)
        layout.setSpacing(6)

        self.input = QtWidgets.QLineEdit()
        self.input.setObjectName("cmdInput")
        self.input.setPlaceholderText("Type to search actions and blocks…")
        self.input.setFrame(False)
        layout.addWidget(self.input)

        self.list = QtWidgets.QListWidget()
        self.list.setObjectName("cmdList")
        self.list.setUniformItemSizes(False)
        layout.addWidget(self.list, 1)

        self.input.textChanged.connect(self._refill)
        self.input.returnPressed.connect(self._run_current)
        self.list.itemActivated.connect(self._run_item)
        self.list.itemClicked.connect(self._run_item)
        self._refill("")

    # ------------------------------------------------------------- api
    def open_at(self, host: QtWidgets.QWidget) -> None:
        self._refill(self.input.text())
        x = host.x() + (host.width() - self.width()) // 2
        self.move(max(x, 12), host.y() + 92)
        self.show()
        self.raise_()
        self.activateWindow()
        self.input.setFocus()
        self.input.selectAll()

    # ---------------------------------------------------------- search
    def _refill(self, query: str) -> None:
        self.list.clear()
        action_hits, block_hits = [], []
        for act in self._actions.values():
            label = act.text().replace("&", "")
            score = fuzzy_score(label, query)
            if score is None:
                continue
            hint = act.shortcut().toString(
                QtGui.QKeySequence.SequenceFormat.NativeText)
            action_hits.append((score, label, label, hint, act))
        for block in get_registry().all():
            label = f"{block.display_name}  ·  {block.category}"
            score = fuzzy_score(label, query)
            if score is None:
                continue
            block_hits.append((score, label, label, "⏎ places on paper",
                               block.type_id))
        action_hits.sort(key=lambda h: (-h[0], h[1]))
        block_hits.sort(key=lambda h: (-h[0], h[1]))

        if action_hits:
            self._section("⚡ Actions")
            self._add_rows(action_hits[:12])
        if block_hits:
            self._section("🧱 Blocks")
            self._add_rows(block_hits[:28])
        if self.list.count():
            self.list.setCurrentRow(1 if action_hits else 0)

    def _add_rows(self, hits) -> None:
        for _score, _sort_key, label, hint, payload in hits:
            item = QtWidgets.QListWidgetItem(f"{label}   {hint}")
            item.setData(_ROLE_KIND,
                         "action" if isinstance(payload, QtGui.QAction)
                         else "block")
            item.setData(_ROLE_PAYLOAD, payload)
            self.list.addItem(item)

    def _section(self, title: str | None) -> None:
        if not title:
            return
        item = QtWidgets.QListWidgetItem(title)
        item.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
        item.setData(_ROLE_SECTION, True)
        self.list.addItem(item)

    # ---------------------------------------------------------- run
    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt name)
        if event.key() in (QtCore.Qt.Key.Key_Escape,):
            self.close()
            return
        if event.key() in (QtCore.Qt.Key.Key_Down, QtCore.Qt.Key.Key_Up):
            QtWidgets.QApplication.sendEvent(self.list, event)
            return
        super().keyPressEvent(event)

    def _run_current(self) -> None:
        item = self.list.currentItem()
        if item is not None:
            self._run_item(item)

    def _run_item(self, item: QtWidgets.QListWidgetItem) -> None:
        kind = item.data(_ROLE_KIND)
        payload = item.data(_ROLE_PAYLOAD)
        self.close()
        if kind == "action" and isinstance(payload, QtGui.QAction):
            payload.trigger()
        elif kind == "block":
            self.place_requested.emit(payload)
