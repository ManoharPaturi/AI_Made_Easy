"""First-run tour (3.3): three short steps (library → canvas → train) on a small card beside
the part of the window it describes. It never blocks the window, can be skipped, and is
remembered (QSettings ``tour_done``). Help ▸ Show the Tour brings it back. The web app has the
same three steps (web/src/components/Tour.tsx).
"""
from __future__ import annotations

from typing import Callable

from PySide6 import QtCore, QtWidgets

STEPS = (
    ("1 · Pick blocks",
     "The library holds every block. Drag one onto the canvas, or double-click it."),
    ("2 · Wire them up",
     "Connect ports from left to right. Problems show live underneath, with one-click fixes."),
    ("3 · Train",
     "Press ▶ Train. Every run is saved and can be compared on the Experiments panel."),
)


class Tour(QtWidgets.QFrame):
    finished = QtCore.Signal()

    def __init__(self, window: QtWidgets.QMainWindow,
                 targets: list[Callable[[], QtWidgets.QWidget | None]]):
        super().__init__(window)
        self.setObjectName("tourCard")
        self.setAccessibleName("Tour")
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setStyleSheet("QFrame#tourCard { border: 1px solid palette(highlight); "
                           "border-radius: 10px; background: palette(window); }")
        self._window = window
        self._targets = targets
        self.step = 0
        self.setFixedWidth(300)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        self.counter = QtWidgets.QLabel()
        self.counter.setEnabled(False)
        self.title = QtWidgets.QLabel()
        font = self.title.font()
        font.setBold(True)
        self.title.setFont(font)
        self.text = QtWidgets.QLabel()
        self.text.setWordWrap(True)
        layout.addWidget(self.counter)
        layout.addWidget(self.title)
        layout.addWidget(self.text)
        row = QtWidgets.QHBoxLayout()
        self.skip = QtWidgets.QPushButton("Skip tour")
        self.skip.setFlat(True)
        self.back = QtWidgets.QPushButton("Back")
        self.next = QtWidgets.QPushButton("Next")
        self.next.setDefault(True)
        row.addWidget(self.skip)
        row.addStretch(1)
        row.addWidget(self.back)
        row.addWidget(self.next)
        layout.addLayout(row)
        self.skip.clicked.connect(self.finish)
        self.back.clicked.connect(lambda: self.show_step(self.step - 1))
        self.next.clicked.connect(self._advance)
        window.installEventFilter(self)
        self.show_step(0)

    def _advance(self) -> None:
        if self.step >= len(STEPS) - 1:
            self.finish()
        else:
            self.show_step(self.step + 1)

    def show_step(self, step: int) -> None:
        self.step = max(0, min(step, len(STEPS) - 1))
        title, text = STEPS[self.step]
        self.counter.setText(f"Step {self.step + 1} of {len(STEPS)}")
        self.title.setText(title)
        self.text.setText(text)
        self.back.setVisible(self.step > 0)
        self.next.setText("Done" if self.step == len(STEPS) - 1 else "Next")
        self.adjustSize()
        self.place()
        self.show()
        self.raise_()

    def place(self) -> None:
        target = self._targets[self.step]() if self.step < len(self._targets) else None
        win = self._window.rect()
        if target is not None and target.isVisible():
            top_left = target.mapTo(self._window, QtCore.QPoint(0, 0))
            rect = QtCore.QRect(top_left, target.size())
            x = rect.right() + 12
            if x + self.width() > win.right() - 12:
                x = rect.left() + 16
            y = rect.top() + 16
        else:
            x, y = 40, 80
        x = max(12, min(x, win.right() - self.width() - 12))
        y = max(12, min(y, win.bottom() - self.height() - 12))
        self.move(x, y)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802, ANN001 — Qt API
        if obj is self._window and event.type() == QtCore.QEvent.Type.Resize:
            self.place()
        return False

    def keyPressEvent(self, event) -> None:  # noqa: N802, ANN001 — Qt API
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self.finish()
        else:
            super().keyPressEvent(event)

    def finish(self) -> None:
        self._window.removeEventFilter(self)
        self.hide()
        self.finished.emit()
        self.deleteLater()
