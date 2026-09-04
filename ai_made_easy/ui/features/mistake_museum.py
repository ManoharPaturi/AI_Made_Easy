"""Mistake Museum: learn from what the model got wrong.

Research-backed (IDC'19 youth ML debugging; IDC'21 ages 7-13): children
diagnose models by studying misclassified examples with visible confidence.
Two tabs — 🔍 Mistakes (cards with remedy chips) and 📊 All predictions
(thumbnail grid with confidence bars). Dumb dialog: reads the run's
workdir, emits nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

_REMEDIES = [
    ("too few examples",
     "Take 10 more photos of this class — models need lots of examples "
     "to learn a look."),
    ("bad lighting",
     "Mix in brighter and darker photos so the model learns the shape, "
     "not the light."),
    ("too similar to another class",
     "These two classes look almost the same — pick clearer differences, "
     "or merge them into one."),
    ("weird background",
     "Add photos with different backgrounds — the model may be learning "
     "the background, not the object! Check 👀 What is it looking at?"),
]


def _bars(probs: list, correct: bool | None = None) -> QtWidgets.QWidget:
    host = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(2)
    for cls, prob in sorted(enumerate(probs), key=lambda kv: -kv[1])[:3]:
        row = QtWidgets.QHBoxLayout()
        name = QtWidgets.QLabel(f"class {cls}")
        name.setFixedWidth(70)
        row.addWidget(name, 0)
        bar = QtWidgets.QProgressBar()
        bar.setRange(0, 1000)
        bar.setValue(int(prob * 1000))
        bar.setTextVisible(False)
        bar.setFixedHeight(10)
        bar.setStyleSheet(
            "QProgressBar { background: #EFECE2; border-radius: 5px; }"
            "QProgressBar::chunk { border-radius: 5px; background: %s; }"
            % ("#2FA96C" if correct else "#E5484D"))
        row.addWidget(bar, 1)
        pct = QtWidgets.QLabel(f"{prob:.0%}")
        pct.setFixedWidth(36)
        row.addWidget(pct, 0)
        layout.addLayout(row)
    return host


class MistakeCard(QtWidgets.QFrame):
    def __init__(self, item: dict, parent=None):
        super().__init__(parent)
        self.setProperty("card", True)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        head = QtWidgets.QHBoxLayout()
        if item.get("file"):
            pic = QtWidgets.QLabel()
            pix = QtGui.QPixmap(item["file"])
            if not pix.isNull():
                pic.setPixmap(pix.scaled(
                    96, 96, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation))
                head.addWidget(pic)
        else:
            placeholder = QtWidgets.QLabel(f"🔢\n#{item['index']}")
            placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            placeholder.setFixedSize(64, 64)
            placeholder.setStyleSheet(
                "background: #EFECE2; border-radius: 8px; font-size: 11px;")
            head.addWidget(placeholder)
        guessed = max(range(len(item["probs"])), key=lambda i: item["probs"][i])
        label = QtWidgets.QLabel(
            f"✖ example {item['index']}\n"
            f"actual: class {item['true']}\n"
            f"guessed: class {guessed} ({item['probs'][guessed]:.0%})")
        label.setStyleSheet("font-weight: 600;")
        head.addWidget(label, 1)
        layout.addLayout(head)

        layout.addWidget(_bars(item["probs"], correct=False))

        tip = QtWidgets.QLabel("What went wrong? Tag it:")
        tip.setStyleSheet("color: #7A7565; font-size: 12px;")
        layout.addWidget(tip)
        chips = QtWidgets.QHBoxLayout()
        self.advice = QtWidgets.QLabel("")
        self.advice.setWordWrap(True)
        self.advice.setStyleSheet(
            "background: #EFECE2; border-radius: 8px; padding: 6px;")
        for name, text in _REMEDIES:
            chip = QtWidgets.QToolButton()
            chip.setText(name)
            chip.setCheckable(True)
            chip.clicked.connect(lambda _=False, t=text, c=chip:
                                 self._show_advice(t, c))
            chips.addWidget(chip)
        layout.addLayout(chips)
        layout.addWidget(self.advice)

    def _show_advice(self, text: str, chip) -> None:  # noqa: ANN001
        self.advice.setText(f"💡 {text}" if chip.isChecked() else "")


class PredictionRow(QtWidgets.QFrame):
    def __init__(self, item: dict, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        if item.get("file"):
            pic = QtWidgets.QLabel()
            pix = QtGui.QPixmap(item["file"])
            if not pix.isNull():
                pic.setPixmap(pix.scaled(
                    48, 48, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation))
                layout.addWidget(pic)
        guessed = max(range(len(item["probs"])), key=lambda i: item["probs"][i])
        ok = guessed == item["true"]
        icon = "✅" if ok else "❌"
        layout.addWidget(QtWidgets.QLabel(
            f"{icon} class {item['true']} → {guessed} "
            f"({item['probs'][guessed]:.0%})"))
        layout.addStretch(1)
        best = max(item["probs"])
        bar = QtWidgets.QProgressBar()
        bar.setRange(0, 1000)
        bar.setValue(int(best * 1000))
        bar.setFixedWidth(120)
        bar.setFixedHeight(8)
        bar.setTextVisible(False)
        bar.setStyleSheet(
            "QProgressBar { background: #EFECE2; border-radius: 4px; }"
            "QProgressBar::chunk { border-radius: 4px; background: %s; }"
            % ("#2FA96C" if ok else "#E5484D"))
        layout.addWidget(bar)


class MistakeMuseumDialog(QtWidgets.QDialog):
    def __init__(self, parent, workdir: Path):
        super().__init__(parent)
        self.setWindowTitle("🔍 Mistake Museum — learn from mistakes")
        self.setModal(True)
        self.resize(900, 560)
        self.workdir = Path(workdir)

        layout = QtWidgets.QVBoxLayout(self)
        tabs = QtWidgets.QTabWidget()

        # ---- mistakes tab
        mistakes = self._read("mistakes.json")
        mistakes_host = QtWidgets.QScrollArea()
        mistakes_host.setWidgetResizable(True)
        inner = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(inner)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setSpacing(8)
        for i, item in enumerate(mistakes):
            grid.addWidget(MistakeCard(item), i // 2, i % 2)
        if not mistakes:
            grid.addWidget(QtWidgets.QLabel(
                "🎉 Nothing in the museum — the model got every checked "
                "example right!"), 0, 0)
        mistakes_host.setWidget(inner)
        tabs.addTab(mistakes_host, f"🔍 Mistakes ({len(mistakes)})")

        # ---- all predictions tab
        preds = self._read("predictions.json")
        preds_host = QtWidgets.QScrollArea()
        preds_host.setWidgetResizable(True)
        pinner = QtWidgets.QWidget()
        playout = QtWidgets.QVBoxLayout(pinner)
        playout.setContentsMargins(8, 8, 8, 8)
        for item in preds[:120]:
            playout.addWidget(PredictionRow(item))
        preds_host.setWidget(pinner)
        tabs.addTab(preds_host, f"📊 All predictions ({len(preds)})")

        layout.addWidget(tabs)
        close = QtWidgets.QPushButton("Back to building 🔧")
        close.clicked.connect(self.accept)
        layout.addWidget(close)

    def _read(self, name: str) -> list:
        path = self.workdir / name
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return []
