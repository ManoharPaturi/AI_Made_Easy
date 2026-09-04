"""Predict-before-training (POE / hypercorrection pedagogy).

Before a valid run, learners commit to a guess (terrible / okay / great).
After the run, a confidently-wrong guess triggers the 😲 Surprise dialog —
surprises are where learning happens (and the museum/meter show WHY).
"""
from __future__ import annotations

from PySide6 import QtWidgets

# accuracy bands shared by the guess mapping and the surprise check
GREAT = 0.85
OKAY = 0.60

_CHOICES = [
    ("terrible", "😵 Terrible", "barely better than guessing"),
    ("okay", "🙂 Okay", "gets most simple ones right"),
    ("great", "🤩 Great", "nearly always right"),
]


def score_band(score: float | None) -> str | None:
    """Map a 0–1 (or 0–100) accuracy to terrible/okay/great."""
    if score is None:
        return None
    s = score / 100.0 if score > 1.5 else score
    if s >= GREAT:
        return "great"
    if s >= OKAY:
        return "okay"
    return "terrible"


def is_surprising(guess: str | None, score: float | None) -> bool:
    """Only confidently-wrong guesses surprise (one band off is fine)."""
    actual = score_band(score)
    if not guess or actual is None:
        return False
    order = {"terrible": 0, "okay": 1, "great": 2}
    return abs(order[guess] - order[actual]) == 2


class PredictGateDialog(QtWidgets.QDialog):
    """One question before the run: how well will it do?"""

    def __init__(self, parent, project_name: str):  # noqa: ANN001
        super().__init__(parent)
        self.setWindowTitle("🔮 Predict first!")
        self.setModal(True)
        self.guess: str | None = None

        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel(
            f"Before we train “{project_name}” — make a prediction.\n"
            "How well will this model do on new examples?")
        title.setWordWrap(True)
        layout.addWidget(title)

        for value, label, blurb in _CHOICES:
            btn = QtWidgets.QPushButton(f"{label} — {blurb}")
            btn.setMinimumHeight(44)
            btn.clicked.connect(lambda _=False, v=value: self._pick(v))
            layout.addWidget(btn)

        skip = QtWidgets.QPushButton("just train, no guess")
        skip.setStyleSheet("color: #7A7565;")
        skip.clicked.connect(self.reject)
        layout.addWidget(skip)

    def _pick(self, value: str) -> None:
        self.guess = value
        self.accept()


_SURPRISE_HINTS = {
    ("great", "terrible"): [
        "Models can't read minds — they need patterns with enough examples.",
        "Check 🔍 Mistake Museum: what IS it getting wrong?",
        "Check 🩺 dataset health (double-click your data block): too few "
        "photos or very uneven classes?",
    ],
    ("terrible", "great"): [
        "Some patterns are genuinely easy — simple shapes separate cleanly.",
        "Great data beats big models. You built a clean dataset!",
        "Try 🔴 Live — does it work on YOUR examples too, not just the test?",
    ],
}


class SurpriseDialog(QtWidgets.QDialog):
    """😲 Your prediction was way off — here's the interesting part."""

    def __init__(self, parent, guess: str, score: float | None):  # noqa: ANN001
        super().__init__(parent)
        self.setWindowTitle("😲 Surprise!")
        self.setModal(True)
        actual = score_band(score)
        shown = score * 100 if (score or 0) <= 1.5 else score
        layout = QtWidgets.QVBoxLayout(self)
        head = QtWidgets.QLabel(
            f"You predicted <b>{guess}</b> — it came out <b>{actual}</b> "
            f"(about {shown:.0f}% right).")
        head.setWordWrap(True)
        layout.addWidget(head)
        why = QtWidgets.QLabel("Real surprises are the best way to learn. "
                               "Where to look:")
        why.setStyleSheet("font-weight: 700;")
        layout.addWidget(why)
        for hint in _SURPRISE_HINTS.get((guess, actual), []):
            line = QtWidgets.QLabel(f"💡 {hint}")
            line.setWordWrap(True)
            layout.addWidget(line)
        close = QtWidgets.QPushButton("interesting… 👍")
        close.clicked.connect(self.accept)
        layout.addWidget(close)
