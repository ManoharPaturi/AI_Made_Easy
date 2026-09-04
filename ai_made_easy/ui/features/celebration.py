"""CelebrationOverlay: a brief confetti moment when a run finishes.

Delight matters for young learners — finishing a training run should feel
like an achievement, not a log line. Pure QWidget painted with QPainter,
auto-fades after a few seconds; no external deps.
"""
from __future__ import annotations

import random

from PySide6 import QtCore, QtGui, QtWidgets

_EMOJI = ["🎉", "⭐", "🧠", "✨", "🏆", "💡"]
_PALETTE = ["#F5D547", "#FF8787", "#74C0FC", "#63E6BE", "#D0BFFF", "#F783AC"]


class _Piece:
    __slots__ = ("x", "y", "vy", "vx", "emoji", "color", "spin")

    def __init__(self, w: int, h: int):
        self.x = random.uniform(0, w)
        self.y = random.uniform(-h * 0.6, 0)
        self.vy = random.uniform(2.2, 4.6)
        self.vx = random.uniform(-0.9, 0.9)
        self.emoji = random.choice(_EMOJI)
        self.color = random.choice(_PALETTE)
        self.spin = random.uniform(-14, 14)


class CelebrationOverlay(QtWidgets.QWidget):
    """Transparent full-parent overlay with a headline and falling confetti."""

    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground)
        self._message = ""
        self._sub = ""
        self._pieces: list[_Piece] = []
        self._timer = QtCore.QTimer(self, interval=33, timeout=self._tick)
        self._ticks = 0

    def celebrate(self, message: str, sub: str = "") -> None:
        self._message = message
        self._sub = sub
        self._ticks = 0
        self.resize(self.parentWidget().size())
        self._pieces = [_Piece(self.width(), self.height()) for _ in range(46)]
        self.raise_()
        self.show()
        self._timer.start()

    def _tick(self) -> None:
        self._ticks += 1
        for p in self._pieces:
            p.y += p.vy
            p.x += p.vx
        if self._ticks > 150:  # ~5 s
            self._timer.stop()
            self.hide()
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802, ANN001
        if not self._pieces:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        fade = max(0.0, 1.0 - max(self._ticks - 110, 0) / 40.0)

        # headline card
        card_w, card_h = min(460, self.width() - 40), 96
        rect = QtCore.QRectF((self.width() - card_w) / 2, 18, card_w, card_h)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(255, 253, 248, int(244 * fade)))
        painter.drawRoundedRect(rect, 18, 18)
        painter.setPen(QtGui.QColor(51, 49, 43, int(255 * fade)))
        font = QtGui.QFont()
        font.setPointSizeF(17)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(rect.adjusted(0, 8, 0, 0),
                         QtCore.Qt.AlignmentFlag.AlignHCenter
                         | QtCore.Qt.AlignmentFlag.AlignTop, self._message)
        font.setPointSizeF(11)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(122, 117, 101, int(255 * fade)))
        painter.drawText(rect.adjusted(0, 42, 0, 0),
                         QtCore.Qt.AlignmentFlag.AlignHCenter
                         | QtCore.Qt.AlignmentFlag.AlignTop, self._sub)

        # confetti
        emoji_font = QtGui.QFont()
        emoji_font.setPointSizeF(15)
        painter.setFont(emoji_font)
        for p in self._pieces:
            alpha = int(255 * fade)
            painter.setPen(QtGui.QColor(31, 35, 40, alpha))
            painter.drawText(QtCore.QPointF(p.x, p.y), p.emoji)
        painter.end()
