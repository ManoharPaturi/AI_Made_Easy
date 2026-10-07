"""Render the application icon (PNG, plus .icns on macOS) into ai_made_easy/assets."""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

ASSETS = Path(__file__).resolve().parent.parent / "ai_made_easy" / "assets"


def render(size: int) -> QtGui.QImage:
    img = QtGui.QImage(size, size, QtGui.QImage.Format.Format_ARGB32)
    img.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(img)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    s = size / 1024.0
    grad = QtGui.QLinearGradient(0, 0, size, size)
    grad.setColorAt(0, QtGui.QColor("#3d7cff"))
    grad.setColorAt(1, QtGui.QColor("#6f4bff"))
    p.setPen(QtCore.Qt.PenStyle.NoPen)
    p.setBrush(grad)
    p.drawRoundedRect(QtCore.QRectF(64 * s, 64 * s, 896 * s, 896 * s), 200 * s, 200 * s)
    # three connected blocks: input -> layer -> output
    pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 230), 34 * s)
    pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    path = QtGui.QPainterPath(QtCore.QPointF(300 * s, 340 * s))
    path.cubicTo(420 * s, 340 * s, 400 * s, 512 * s, 512 * s, 512 * s)
    path.moveTo(512 * s, 512 * s)
    path.cubicTo(624 * s, 512 * s, 604 * s, 684 * s, 724 * s, 684 * s)
    p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    p.drawPath(path)
    p.setPen(QtCore.Qt.PenStyle.NoPen)
    for cx, cy, color in ((300, 340, "#ffffff"), (512, 512, "#ffd166"), (724, 684, "#ffffff")):
        p.setBrush(QtGui.QColor(color))
        p.drawRoundedRect(QtCore.QRectF((cx - 95) * s, (cy - 70) * s, 190 * s, 140 * s),
                          36 * s, 36 * s)
    p.end()
    return img


def main() -> int:
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    ASSETS.mkdir(exist_ok=True)
    render(1024).save(str(ASSETS / "icon.png"))
    render(256).save(str(ASSETS / "icon_256.png"))
    if sys.platform == "darwin" and shutil.which("iconutil"):
        with tempfile.TemporaryDirectory() as tmp:
            iconset = Path(tmp) / "icon.iconset"
            iconset.mkdir()
            for size in (16, 32, 64, 128, 256, 512):
                render(size).save(str(iconset / f"icon_{size}x{size}.png"))
                render(size * 2).save(str(iconset / f"icon_{size}x{size}@2x.png"))
            subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o",
                            str(ASSETS / "icon.icns")], check=True)
    print("icons written to", ASSETS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
