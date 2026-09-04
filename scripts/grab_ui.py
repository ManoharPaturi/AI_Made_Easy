"""Render the app window to PNGs (for design iteration) and exit.

Usage: python scripts/grab_ui.py [out_prefix]
"""
from __future__ import annotations

import sys
from pathlib import Path

from ai_made_easy.ui.app import _ensure_qt_plugin_path

_ensure_qt_plugin_path()

from PySide6 import QtCore, QtWidgets  # noqa: E402

from ai_made_easy.ui.context import AppContext  # noqa: E402
from ai_made_easy.ui.theme import apply_dark_theme  # noqa: E402
from ai_made_easy.ui.workbench import Workbench  # noqa: E402


def main() -> int:
    prefix = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/aime_ui")
    app = QtWidgets.QApplication(sys.argv[:1])
    apply_dark_theme(app)
    win = Workbench(AppContext())
    win.resize(1500, 900)
    win.show()

    def capture() -> None:
        # capture the true boot view — no extra zooming
        QtCore.QTimer.singleShot(600, shoot)  # let cached pixmaps repaint

    def shoot() -> None:
        win.grab().save(f"{prefix}_full.png")
        win.ctx.canvas_area.grab().save(f"{prefix}_canvas.png")
        print(f"saved {prefix}_full.png and {prefix}_canvas.png", flush=True)
        app.quit()

    QtCore.QTimer.singleShot(4000, capture)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
