"""Render the workbench to PNG files (design review / documentation).

Usage: python scripts/grab_ui.py [out_prefix] [dark|light]
"""
from __future__ import annotations

import sys
from pathlib import Path

from ai_made_easy.ui.app import _ensure_qt_plugin_path

_ensure_qt_plugin_path()

from PySide6 import QtCore, QtWidgets  # noqa: E402


def main() -> int:
    prefix = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/aime_ui")
    theme = sys.argv[2] if len(sys.argv) > 2 else "dark"
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setOrganizationName("AI Made Easy")
    app.setApplicationName("AI Made Easy (capture)")
    from ai_made_easy.ui.context import AppContext
    from ai_made_easy.ui.workbench import Workbench

    ctx = AppContext()
    win = Workbench(ctx)
    win.resize(1600, 960)
    win.show()

    def select_first() -> None:
        ctx._apply_theme(theme, announce=False)
        ids = [n.id for n in ctx.canvas.node_graph.all_nodes()]
        conv = [i for i in ids if ctx.canvas.block_of(i) and
                ctx.canvas.block_of(i).type_id == "core.conv2d"]
        if conv:
            ctx._locate(conv[0])
            ctx.canvas.center_view()

    def shoot() -> None:
        ctx.project_store.mark_clean()
        win.grab().save(f"{prefix}_full.png")
        print(f"saved {prefix}_full.png", flush=True)
        app.quit()

    QtCore.QTimer.singleShot(1500, select_first)
    QtCore.QTimer.singleShot(3500, shoot)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
