"""Application entrypoint: ``python -m ai_made_easy``.

Owns only bootstrap concerns — the macOS/iCloud Qt-plugin workaround,
QApplication creation, theme, context + workbench composition.
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path


def _fresh_copytree(src: Path, dst: Path) -> None:
    """Copy with real byte writes.

    shutil.copytree uses fcopyfile/APFS clones on macOS, and Qt's plugin
    loader has been observed refusing clone-copied dylibs on iCloud-managed
    volumes. Plain chunked writes produce ordinary files that load fine.
    """
    dst.mkdir(parents=True, exist_ok=True)
    for item in sorted(src.rglob("*")):
        target = dst / item.relative_to(src)
        if item.is_dir():
            target.mkdir(exist_ok=True)
        elif item.is_file():
            with open(item, "rb") as fsrc, open(target, "wb") as fdst:
                while chunk := fsrc.read(1024 * 1024):
                    fdst.write(chunk)
            os.chmod(target, stat.S_IMODE(item.stat().st_mode))


def _ensure_qt_plugin_path() -> None:
    """Help Qt find its platform plugins.

    On iCloud-synced folders (Desktop & Documents) Qt's plugin directory
    scan can intermittently return empty. When PySide6 lives under such a
    folder we mirror the plugins tree into /tmp and point Qt there.
    No-op when the env variable is already set.
    """
    if os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH"):
        return
    try:
        import PySide6

        platforms = Path(PySide6.__file__).parent / "Qt" / "plugins" / "platforms"
        if not platforms.is_dir():
            return
        home = Path.home().resolve()
        resolved = platforms.resolve()
        icloud_roots = (home / "Desktop", home / "Documents")
        if any(str(resolved).startswith(str(r)) for r in icloud_roots):
            # We mirror the *whole* plugins tree — Qt also resolves sibling
            # plugin categories (styles, imageformats) relative to this path.
            cached = Path("/tmp") / f"aime-qt-plugins-{PySide6.__version__}"
            if not (cached / "platforms" / "libqcocoa.dylib").exists():
                _fresh_copytree(platforms.parent, cached)
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(cached / "platforms")
        else:
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platforms)
    except Exception:
        pass


def run(argv: list[str] | None = None) -> int:
    _ensure_qt_plugin_path()

    from PySide6 import QtWidgets

    from ai_made_easy.ui.context import AppContext
    from ai_made_easy.ui.theme import apply_dark_theme
    from ai_made_easy.ui.workbench import Workbench

    app = QtWidgets.QApplication(argv or sys.argv)
    app.setApplicationName("AI Made Easy")
    app.setOrganizationName("AI Made Easy")
    apply_dark_theme(app)

    ctx = AppContext()
    window = Workbench(ctx)
    window.show()
    return app.exec()
