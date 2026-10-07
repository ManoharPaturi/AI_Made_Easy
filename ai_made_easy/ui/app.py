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


def data_dir() -> Path:
    """Per-user application data: logs, autosave, custom blocks."""
    path = Path.home() / ".aime"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _setup_logging() -> Path:
    import logging
    from logging.handlers import RotatingFileHandler

    log_dir = data_dir() / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "ai_made_easy.log"
    handler = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3,
                                  encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    return log_file


def _install_excepthook(log_file: Path) -> None:
    """Log unexpected exceptions and show them instead of failing silently."""
    import logging
    import traceback

    def hook(exc_type, exc, tb) -> None:  # noqa: ANN001
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        logging.getLogger("ai_made_easy").error("unhandled exception\n%s", text)
        try:
            from PySide6 import QtWidgets

            if QtWidgets.QApplication.instance() is not None:
                box = QtWidgets.QMessageBox()
                box.setIcon(QtWidgets.QMessageBox.Icon.Critical)
                box.setWindowTitle("Unexpected error")
                box.setText(f"{exc_type.__name__}: {exc}")
                box.setInformativeText(f"The full report was written to {log_file}.")
                box.setDetailedText(text)
                box.exec()
        except Exception:  # noqa: BLE001 — never raise from the hook
            sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = hook


def run(argv: list[str] | None = None) -> int:
    _ensure_qt_plugin_path()
    log_file = _setup_logging()

    from PySide6 import QtGui, QtWidgets

    from ai_made_easy import __version__
    from ai_made_easy.ui.context import AppContext
    from ai_made_easy.ui.workbench import Workbench

    app = QtWidgets.QApplication(argv or sys.argv)
    app.setApplicationName("AI Made Easy")
    app.setOrganizationName("AI Made Easy")
    app.setApplicationVersion(__version__)
    icon = Path(__file__).resolve().parent.parent / "assets" / "icon.png"
    if icon.exists():
        app.setWindowIcon(QtGui.QIcon(str(icon)))
    _install_excepthook(log_file)

    ctx = AppContext()
    window = Workbench(ctx)
    window.show()
    return app.exec()
