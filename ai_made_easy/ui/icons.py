"""Vector icon set: 24×24 stroke icons rendered in the active theme colour.

Icons are SVG path data (drawn for this application) rendered through
QSvgRenderer, so they stay crisp at any device-pixel ratio.
"""
from __future__ import annotations

from functools import lru_cache

_PATHS: dict[str, str] = {
    "play": '<path d="M7 5v14l11-7z"/>',
    "stop": '<rect x="6" y="6" width="12" height="12" rx="1.5"/>',
    "check": '<circle cx="12" cy="12" r="9"/><path d="M8 12.5l2.7 2.7L16.5 9.5"/>',
    "zap": '<path d="M13 3L5 14h6l-1 7 8-11h-6z"/>',
    "download": '<path d="M12 4v11"/><path d="M7.5 10.5L12 15l4.5-4.5"/><path d="M5 19h14"/>',
    "code": '<path d="M9 7l-5 5 5 5"/><path d="M15 7l5 5-5 5"/>',
    "layers": '<path d="M12 4l8 4-8 4-8-4z"/><path d="M4 12l8 4 8-4"/><path d="M4 16l8 4 8-4"/>',
    "search": '<circle cx="11" cy="11" r="6"/><path d="M20 20l-4.5-4.5"/>',
    "warning": '<path d="M12 4l9 16H3z"/><path d="M12 10v4"/><path d="M12 17.2v.3"/>',
    "error": '<circle cx="12" cy="12" r="9"/><path d="M9 9l6 6M15 9l-6 6"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6"/><path d="M12 7.5v.3"/>',
    "save": '<path d="M5 4h11l3 3v13H5z"/><path d="M8 4v5h7V4"/><path d="M8 20v-6h8v6"/>',
    "open": '<path d="M4 6h6l2 2h8v11H4z"/>',
    "new": '<path d="M6 3h8l4 4v14H6z"/><path d="M14 3v4h4"/><path d="M12 11v6M9 14h6"/>',
    "undo": '<path d="M9 7L4 12l5 5"/><path d="M4 12h10a5 5 0 010 10h-2"/>',
    "redo": '<path d="M15 7l5 5-5 5"/><path d="M20 12H10a5 5 0 000 10h2"/>',
    "zoom_in": '<circle cx="11" cy="11" r="6"/><path d="M20 20l-4.5-4.5M11 8v6M8 11h6"/>',
    "zoom_out": '<circle cx="11" cy="11" r="6"/><path d="M20 20l-4.5-4.5M8 11h6"/>',
    "fit": '<path d="M4 9V4h5M15 4h5v5M20 15v5h-5M9 20H4v-5"/>',
    "image": '<rect x="4" y="5" width="16" height="14" rx="2"/><circle cx="9" cy="10" r="1.6"/>'
             '<path d="M5 18l5-5 4 4 2-2 3 3"/>',
    "chart": '<path d="M4 20h16"/><path d="M7 16v-5M12 16V7M17 16v-8"/>',
    "settings": '<circle cx="12" cy="12" r="3"/><path d="M12 3v3M12 18v3M3 12h3M18 12h3'
                'M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M5.6 18.4l2.1-2.1M16.3 7.7l2.1-2.1"/>',
    "terminal": '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M7 10l3 2-3 2M12 15h5"/>',
    "list": '<path d="M9 6h11M9 12h11M9 18h11"/><path d="M4.5 6h.5M4.5 12h.5M4.5 18h.5"/>',
    "wand": '<path d="M4 20L15 9"/><path d="M14 4v2M19 9h2M17.5 5.5l1.5-1.5M18 12l1 1M11 5l1 1"/>',
    "eye": '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
    "card": '<rect x="4" y="4" width="16" height="16" rx="2"/><path d="M8 9h8M8 13h8M8 17h5"/>',
    "block": '<rect x="4" y="6" width="16" height="12" rx="2.5"/><path d="M4 10h16"/>',
    "expand": '<rect x="3" y="9" width="6" height="6" rx="1"/><rect x="15" y="3" width="6" height="6" rx="1"/>'
              '<rect x="15" y="15" width="6" height="6" rx="1"/><path d="M9 12h3M12 6v12M12 6h3M12 18h3"/>',
    "plus_box": '<rect x="4" y="4" width="16" height="16" rx="2.5"/><path d="M12 8v8M8 12h8"/>',
    "sun": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4'
           'M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
    "moon": '<path d="M20 14.5A8 8 0 019.5 4 8 8 0 1020 14.5z"/>',
    "keyboard": '<rect x="2" y="6" width="20" height="12" rx="2"/>'
                '<path d="M6 10h.01M10 10h.01M14 10h.01M18 10h.01M7 14h10"/>',
    "help": '<circle cx="12" cy="12" r="9"/><path d="M9.5 9.5a2.5 2.5 0 015 .5c0 1.6-2.5 2-2.5 4"/>'
            '<path d="M12 17.3v.2"/>',
    "copy": '<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V5a1 1 0 00-1-1H5a1 1 0 00-1 1v10a1 1 0 001 1h3"/>',
    "trash": '<path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/>',
    "folder": '<path d="M4 6h6l2 2h8v11H4z"/>',
    "cpu": '<rect x="6" y="6" width="12" height="12" rx="2"/><rect x="9.5" y="9.5" width="5" height="5"/>'
           '<path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>',
    "package": '<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M4 7.5l8 4.5 8-4.5M12 12v9"/>',
    "palette": '<circle cx="12" cy="12" r="9"/><circle cx="8" cy="10" r="1"/><circle cx="12" cy="7.5" r="1"/>'
               '<circle cx="16" cy="10" r="1"/><path d="M12 21a2 2 0 010-4h1.5a2.5 2.5 0 000-5"/>',
    "sparkle": '<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/><path d="M19 17v4M17 19h4"/>',
    "dataset": '<ellipse cx="12" cy="6" rx="7" ry="3"/><path d="M5 6v12c0 1.7 3.1 3 7 3s7-1.3 7-3V6"/>'
               '<path d="M5 12c0 1.7 3.1 3 7 3s7-1.3 7-3"/>',
}

_COLORS = {"normal": "#a0a6af", "inverse": "#ffffff"}


def set_color(normal: str, inverse: str = "#ffffff") -> None:
    """Called by ThemeService when the theme changes."""
    _COLORS["normal"], _COLORS["inverse"] = normal, inverse
    _pixmap.cache_clear()


def names() -> list[str]:
    return list(_PATHS)


def svg(name: str, color: str) -> str:
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="1.7" stroke-linecap="round" '
            f'stroke-linejoin="round">{_PATHS[name]}</svg>')


@lru_cache(maxsize=512)
def _pixmap(name: str, color: str, size: int, dpr: float):
    from PySide6 import QtCore, QtGui, QtSvg

    renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(svg(name, color).encode()))
    pm = QtGui.QPixmap(int(size * dpr), int(size * dpr))
    pm.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pm)
    renderer.render(painter)
    painter.end()
    pm.setDevicePixelRatio(dpr)
    return pm


def icon(name: str, color: str | None = None, size: int = 18):
    """QIcon for ``name`` in the theme colour (or an explicit colour)."""
    from PySide6 import QtGui, QtWidgets

    app = QtWidgets.QApplication.instance()
    dpr = app.devicePixelRatio() if app is not None else 2.0
    ic = QtGui.QIcon()
    ic.addPixmap(_pixmap(name, color or _COLORS["normal"], size, max(dpr, 2.0)))
    return ic
