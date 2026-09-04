"""ThemeService: swappable app themes (Ryven Design-hub logic, no globals).

One instance per app, held by the AppContext. Every theme defines palette
anchors; the QSS and canvas colors are derived from them.

Default theme is "classroom" — the Sunlit Classroom look: warm light chrome
that flows straight into the dusty graph-paper canvas (one continuous
notebook world), big rounded controls and large type for young learners.
"""
from __future__ import annotations

# The canvas look is a design constant (like the block family palette):
# dusty-white graph paper — fine grey lines every 50px, darker major lines
# (the scene derives them from the background colour). All app themes share
# it so switching chrome never changes the paper.
CANVAS_BG = (237, 238, 232)      # #EDEEE8 dusty white
CANVAS_GRID = (201, 203, 195)    # #C9CBC3 fine grey lines

THEMES: dict[str, dict] = {
    "classroom": {
        "BG": "#F6F4ED", "PANEL": "#FFFDF8", "INPUT": "#EFECE2",
        "BORDER": "#E1DDCF", "TEXT": "#33312B", "TEXT_DIM": "#7A7565",
        "ACCENT": "#6C5CE7", "PRIMARY_BG": "#2FA96C", "PRIMARY_HOVER": "#26935D",
        "SURFACE": "#FFFFFF",
        "CANVAS_BG": CANVAS_BG, "CANVAS_GRID": CANVAS_GRID,
    },
    "dark": {
        "BG": "#16191f", "PANEL": "#1d2129", "INPUT": "#242933",
        "BORDER": "#333a46", "TEXT": "#e6edf3", "TEXT_DIM": "#9aa4b2",
        "ACCENT": "#4a9eff", "PRIMARY_BG": "#238636", "PRIMARY_HOVER": "#2ea043",
        "SURFACE": "#20242d",
        "CANVAS_BG": CANVAS_BG, "CANVAS_GRID": CANVAS_GRID,
    },
    "light": {
        "BG": "#f5f6f8", "PANEL": "#ffffff", "INPUT": "#eef1f5",
        "BORDER": "#d0d7e2", "TEXT": "#1f2328", "TEXT_DIM": "#57606a",
        "ACCENT": "#0969da", "PRIMARY_BG": "#1f883d", "PRIMARY_HOVER": "#1a7f37",
        "SURFACE": "#ffffff",
        "CANVAS_BG": CANVAS_BG, "CANVAS_GRID": CANVAS_GRID,
    },
}

DEFAULT_THEME = "classroom"


def _build_qss(t: dict) -> str:
    return f"""
QWidget {{ background-color: {t['BG']}; color: {t['TEXT']}; font-size: 14px; }}
/* text controls must never paint page-colour patches onto card surfaces */
QLabel, QCheckBox, QRadioButton {{ background: transparent; }}
QMainWindow::separator {{ background: {t['BORDER']}; width: 2px; height: 2px; }}

#workspacePage {{ background-color: {t['BG']}; }}
QFrame[card="true"] {{
    background: {t['PANEL']}; border: 1px solid {t['BORDER']};
    border-radius: 14px;
}}
QFrame[hline="true"] {{ background: {t['BORDER']}; border: none; max-height: 1px; }}
QFrame[vline="true"] {{ background: {t['BORDER']}; border: none; max-width: 1px; }}
QFrame#issueRow {{ background: {t['INPUT']}; border-radius: 8px; }}

#microLabel {{
    color: {t['TEXT_DIM']}; font-size: 11px; font-weight: 700;
    background: transparent; padding: 0 2px;
}}
#cardTitle {{ font-size: 14px; font-weight: 800; background: transparent; }}
#bandTag {{
    color: {t['TEXT_DIM']}; font-size: 12px; font-style: italic;
    background: transparent;
}}
QLabel[chip="true"] {{
    background: {t['INPUT']}; color: {t['TEXT_DIM']};
    border: 1px solid {t['BORDER']}; border-radius: 999px;
    padding: 4px 12px; font-size: 12px; font-weight: 600;
}}

QSplitter::handle {{ background: transparent; }}
QSplitter::handle:horizontal {{ width: 8px; }}
QSplitter::handle:vertical {{ height: 8px; }}
QSplitter::handle:hover {{ background: {t['ACCENT']}; border-radius: 3px; }}

QToolBar {{
    background: {t['PANEL']}; border: none;
    border-bottom: 1px solid {t['BORDER']};
    padding: 10px 14px; spacing: 8px;
}}
QToolBar QToolButton, QToolBar QPushButton {{
    background: transparent; color: {t['TEXT']};
    border: 1px solid transparent; border-radius: 10px;
    padding: 8px 16px; font-weight: 600; min-height: 40px;
}}
QToolBar QToolButton:hover, QToolBar QPushButton:hover {{
    background: {t['INPUT']}; border-color: {t['BORDER']};
}}
QToolBar QToolButton:pressed, QToolBar QPushButton:pressed {{
    background: {t['BORDER']};
}}
QToolBar QToolButton::menu-indicator {{ image: none; }}
QToolBar QLineEdit {{
    background: {t['INPUT']}; border: 1px solid {t['BORDER']};
    border-radius: 10px; padding: 5px 12px; font-weight: 600;
    min-height: 30px; selection-background-color: {t['ACCENT']};
}}

QDockWidget {{ color: {t['TEXT_DIM']}; font-weight: 800; font-size: 13px; }}
QDockWidget::title {{ background: {t['PANEL']}; padding: 9px 12px; }}

QMenu {{ background: {t['PANEL']}; border: 1px solid {t['BORDER']}; border-radius: 10px; padding: 6px; }}
QMenu::item {{ padding: 8px 26px; border-radius: 6px; }}
QMenu::item:selected {{ background: {t['INPUT']}; color: {t['TEXT']}; }}
QMenu::item:disabled {{ color: {t['TEXT_DIM']}; font-size: 11px; font-weight: 700; }}
QMenu::separator {{ height: 1px; background: {t['BORDER']}; margin: 5px 10px; }}

QTextEdit, QPlainTextEdit, QTextBrowser {{
    background: {t['SURFACE']}; color: {t['TEXT']};
    border: 1px solid {t['BORDER']}; border-radius: 8px;
    selection-background-color: {t['ACCENT']};
}}
#sideCode {{ font-family: 'Menlo', 'Courier New', monospace; font-size: 12px; }}
QTableWidget, QTableWidget::item {{
    background: {t['SURFACE']}; alternate-background-color: {t['PANEL']};
    selection-background-color: {t['ACCENT']};
}}
QTableView QHeaderView::section, QTableWidget QHeaderView::section {{
    background: {t['INPUT']}; color: {t['TEXT_DIM']}; border: none;
    border-right: 1px solid {t['BORDER']}; padding: 6px 10px; font-weight: 600;
}}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QListWidget {{
    background: {t['INPUT']}; color: {t['TEXT']};
    border: 1px solid {t['BORDER']}; border-radius: 8px;
    padding: 6px 10px; selection-background-color: {t['ACCENT']};
}}
QListWidget {{ padding: 3px; }}
QListWidget::item {{ padding: 7px 10px; border-radius: 6px; }}
QListWidget::item:selected {{ background: {t['ACCENT']}; color: #ffffff; }}
QListWidget::item:hover {{ background: {t['INPUT']}; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {t['ACCENT']};
}}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox QAbstractItemView {{
    background: {t['INPUT']}; border: 1px solid {t['BORDER']};
    selection-background-color: {t['ACCENT']};
}}
QCheckBox {{ spacing: 9px; }}

QPushButton {{
    background: {t['INPUT']}; color: {t['TEXT']}; border: 1px solid {t['BORDER']};
    border-radius: 10px; padding: 8px 18px; font-weight: 600; min-height: 38px;
}}
QPushButton:hover {{ border-color: {t['ACCENT']}; }}
QPushButton:disabled {{ color: {t['TEXT_DIM']}; background: {t['PANEL']}; }}

#primaryBtn {{
    background: {t['PRIMARY_BG']}; color: #ffffff;
    border: 1px solid {t['PRIMARY_HOVER']}; border-radius: 12px;
    padding: 9px 26px; font-size: 15px; font-weight: 700; min-height: 40px;
}}
#primaryBtn:hover {{ background: {t['PRIMARY_HOVER']}; }}
#primaryBtn:disabled {{
    background: {t['PANEL']}; color: {t['TEXT_DIM']};
    border-color: {t['BORDER']};
}}

#appTitle {{ font-size: 18px; font-weight: 800; padding: 0 8px 0 4px; }}

#canvasControls {{
    background: {t['PANEL']}; border: 1px solid {t['BORDER']};
    border-radius: 12px;
}}
#canvasControls QPushButton {{
    background: transparent; color: {t['TEXT']};
    border: 1px solid transparent; border-radius: 9px;
    padding: 4px 10px; margin: 0; min-height: 30px; font-weight: 600;
}}
#canvasControls QPushButton:hover {{
    background: {t['INPUT']}; border-color: {t['BORDER']};
}}
#canvasControls QPushButton:pressed {{ background: {t['BORDER']}; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{
    background: {t['BORDER']}; border-radius: 5px; min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: {t['ACCENT']}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {t['BORDER']}; border-radius: 5px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}

QTabWidget::pane {{ border: none; }}
QTabBar {{ font-size: 13px; font-weight: 600; background: transparent; }}
QTabBar::tab {{
    background: transparent; color: {t['TEXT_DIM']}; padding: 7px 10px;
    margin: 5px 2px; border: 1px solid transparent; border-radius: 9px;
}}
QTabBar::tab:hover {{ background: {t['INPUT']}; color: {t['TEXT']}; }}
QTabBar::tab:selected {{
    background: {t['INPUT']}; color: {t['TEXT']};
    border: 1px solid {t['BORDER']}; font-weight: 700;
}}
QTabBar QToolButton {{ /* tab-scroll arrows stay in the family */
    background: transparent; color: {t['TEXT_DIM']};
    border: none; border-radius: 6px; padding: 2px;
}}
QTabBar QToolButton:hover {{ background: {t['INPUT']}; color: {t['TEXT']}; }}

QStatusBar {{
    background: {t['BG']}; color: {t['TEXT_DIM']};
    border-top: 1px solid {t['BORDER']}; font-size: 13px; padding: 5px 12px;
}}
QStatusBar::item {{ border: none; }}
#statusTrust {{ background: transparent; color: {t['TEXT_DIM']}; font-size: 12px; }}
QToolTip {{
    background: {t['INPUT']}; color: {t['TEXT']};
    border: 1px solid {t['BORDER']}; border-radius: 6px; padding: 6px 8px;
}}
"""


class ThemeService:
    """Owns the active theme; apply() is repeatable for live switching."""

    def __init__(self) -> None:
        self._active = DEFAULT_THEME

    def active(self) -> str:
        return self._active

    def names(self) -> list[str]:
        return list(THEMES)

    def canvas_colors(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        t = THEMES[self._active]
        return t["CANVAS_BG"], t["CANVAS_GRID"]

    def apply(self, app, name: str = DEFAULT_THEME) -> None:
        self._active = name if name in THEMES else DEFAULT_THEME
        app.setStyle("Fusion")
        app.setStyleSheet(_build_qss(THEMES[self._active]))
        font = app.font()
        font.setPointSize(12)
        app.setFont(font)


# ---- back-compat shims (grab script / older callers) ----
def apply_dark_theme(app) -> None:
    ThemeService().apply(app, DEFAULT_THEME)
