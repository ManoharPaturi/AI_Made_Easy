"""Actions catalog: actions as DATA (Langflow convention).

Every menu action + shortcut is declared once here; the Workbench builds
QActions from specs, menus assemble them, the ShortcutsDialog documents
them, and the completeness test asserts every slot resolves on the context.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ActionSpec:
    id: str                 # stable objectName (Orange convention)
    text: str
    menu: str               # "" = toolbar/header only
    slot: str               # method name on AppContext
    shortcut: str = ""
    tooltip: str = ""
    checkable: bool = False
    separator_after: bool = False


CATALOG: list[ActionSpec] = [
    # ---- File ----
    ActionSpec("file.new", "&New Project", "&File", "act_new", "Ctrl+N"),
    ActionSpec("file.open", "&Open Project...", "&File", "act_open", "Ctrl+O"),
    ActionSpec("file.save", "&Save Project", "&File", "act_save", "Ctrl+S"),
    ActionSpec("file.save_as", "Save Project &As...", "&File", "act_save_as",
               "Ctrl+Shift+S"),
    ActionSpec("file.samples", "Sample &Projects...", "&File", "act_samples",
               "Ctrl+Shift+O",
               "One-click editable starter graphs", separator_after=True),
    ActionSpec("file.export_png", "Export Canvas as &PNG...", "&File",
               "act_export_png", "", "Share the graph as an image",
               separator_after=True),
    ActionSpec("file.export_bundle", "Export .aime &Bundle...", "&File",
               "act_export_bundle", "",
               "One shareable file: graph + card + dataset + checkpoint"),
    ActionSpec("file.open_bundle", "&Open a Friend's Bundle...", "&File",
               "act_open_bundle", "",
               "Load their project or swap-test their model on your photos",
               separator_after=True),
    ActionSpec("file.quit", "&Quit", "&File", "act_quit", "Ctrl+Q"),

    # ---- Edit ----
    ActionSpec("edit.undo", "&Undo", "&Edit", "act_undo", "Ctrl+Z"),
    ActionSpec("edit.redo", "&Redo", "&Edit", "act_redo", "Ctrl+Shift+Z",
               separator_after=True),
    ActionSpec("edit.find", "&Find Block...", "&Edit", "act_find", "",
               "Opens the in-canvas search (or press Tab on the canvas)"),

    # ---- View ----
    ActionSpec("view.theme_classroom", "&Classroom Theme", "&View",
               "act_theme_classroom", checkable=True),
    ActionSpec("view.theme_dark", "&Dark Theme", "&View", "act_theme_dark",
               checkable=True),
    ActionSpec("view.theme_light", "&Light Theme", "&View", "act_theme_light",
               checkable=True, separator_after=True),
    ActionSpec("view.zoom_in", "Zoom &In", "&View", "act_zoom_in", "Ctrl++"),
    ActionSpec("view.zoom_out", "Zoom &Out", "&View", "act_zoom_out", "Ctrl+-"),
    ActionSpec("view.zoom_fit", "&Fit Graph", "&View", "act_zoom_fit", "Ctrl+0"),

    # ---- Help ----
    ActionSpec("help.shortcuts", "Keyboard &Shortcuts", "&Help", "act_shortcuts"),
]

MENU_ORDER = ["&File", "&Edit", "&View", "&Help"]


def build_actions(context, parent) -> dict[str, object]:
    """Create QActions from the catalog; objectName = spec.id."""
    from PySide6 import QtGui

    actions: dict = {}
    for spec in CATALOG:
        action = QtGui.QAction(spec.text, parent)
        action.setObjectName(spec.id)
        if spec.shortcut:
            action.setShortcut(QtGui.QKeySequence(spec.shortcut))
        if spec.tooltip:
            action.setToolTip(spec.tooltip)
        action.setCheckable(spec.checkable)
        handler = getattr(context, spec.slot, None)
        if handler is not None:
            action.triggered.connect(handler)
        actions[spec.id] = action
    return actions
