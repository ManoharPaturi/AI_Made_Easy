"""Workbench: the shell, and ONLY the shell (Orange CanvasMainWindow logic).

Build phases in order — setup_actions() -> setup_ui() -> setup_menus() —
then restore saved geometry/state. All behavior lives in AppContext
services; this class owns layout, chrome, and lifecycle only.
"""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from ai_made_easy.ui import context as context_mod
from ai_made_easy.ui.actions_catalog import CATALOG, MENU_ORDER, build_actions

_SETTINGS_KEY = "aime/workbench"
SETTINGS_VERSION = 2  # v2: dock state -> one-page workspace splitters

_APP_TITLE = "AI Made Easy — Visual Model Builder"


class Workbench(QtWidgets.QMainWindow):
    def __init__(self, ctx: "context_mod.AppContext"):
        super().__init__()
        self.ctx = ctx
        self.setWindowTitle(self.tr(_APP_TITLE))  # i18n groundwork (tr pass)
        self.resize(1500, 900)

        self.actions = build_actions(ctx, self)
        self.setup_actions()
        self.setup_ui()
        self.setup_menus()
        self._restore()
        ctx.status_message.connect(self.statusBar().showMessage)

    # ------------------------------------------------------------- phases

    def setup_actions(self) -> None:
        """QActions exist before UI and menus reference them (Orange rule)."""
        for window in (self,):
            for action in self.actions.values():
                window.addAction(action)  # shortcuts work app-window-wide
        classroom = self.actions.get("view.theme_classroom")
        dark = self.actions.get("view.theme_dark")
        light = self.actions.get("view.theme_light")
        group = QtGui.QActionGroup(self)
        for act in (classroom, dark, light):
            if act is not None:
                group.addAction(act)
        if classroom is not None:
            classroom.setChecked(ctx_theme_is(self.ctx, "classroom"))

    def setup_ui(self) -> None:
        header_bar = self.addToolBar("Header")
        header_bar.setObjectName("toolbar.header")
        header_bar.setMovable(False)
        header_bar.setAllowedAreas(QtCore.Qt.ToolBarArea.TopToolBarArea)
        header_bar.addWidget(self.ctx.header)

        from ai_made_easy.ui.features.workspace import Workspace

        self.workspace = Workspace(self.ctx)
        self.setCentralWidget(self.workspace)
        # ⌨️ blocks ↔ python: toggle lives on the canvas, pane in the workspace
        self.ctx.canvas_controls.code_toggle_clicked.connect(
            self.workspace.toggle_code_pane)
        self.ctx.side_code.connect(self.workspace.set_side_code)

    def setup_menus(self) -> None:
        """Menus only assemble existing actions (Orange rule)."""
        for menu_name in MENU_ORDER:
            menu = self.menuBar().addMenu(menu_name)
            for spec in CATALOG:
                if spec.menu != menu_name:
                    continue
                menu.addAction(self.actions[spec.id])
                if spec.separator_after:
                    menu.addSeparator()

    # ---------------------------------------------------------- lifecycle

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt name)
        if self.ctx.run_store.is_running:
            self.ctx.process_service.stop()
        if self.ctx.project_store.dirty and not self._confirm_discard():
            event.ignore()
            return
        self._save_state()
        super().closeEvent(event)

    def _confirm_discard(self) -> bool:
        answer = QtWidgets.QMessageBox.question(
            self, "Unsaved changes",
            f"'{self.ctx.project_store.name}' has unsaved changes.",
            QtWidgets.QMessageBox.StandardButton.Save
            | QtWidgets.QMessageBox.StandardButton.Discard
            | QtWidgets.QMessageBox.StandardButton.Cancel)
        if answer == QtWidgets.QMessageBox.StandardButton.Save:
            self.ctx.act_save()
            return not self.ctx.project_store.dirty
        return answer == QtWidgets.QMessageBox.StandardButton.Discard

    def _save_state(self) -> None:
        settings = QtCore.QSettings(_SETTINGS_KEY)
        settings.beginGroup("workbench")
        settings.setValue("version", SETTINGS_VERSION)
        settings.setValue("geometry", self.saveGeometry())
        self.workspace.save_state(settings)
        settings.endGroup()

    def _restore(self) -> None:
        settings = QtCore.QSettings(_SETTINGS_KEY)
        settings.beginGroup("workbench")
        if int(settings.value("version", 0)) == SETTINGS_VERSION:
            geometry = settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
            self.workspace.restore_state(settings)
        settings.endGroup()


def ctx_theme_is(ctx, name: str) -> bool:
    return ctx.theme.active() == name
