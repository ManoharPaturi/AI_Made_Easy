"""Workbench: the main window shell — layout, chrome and lifecycle only.

Build phases: setup_actions() → setup_ui() → setup_menus(), then restore the
saved layout. All behaviour lives in AppContext; this class arranges the
toolbar, dock panels, menus and status bar around the canvas.
"""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from ai_made_easy import __version__
from ai_made_easy.ui import context as context_mod
from ai_made_easy.ui import icons
from ai_made_easy.ui.actions_catalog import CATALOG, MENU_ORDER, TOOLBAR, build_actions

_SETTINGS_KEY = "aime/workbench"
SETTINGS_VERSION = 5  # v5: Data dock

_APP_TITLE = "AI Made Easy"


class Workbench(QtWidgets.QMainWindow):
    def __init__(self, ctx: "context_mod.AppContext"):
        super().__init__()
        self.ctx = ctx
        self.setWindowTitle(_APP_TITLE)
        self.resize(1560, 940)
        self.setDockNestingEnabled(True)
        self.actions = build_actions(ctx, self)
        self.setup_actions()
        self.setup_ui()
        self.setup_menus()
        self._restore()
        ctx.attach_window(self)

    # ------------------------------------------------------------- phases

    def setup_actions(self) -> None:
        for action in self.actions.values():
            self.addAction(action)
        self.cmdk_action = QtGui.QAction("Command Palette…", self)
        self.cmdk_action.setShortcut(QtGui.QKeySequence("Ctrl+K"))
        self.cmdk_action.setIcon(icons.icon("search"))
        self.addAction(self.cmdk_action)
        group = QtGui.QActionGroup(self)
        for key in ("view.theme_dark", "view.theme_light"):
            group.addAction(self.actions[key])
        self.actions[f"view.theme_{self.ctx.theme.active()}"].setChecked(True)

    def setup_ui(self) -> None:
        bar = QtWidgets.QToolBar("Main")
        bar.setObjectName("toolbar.main")
        bar.setMovable(False)
        bar.setIconSize(QtCore.QSize(18, 18))
        bar.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, bar)
        bar.addWidget(self.ctx.project_field)
        bar.addSeparator()
        for key in TOOLBAR:
            if key == "|":
                bar.addSeparator()
                continue
            bar.addAction(self.actions[key])
            button = bar.widgetForAction(self.actions[key])
            if key == "run.train":
                button.setObjectName("primaryAction")
                button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
                self.actions[key].setIcon(icons.icon("play", color="#ffffff"))
            elif key == "run.stop":
                button.setObjectName("stopAction")
        export = QtWidgets.QToolButton()
        export.setText("Export")
        export.setIcon(icons.icon("download"))
        export.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        export.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.export_menu = QtWidgets.QMenu(export)
        export.setMenu(self.export_menu)
        bar.addWidget(export)
        spacer = QtWidgets.QWidget()
        spacer.setObjectName("toolbarSpacer")
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                             QtWidgets.QSizePolicy.Policy.Preferred)
        bar.addWidget(spacer)
        palette_button = QtWidgets.QToolButton()
        palette_button.setDefaultAction(self.cmdk_action)
        palette_button.setText("Search  ⌘K")
        palette_button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        bar.addWidget(palette_button)

        self.setCentralWidget(self.ctx.canvas_area)
        self.docks = {}
        for key, title, widget, area in (
                ("library", "Block Library", self.ctx.library,
                 QtCore.Qt.DockWidgetArea.LeftDockWidgetArea),
                ("inspector", "Inspector", self.ctx.inspector_tabs,
                 QtCore.Qt.DockWidgetArea.RightDockWidgetArea),
                ("problems", "Problems", self.ctx.problems,
                 QtCore.Qt.DockWidgetArea.BottomDockWidgetArea),
                ("output", "Output", self.ctx.output_page,
                 QtCore.Qt.DockWidgetArea.BottomDockWidgetArea),
                ("training", "Training", self.ctx.training_page,
                 QtCore.Qt.DockWidgetArea.BottomDockWidgetArea),
                ("experiments", "Experiments", self.ctx.experiments_page,
                 QtCore.Qt.DockWidgetArea.BottomDockWidgetArea),
                ("data", "Data", self.ctx.data_page,
                 QtCore.Qt.DockWidgetArea.BottomDockWidgetArea)):
            dock = QtWidgets.QDockWidget(title, self)
            dock.setObjectName(f"dock.{key}")
            dock.setWidget(widget)
            dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
                             | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
                             | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable)
            self.addDockWidget(area, dock)
            self.docks[key] = dock
        self.tabifyDockWidget(self.docks["problems"], self.docks["output"])
        self.tabifyDockWidget(self.docks["output"], self.docks["training"])
        self.tabifyDockWidget(self.docks["training"], self.docks["experiments"])
        self.tabifyDockWidget(self.docks["experiments"], self.docks["data"])
        self.docks["problems"].raise_()
        self.resizeDocks([self.docks["library"], self.docks["inspector"]], [270, 360],
                         QtCore.Qt.Orientation.Horizontal)
        self.resizeDocks([self.docks["problems"]], [230], QtCore.Qt.Orientation.Vertical)

        status = self.statusBar()
        status.setSizeGripEnabled(False)
        for chip in self.ctx.status_chips:
            status.addPermanentWidget(chip)

        from ai_made_easy.ui.features.command_palette import CommandPalette
        from ai_made_easy.ui.features.toasts import ToastLayer

        self.command_palette = CommandPalette(self.actions, self)
        self.command_palette.place_requested.connect(self.ctx.place_block)
        self.cmdk_action.triggered.connect(lambda: self.command_palette.open_at(self))
        self.toasts = ToastLayer(self)

    def setup_menus(self) -> None:
        for menu_name in MENU_ORDER:
            menu = self.menuBar().addMenu(menu_name)
            for spec in CATALOG:
                if spec.menu != menu_name:
                    continue
                menu.addAction(self.actions[spec.id])
                if spec.menu == "E&xport":
                    self.export_menu.addAction(self.actions[spec.id])
                if spec.separator_after:
                    menu.addSeparator()
                    if spec.menu == "E&xport":
                        self.export_menu.addSeparator()
            if menu_name == "&File":
                recent = QtWidgets.QMenu("Open &Recent", menu)
                recent.aboutToShow.connect(lambda m=recent: self.ctx.populate_recent(m))
                menu.insertMenu(self.actions["file.examples"], recent)
            if menu_name == "&View":
                panels = menu.addMenu("Panels")
                for dock in self.docks.values():
                    panels.addAction(dock.toggleViewAction())
            if menu_name == "&Edit":
                menu.addSeparator()
                menu.addAction(self.cmdk_action)

    # ---------------------------------------------------------- lifecycle

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt name)
        if not self.ctx.confirm_close(self):
            event.ignore()
            return
        self._save_state()
        self.ctx.data_service.shutdown()
        super().closeEvent(event)

    def _save_state(self) -> None:
        settings = QtCore.QSettings(_SETTINGS_KEY)
        settings.setValue("version", SETTINGS_VERSION)
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("state", self.saveState(SETTINGS_VERSION))
        settings.setValue("theme", self.ctx.theme.active())

    def _restore(self) -> None:
        settings = QtCore.QSettings(_SETTINGS_KEY)
        if int(settings.value("version", 0) or 0) != SETTINGS_VERSION:
            return
        geometry, state = settings.value("geometry"), settings.value("state")
        if geometry:
            self.restoreGeometry(geometry)
        if state:
            self.restoreState(state, SETTINGS_VERSION)

    def version(self) -> str:
        return __version__
