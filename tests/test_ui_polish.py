"""3.3 Phase 3 — UI refinement: one action list per app that agree, a shared chart palette,
in-app dialogs instead of the browser's, the desktop minimap, the first-run tour and
accessible names. (The web behaviour itself is covered by web/e2e/ui.spec.ts.)"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "src"


def _web_actions() -> list[tuple[str, str, str]]:
    text = (WEB / "actions.ts").read_text(encoding="utf-8")
    return re.findall(r'\{ id: "([^"]+)", label: "([^"]+)", group: "[^"]+"(?:, keys: "([^"]+)")?',
                      text)


# commands only the web app has (pages instead of docks, the theme toggle, the palette's own)
WEB_ONLY = {"file.download", "view.theme", "go.design", "go.data", "go.experiments", "go.models",
            "help.palette"}


def test_web_and_desktop_actions_agree():
    from ai_made_easy.ui.actions_catalog import CATALOG

    desk = {s.id: s.shortcut for s in CATALOG}
    web = _web_actions()
    assert len(web) >= 30
    ids = [i for i, _, _ in web]
    assert len(ids) == len(set(ids)), "duplicate web action ids"
    unknown = [i for i in ids if i not in desk and i not in WEB_ONLY]
    assert not unknown, f"web actions the desktop doesn't have (rename or add to WEB_ONLY): {unknown}"
    # where both apps have a shortcut it is the same one (Mod = Ctrl, or ⌘ on a Mac)
    clash = [(i, keys, desk[i]) for i, _, keys in web
             if keys and desk.get(i) and keys.replace("Mod", "Ctrl") != desk[i]]
    assert not clash, clash


def test_web_shortcuts_are_unique():
    keys = [k for _, _, k in _web_actions() if k]
    assert len(keys) == len(set(keys))


def test_one_series_palette_for_both_apps():
    from ai_made_easy.ui.features import experiments, runconsole
    from ai_made_easy.ui.theme import SERIES

    web = re.findall(r'"(#[0-9a-fA-F]{6})"', (WEB / "theme.ts").read_text(encoding="utf-8"))
    assert tuple(web) == SERIES
    assert runconsole._SERIES_COLORS is SERIES and experiments._SERIES is SERIES
    assert "SERIES" in (WEB / "components" / "LineChart.tsx").read_text(encoding="utf-8")


def test_no_native_browser_dialogs():
    """confirm() / prompt() / alert() block the page and ignore the theme: use useDialogs()."""
    offenders = []
    for path in WEB.rglob("*.ts*"):
        text = path.read_text(encoding="utf-8")
        own = path.name == "Dialogs.tsx" or "useDialogs()" in text   # the in-app ones
        for n, line in enumerate(text.splitlines(), 1):
            native = re.search(r"window\.(confirm|prompt|alert)\(", line) or (
                not own and re.search(r"(?<![\w.])(confirm|prompt|alert)\(", line))
            if native:
                offenders.append(f"{path.relative_to(WEB)}:{n}: {line.strip()}")
    assert not offenders, offenders


def test_web_files_stay_small():
    """The 3.3 refactor split App.tsx and ExperimentsPage.tsx; keep them from regrowing."""
    for name, limit in (("App.tsx", 300), ("pages/ExperimentsPage.tsx", 120)):
        assert len((WEB / name).read_text(encoding="utf-8").splitlines()) <= limit, name


def test_context_split_into_mixins():
    from ai_made_easy.ui import context
    from ai_made_easy.ui.context_data import DataActions
    from ai_made_easy.ui.context_exports import ExportActions
    from ai_made_easy.ui.context_files import FileActions
    from ai_made_easy.ui.context_runs import RunActions

    pytest.importorskip("PySide6")
    for mixin in (FileActions, ExportActions, RunActions, DataActions):
        assert issubclass(context.AppContext, mixin)
    assert hasattr(context.AppContext, "act_open") and hasattr(context.AppContext, "act_train")
    assert hasattr(context, "_load_trained")      # still importable from the old place
    assert len(Path(context.__file__).read_text(encoding="utf-8").splitlines()) < 900


# ---------------------------------------------------------------------- desktop


@pytest.fixture()
def desk(isolated_home):
    pytest.importorskip("PySide6")
    from PySide6 import QtCore, QtWidgets

    from ai_made_easy.ui.app import _ensure_qt_plugin_path
    from ai_made_easy.ui.context import _SETTINGS

    _ensure_qt_plugin_path()
    QtCore.QCoreApplication.setOrganizationName("aime-tests")
    QtCore.QCoreApplication.setApplicationName("ui-polish")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    settings = QtCore.QSettings(_SETTINGS)
    saved = {k: settings.value(k) for k in ("minimap", "tour_done") if settings.contains(k)}
    from ai_made_easy.ui.context import AppContext
    from ai_made_easy.ui.workbench import Workbench

    ctx = AppContext()
    win = Workbench(ctx)
    win.resize(1200, 800)
    win.show()
    app.processEvents()
    yield ctx, win, app
    for k in ("minimap", "tour_done"):   # leave the user's own settings as they were
        if k in saved:
            settings.setValue(k, saved[k])
        else:
            settings.remove(k)
    ctx.project_store.mark_clean()
    win.close()
    win.deleteLater()


def test_minimap_toggles_follows_and_moves_the_view(desk):
    from PySide6 import QtCore, QtGui, QtTest

    from ai_made_easy.core.graph import Graph
    from tests.conftest import tiny_classifier_dict

    ctx, win, app = desk
    design = tiny_classifier_dict()
    for i, node in enumerate(design["nodes"]):
        node["position"] = [i * 400, (i % 2) * 300]
    ctx.canvas.load_ir(Graph.from_dict(design))
    app.processEvents()
    minimap = ctx.canvas_area.minimap
    assert not minimap.isVisible()
    win.actions["view.minimap"].trigger()
    app.processEvents()
    assert minimap.isVisible() and win.actions["view.minimap"].isChecked()
    assert QtCore.QSettings("aime/workbench").value("minimap") in (True, "true")
    # sits in the canvas's bottom-right corner
    area = ctx.canvas_area
    assert minimap.geometry().right() <= area.width() and minimap.geometry().bottom() <= area.height()
    assert area.width() - minimap.geometry().right() <= area.MARGIN + 2
    assert area.height() - minimap.geometry().bottom() <= area.MARGIN + 2
    # the whole design fits in it
    minimap.refresh()
    items = minimap.scene().itemsBoundingRect()
    shown = minimap.mapToScene(minimap.viewport().rect()).boundingRect()
    assert shown.contains(items.center())
    # a click moves the main canvas there
    viewer = ctx.canvas.node_graph.viewer()
    target = items.bottomRight() - QtCore.QPointF(50, 50)
    QtTest.QTest.mouseClick(minimap.viewport(), QtCore.Qt.MouseButton.LeftButton,
                            QtCore.Qt.KeyboardModifier.NoModifier, minimap.mapFromScene(target))
    app.processEvents()
    centre = viewer.mapToScene(viewer.viewport().rect().center())
    assert (QtGui.QVector2D(centre - target)).length() < 120
    win.actions["view.minimap"].trigger()
    app.processEvents()
    assert not minimap.isVisible()


def test_tour_has_three_steps_and_is_remembered(desk):
    from PySide6 import QtCore

    ctx, win, app = desk
    settings = QtCore.QSettings("aime/workbench")
    settings.remove("tour_done")
    ctx._maybe_tour()   # offscreen Qt (tests, CI): never shown by itself
    assert getattr(ctx, "tour", None) is None
    tour = ctx.act_tour()
    app.processEvents()
    assert tour.isVisible() and "Step 1 of 3" in tour.counter.text()
    assert win.rect().contains(tour.geometry())
    tour.next.click()
    assert "Wire" in tour.title.text() and tour.back.isVisible()
    tour.next.click()
    assert tour.next.text() == "Done"
    tour.next.click()
    app.processEvents()
    assert not tour.isVisible()
    assert str(settings.value("tour_done")).lower() in ("true", "1")
    assert ctx.tour is None


def test_key_widgets_have_accessible_names(desk):
    ctx, win, _app = desk
    for key, dock in win.docks.items():
        assert dock.widget().accessibleName(), key
    assert ctx.library.search.accessibleName() == "Search blocks"
    assert ctx.project_field.accessibleName() == "Project name"
    assert ctx.canvas_area.accessibleName() == "Design canvas"
    assert ctx.problems.tree.accessibleName()
