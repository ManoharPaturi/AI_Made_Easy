"""AppContext: the composition root. It builds every store, service and
feature, and it is the ONLY place signal wiring happens (Langflow page
convention; Ryven constructor injection).

Action slots (act_*) live here as thin intent handlers — the Workbench
shell and menus call them via the actions catalog.
"""
from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtWidgets

from ai_made_easy.core.graph import Graph
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.summary import summarize
from ai_made_easy.ui.canvas import CanvasArea, CanvasController
from ai_made_easy.ui.dialogs import (
    SampleGalleryDialog,
    SaveTemplateDialog,
    ShortcutsDialog,
)
from ai_made_easy.ui.features.canvas_controls import CanvasControls
from ai_made_easy.ui.features.header import HeaderBar
from ai_made_easy.ui.features.inspector import (
    AssistantPage,
    InspectorStack,
    PreviewPage,
    SummaryPage,
)
from ai_made_easy.ui.features.palette import PaletteFeature
from ai_made_easy.ui.features.runconsole import (
    ConsolePage,
    RunConsoleStack,
    TrainingPage,
)
from ai_made_easy.ui.services.export_service import ExportService
from ai_made_easy.ui.services.graph_service import GraphService
from ai_made_easy.ui.services.process_service import ProcessService
from ai_made_easy.ui.services.project_service import DEMO_SEED, ProjectService
from ai_made_easy.ui.stores import (
    LogBus,
    ProjectStore,
    RunStore,
    ValidationStore,
)
from ai_made_easy.ui.theme import ThemeService

_TRAINER_BLOCK = "train.trainer"


class AppContext(QtCore.QObject):
    """Builds + wires everything. One instance per application."""

    status_message = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        # ---- services & stores (no widgets) ----
        self.theme = ThemeService()
        self.log_bus = LogBus(self)
        self.log = self.log_bus
        self.project_store = ProjectStore(self)
        self.run_store = RunStore(self)
        self.validation_store = ValidationStore(self)
        self._seen_issue_msgs: set[str] = set()
        self._celebration = None

        self.canvas = CanvasController()
        self.graph_service = GraphService(self.canvas, self.log_bus, self)
        self.export_service = ExportService(self.log_bus)
        self.process_service = ProcessService(self.log_bus, self.run_store, self)
        self.project_service = ProjectService(self.project_store,
                                              self.graph_service, self.log_bus)

        # ---- features (widgets) ----
        self.canvas_area = CanvasArea(self.canvas)
        self.header = HeaderBar(self.project_store)
        self.palette = PaletteFeature(self.canvas_area.make_palette_widget())
        self.summary_page = SummaryPage()
        self.properties_page = self.canvas_area.make_properties_widget()
        self.preview_page = PreviewPage()
        self.assistant_page = AssistantPage()
        self.inspector = InspectorStack(self.properties_page, self.summary_page,
                                        self.preview_page, self.assistant_page)
        self.console_page = ConsolePage(self.log_bus)
        self.training_page = TrainingPage(self.run_store)
        self.runconsole = RunConsoleStack(self.console_page, self.training_page,
                                          self.run_store)
        self.canvas_controls = CanvasControls(self.canvas_area, self.canvas)

        self._wire()
        self._boot_demo()

    # ============================================================== wiring

    def _wire(self) -> None:
        gs = self.graph_service

        # settle pipeline -> validation/preview/summary/assistant/dirty
        gs.graph_settled.connect(self._on_graph_settled)

        # wire-guard notices -> statusbar + console
        gs.guard_message.connect(
            lambda msg: (self.status_message.emit(msg), self.log_bus.info(msg)))

        # issue rows -> jump to the block on canvas / one-click fix
        self.summary_page.node_requested.connect(
            lambda nid: self.canvas.select_and_center(nid))
        self.summary_page.fix_requested.connect(self._apply_fix)

    def _apply_fix(self, issue) -> None:
        from ai_made_easy.core.fixes import fix_for_issue

        ir = self.graph_service.snapshot()
        result = fix_for_issue(ir, issue)
        if result is None:
            self.log_bus.info("no automatic fix for this one — follow the 💡 tip")
            return
        label, description, fixed = result
        self.graph_service.load(fixed)
        self.log_bus.info(f"🔧 fixed: {description}")
        self.status_message.emit(f"🔧 fixed — {description}")

        # inspector switching policy (selection -> Properties)
        graph = self.canvas.node_graph
        graph.node_selected.connect(lambda _n: self.inspector.show_properties())
        graph.node_selection_changed.connect(
            lambda sel, _desel: None if sel else self.inspector.show_summary())

        # palette search -> place
        self.palette.place_requested.connect(gs.place_block)

        # missions -> load the sample, follow progress on every settle
        self.palette.missions.mission_selected.connect(self._open_mission)

        # data blocks open a preview on double-click
        graph.node_double_clicked.connect(self._preview_data_block)

        # header intents -> services
        self.header.train_clicked.connect(self.act_train)
        self.header.test_clicked.connect(self.act_test_run)
        self.header.validate_clicked.connect(gs.settle_now)
        self.header.llm_clicked.connect(lambda: self.act_export("llm"))
        self.header.expand_clicked.connect(gs.expand_selected)
        self.header.save_selection_clicked.connect(self.act_save_selection)
        self.header.export_requested.connect(
            lambda framework, kind: self.act_export(
                kind if kind == "llm" else f"{framework}_{kind}"))
        self.header.runtime_export_requested.connect(
            lambda kind: self.act_runtime_export(kind))

        # training page buttons
        self.training_page.train_clicked.connect(self.act_train)
        self.training_page.stop_clicked.connect(self.process_service.stop)

        # run lifecycle: statusbar text, trainer node status, plots, header
        self.run_store.state_changed.connect(
            lambda state, kind: self.header.set_running(
                state == RunStore.RUNNING))
        self.process_service.epoch_received.connect(self.training_page.on_epoch)
        self.process_service.epoch_received.connect(
            lambda e: gs.set_node_status(
                _TRAINER_BLOCK,
                f"Trainer · epoch {e.get('epoch')}/{e.get('total')}"))
        self.process_service.epoch_received.connect(self._on_epoch_progress)
        self.process_service.log_received.connect(self.log_bus.info)
        self.process_service.error_received.connect(self.log_bus.error)
        self.process_service.finished.connect(
            lambda code, kind: gs.set_node_status(_TRAINER_BLOCK, None))
        self.process_service.finished.connect(self._on_run_finished)

        # assistant round-trip
        self.assistant_page.apply_requested.connect(
            lambda data: self.assistant_page.applied(
                self.project_service.apply_graph_dict(data)))

        # canvas snapshot button
        self.canvas_controls.snapshot_clicked.connect(self.act_export_png)

        # preview target changes re-render through the service
        self.preview_page.target_changed.connect(
            lambda _t: self._refresh_preview(gs.snapshot()))

    def _boot_demo(self) -> None:
        import json

        self.project_store.reset("mnist_cnn")
        self.graph_service.load(Graph.from_dict(json.loads(DEMO_SEED.read_text())))
        self.project_store.mark_clean()
        self.log_bus.info("Welcome to AI Made Easy. Drag blocks from the "
                          "palette, wire them up, then export model or "
                          "training code.")

    # ============================================================== settle

    def _on_graph_settled(self, ir: Graph) -> None:
        issues = ir.validate()
        self.validation_store.update(issues)
        self.graph_service.note_shapes(ir)
        self.graph_service.apply_validation(issues)
        self.summary_page.set_issues(issues, ir)
        self.palette.missions.check(ir)
        self._log_issues(issues)
        self._refresh_preview(ir)
        try:
            self.summary_page.set_summary(summarize(ir))
        except Exception:
            self.summary_page.set_summary(None)
        self.assistant_page.set_graph(ir.to_dict())
        self.project_store.mark_dirty()
        errors, warnings = (len(self.validation_store.errors),
                            len([i for i in issues if i.severity == "warning"]))
        if not errors and not warnings:
            self.status_message.emit("✓ graph is valid")
        elif errors:
            self.status_message.emit(
                f"✖ {errors} error(s), {warnings} warning(s) — "
                "see 📋 Summary in the Inspector")
        else:
            self.status_message.emit(f"⚠ {warnings} warning(s) — "
                                     "see 📋 Summary in the Inspector")

    def _log_issues(self, issues: list) -> None:
        """New issues go to the Console so 'see Console' is honest."""
        msgs = {i.message for i in issues}
        fresh = [i for i in issues if i.message not in self._seen_issue_msgs]
        fixed = self._seen_issue_msgs - msgs
        for issue in fresh:
            self.log_bus.error(str(issue)) if issue.severity == "error" \
                else self.log_bus.warning(str(issue))
        for gone in fixed:
            self.log_bus.info(f"resolved: {gone}")
        self._seen_issue_msgs = msgs

    def _refresh_preview(self, ir: Graph) -> None:
        try:
            code = self.export_service.render(ir,
                                              self.preview_page.current_target())
            self.preview_page.set_code(code)
        except Exception as exc:
            self.preview_page.set_error(f"{type(exc).__name__}: {exc}")

    # ============================================================= delight

    def _open_mission(self, sample: str) -> None:
        from ai_made_easy.ui.services.project_service import ProjectService

        path = ProjectService.samples_dir() / sample
        if path.exists():
            self.project_service.open_sample(path)
            self.status_message.emit("🚀 mission loaded — follow the checklist")

    def _preview_data_block(self, node) -> None:
        definition = self.canvas.definition_of(node)
        if definition is None or not definition.type_id.startswith("data."):
            return
        from ai_made_easy.ui.features.data_preview import DataPreviewDialog

        params = {}
        reg = get_registry()
        spec = reg.get(definition.type_id)
        if spec is not None:
            params = {p.name: node.get_property(p.name) for p in spec.params}
        DataPreviewDialog(None, definition, params).exec()

    def _on_epoch_progress(self, event: dict) -> None:
        epoch, total = event.get("epoch"), event.get("total")
        if total:
            self.canvas.set_node_progress(
                _TRAINER_BLOCK, min(float(epoch) / float(total), 1.0))

    def _on_run_finished(self, code: int, kind: str) -> None:
        self.canvas.set_node_progress(_TRAINER_BLOCK, None)
        if code != 0 or kind != "train" \
                or self.run_store.state != RunStore.FINISHED:
            return
        if self._celebration is None:
            from ai_made_easy.ui.features.celebration import CelebrationOverlay

            self._celebration = CelebrationOverlay(self.canvas_area)
        self._celebration.celebrate(
            "🎉 Training finished — great job!",
            "Open 📈 Training below to see how your model learned")
        self.missions_panel.notify_run_finished()

    # ========================================================= act_* slots

    # ---- file ----
    def act_new(self, *_):
        self.project_service.new_project()

    def act_open(self, *_):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "Open Project", str(self.project_service.samples_dir()),
            "AI Made Easy (*.json)")
        if path:
            self.project_service.open_file(path)

    def act_save(self, *_):
        if self.project_store.path is None:
            self.act_save_as()
        else:
            self.project_service.save()

    def act_save_as(self, *_):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            None, "Save Project", f"{self.project_store.name}.json",
            "AI Made Easy (*.json)")
        if path:
            self.project_service.save_as(path)

    def act_samples(self, *_):
        entries = self.project_service.list_samples()
        if not entries:
            self.log.error("no sample projects found in ./samples")
            return
        dialog = SampleGalleryDialog(None, entries)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.project_service.open_sample(entries[dialog.chosen_index()][0])

    def act_export_png(self, *_):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            None, "Export Canvas as PNG", f"{self.project_store.name}.png",
            "PNG (*.png)")
        if not path:
            return

        def done(saved, error):
            (self.log.error if error else self.log.info)(
                f"canvas export failed: {error}" if error
                else f"canvas image saved → {saved}")

        self.canvas.export_canvas_png(Path(path), on_done=done)

    def act_quit(self, *_):
        from PySide6 import QtWidgets

        for window in QtWidgets.QApplication.topLevelWidgets():
            if isinstance(window, QtWidgets.QMainWindow):
                window.close()

    # ---- edit ----
    def act_undo(self, *_):
        self.canvas.node_graph.undo_stack().undo()

    def act_redo(self, *_):
        self.canvas.node_graph.undo_stack().redo()

    def act_find(self, *_):
        self.canvas.node_graph.toggle_node_search()

    # ---- view ----
    def act_theme_classroom(self, *_):
        self._apply_theme("classroom")

    def act_theme_dark(self, *_):
        self._apply_theme("dark")

    def act_theme_light(self, *_):
        self._apply_theme("light")

    def _apply_theme(self, name: str) -> None:
        from PySide6 import QtWidgets

        self.theme.apply(QtWidgets.QApplication.instance(), name)
        self.canvas.apply_theme(self.theme.canvas_colors())
        self.log.info(f"theme switched to {name}")

    def act_zoom_in(self, *_):
        self.canvas.zoom(+1)

    def act_zoom_out(self, *_):
        self.canvas.zoom(-1)

    def act_zoom_fit(self, *_):
        self.canvas.center_view()

    # ---- help ----
    def act_shortcuts(self, *_):
        from ai_made_easy.ui.actions_catalog import CATALOG

        ShortcutsDialog(None, CATALOG).exec()

    # ---- run & export ----
    def act_train(self, *_):
        if not self._guard_run("train"):
            return
        self.training_page.reset()
        self.process_service.run_training(self.project_service.snapshot())

    def act_test_run(self, *_):
        if not self._guard_run("test run"):
            return
        self.process_service.run_test(self.project_service.snapshot())

    def _guard_run(self, what: str) -> bool:
        """Block runs while the graph has errors; show a friendly checklist."""
        if self.validation_store.valid:
            return True
        issues = [i for i in self.validation_store.issues
                  if i.severity == "error"] or self.validation_store.issues
        lines = [f"✖  {i.message}" for i in issues[:6]]
        if len(issues) > 6:
            lines.append(f"…and {len(issues) - 6} more (see 📋 Summary)")
        QtWidgets.QMessageBox.warning(
            None, f"Fix {len(issues)} thing(s) before the {what}",
            "Your model isn't ready to run yet:\n\n"
            + "\n".join(lines)
            + "\n\nOpen 📋 Summary in the Inspector and click a problem "
              "to jump to its block.")
        return False

    def act_export(self, target: str, *_):
        try:
            self.export_service.write(self.project_service.snapshot(), target,
                                      Path.cwd() / "exports")
        except Exception as exc:
            self.log.error(f"code generation failed: {exc}")

    def act_runtime_export(self, kind: str, *_):
        try:
            script = self.export_service.write_runtime_script(
                self.project_service.snapshot(), kind, Path.cwd() / "exports")
        except Exception as exc:
            self.log.error(f"export failed: {exc}")
            return
        self.process_service.run_script(script, Path.cwd(), kind)

    def act_save_selection(self, *_):
        dialog = SaveTemplateDialog(None)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        name = dialog.template_name()
        if not name:
            return
        try:
            path = self.graph_service.save_selection_template(name)
        except Exception as exc:
            self.log.error(str(exc))
            return
        self.log.info(f"saved template → {path} (find it in the Custom palette)")
