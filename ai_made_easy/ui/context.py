"""AppContext: the composition root.

Builds every store, service and panel, and is the only place signals are
wired. ``act_*`` methods are the intent slots the actions catalog binds to;
the Workbench only arranges what the context builds.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from ai_made_easy import __version__
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core import budget
from ai_made_easy.core.summary import summarize
from ai_made_easy.ui.canvas import CanvasArea, CanvasController, block_mime_data
from ai_made_easy.ui.dialogs import (
    AboutDialog,
    ExamplesDialog,
    SaveTemplateDialog,
    ShortcutsDialog,
)
from ai_made_easy.ui.features.data_workspace import DataPage
from ai_made_easy.ui.features.experiments import CompareDialog, ExperimentsPage, SweepDialog
from ai_made_easy.ui.features.inspector import AssistantPage, CodePage, SummaryPage
from ai_made_easy.ui.features.library import BlockLibrary
from ai_made_easy.ui.features.problems import ProblemsPanel
from ai_made_easy.ui.features.project_field import ProjectNameField
from ai_made_easy.ui.features.properties import PropertyInspector
from ai_made_easy.ui.features.runconsole import OutputPage, TrainingPage
from ai_made_easy.ui.services.data_service import DataService
from ai_made_easy.ui.services.experiment_service import ExperimentService
from ai_made_easy.ui.services.export_service import ExportService, default_filename
from ai_made_easy.ui.services.graph_service import GraphService
from ai_made_easy.ui.services.process_service import ProcessService, python_executable
from ai_made_easy.ui.services.project_service import (
    DEMO_SEED,
    ProjectService,
    project_kind,
)
from ai_made_easy.ui.stores import LogBus, ProjectStore, RunStore, ValidationStore
from ai_made_easy.ui.theme import ThemeService

_TRAINER_BLOCK = "train.trainer"
_SETTINGS = "aime/workbench"


class StatusChip(QtWidgets.QLabel):
    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setObjectName("statusChip")

    def set_state(self, text: str, state: str = "") -> None:
        self.setText(text)
        self.setProperty("state", state)
        self.style().unpolish(self)
        self.style().polish(self)


class AppContext(QtCore.QObject):
    """Builds and wires everything. One instance per application."""

    status_message = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.window: QtWidgets.QMainWindow | None = None
        # ---- services & stores
        self.theme = ThemeService()
        self.log_bus = LogBus(self)
        self.log = self.log_bus
        self.project_store = ProjectStore(self)
        self.run_store = RunStore(self)
        self.validation_store = ValidationStore(self)
        self.canvas = CanvasController()
        self.graph_service = GraphService(self.canvas, self.log_bus, self)
        self.export_service = ExportService(self.log_bus)
        self.process_service = ProcessService(self.log_bus, self.run_store, self)
        self.experiment_service = ExperimentService(self.process_service.history, self.log_bus,
                                                    self, python=python_executable())
        self.project_service = ProjectService(self.project_store, self.graph_service,
                                              self.log_bus)
        self.data_service = DataService(self, python=python_executable())
        self._last_ir: Graph | None = None
        self._run_kind = ""
        self._known_errors: set[tuple] = set()

        # ---- panels
        self.canvas_area = CanvasArea(self.canvas)
        self.project_field = ProjectNameField(self.project_store)
        self.library = BlockLibrary(block_mime_data)
        self.properties = PropertyInspector()
        self.summary_page = SummaryPage()
        self.code_page = CodePage()
        self.assistant_page = AssistantPage()
        self.inspector_tabs = QtWidgets.QTabWidget()
        self.inspector_tabs.setDocumentMode(True)
        for page, title in ((self.properties, "Properties"), (self.summary_page, "Summary"),
                            (self.code_page, "Code"), (self.assistant_page, "Assistant")):
            self.inspector_tabs.addTab(page, title)
        self.problems = ProblemsPanel(self.canvas.node_title, self._fixable)
        self.output_page = OutputPage(self.log_bus)
        self.training_page = TrainingPage(self.run_store)
        self.experiments_page = ExperimentsPage()
        self.data_page = DataPage()
        self._data_key = None
        self.validation_chip = StatusChip("Not validated")
        self.params_chip = StatusChip("")
        self.kind_chip = StatusChip("")
        self.device_chip = StatusChip("")
        self.status_chips = [self.validation_chip, self.params_chip, self.kind_chip,
                             self.device_chip, StatusChip(f"v{__version__}")]
        self._wire()

    # ================================================================ wiring

    def _wire(self) -> None:
        gs = self.graph_service
        gs.graph_settled.connect(self._on_graph_settled)
        gs.guard_message.connect(lambda msg: (self.status_message.emit(msg),
                                              self.log_bus.warning(msg)))
        graph = self.canvas.node_graph
        graph.node_selection_changed.connect(lambda *_: self._on_selection())
        graph.node_selected.connect(lambda *_: self._on_selection())
        graph.node_double_clicked.connect(self._preview_data_block)

        self.library.place_requested.connect(self.place_block)
        self.properties.param_changed.connect(self.canvas.set_param)
        self.properties.reset_requested.connect(self._reset_params)
        self.properties.table_edit_requested.connect(self._edit_table)
        self.problems.locate_requested.connect(self._locate)
        self.problems.fix_requested.connect(self._apply_fix)
        self.summary_page.budget_changed.connect(self._on_budget_changed)
        self.code_page.target_changed.connect(lambda _t: self._refresh_preview())
        self.code_page.export_requested.connect(self.act_export)
        self.assistant_page.apply_requested.connect(
            lambda data: self.assistant_page.applied(self.project_service.apply_graph_dict(data)))

        self.training_page.train_clicked.connect(self.act_train)
        self.training_page.stop_clicked.connect(self.act_stop)
        self.training_page.errors_clicked.connect(self.act_error_analysis)
        self.training_page.saliency_clicked.connect(self.act_saliency)
        self.training_page.card_clicked.connect(self.act_model_card)
        self.training_page.folder_clicked.connect(self.act_open_run_folder)

        ep, es = self.experiments_page, self.experiment_service
        ep.scope.currentIndexChanged.connect(lambda *_: self._refresh_experiments())
        ep.compare_requested.connect(self._compare_runs)
        ep.stop_pipeline_requested.connect(lambda _pid: self.experiment_service.stop_pipeline())
        ep.resume_pipeline_requested.connect(
            lambda pid: self._run_pipeline(None, resume_from=pid))
        ep.results_requested.connect(self._load_run_results)
        ep.restore_requested.connect(self._restore_run)
        ep.folder_requested.connect(self._open_run_folder)
        ep.delete_requested.connect(self._delete_runs)
        ep.tag_requested.connect(self._tag_run)
        ep.new_sweep_requested.connect(self.act_new_sweep)
        ep.stop_sweep_requested.connect(lambda _sid: es.stop_sweep())
        ep.apply_best_requested.connect(self._apply_best_sweep)
        ep.register_requested.connect(self._register_run)
        ep.deploy_run_requested.connect(self._deploy_run)
        ep.model_deploy_requested.connect(self._deploy_model)
        ep.model_stage_requested.connect(self._set_model_stage)
        ep.model_delete_requested.connect(self._delete_model)
        es.changed.connect(self._refresh_experiments)
        es.sweep_finished.connect(self._on_sweep_finished)
        self.process_service.history_changed.connect(self._refresh_experiments)
        self.project_store.name_changed.connect(lambda *_: self._refresh_experiments())

        dp, ds = self.data_page, self.data_service
        dp.dataset_selected.connect(lambda _nid: self._profile_selected())
        dp.refresh_requested.connect(lambda: self._profile_selected(refresh=True))
        dp.open_file_requested.connect(self.act_profile_file)
        dp.rows_requested.connect(self._load_rows)
        dp.split_requested.connect(self.act_split_preview)
        dp.augment_requested.connect(self.act_augmentation_preview)
        ds.profiled.connect(self._on_profiled)
        ds.rows_ready.connect(lambda key, frame: dp.set_rows(frame)
                              if key == self._data_key else None)
        ds.split_ready.connect(dp.set_split)
        ds.preview_ready.connect(dp.set_augmentation)
        ds.changed.connect(self._republish_issues)

        self.run_store.state_changed.connect(self._on_run_state)
        ps = self.process_service
        ps.epoch_received.connect(self.training_page.on_epoch)
        ps.samples_received.connect(self.training_page.on_samples)
        ps.epoch_received.connect(self._on_epoch_progress)
        ps.log_received.connect(self.log_bus.info)
        ps.error_received.connect(self.log_bus.error)
        ps.finished.connect(self._on_run_finished)

    def attach_window(self, window: QtWidgets.QMainWindow) -> None:
        """Called by the Workbench once its chrome exists."""
        self.window = window
        self.status_message.connect(lambda m: window.statusBar().showMessage(m, 6000))
        self.status_message.connect(window.toasts.toast)
        theme = QtCore.QSettings(_SETTINGS).value("theme", self.theme.active())
        self._apply_theme(str(theme), announce=False)
        self._on_run_state(self.run_store.state, "")
        QtCore.QTimer.singleShot(0, self._boot)
        QtCore.QTimer.singleShot(400, self._detect_device)
        self._autosave_timer = QtCore.QTimer(self, interval=120_000, timeout=self._autosave)
        self._autosave_timer.start()

    def _boot(self) -> None:
        self.project_store.reset("mnist_cnn")
        self.graph_service.load(Graph.from_dict(json.loads(DEMO_SEED.read_text())))
        self.project_store.set_name(json.loads(DEMO_SEED.read_text()).get("name", "untitled"))
        self.project_store.mark_clean()
        self.canvas.widget.setFocus()
        self.log_bus.info(f"AI Made Easy {__version__} ready — {len(get_registry().all())} "
                          "blocks available")
        QtCore.QTimer.singleShot(0, self._offer_recovery)

    # ======================================================= autosave / recent

    @staticmethod
    def _recovery_file() -> Path:
        from ai_made_easy.ui.app import data_dir

        name = QtCore.QCoreApplication.applicationName() or "app"
        folder = data_dir() / "autosave"
        folder.mkdir(exist_ok=True)
        return folder / f"{''.join(c if c.isalnum() else '_' for c in name)}.json"

    def _autosave(self) -> None:
        if not self.project_store.dirty:
            return
        data = self.project_service.snapshot().to_dict()
        data.setdefault("meta", {})["autosave_of"] = str(self.project_store.path or "")
        try:
            self._recovery_file().write_text(json.dumps(data))
        except OSError as exc:
            self.log_bus.warning(f"autosave failed: {exc}")

    def _clear_recovery(self) -> None:
        self._recovery_file().unlink(missing_ok=True)

    def _offer_recovery(self) -> None:
        path = self._recovery_file()
        if not path.exists() or self.window is None:
            return
        answer = QtWidgets.QMessageBox.question(
            self.window, "Recover unsaved work",
            "AI Made Easy closed with unsaved changes. Restore the last autosaved design?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
        if answer == QtWidgets.QMessageBox.StandardButton.Yes:
            try:
                data = json.loads(path.read_text())
                self.graph_service.load(Graph.from_dict(data))
                self.project_store.set_name(data.get("name", "recovered"))
                original = data.get("meta", {}).get("autosave_of")
                self.project_store.set_path(Path(original) if original else None)
                self.project_store.mark_dirty()
                self.log_bus.info("restored autosaved work")
            except (OSError, ValueError) as exc:
                self.log_bus.error(f"could not restore autosave: {exc}")
        self._clear_recovery()

    def recent_files(self) -> list[str]:
        value = QtCore.QSettings(_SETTINGS).value("recent_files", []) or []
        files = [value] if isinstance(value, str) else list(value)
        return [f for f in files if Path(f).exists()]

    def _add_recent(self, path) -> None:  # noqa: ANN001
        files = [str(Path(path))] + [f for f in self.recent_files() if f != str(Path(path))]
        QtCore.QSettings(_SETTINGS).setValue("recent_files", files[:10])

    def populate_recent(self, menu: QtWidgets.QMenu) -> None:
        menu.clear()
        files = self.recent_files()
        for f in files:
            action = menu.addAction(Path(f).name)
            action.setToolTip(f)
            action.triggered.connect(lambda _=False, f=f: self.open_path(f))
        if not files:
            menu.addAction("No recent projects").setEnabled(False)

    def open_path(self, path: str) -> None:
        if self._confirm_discard() and self.project_service.open_file(path):
            self._add_recent(path)

    def _detect_device(self) -> None:
        import importlib.util

        if importlib.util.find_spec("torch") is None:
            self.device_chip.set_state("PyTorch not installed", "warning")
            return
        try:
            import torch

            device = ("CUDA · " + torch.cuda.get_device_name(0) if torch.cuda.is_available()
                      else "Apple GPU (MPS)" if torch.backends.mps.is_available() else "CPU")
        except Exception:  # noqa: BLE001
            device = "CPU"
        self.device_chip.set_state(device)

    # ================================================================ settle

    def _on_graph_settled(self, ir: Graph) -> None:
        self._last_ir = ir
        self._publish_issues(ir)
        self._refresh_datasets(ir)
        kind = project_kind(ir.to_dict())
        task = None
        if kind == "Neural network":
            from ai_made_easy.core.tasks import task_of

            task = task_of(ir)
        self.kind_chip.set_state(f"{kind} · {task.label}" if task else kind)
        self.summary_page.set_kind(kind.split()[0] if kind != "Neural network" else "Neural net")
        try:
            summary = summarize(ir) if kind == "Neural network" else None
        except Exception:  # noqa: BLE001 — summaries need a valid model
            summary = None
        estimate, checks = None, []
        if summary is not None:
            try:
                estimate = budget.estimate(ir)
                checks = budget.check(ir, estimate)
            except Exception:  # noqa: BLE001 — an estimate never blocks the summary
                estimate = None
        flops = ([c.flops for c in estimate.layers]
                 if estimate and len(estimate.layers) == len(summary.layers) else None)
        self.summary_page.set_summary(
            summary, "" if summary else ("Classic ML pipelines have no layer summary."
                                         if kind.startswith("Classic") else ""), flops)
        self.summary_page.set_budget(budget.budget_of(ir))
        self.summary_page.set_estimate(estimate, checks)
        self.params_chip.set_state(f"{summary.total_params_display} parameters"
                                   if summary else "")
        self._refresh_preview(ir)
        self.assistant_page.set_graph(ir.to_dict())
        self._refresh_properties()
        if not self.graph_service.settled_from_load:
            self.project_store.mark_dirty()

    def _on_budget_changed(self, values: dict) -> None:
        """The Summary's budget controls edited ``meta["budget"]``: re-check the design."""
        self.graph_service.adapter.meta["budget"] = values
        self.graph_service.schedule()

    def _republish_issues(self) -> None:
        """Data profiles landed in the background: refresh the problems only."""
        if self._last_ir is not None:
            self._publish_issues(self._last_ir)

    def _publish_issues(self, ir: Graph) -> None:
        from ai_made_easy.core.data.lints import project_base

        self.data_service.base = project_base(self.project_store.path)
        issues = ir.validate() + self.data_service.issues_for(ir)
        self.validation_store.update(issues)
        self.graph_service.note_shapes(ir)
        self.graph_service.apply_validation(issues)
        self.problems.set_issues(issues)
        current = {(i.node_id, i.message) for i in issues if i.severity == "error"}
        fresh = [m for nid, m in current - self._known_errors]
        if fresh and not self.graph_service.settled_from_load:
            self.status_message.emit(fresh[0].split(" Fix:")[0])
        self._known_errors = current
        errors = len([i for i in issues if i.severity == "error"])
        warnings = len(issues) - errors
        if errors:
            self.validation_chip.set_state(f"{errors} error{'s' * (errors != 1)}"
                                           + (f", {warnings} warning{'s' * (warnings != 1)}"
                                              if warnings else ""), "error")
        elif warnings:
            self.validation_chip.set_state(f"{warnings} warning{'s' * (warnings != 1)}",
                                           "warning")
        else:
            self.validation_chip.set_state("No problems", "ok")

    def _refresh_preview(self, ir: Graph | None = None) -> None:
        ir = ir or self._last_ir
        if ir is None:
            return
        target = self.code_page.current_target()
        try:
            self.code_page.set_code(self.export_service.render(ir, target))
        except Exception as exc:  # noqa: BLE001 — shown in the code panel
            self.code_page.set_error(str(exc))

    # ============================================================ selection

    def _on_selection(self) -> None:
        ids = self.canvas.selected_ids()
        if len(ids) != 1:
            self.properties.clear()
            return
        node_id = ids[0]
        block = self.canvas.block_of(node_id)
        if block is None:
            self.properties.clear()
            return
        if self.properties.node_id != node_id:
            issues, shape, count = self._node_facts(node_id)
            self.properties.show_node(node_id, block, self.canvas.params_of(node_id),
                                      issues, shape, count, why=self._why(node_id))
            self.inspector_tabs.setCurrentWidget(self.properties)

    def _refresh_properties(self) -> None:
        node_id = self.properties.node_id
        if node_id is None:
            return
        if self.canvas.block_of(node_id) is None:
            self.properties.clear()
            return
        self.properties.update_status(*self._node_facts(node_id))

    def _edit_table(self, node_id: str, param: str) -> None:
        """Open the grid editor for a table parameter (a variable's probability table)."""
        from ai_made_easy.core.pgm.network import layout
        from ai_made_easy.ui.features.table_editor import TableEditorDialog

        graph = self.canvas.to_ir()
        info = layout(graph, node_id)
        if "error" in info:
            self.status_message.emit(info["error"])
            QtWidgets.QMessageBox.information(self.window, "Probability table", info["error"])
            return
        dialog = TableEditorDialog(self.window, info)
        if dialog.exec() and dialog.result_text is not None:
            self.canvas.set_param(node_id, param, dialog.result_text)
            self.properties.show_node(node_id, graph.nodes[node_id].definition(),
                                      self.canvas.params_of(node_id),
                                      *self._node_facts(node_id))

    def _why(self, node_id: str) -> str:
        """The recipe's note on why this block is in the design ("" when none)."""
        recipe = (getattr(self.canvas, "meta", None) or {}).get("recipe") or {}
        return str((recipe.get("why") or {}).get(node_id, ""))

    def _node_facts(self, node_id: str):
        issues = [i for i in self.validation_store.issues if i.node_id == node_id]
        shape = self.graph_service.last_shapes.get(node_id)
        count = None
        ir = self._last_ir
        if ir is not None and node_id in ir.nodes:
            node = ir.nodes[node_id]
            defn = node.definition()
            if defn.param_fn is not None:
                in_shapes = [self.graph_service.last_shapes.get(e.source_id)
                             for e in ir.incoming(node_id)]
                if all(s is not None for s in in_shapes) and in_shapes:
                    try:
                        count = int(defn.param_fn(in_shapes, node.resolved_params()))
                    except Exception:  # noqa: BLE001
                        count = None
        return issues, shape, count

    def _reset_params(self, node_id: str) -> None:
        block = self.canvas.block_of(node_id)
        if block is None:
            return
        for spec in block.params:
            self.canvas.set_param(node_id, spec.name, spec.default)
        issues, shape, count = self._node_facts(node_id)
        self.properties.show_node(node_id, block, self.canvas.params_of(node_id), issues,
                                  shape, count)

    def _locate(self, node_id: str) -> None:
        self.canvas.select_and_center(node_id)
        self._on_selection()

    # ============================================================== fixes

    def _fixable(self, issue) -> bool:
        from ai_made_easy.core.fixes import fix_for_issue

        if self._last_ir is None:
            return False
        try:
            return fix_for_issue(self._last_ir, issue) is not None
        except Exception:  # noqa: BLE001
            return False

    def _apply_fix(self, issue) -> None:
        from ai_made_easy.core.fixes import fix_for_issue

        ir = self.project_service.snapshot()
        result = fix_for_issue(ir, issue)
        if result is None:
            self.status_message.emit("No automatic fix is available for this problem")
            return
        label, description, fixed = result
        self.graph_service.load(fixed)
        self.project_store.mark_dirty()
        self.log_bus.info(f"Quick fix applied: {description}")
        self.status_message.emit(f"{label}: {description}")

    # ============================================================ canvas

    def place_block(self, type_id: str) -> None:
        self.graph_service.place_block(type_id)

    def _preview_data_block(self, node) -> None:  # noqa: ANN001
        definition = self.canvas.definition_of(node)
        if definition is None or not definition.type_id.startswith("data."):
            return
        self.data_page.select(node.id)
        self.act_data()

    # ================================================================ data

    def act_data(self, *_):
        if self.window is not None:
            self.window.docks["data"].show()
            self.window.docks["data"].raise_()
        self._profile_selected()

    def _refresh_datasets(self, ir: Graph) -> None:
        from ai_made_easy.core.data.lints import dataset_nodes

        items = []
        for node_id, type_id, params in dataset_nodes(ir):
            source = params.get("root") or params.get("path") or params.get("dataset") or \
                params.get("repo_id") or params.get("kind") or ""
            items.append((node_id, f"{self.canvas.node_title(node_id)} · {source}"))
        previous = self.data_page.current_dataset()
        self.data_page.set_datasets(items)
        if not items:
            self._data_key = None
            self.data_page.clear()
        elif self.data_page.current_dataset() == previous:
            self._profile_selected()  # params may have changed

    def _profile_selected(self, refresh: bool = False) -> None:
        node_id = self.data_page.current_dataset()
        ir = self._last_ir
        if not node_id or ir is None or node_id not in ir.nodes:
            return
        node = ir.nodes[node_id]
        key = self.data_service.cache.key(node.type_id, dict(node.resolved_params()),
                                          self.data_service.base)
        if key == self._data_key and not refresh and self.data_page.profile is not None:
            return
        self._data_key = key
        self.data_page.set_loading()
        self.data_service.request(node.type_id, dict(node.resolved_params()), refresh=refresh)

    def _on_profiled(self, key, profile) -> None:  # noqa: ANN001
        if key == self._data_key:
            self.data_page.set_profile(profile)

    def _load_rows(self) -> None:
        if self.data_page.profile is not None:
            self.data_service.load_rows(self._data_key, self.data_page.profile)

    def act_profile_file(self, *_):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.window, "Profile a Data File", str(self._last_dir()),
            "Data files (*.csv *.tsv *.parquet *.xlsx *.xls *.json *.jsonl *.npz);;"
            "All files (*)")
        if not path:
            return
        self.data_page.set_loading(f"Profiling {Path(path).name}…")
        self._data_key = self.data_service.request_path(path)

    def act_data_split(self, *_):
        self.act_data()
        self.data_page.tabs.setCurrentIndex(4)
        self.act_split_preview()

    def act_data_augment(self, *_):
        self.act_data()
        self.data_page.tabs.setCurrentIndex(5)
        self.act_augmentation_preview()

    def act_split_preview(self, *_):
        if self._last_ir is None:
            return
        self.data_page.split_btn.setEnabled(False)
        self.data_page.split_info.setText("Computing the split…")
        self.data_service.split(self._last_ir)

    def act_augmentation_preview(self, *_):
        if self._last_ir is None:
            return
        self.data_page.augment_btn.setEnabled(False)
        self.data_page.augment_info.setText("Rendering training views…")
        self.data_service.augment(self._last_ir)

    # ================================================================ runs

    def _on_run_state(self, state: str, kind: str) -> None:
        if self.window is None:
            return
        running = state == RunStore.RUNNING
        acts = self.window.actions
        for key in ("run.train", "run.test"):
            acts[key].setEnabled(not running)
        acts["run.stop"].setEnabled(running)
        self.canvas.set_wire_flow(running and kind == "train")
        if running and kind == "train":
            self.window.docks["training"].raise_()

    def _guard_run(self, what: str) -> bool:
        if self.validation_store.valid:
            return True
        errors = self.validation_store.errors
        lines = "\n".join(f"• {i.message}" for i in errors[:8])
        more = f"\n…and {len(errors) - 8} more" if len(errors) > 8 else ""
        box = QtWidgets.QMessageBox(self.window)
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        box.setWindowTitle(f"Cannot {what}")
        box.setText(f"The design has {len(errors)} error(s). Resolve them before you {what}.")
        box.setInformativeText(lines + more)
        show = box.addButton("Show Problems", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QtWidgets.QMessageBox.StandardButton.Close)
        box.exec()
        if box.clickedButton() is show and self.window is not None:
            self.window.docks["problems"].show()
            self.window.docks["problems"].raise_()
        return False

    def act_train(self, *_):
        ir = self.project_service.snapshot()
        from ai_made_easy.core.families import family_of

        if family_of(ir).id == "pipeline":
            self._run_pipeline(ir)
            return
        if project_kind(ir.to_dict()) == "LLM workflow":
            self.status_message.emit("LLM workflows run as scripts: Export ▸ LLM Workflow Script")
            return
        if self.experiment_service.is_running():
            self.status_message.emit("A sweep is running — stop it before training")
            return
        if not self._guard_run("train"):
            return
        self.training_page.reset()
        self._run_kind = "train"
        self.process_service.run_training(ir, project=self.project_store.name)

    def _run_pipeline(self, ir, resume_from: str = "") -> None:  # noqa: ANN001
        if not resume_from and not self._guard_run("run the pipeline"):
            return
        path = self.project_store.path
        pipeline_id = self.experiment_service.start_pipeline(
            ir, self.project_store.name, base=str(path.parent) if path else None,
            resume_from=resume_from)
        if pipeline_id:
            self.act_experiments()
            self.experiments_page.show_pipeline(pipeline_id)
            self.status_message.emit(f"Pipeline {pipeline_id} running — Experiments ▸ "
                                     "Pipelines shows its stages")

    def act_test_run(self, *_):
        ir = self.project_service.snapshot()
        if project_kind(ir.to_dict()) != "Neural network":
            self.status_message.emit("Forward-pass tests apply to neural networks")
            return
        if self._guard_run("run a forward pass"):
            self.process_service.run_test(ir)

    def act_stop(self, *_):
        if self.experiment_service.is_running():
            self.experiment_service.stop_sweep()
        self.process_service.stop()
        self.run_store.set(RunStore.STOPPED, "train")

    def _on_epoch_progress(self, event: dict) -> None:
        epoch, total = event.get("epoch"), event.get("total")
        if total:
            self.canvas.set_node_progress(_TRAINER_BLOCK, min(float(epoch) / float(total), 1.0))

    def _on_run_finished(self, code: int, kind: str) -> None:
        self.canvas.set_node_progress(_TRAINER_BLOCK, None)
        if kind == "inspect":
            if code == 0:
                self._show_saliency()
            else:
                self.log_bus.error("saliency computation failed — see Output")
            return
        if kind == "train":
            self.training_page.set_results_available(self.process_service.last_workdir)
            if code == 0:
                metrics = self.training_page.last_metrics()
                headline = ", ".join(f"{k} {v:.4g}" for k, v in list(metrics.items())[:3])
                self.status_message.emit("Training finished" + (f" — {headline}" if headline
                                                                 else ""))
            else:
                self.status_message.emit("Training failed — see Output")
                if self.window is not None:
                    self.window.docks["output"].raise_()
        elif kind == "test":
            self.status_message.emit("Forward pass succeeded" if code == 0
                                     else "Forward pass failed — see Output")

    def _run_workdir(self) -> Path | None:
        wd = self.process_service.last_workdir
        return Path(wd) if wd and Path(wd).exists() else None

    # ========================================================== analysis

    def act_error_analysis(self, *_):
        wd = self._run_workdir()
        overlays = wd is not None and any((wd / "eval_samples").glob("*.png"))
        if wd is None or not ((wd / "predictions.json").exists() or overlays):
            self.status_message.emit("Train a classification or vision model first")
            return
        from ai_made_easy.ui.features.analysis import ErrorAnalysisDialog, VisionAnalysisDialog

        if (wd / "eval_samples").is_dir() and not (wd / "predictions.json").exists():
            VisionAnalysisDialog(self.window, wd).exec()
        else:
            ErrorAnalysisDialog(self.window, wd).exec()

    def act_saliency(self, *_, sample: str = "0"):
        wd = self._run_workdir()
        if wd is None or not any(wd.glob("*_best.pt")):
            self.status_message.emit("Train a PyTorch classification model first")
            return
        from ai_made_easy.core.codegen.training_gen import generate_inspect

        try:
            script_body = generate_inspect(self.project_service.snapshot())
        except Exception as exc:  # noqa: BLE001
            self.log_bus.error(f"saliency: {exc}")
            return
        script = wd / "aime_inspect.py"
        script.write_text(script_body)
        image = sample.startswith("--image")
        arg = sample[len("--image"):].strip() if image else sample
        launcher = wd / "aime_inspect_launch.py"
        argv = ["inspect.py", "--image", arg] if image else ["inspect.py", arg]
        launcher.write_text(f"import sys\nsys.argv = {argv!r}\n"
                            f"exec(compile(open({str(script)!r}).read(), 'inspect', 'exec'))\n")
        self._inspect_workdir = wd
        self.process_service.run_script(launcher, wd, "inspect")

    def _show_saliency(self) -> None:
        from ai_made_easy.ui.features.inspect_view import InspectDialog

        dialog = InspectDialog(self.window, self._inspect_workdir)
        dialog.rerun_requested.connect(lambda arg: (dialog.accept(),
                                                    self.act_saliency(sample=arg)))
        dialog.exec()

    def act_model_card(self, *_):
        from ai_made_easy.core.classic.generate import is_classic
        from ai_made_easy.core.codegen.training_gen import collect_spec, dataset_comment
        from ai_made_easy.ui.features.analysis import ModelCardDialog

        ir = self.project_service.snapshot()
        details: dict = {}
        try:
            if is_classic(ir):
                from ai_made_easy.core.classic.generate import collect_classic

                spec = collect_classic(ir)
                comment, trainer = spec.dataset.get("block", ""), {"seed": spec.seed}
                details = {"framework": spec.estimator.package, "task": spec.task,
                           "architecture": spec.estimator.name}
            else:
                spec = collect_spec(ir)
                comment, trainer = dataset_comment(spec), spec.trainer
                summary = summarize(ir)
                details = {"framework": "PyTorch", "task": spec.task,
                           "parameters": f"{summary.total_params:,}",
                           "architecture": " → ".join(layer.name for layer in summary.layers[1:-1][:12]),
                           "optimizer": spec.optimizer["kind"].split(".")[-1],
                           "loss": spec.loss["kind"].split(".")[-1].replace("loss_", "")}
        except Exception as exc:  # noqa: BLE001
            self.log_bus.error(f"model card: {exc}")
            return
        ModelCardDialog(self.window, self.project_store.name, comment, trainer,
                        self._run_workdir(), details).exec()

    def act_open_run_folder(self, *_):
        wd = self._run_workdir()
        if wd is not None:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(wd)))

    # ======================================================== experiments

    def act_experiments(self, *_):
        self._refresh_experiments()
        if self.window is not None:
            self.window.docks["experiments"].show()
            self.window.docks["experiments"].raise_()

    def _experiment_project(self) -> str | None:
        return self.project_store.name if self.experiments_page.project_scope() else None

    def _refresh_experiments(self) -> None:
        try:
            project = self._experiment_project()
            self.experiments_page.set_runs(self.experiment_service.runs(project))
            self.experiments_page.set_sweeps(self.experiment_service.sweep_records(project))
            self.experiments_page.set_pipelines(
                self.experiment_service.pipeline_records(project))
            self.experiments_page.set_models(self.experiment_service.registry.list())
        except OSError as exc:
            self.log_bus.warning(f"could not read the run history: {exc}")

    def act_new_sweep(self, *_):
        from ai_made_easy.core.sweeps import sweepable_params

        if self.experiment_service.is_running():
            self.status_message.emit("A sweep is already running")
            return
        if self.run_store.is_running:
            self.status_message.emit("Wait for the current run to finish")
            return
        ir = self.project_service.snapshot()
        if project_kind(ir.to_dict()) == "LLM workflow":
            self.status_message.emit("Sweeps apply to trainable models")
            return
        if not self._guard_run("start a sweep"):
            return
        params = sweepable_params(ir)
        metrics = ["val_loss", "accuracy", "f1", "roc_auc", "mae", "rmse", "r2",
                   "train_loss", "loss"]
        dialog = SweepDialog(self.window, params, metrics)
        for key in ("opt.lr",):
            dialog.check_row(key)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted or dialog.spec() is None:
            return
        if self.experiment_service.start_sweep(ir, dialog.spec(), self.project_store.name):
            self.experiments_page.tabs.setCurrentIndex(1)
            self.act_experiments()
            self.status_message.emit("Sweep started — trials appear in Experiments ▸ Sweeps")

    def _on_sweep_finished(self, sweep_id: str, state: str) -> None:
        self.status_message.emit(f"Sweep {state}")

    def _apply_best_sweep(self, sweep_id: str) -> None:
        from ai_made_easy.core.sweeps import SweepError, best_graph

        try:
            data = best_graph(self.experiment_service.sweeps.get(sweep_id))
        except (KeyError, SweepError) as exc:
            self.status_message.emit(str(exc))
            return
        if self._confirm_discard() and self.project_service.apply_graph_dict(data):
            self.status_message.emit("Best sweep parameters applied to the design")

    def _compare_runs(self, run_ids: list) -> None:
        history = self.experiment_service.history
        try:
            records = [history.get(r) for r in run_ids]
        except KeyError as exc:
            self.status_message.emit(str(exc))
            return
        CompareDialog(self.window, records, {r: history.epochs(r) for r in run_ids}).exec()

    def _load_run_results(self, run_id: str) -> None:
        history = self.experiment_service.history
        if self.run_store.is_running:
            self.status_message.emit("Wait for the current run to finish")
            return
        self.training_page.reset()
        for event in history.epochs(run_id):
            self.training_page.on_epoch(event)
        self.process_service.last_workdir = history.path(run_id)
        self.training_page.set_results_available(history.path(run_id))
        rec = history.get(run_id)
        self.training_page.status.setText(f"Run {rec.name} ({run_id}) — {rec.status}")
        if self.window is not None:
            self.window.docks["training"].raise_()

    def _restore_run(self, run_id: str) -> None:
        rec = self.experiment_service.history.get(run_id)
        if self._confirm_discard() and self.project_service.apply_graph_dict(rec.graph):
            self.status_message.emit(f"Restored the design of run {run_id}")

    def _open_run_folder(self, run_id: str) -> None:
        path = self.experiment_service.history.path(run_id)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))

    def _delete_runs(self, run_ids: list) -> None:
        if not run_ids:
            return
        live = self.process_service.current_run_id if self.run_store.is_running else None
        if live in run_ids:
            self.status_message.emit("Stop the running training before deleting it")
            return
        answer = QtWidgets.QMessageBox.question(
            self.window, "Delete runs",
            f"Delete {len(run_ids)} run(s) and their checkpoints and files? "
            "This cannot be undone.")
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        for run_id in run_ids:
            try:
                self.experiment_service.history.delete(run_id)
            except (KeyError, OSError) as exc:
                self.log_bus.error(f"could not delete {run_id}: {exc}")
        self._refresh_experiments()

    def _tag_run(self, run_id: str) -> None:
        history = self.experiment_service.history
        rec = history.get(run_id)
        text, ok = QtWidgets.QInputDialog.getText(
            self.window, "Run tags", "Comma-separated tags:", text=", ".join(rec.tags))
        if ok:
            history.update(run_id, tags=[t.strip() for t in text.split(",") if t.strip()])
            self._refresh_experiments()

    # ============================================================= import

    def act_import_model(self, *_):
        from ai_made_easy.ui.features.import_model import ImportModelDialog

        if not self._confirm_discard():
            return
        dialog = ImportModelDialog(self.window, python=python_executable(),
                                   start_dir=self._last_dir())
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted or dialog.result is None:
            return
        self.open_imported(dialog.result.graph)

    def open_imported(self, graph: dict) -> bool:
        """Load an imported graph as a new, unsaved project and arrange it."""
        if not self.project_service.apply_graph_dict(graph):
            return False
        self.project_store.set_path(None)
        self.canvas.auto_layout(self.project_service.snapshot())
        self.canvas.center_view()
        self.project_store.mark_dirty()
        self.status_message.emit(f"Imported {graph.get('name', 'model')} — "
                                 f"{len(graph.get('nodes', []))} blocks")
        return True

    # ============================================================= deploy

    def act_deploy(self, *_):
        """Deploy the selected run, else this project's latest finished run."""
        history = self.experiment_service.history
        selected = self.experiments_page.selected_run_ids()
        candidates = [history.get(r) for r in selected] if len(selected) == 1 else \
            history.list(project=self.project_store.name)
        run = next((r for r in candidates if r.status == "finished"), None)
        if run is None:
            self.status_message.emit("Train the model first — deployment needs a finished run")
            return
        self._deploy_run(run.run_id)

    def _deploy_run(self, run_id: str) -> None:
        from ai_made_easy.ui.features.deploy import DeployDialog

        rec = self.experiment_service.history.get(run_id)
        DeployDialog(self.window, self.experiment_service.history.path(run_id), rec.framework,
                     rec.project or rec.name, default_dir=self._last_dir(),
                     python=python_executable(),
                     subtitle=f"Source: run {run_id} ({rec.name})").exec()

    def _deploy_model(self, name: str, version: int) -> None:
        from ai_made_easy.ui.features.deploy import DeployDialog

        registry = self.experiment_service.registry
        mv = registry.get(name, version)
        DeployDialog(self.window, registry.path(name, version), mv.framework, name,
                     version=str(version), default_dir=self._last_dir(),
                     python=python_executable(),
                     subtitle=f"Source: {name} v{version} ({mv.stage})").exec()

    def _register_run(self, run_id: str) -> None:
        import re

        from ai_made_easy.core.deploy import RegistryError

        rec = self.experiment_service.history.get(run_id)
        default = re.sub(r"[^A-Za-z0-9._-]+", "-", rec.project or rec.name).strip("-") or "model"
        name, ok = QtWidgets.QInputDialog.getText(
            self.window, "Register Model", "Model name (letters, digits, . _ -):",
            text=default)
        if not ok or not name.strip():
            return
        try:
            mv = self.experiment_service.registry.register(
                self.experiment_service.history.path(run_id), name.strip())
        except (RegistryError, OSError) as exc:
            QtWidgets.QMessageBox.warning(self.window, "Register Model", str(exc))
            return
        self._refresh_experiments()
        self.experiments_page.tabs.setCurrentIndex(2)
        self.status_message.emit(f"Registered {mv.name} v{mv.version}")

    def _set_model_stage(self, name: str, version: int, stage: str) -> None:
        self.experiment_service.registry.set_stage(name, version, stage)
        self._refresh_experiments()
        self.status_message.emit(f"{name} v{version} → {stage}")

    def _delete_model(self, name: str, version: int) -> None:
        answer = QtWidgets.QMessageBox.question(
            self.window, "Delete model version",
            f"Delete {name} v{version} from the registry? This cannot be undone.")
        if answer == QtWidgets.QMessageBox.StandardButton.Yes:
            self.experiment_service.registry.delete(name, version)
            self._refresh_experiments()

    # ============================================================== file

    def act_new(self, *_):
        if self._confirm_discard():
            self.project_service.new_project()

    def act_new_from_task(self, *_):
        from ai_made_easy.ui.features.wizard import NewProjectWizard

        dialog = NewProjectWizard(self.window, start_dir=self._last_dir(),
                                  project=self.project_store.name)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        if dialog.automl_spec is not None:
            if self.experiment_service.start_automl(dialog.automl_spec,
                                                    self.project_store.name):
                self.act_experiments()
            return
        if dialog.result_graph is not None and self._confirm_discard():
            graph = dialog.result_graph
            if self.open_imported(graph):
                recipe = (graph.get("meta") or {}).get("recipe") or {}
                self.status_message.emit(f"Created a {recipe.get('title', 'recipe')} design — "
                                         "Model › Explain This Design says why each block "
                                         "is there")

    def act_explain(self, *_):
        from html import escape

        from ai_made_easy.core.recipes import explain

        info = explain(self.canvas.to_ir())
        recipe, task = info["recipe"], info["task"]
        parts = []
        if recipe:
            parts.append(f"<p><b>{escape(recipe['title'])}</b> recipe ({escape(recipe['tier'])})"
                         f" — {escape(recipe.get('description') or '')}</p>")
            if recipe.get("adapted"):
                parts.append("<p>Adapted to your data: "
                             f"{escape('; '.join(recipe['adapted']))}.</p>")
        else:
            parts.append("<p>Not built from a recipe: each block's own description is "
                         "shown. File › New from Task starts from a recipe.</p>")
        if task:
            parts.append(f"<p><i>Task: {escape(task['label'])} — "
                         f"{escape(task['description'])}</i></p>")
        parts.append("<table cellspacing='4'>" + "".join(
            f"<tr><td valign='top'><b>{escape(b['name'])}</b></td>"
            f"<td>{escape(b['why'])}</td></tr>" for b in info["blocks"]) + "</table>")
        box = QtWidgets.QDialog(self.window)
        box.setWindowTitle("Explain This Design")
        box.resize(720, 560)
        layout = QtWidgets.QVBoxLayout(box)
        text = QtWidgets.QTextBrowser()
        text.setHtml("".join(parts))
        layout.addWidget(text)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(box.accept)
        layout.addWidget(close, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        box.exec()

    def act_open(self, *_):
        if not self._confirm_discard():
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.window, "Open Project", str(self._last_dir()), "AI Made Easy project (*.json)")
        if path:
            self._remember_dir(path)
            if self.project_service.open_file(path):
                self._add_recent(path)

    def act_save(self, *_):
        if self.project_store.path is None:
            self.act_save_as()
        elif self.project_service.save():
            self._clear_recovery()

    def act_save_as(self, *_):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.window, "Save Project", str(self._last_dir() / f"{self.project_store.name}.json"),
            "AI Made Easy project (*.json)")
        if path:
            self._remember_dir(path)
            if self.project_service.save_as(path):
                self._add_recent(path)
                self._clear_recovery()

    def act_samples(self, *_):
        entries = self.project_service.list_samples()
        if not entries:
            self.log_bus.error("no example projects found")
            return
        dialog = ExamplesDialog(self.window, entries)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted and self._confirm_discard():
            self.project_service.open_sample(entries[dialog.chosen_index()][0])
            self.project_store.set_path(None)
            self.canvas.center_view()

    def act_export_png(self, *_):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.window, "Export Canvas as Image",
            str(self._last_dir() / f"{self.project_store.name}.png"), "PNG image (*.png)")
        if not path:
            return

        def done(saved, error):
            (self.log_bus.error if error else self.log_bus.info)(
                f"canvas export failed: {error}" if error else f"canvas image saved → {saved}")

        self.canvas.export_canvas_png(Path(path), on_done=done)

    def act_export_bundle(self, *_):
        from ai_made_easy.core.bundle import write_bundle

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.window, "Export Project Archive",
            str(self._last_dir() / f"{self.project_store.name}.aime"),
            "AI Made Easy archive (*.aime)")
        if not path:
            return
        thumb = self.canvas_area.grab().scaledToWidth(640).toImage()
        buf = QtCore.QBuffer()
        buf.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
        thumb.save(buf, "PNG")
        try:
            out = write_bundle(Path(path), self.project_service.snapshot().to_dict(),
                               name=self.project_store.name, thumbnail_png=bytes(buf.data()),
                               workdir=self._run_workdir())
            self.log_bus.info(f"project archive saved → {out} "
                              f"({out.stat().st_size / 1e6:.1f} MB)")
        except Exception as exc:  # noqa: BLE001
            self.log_bus.error(f"archive export failed: {exc}")

    def act_open_bundle(self, *_):
        from ai_made_easy.core.bundle import read_bundle

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.window, "Open Project Archive", str(self._last_dir()),
            "AI Made Easy archive (*.aime)")
        if not path or not self._confirm_discard():
            return
        try:
            bundle = read_bundle(Path(path))
        except (ValueError, OSError) as exc:
            self.log_bus.error(str(exc))
            return
        if bundle.get("custom_blocks"):
            from ai_made_easy.core.block_packs import install_for_project
            from ai_made_easy.ui.canvas import templates as template_store
            from ai_made_easy.ui.canvas.node_factory import make_node_class

            installed = install_for_project(bundle["graph"], bundle["custom_blocks"])
            template_store.register_user_templates(get_registry().register, make_node_class)
            self.library.rebuild()
            if installed["imported"] or installed["renamed"]:
                self.log_bus.info(f"installed custom blocks from the archive: {installed}")
        self.graph_service.load(Graph.from_dict(bundle["graph"]))
        self.project_store.set_name(bundle["manifest"].get("name", "untitled"))
        self.project_store.set_path(None)
        from ai_made_easy.core.bundle import restore_run

        run_id = restore_run(bundle, self.experiment_service.history)
        if run_id:
            self._refresh_experiments()
            self._load_run_results(run_id)
            self.log_bus.info(f"the archive's trained run is available as {run_id}")
        self.log_bus.info(f"opened archive {path}")

    def act_quit(self, *_):
        if self.window is not None:
            self.window.close()

    # ============================================================== edit

    def act_undo(self, *_):
        self.canvas.node_graph.undo_stack().undo()

    def act_redo(self, *_):
        self.canvas.node_graph.undo_stack().redo()

    def act_delete(self, *_):
        focus = QtWidgets.QApplication.focusWidget()
        if isinstance(focus, (QtWidgets.QLineEdit, QtWidgets.QPlainTextEdit,
                              QtWidgets.QAbstractSpinBox, QtWidgets.QTextEdit)):
            return
        if self.canvas.delete_selected():
            self.properties.clear()

    def act_select_all(self, *_):
        self.canvas.select_all()

    def act_find(self, *_):
        if self.window is not None:
            self.window.docks["library"].show()
        self.library.focus_search()

    # ============================================================== view

    def act_theme_dark(self, *_):
        self._apply_theme("dark")

    def act_theme_light(self, *_):
        self._apply_theme("light")

    def _apply_theme(self, name: str, announce: bool = True) -> None:
        app = QtWidgets.QApplication.instance()
        self.theme.apply(app, name)
        tokens = self.theme.tokens()
        self.canvas.apply_theme(tokens)
        self.training_page.set_theme(tokens)
        self.code_page.set_theme(self.theme.active())
        if self.window is not None:
            from ai_made_easy.ui import icons
            from ai_made_easy.ui.actions_catalog import CATALOG

            for spec in CATALOG:
                if spec.icon and spec.id != "run.train":
                    self.window.actions[spec.id].setIcon(icons.icon(spec.icon))
            self.window.actions[f"view.theme_{self.theme.active()}"].setChecked(True)
        if announce:
            self.log_bus.info(f"theme: {self.theme.active()}")

    def act_zoom_in(self, *_):
        self.canvas.zoom(+1)

    def act_zoom_out(self, *_):
        self.canvas.zoom(-1)

    def act_zoom_fit(self, *_):
        self.canvas.center_view()

    def act_auto_layout(self, *_):
        self.canvas.auto_layout(self.project_service.snapshot())

    # ============================================================= model

    def act_validate(self, *_):
        self.graph_service.settle_now()
        if self.window is not None:
            self.window.docks["problems"].show()
            self.window.docks["problems"].raise_()
        errors, warnings = self.problems.counts
        self.status_message.emit("No problems found" if not errors + warnings
                                 else f"{errors} error(s), {warnings} warning(s)")

    def act_expand(self, *_):
        self.graph_service.expand_selected()

    def act_save_selection(self, *_):
        dialog = SaveTemplateDialog(self.window)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted or not dialog.template_name():
            return
        try:
            path = self.graph_service.save_selection_template(dialog.template_name())
        except Exception as exc:  # noqa: BLE001
            self.log_bus.error(str(exc))
            return
        self.library.rebuild()
        self.log_bus.info(f"saved custom block → {path}")
        self.status_message.emit(f"Saved custom block '{dialog.template_name()}'")

    def act_export_blocks(self, *_):
        from ai_made_easy.core.block_packs import PackError, export_pack, list_blocks

        if not list_blocks():
            self.status_message.emit("No custom blocks yet — use Model ▸ Save Selection as Block")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.window, "Export Custom Blocks", str(self._last_dir() / "blocks.aimeblocks"),
            "Custom block packs (*.aimeblocks)")
        if not path:
            return
        try:
            export_pack(path)
        except (PackError, OSError) as exc:
            QtWidgets.QMessageBox.warning(self.window, "Export Custom Blocks", str(exc))
            return
        self._remember_dir(path)
        self.status_message.emit(f"Exported {len(list_blocks())} custom block(s)")

    def act_import_blocks(self, *_):
        from ai_made_easy.core.block_packs import PackError, import_pack
        from ai_made_easy.core.registry import get_registry
        from ai_made_easy.ui.canvas import templates as template_store
        from ai_made_easy.ui.canvas.node_factory import make_node_class

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.window, "Import Custom Blocks", str(self._last_dir()),
            "Custom block packs (*.aimeblocks)")
        if not path:
            return
        try:
            result = import_pack(path)
        except (PackError, OSError) as exc:
            QtWidgets.QMessageBox.warning(self.window, "Import Custom Blocks", str(exc))
            return
        template_store.register_user_templates(get_registry().register, make_node_class)
        self.library.rebuild()
        parts = [f"{len(result['imported'])} imported"]
        if result["skipped"]:
            parts.append(f"{len(result['skipped'])} already installed")
        if result["renamed"]:
            parts.append("renamed: " + ", ".join(f"{a} → {b}"
                                                 for a, b in result["renamed"].items()))
        self.status_message.emit("Custom blocks: " + "; ".join(parts))

    # ============================================================ export

    def act_export(self, target: str, *_):
        ir = self.project_service.snapshot()
        try:
            code = self.export_service.render(ir, target)
        except Exception as exc:  # noqa: BLE001
            self._export_error(target, exc)
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.window, "Export", str(self._last_dir() / default_filename(ir, target)),
            "Python (*.py)")
        if path:
            self._remember_dir(path)
            Path(path).write_text(code)
            self.log_bus.info(f"exported → {path}")
            self.status_message.emit(f"Exported {Path(path).name}")

    def _export_error(self, target: str, exc: Exception) -> None:
        from ai_made_easy.ui.services.export_service import target_label

        QtWidgets.QMessageBox.warning(self.window, "Export not available",
                                      f"{target_label(target)} cannot be generated:\n\n{exc}")

    def act_export_pytorch_model(self, *_):
        self.act_export("pytorch_model")

    def act_export_pytorch_train(self, *_):
        self.act_export("pytorch_train")

    def act_export_keras_model(self, *_):
        self.act_export("keras_model")

    def act_export_keras_train(self, *_):
        self.act_export("keras_train")

    def act_export_sklearn(self, *_):
        self.act_export("sklearn_train")

    def act_export_llm(self, *_):
        self.act_export("llm")

    def act_export_onnx(self, *_):
        self.act_runtime_export("onnx")

    def act_export_jit(self, *_):
        self.act_runtime_export("jit")

    def act_runtime_export(self, kind: str, *_):
        ir = self.project_service.snapshot()
        suffix = ".onnx" if kind == "onnx" else ".pt"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.window, f"Export {'ONNX' if kind == 'onnx' else 'TorchScript'}",
            str(self._last_dir() / f"{ir.name}{suffix}"), f"Model (*{suffix})")
        if not path:
            return
        try:
            script = self.export_service.write_runtime_script(ir, kind, Path(path))
        except Exception as exc:  # noqa: BLE001
            self._export_error("pytorch_model", exc)
            return
        self._remember_dir(path)
        self.process_service.run_script(script, script.parent, kind)

    def act_export_web(self, *_):
        from ai_made_easy.core.codegen.training_gen import collect_spec
        from ai_made_easy.core.web_export import build_web_demo, layers_from_torch

        wd = self._run_workdir()
        if wd is None or not any(wd.glob("*_best.pt")):
            self.status_message.emit("The web demo needs a trained PyTorch image model")
            return
        try:
            spec = collect_spec(self.project_service.snapshot())
            model, shape, norm = _load_trained(wd, spec.class_name)
            classes = _json_or(wd / "classes.json", None) or [str(i) for i in
                                                              range(spec.num_outputs)]
            html = build_web_demo(layers_from_torch(model), classes, list(shape), norm,
                                  title=self.project_store.name)
        except Exception as exc:  # noqa: BLE001
            self._export_error("pytorch_model", exc)
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.window, "Export Web Demo",
            str(self._last_dir() / f"{self.project_store.name}_demo.html"), "HTML (*.html)")
        if path:
            Path(path).write_text(html)
            self.log_bus.info(f"web demo saved → {path}")

    # ============================================================== help

    def act_shortcuts(self, *_):
        from ai_made_easy.ui.actions_catalog import CATALOG

        ShortcutsDialog(self.window, CATALOG).exec()

    def act_about(self, *_):
        import importlib.util

        env = [("Python", sys.version.split()[0])]
        for module, label in (("torch", "PyTorch"), ("keras", "Keras"),
                              ("sklearn", "scikit-learn"), ("PySide6", "Qt for Python")):
            if importlib.util.find_spec(module) is not None:
                try:
                    env.append((label, __import__(module).__version__))
                except Exception:  # noqa: BLE001
                    env.append((label, "installed"))
            else:
                env.append((label, "not installed"))
        env.append(("Blocks", str(len(get_registry().all()))))
        AboutDialog(self.window, __version__, env).exec()

    # ============================================================ helpers

    def confirm_close(self, window) -> bool:  # noqa: ANN001
        if self.run_store.is_running:
            self.process_service.stop()
        self.experiment_service.shutdown()
        ok = self._confirm_discard()
        if ok:
            self._clear_recovery()
        return ok

    def _confirm_discard(self) -> bool:
        if not self.project_store.dirty or self.window is None:
            return True
        answer = QtWidgets.QMessageBox.question(
            self.window, "Unsaved changes",
            f"Save changes to '{self.project_store.name}' before continuing?",
            QtWidgets.QMessageBox.StandardButton.Save
            | QtWidgets.QMessageBox.StandardButton.Discard
            | QtWidgets.QMessageBox.StandardButton.Cancel)
        if answer == QtWidgets.QMessageBox.StandardButton.Save:
            self.act_save()
            return not self.project_store.dirty
        return answer == QtWidgets.QMessageBox.StandardButton.Discard

    def _last_dir(self) -> Path:
        value = QtCore.QSettings(_SETTINGS).value("last_dir", "")
        path = Path(str(value)) if value else Path.home()
        return path if path.exists() else Path.home()

    def _remember_dir(self, path: str) -> None:
        QtCore.QSettings(_SETTINGS).setValue("last_dir", str(Path(path).parent))


def _json_or(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def _load_trained(workdir: Path, class_name: str):
    """Import the run's training script as a module and load its checkpoint."""
    import importlib.util

    import torch

    script = next(iter(sorted(workdir.glob("*_train_pytorch.py"))), None)
    if script is None:
        raise RuntimeError("no PyTorch training script in the run folder")
    spec = importlib.util.spec_from_file_location("aime_trained", script)
    module = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    try:
        os.chdir(workdir)
        spec.loader.exec_module(module)
    finally:
        os.chdir(cwd)
    model = getattr(module, class_name)()
    model.load_state_dict(torch.load(workdir / module.CHECKPOINT, map_location="cpu"))
    model.eval()
    norm = ((tuple(module.NORM_MEAN), tuple(module.NORM_STD))
            if hasattr(module, "NORM_MEAN") else None)
    return model, tuple(module.INPUT_SHAPE), norm

