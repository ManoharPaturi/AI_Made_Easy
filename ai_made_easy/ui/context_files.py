"""File menu, autosave and recent files.

The ``act_*`` slots of :class:`~ai_made_easy.ui.context.AppContext` for this area,
as a mixin: the methods use the stores, services and panels AppContext builds (split out of
context.py in 3.3; no behaviour change).
"""
from __future__ import annotations

import json
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from ai_made_easy.core.graph import Graph
from ai_made_easy.core.registry import get_registry
from ai_made_easy.ui.context_common import _SETTINGS
from ai_made_easy.ui.dialogs import (
    ExamplesDialog,
)
from ai_made_easy.ui.services.project_service import (
    recent_kind,
)


class FileActions:
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
            kind = recent_kind(f)
            # text after a tab sits in the right-hand (shortcut) column
            action = menu.addAction(f"{Path(f).name}\t{kind}" if kind else Path(f).name)
            action.setToolTip(f)
            action.triggered.connect(lambda _=False, f=f: self.open_path(f))
        if not files:
            menu.addAction("No recent projects").setEnabled(False)

    def open_path(self, path: str) -> None:
        if self._confirm_discard() and self.project_service.open_file(path):
            self._add_recent(path)

    def _detect_device(self) -> None:
        """Fill the device chip. Importing torch takes seconds, so it happens on a worker
        thread and the chip updates through a queued signal."""
        import importlib.util
        import threading

        if importlib.util.find_spec("torch") is None:
            self.device_chip.set_state("PyTorch not installed", "warning")
            return
        self._device_found.connect(self.device_chip.set_state)

        def probe() -> None:
            try:
                import torch

                device = ("CUDA · " + torch.cuda.get_device_name(0)
                          if torch.cuda.is_available()
                          else "Apple GPU (MPS)" if torch.backends.mps.is_available()
                          else "CPU")
            except Exception:  # noqa: BLE001
                device = "CPU"
            self._device_found.emit(device, "")

        threading.Thread(target=probe, name="aime-device", daemon=True).start()

    # ============================================================== file

    def act_new(self, *_):
        if self._confirm_discard():
            self.project_service.new_project()

    def act_new_from_task(self, *_, initial_path: str = ""):
        from ai_made_easy.ui.features.wizard import NewProjectWizard

        dialog = NewProjectWizard(self.window, start_dir=self._last_dir(),
                                  project=self.project_store.name)
        if initial_path:
            dialog.path.setText(initial_path)
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
        from ai_made_easy.core.migrate import stamp

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
            out = write_bundle(Path(path), stamp(self.project_service.snapshot().to_dict()),
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
