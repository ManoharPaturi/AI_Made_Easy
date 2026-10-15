"""Training runs, analysis and experiments.

The ``act_*`` slots of :class:`~ai_made_easy.ui.context.AppContext` for this area,
as a mixin: the methods use the stores, services and panels AppContext builds (split out of
context.py in 3.3; no behaviour change).
"""
from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from ai_made_easy.core.summary import summarize
from ai_made_easy.ui.context_common import _TRAINER_BLOCK
from ai_made_easy.ui.features.experiments import CompareDialog, SweepDialog
from ai_made_easy.ui.services.project_service import (
    project_kind,
)
from ai_made_easy.ui.stores import RunStore


class RunActions:
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
        self.process_service.run_training(ir, project=self.project_store.name,
                                          on=self.training_page.run_on.selected())

    def act_import_results(self, *_):
        """Complete a Colab hand-off run from the results zip its notebook downloaded."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.window, "Import run results", "", "Results zip (*.zip)")
        if not path:
            return
        try:
            status = self.experiment_service.manager.import_results(path)
        except (ValueError, KeyError, OSError) as exc:
            QtWidgets.QMessageBox.warning(self.window, "Import run results",
                                          str(exc).strip("'\""))
            return
        self.status_message.emit(f"Imported run {status['run_id']}: {status['state']}")
        self.process_service.history_changed.emit()

    def act_backends(self, *_):
        from ai_made_easy.ui.features.backends import BackendsDialog

        BackendsDialog(self.window).exec()
        self.training_page.run_on.refresh()

    def _run_pipeline(self, ir, resume_from: str = "") -> None:  # noqa: ANN001
        if not resume_from and not self._guard_run("run the pipeline"):
            return
        path = self.project_store.path
        pipeline_id = self.experiment_service.start_pipeline(
            ir, self.project_store.name, base=str(path.parent) if path else None,
            resume_from=resume_from, on=self.training_page.run_on.selected())
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
        spec = dict(dialog.spec())
        on = self.training_page.run_on.selected()
        spec["on"] = "" if on == "local" else on  # trials train where Train would
        if self.experiment_service.start_sweep(ir, spec, self.project_store.name):
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
