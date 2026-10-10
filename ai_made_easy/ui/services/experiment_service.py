"""ExperimentService: run history access and hyperparameter sweeps for the UI.

Sweeps run on the headless ``core`` RunManager (background threads); this
service polls the sweep record while one is active and emits ``changed`` so
the Experiments panel can refresh. One sweep at a time.
"""
from __future__ import annotations

from PySide6 import QtCore

from ai_made_easy.core.deploy import ModelRegistry
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.runner.manager import RunManager
from ai_made_easy.core.runs.history import RunHistory
from ai_made_easy.core.sweeps import SweepRunner, SweepSpec, SweepStore


class ExperimentService(QtCore.QObject):
    changed = QtCore.Signal()
    sweep_finished = QtCore.Signal(str, str)  # sweep_id, state

    def __init__(self, history: RunHistory, log, parent=None,  # noqa: ANN001
                 python: str | None = None, sweeps: SweepStore | None = None):
        super().__init__(parent)
        self.history = history
        self.log = log
        self.sweeps = sweeps if sweeps is not None else SweepStore()
        self.registry = ModelRegistry()
        self.manager = RunManager(history, python=python)
        self.runner: SweepRunner | None = None
        self._last_trials = -1
        self._timer = QtCore.QTimer(self, interval=1000, timeout=self._poll)
        from ai_made_easy.core.pipelines.runner import PipelineStore

        self.pipelines = PipelineStore()
        self.pipeline = None
        self._pipeline_timer = QtCore.QTimer(self, interval=1000, timeout=self._poll_pipeline)

    # ---------------------------------------------------------------- runs
    def runs(self, project: str | None = None):
        return self.history.list(project=project)

    def sweep_records(self, project: str | None = None):
        for rec in self.sweeps.list(project):
            if rec.state == "running" and (self.runner is None
                                           or self.runner.sweep_id != rec.sweep_id):
                rec.state = "stopped"  # left over from an earlier session
                rec.message = rec.message or "interrupted"
                self.sweeps.save(rec)
        return self.sweeps.list(project)

    # -------------------------------------------------------------- sweeps
    def is_running(self) -> bool:
        return self.runner is not None and self.runner.thread is not None \
            and self.runner.thread.is_alive()

    def start_sweep(self, graph: Graph, spec: dict, project: str = "") -> str | None:
        if self.is_running():
            self.log.error("a sweep is already running")
            return None
        try:
            runner = SweepRunner(self.manager, self.sweeps, graph, SweepSpec.from_dict(spec),
                                 project=project)
        except (ValueError, TypeError) as exc:
            self.log.error(f"could not start the sweep: {exc}")
            return None
        self.runner = runner
        runner.start()
        self._last_trials = -1
        self._timer.start()
        self.log.info(f"sweep {runner.sweep_id} started — {spec.get('strategy')} search, "
                      f"up to {spec.get('max_trials')} trials on {spec.get('metric')}")
        self.changed.emit()
        return runner.sweep_id

    def start_automl(self, spec: dict, project: str = "") -> str | None:
        """Search recipes and their settings (an AutoML sweep record)."""
        from ai_made_easy.core import automl

        if self.is_running():
            self.log.error("a sweep is already running")
            return None
        try:
            runner = automl.start(self.manager, self.sweeps, spec, project=project)
        except (automl.AutoMLError, ValueError, TypeError) as exc:
            self.log.error(f"could not start AutoML: {exc}")
            return None
        self.runner = runner
        self._last_trials = -1
        self._timer.start()
        self.log.info(f"AutoML {runner.sweep_id} started — {len(runner.recipes)} recipes, "
                      f"up to {runner.spec.max_trials} trials on {runner.spec.metric}")
        self.changed.emit()
        return runner.sweep_id

    # ------------------------------------------------------------ pipelines
    def pipeline_records(self, project: str | None = None):
        live = {self.pipeline.pipeline_id} if self.pipeline_running() else set()
        records = self.pipelines.list(project)
        for rec in records:
            if rec.state == "running" and rec.pipeline_id not in live:
                rec.state, rec.message = "stopped", rec.message or "interrupted"
                self.pipelines.save(rec)
        return records

    def pipeline_running(self) -> bool:
        return self.pipeline is not None and self.pipeline.thread is not None \
            and self.pipeline.thread.is_alive()

    def start_pipeline(self, graph: Graph, project: str = "", base=None,  # noqa: ANN001
                       resume_from: str = "") -> str | None:
        from ai_made_easy.core.pipelines.runner import PipelineError, PipelineRunner, resume

        if self.pipeline_running():
            self.log.error("a pipeline is already running")
            return None
        try:
            runner = resume(self.manager, self.pipelines, resume_from, base) if resume_from \
                else PipelineRunner(self.manager, self.pipelines, graph, project, base)
        except (PipelineError, KeyError) as exc:
            self.log.error(f"could not start the pipeline: {exc}")
            return None
        if not resume_from:
            runner.start()
        self.pipeline = runner
        self._pipeline_timer.start()
        self.log.info(f"pipeline {runner.pipeline_id} started — {len(runner.record.order)} "
                      "stages")
        self.changed.emit()
        return runner.pipeline_id

    def stop_pipeline(self) -> None:
        if self.pipeline_running():
            self.pipeline.stop()
            self.log.info("stopping the pipeline")

    def _poll_pipeline(self) -> None:
        runner = self.pipeline
        if runner is None:
            self._pipeline_timer.stop()
            return
        self.changed.emit()
        if not runner.thread.is_alive():
            self._pipeline_timer.stop()
            record = self.pipelines.get(runner.pipeline_id)
            self.pipeline = None
            self.log.info(f"pipeline {record.pipeline_id} {record.state}")
            self.changed.emit()

    def stop_sweep(self) -> None:
        if self.is_running():
            self.runner.stop()
            self.log.info("stopping the sweep after the current trial")

    def _poll(self) -> None:
        runner = self.runner
        if runner is None:
            self._timer.stop()
            return
        try:
            record = self.sweeps.get(runner.sweep_id)
        except KeyError:
            return
        signature = sum(1 for t in record.trials if t["state"] not in ("pending", "running"))
        signature = signature * 10 + len(record.trials)
        if signature != self._last_trials:
            self._last_trials = signature
            self.changed.emit()
        if not runner.thread.is_alive():
            self._timer.stop()
            self.runner = None
            best = record.best
            self.log.info(f"sweep {record.sweep_id} {record.state}"
                          + (f" — best {record.spec.get('metric')} = {best['score']:.4g} "
                             f"(trial {best['number']})" if best else ""))
            self.changed.emit()
            self.sweep_finished.emit(record.sweep_id, record.state)

    def shutdown(self) -> None:
        if self.is_running():
            self.runner.stop()
        if self.pipeline_running():
            self.pipeline.stop()
        self.manager.stop_all()
