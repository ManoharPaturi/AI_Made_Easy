"""RunConsoleStack: the bottom pages (Console + Training) and their
switching policy. TrainingPage is plots + buttons ONLY — its status label
has exactly one writer: the RunStore subscription.
"""
from __future__ import annotations

from datetime import datetime

import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ai_made_easy.ui.stores import LogBus, RunStore


class ConsolePage(QtWidgets.QPlainTextEdit):
    """The only renderer of the LogBus."""

    def __init__(self, log_bus: LogBus, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(5000)
        self.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        log_bus.logged.connect(self._append)

    def _append(self, level: str, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        prefix = {"error": "ERROR", "warning": "WARN"}.get(level, "info")
        self.appendPlainText(f"[{stamp}] {prefix}: {message}")


class TrainingPage(QtWidgets.QWidget):
    """Loss/score plots + run controls. Subscribes to RunStore + epochs."""

    train_clicked = QtCore.Signal()
    stop_clicked = QtCore.Signal()
    museum_clicked = QtCore.Signal()
    inspect_clicked = QtCore.Signal()
    card_clicked = QtCore.Signal()

    def __init__(self, run_store: RunStore, parent=None):
        super().__init__(parent)
        self._run_store = run_store

        pg.setConfigOptions(antialias=True, background="#26292f",
                            foreground="#c9d1d9")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        row = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("▶ Train (PyTorch)")
        self.start_btn.setToolTip("Generate the training script and run it "
                                  "in a managed subprocess")
        self.stop_btn = QtWidgets.QPushButton("■ Stop")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.train_clicked.emit)
        self.stop_btn.clicked.connect(self.stop_clicked.emit)
        row.addWidget(self.start_btn)
        row.addWidget(self.stop_btn)
        self.status = QtWidgets.QLabel("idle")
        row.addWidget(self.status, stretch=1)
        layout.addLayout(row)

        # Wave-1 insight buttons — enabled once a run leaves artifacts
        results = QtWidgets.QHBoxLayout()
        self.museum_btn = QtWidgets.QPushButton("🔍 Mistake Museum")
        self.museum_btn.setToolTip(
            "Browse what the model got wrong and learn how to fix it")
        self.inspect_btn = QtWidgets.QPushButton("👀 What is it looking at?")
        self.inspect_btn.setToolTip(
            "See WHERE the model looked (Grad-CAM heatmap + first layer)")
        self.card_btn = QtWidgets.QPushButton("🪪 Report Card")
        self.card_btn.setToolTip("A shareable card about your model")
        for btn, sig in ((self.museum_btn, self.museum_clicked),
                         (self.inspect_btn, self.inspect_clicked),
                         (self.card_btn, self.card_clicked)):
            btn.setEnabled(False)
            btn.clicked.connect(sig.emit)
            results.addWidget(btn)
        results.addStretch(1)
        layout.addLayout(results)

        plots = pg.GraphicsLayoutWidget()
        self.loss_plot = plots.addPlot(row=0, col=0, title="Loss")
        self.loss_plot.addLegend(offset=(10, 10))
        self.loss_plot.showGrid(x=True, y=True, alpha=0.2)
        self.loss_plot.setLabel("left", "loss")
        self.loss_plot.setLabel("bottom", "epoch")
        self.score_plot = plots.addPlot(row=1, col=0, title="Scores")
        self.score_plot.addLegend(offset=(10, 10))
        self.score_plot.showGrid(x=True, y=True, alpha=0.2)
        self.score_plot.setLabel("left", "score")
        self.score_plot.setLabel("bottom", "epoch")
        self.score_plot.setYRange(0.0, 1.02)
        layout.addWidget(plots)

        self._loss_curves: dict[str, pg.PlotDataItem] = {}
        self._score_curves: dict[str, pg.PlotDataItem] = {}
        self._epoch_x: list[float] = []
        self._series: dict[str, list[float]] = {}

        run_store.state_changed.connect(self._on_state)

    # single writer for button/status state
    def _on_state(self, state: str, kind: str) -> None:
        running = state == RunStore.RUNNING
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        if state == RunStore.IDLE:
            self.status.setText("idle")
        elif state == RunStore.RUNNING:
            self.status.setText(f"running ({kind})…")
        elif state == RunStore.FINISHED:
            self.status.setText(f"finished ({kind}) — {len(self._epoch_x)} epoch(s)")
        elif state == RunStore.FAILED:
            self.status.setText(f"failed ({kind}) — see Console")
        elif state == RunStore.STOPPED:
            self.status.setText(f"stopped after {len(self._epoch_x)} epoch(s)")

    def set_results_available(self, workdir) -> None:  # noqa: ANN001
        """Enable the insight buttons once a run left artifacts behind."""
        from pathlib import Path
        wd = Path(workdir)
        has = wd.joinpath("predictions.json").exists()
        for btn in (self.museum_btn, self.inspect_btn, self.card_btn):
            btn.setEnabled(has)

    # ------------------------------------------------------------- data

    def reset(self) -> None:
        self._epoch_x.clear()
        self._series.clear()
        for curves in (self._loss_curves, self._score_curves):
            for curve in curves.values():
                curve.setData([], [])
            curves.clear()

    def on_epoch(self, event: dict) -> None:
        epoch = int(event.get("epoch", len(self._epoch_x) + 1))
        total = int(event.get("total") or (epoch + 1))
        self._epoch_x.append(float(epoch))
        for key, value in event.get("metrics", {}).items():
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            self._series.setdefault(key, []).append(value)
            curves = self._loss_curves if "loss" in key else self._score_curves
            curve = curves.get(key)
            if curve is None:
                color = "#ff7b72" if key.startswith("train") else (
                    "#d29922" if "loss" in key else "#3fb950")
                plot = self.loss_plot if "loss" in key else self.score_plot
                curve = plot.plot(pen=pg.mkPen(color, width=2), name=key,
                                  symbol="o", symbolSize=4)
                curves[key] = curve
            curve.setData(self._epoch_x, self._series[key])
        ticks = [[(i, str(i)) for i in range(1, total + 1)]]
        self.loss_plot.getAxis("bottom").setTicks(ticks)
        self.score_plot.getAxis("bottom").setTicks(ticks)
        self.status.setText(f"running — epoch {epoch}/{total}")


class RunConsoleStack(QtWidgets.QTabWidget):
    """Bottom stack; auto-raises Training on run start, restores after."""

    def __init__(self, console_page: ConsolePage, training_page: TrainingPage,
                 run_store: RunStore, parent=None):
        super().__init__(parent)
        self.console_page = console_page
        self.training_page = training_page
        self._run_store = run_store
        self._previous = 0
        self.addTab(console_page, "🖨️ Console")
        self.addTab(training_page, "📈 Training")
        run_store.state_changed.connect(self._on_state)

    def _on_state(self, state: str, _kind: str) -> None:
        if state == RunStore.RUNNING:
            if self.currentIndex() != 1:
                self._previous = self.currentIndex()
            self.setCurrentIndex(1)
        elif state in (RunStore.FINISHED, RunStore.FAILED, RunStore.STOPPED):
            self.setCurrentIndex(self._previous)
