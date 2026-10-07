"""Experiments panel: run history, run comparison and hyperparameter sweeps.

Presentation only — records come from ``core.runs`` / ``core.sweeps``; the
AppContext owns the run manager and reacts to the signals emitted here.
"""
from __future__ import annotations

import time

import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ai_made_easy.core.runs.history import RunRecord, compare
from ai_made_easy.ui import icons

_SERIES = ("#4c8dff", "#f0883e", "#3fb950", "#d2a8ff", "#e3a008", "#56d4dd", "#ff7b72")
_MAX_METRIC_COLUMNS = 5


def _fmt(value) -> str:  # noqa: ANN001
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _when(ts: float | None) -> str:
    return time.strftime("%b %d %H:%M", time.localtime(ts)) if ts else "—"


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    seconds = int(seconds)
    return f"{seconds // 60}m {seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


def _readonly_table(columns: list[str]) -> QtWidgets.QTableWidget:
    table = QtWidgets.QTableWidget(0, len(columns))
    table.setHorizontalHeaderLabels(columns)
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
    table.setAlternatingRowColors(True)
    table.setSortingEnabled(True)
    table.horizontalHeader().setStretchLastSection(True)
    return table


class _Item(QtWidgets.QTableWidgetItem):
    """Sorts numerically when both cells hold numbers."""

    def __init__(self, text: str, sort_value=None):  # noqa: ANN001
        super().__init__(text)
        self._sort = sort_value

    def __lt__(self, other):  # noqa: ANN001
        a, b = self._sort, getattr(other, "_sort", None)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a < b
        return self.text().lower() < other.text().lower()


class ExperimentsPage(QtWidgets.QWidget):
    """Runs and Sweeps tabs over the persistent history."""

    restore_requested = QtCore.Signal(str)     # run_id
    results_requested = QtCore.Signal(str)     # run_id → load into Training panel
    folder_requested = QtCore.Signal(str)      # run_id
    delete_requested = QtCore.Signal(list)     # run_ids
    compare_requested = QtCore.Signal(list)    # run_ids
    tag_requested = QtCore.Signal(str)         # run_id
    new_sweep_requested = QtCore.Signal()
    stop_sweep_requested = QtCore.Signal(str)  # sweep_id
    apply_best_requested = QtCore.Signal(str)  # sweep_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("dockBody")
        self._records: list[RunRecord] = []
        self._sweeps: list = []
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self._build_runs(), "Runs")
        self.tabs.addTab(self._build_sweeps(), "Sweeps")
        layout.addWidget(self.tabs)

    # ------------------------------------------------------------- runs tab
    def _build_runs(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        box = QtWidgets.QVBoxLayout(page)
        box.setContentsMargins(0, 6, 0, 0)
        bar = QtWidgets.QHBoxLayout()
        self.scope = QtWidgets.QComboBox()
        self.scope.addItems(["This project", "All projects"])
        self.scope.setToolTip("Which runs to list")
        bar.addWidget(self.scope)
        self.count_label = QtWidgets.QLabel("")
        self.count_label.setObjectName("blockMeta")
        bar.addWidget(self.count_label)
        bar.addStretch(1)
        self.compare_btn = QtWidgets.QPushButton("Compare")
        self.compare_btn.setIcon(icons.icon("chart"))
        self.compare_btn.setToolTip("Compare the selected runs (select two or more)")
        self.results_btn = QtWidgets.QPushButton("Load Results")
        self.results_btn.setIcon(icons.icon("list"))
        self.results_btn.setToolTip("Show this run's curves and enable the analysis tools")
        self.restore_btn = QtWidgets.QPushButton("Restore Design")
        self.restore_btn.setIcon(icons.icon("undo"))
        self.restore_btn.setToolTip("Replace the canvas with the design this run trained")
        self.tag_btn = QtWidgets.QPushButton("Tags…")
        self.folder_btn = QtWidgets.QPushButton("")
        self.folder_btn.setIcon(icons.icon("folder"))
        self.folder_btn.setToolTip("Open the run folder")
        self.delete_btn = QtWidgets.QPushButton("")
        self.delete_btn.setIcon(icons.icon("trash"))
        self.delete_btn.setToolTip("Delete the selected runs and their files")
        for btn in (self.compare_btn, self.results_btn, self.restore_btn, self.tag_btn,
                    self.folder_btn, self.delete_btn):
            bar.addWidget(btn)
        box.addLayout(bar)

        self.runs_table = _readonly_table(["Run", "Status", "Framework", "Started",
                                           "Duration", "Epochs", "Tags"])
        self.runs_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.runs_table.itemSelectionChanged.connect(self._update_buttons)
        self.runs_table.doubleClicked.connect(lambda *_: self._emit_single(
            self.results_requested))
        box.addWidget(self.runs_table, 1)

        self.empty_label = QtWidgets.QLabel(
            "No runs yet. Every training run is recorded here with its design, "
            "parameters and metrics.")
        self.empty_label.setObjectName("blockMeta")
        self.empty_label.setWordWrap(True)
        box.addWidget(self.empty_label)

        self.compare_btn.clicked.connect(
            lambda: self.compare_requested.emit(self.selected_run_ids()))
        self.results_btn.clicked.connect(lambda: self._emit_single(self.results_requested))
        self.restore_btn.clicked.connect(lambda: self._emit_single(self.restore_requested))
        self.tag_btn.clicked.connect(lambda: self._emit_single(self.tag_requested))
        self.folder_btn.clicked.connect(lambda: self._emit_single(self.folder_requested))
        self.delete_btn.clicked.connect(
            lambda: self.delete_requested.emit(self.selected_run_ids()))
        self._update_buttons()
        return page

    def project_scope(self) -> bool:
        return self.scope.currentIndex() == 0

    def _emit_single(self, signal) -> None:  # noqa: ANN001
        ids = self.selected_run_ids()
        if len(ids) == 1:
            signal.emit(ids[0])

    def selected_run_ids(self) -> list[str]:
        rows = sorted({i.row() for i in self.runs_table.selectedIndexes()})
        out = []
        for row in rows:
            item = self.runs_table.item(row, 0)
            if item is not None:
                out.append(item.data(QtCore.Qt.ItemDataRole.UserRole))
        return out

    def _update_buttons(self) -> None:
        n = len(self.selected_run_ids())
        self.compare_btn.setEnabled(n >= 2)
        for btn in (self.results_btn, self.restore_btn, self.tag_btn, self.folder_btn):
            btn.setEnabled(n == 1)
        self.delete_btn.setEnabled(n >= 1)

    def set_runs(self, records: list[RunRecord]) -> None:
        selected = set(self.selected_run_ids())
        self._records = records
        metric_keys: list[str] = []
        for rec in records:
            for key in rec.final_metrics or rec.best_metrics:
                if key not in metric_keys:
                    metric_keys.append(key)
        metric_keys = metric_keys[:_MAX_METRIC_COLUMNS]
        columns = ["Run", "Status", "Framework", "Started", "Duration", "Epochs",
                   *metric_keys, "Tags"]
        table = self.runs_table
        table.setSortingEnabled(False)
        table.clear()
        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels(columns)
        table.setRowCount(len(records))
        for row, rec in enumerate(records):
            label = rec.name + (f"  · trial {rec.trial}" if rec.parent and rec.trial else "")
            first = _Item(label)
            first.setData(QtCore.Qt.ItemDataRole.UserRole, rec.run_id)
            first.setToolTip(f"{rec.run_id}\nproject: {rec.project or '—'}"
                             + (f"\n{rec.note}" if rec.note else "")
                             + (f"\n{rec.error[-300:]}" if rec.error else ""))
            cells = [first, _Item(rec.status), _Item(rec.framework),
                     _Item(_when(rec.created_at), rec.created_at),
                     _Item(_duration(rec.duration), rec.duration or 0),
                     _Item(str(rec.epochs_done), rec.epochs_done)]
            metrics = rec.final_metrics or rec.best_metrics
            cells += [_Item(_fmt(metrics.get(k)), metrics.get(k)) for k in metric_keys]
            cells.append(_Item(", ".join(rec.tags)))
            for col, item in enumerate(cells):
                table.setItem(row, col, item)
            if rec.run_id in selected:
                table.selectRow(row)
        table.setSortingEnabled(True)
        table.resizeColumnsToContents()
        self.count_label.setText(f"{len(records)} run(s)")
        self.empty_label.setVisible(not records)
        self._update_buttons()

    # ----------------------------------------------------------- sweeps tab
    def _build_sweeps(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        box = QtWidgets.QVBoxLayout(page)
        box.setContentsMargins(0, 6, 0, 0)
        bar = QtWidgets.QHBoxLayout()
        self.new_sweep_btn = QtWidgets.QPushButton("New Sweep…")
        self.new_sweep_btn.setObjectName("primaryButton")
        self.new_sweep_btn.setIcon(icons.icon("sparkle", color="#ffffff"))
        self.stop_sweep_btn = QtWidgets.QPushButton("Stop")
        self.stop_sweep_btn.setIcon(icons.icon("stop"))
        self.apply_best_btn = QtWidgets.QPushButton("Apply Best Parameters")
        self.apply_best_btn.setIcon(icons.icon("check"))
        self.apply_best_btn.setToolTip("Load the design with the best trial's values")
        bar.addWidget(self.new_sweep_btn)
        bar.addStretch(1)
        bar.addWidget(self.stop_sweep_btn)
        bar.addWidget(self.apply_best_btn)
        box.addLayout(bar)
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.sweeps_table = _readonly_table(["Sweep", "Status", "Strategy", "Trials",
                                             "Objective", "Best"])
        self.sweeps_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.sweeps_table.itemSelectionChanged.connect(self._show_trials)
        self.trials_table = _readonly_table(["#", "State", "Score", "Values", "Note"])
        split.addWidget(self.sweeps_table)
        split.addWidget(self.trials_table)
        split.setSizes([380, 520])
        box.addWidget(split, 1)
        self.new_sweep_btn.clicked.connect(self.new_sweep_requested.emit)
        self.stop_sweep_btn.clicked.connect(lambda: self._emit_sweep(self.stop_sweep_requested))
        self.apply_best_btn.clicked.connect(lambda: self._emit_sweep(self.apply_best_requested))
        self._update_sweep_buttons()
        return page

    def _emit_sweep(self, signal) -> None:  # noqa: ANN001
        sweep = self.selected_sweep()
        if sweep is not None:
            signal.emit(sweep.sweep_id)

    def selected_sweep(self):
        rows = {i.row() for i in self.sweeps_table.selectedIndexes()}
        if len(rows) != 1:
            return None
        item = self.sweeps_table.item(rows.pop(), 0)
        sweep_id = item.data(QtCore.Qt.ItemDataRole.UserRole) if item else None
        return next((s for s in self._sweeps if s.sweep_id == sweep_id), None)

    def _update_sweep_buttons(self) -> None:
        sweep = self.selected_sweep()
        self.stop_sweep_btn.setEnabled(bool(sweep and sweep.state == "running"))
        self.apply_best_btn.setEnabled(bool(sweep and sweep.best))

    def set_sweeps(self, sweeps: list) -> None:
        current = self.selected_sweep()
        current_id = current.sweep_id if current else (sweeps[0].sweep_id if sweeps else None)
        self._sweeps = sweeps
        table = self.sweeps_table
        table.setSortingEnabled(False)
        table.setRowCount(len(sweeps))
        for row, sw in enumerate(sweeps):
            spec = sw.spec
            done = sum(1 for t in sw.trials if t["state"] not in ("pending", "running"))
            direction = spec.get("direction") or ""
            first = _Item(f"{sw.name}  {_when(sw.created_at)}", sw.created_at)
            first.setData(QtCore.Qt.ItemDataRole.UserRole, sw.sweep_id)
            first.setToolTip(sw.sweep_id + (f"\n{sw.message}" if sw.message else ""))
            cells = [first, _Item(sw.state), _Item(spec.get("strategy", "")),
                     _Item(f"{done}/{spec.get('max_trials', '?')}", done),
                     _Item(f"{spec.get('metric', '')} {direction}".strip()),
                     _Item(_fmt(sw.best["score"]) if sw.best else "—",
                           sw.best["score"] if sw.best else None)]
            for col, item in enumerate(cells):
                table.setItem(row, col, item)
            if sw.sweep_id == current_id:
                table.selectRow(row)
        table.setSortingEnabled(True)
        table.resizeColumnsToContents()
        self._show_trials()

    def _show_trials(self) -> None:
        sweep = self.selected_sweep()
        trials = sweep.trials if sweep else []
        best = sweep.best["number"] if sweep and sweep.best else None
        table = self.trials_table
        table.setSortingEnabled(False)
        table.setRowCount(len(trials))
        for row, t in enumerate(trials):
            values = ", ".join(f"{k.split('.', 1)[-1]}={_fmt(v)}" for k, v in t["values"].items())
            cells = [_Item(str(t["number"]) + ("  ★" if t["number"] == best else ""),
                           t["number"]),
                     _Item(t["state"]), _Item(_fmt(t["score"]), t["score"]),
                     _Item(values), _Item(t.get("message", ""))]
            cells[3].setToolTip("\n".join(f"{k} = {v}" for k, v in t["values"].items()))
            for col, item in enumerate(cells):
                table.setItem(row, col, item)
        table.setSortingEnabled(True)
        table.resizeColumnsToContents()
        self._update_sweep_buttons()


# ------------------------------------------------------------------ dialogs

class CompareDialog(QtWidgets.QDialog):
    """Metrics side by side, the parameters that differ, overlaid curves."""

    def __init__(self, parent, records: list[RunRecord], epochs: dict[str, list[dict]]):
        super().__init__(parent)
        self.setWindowTitle(f"Compare {len(records)} Runs")
        self.resize(1000, 680)
        self._records = records
        self._epochs = epochs
        data = compare(records)
        layout = QtWidgets.QVBoxLayout(self)
        names = [f"{r.name}\n{r.run_id}" for r in records]

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        rows: list[tuple[str, str, list]] = []
        rows += [("Metric (test)", k, v) for k, v in data["final_metrics"].items()
                 if any(x is not None for x in v)]
        rows += [("Best epoch", k, v) for k, v in data["best_metrics"].items()
                 if any(x is not None for x in v)]
        rows += [("Parameter", k, v) for k, v in data["params"].items()]
        rows += [("Run", "status", data["status"]),
                 ("Run", "duration", [_duration(d) for d in data["duration"]])]
        self.table = QtWidgets.QTableWidget(len(rows), len(records) + 2)
        self.table.setHorizontalHeaderLabels(["Section", "Name", *names])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        for row, (section, name, values) in enumerate(rows):
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(section))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(name))
            numeric = [v for v in values if isinstance(v, (int, float)) and not isinstance(
                v, bool)]
            best = None
            if section != "Parameter" and len(numeric) > 1:
                from ai_made_easy.core.runs.history import lower_is_better

                best = min(numeric) if lower_is_better(name) else max(numeric)
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(_fmt(value))
                if best is not None and value == best:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                self.table.setItem(row, col + 2, item)
        self.table.resizeColumnsToContents()
        split.addWidget(self.table)

        chart = QtWidgets.QWidget()
        chart_box = QtWidgets.QVBoxLayout(chart)
        chart_box.setContentsMargins(0, 0, 0, 0)
        picker = QtWidgets.QHBoxLayout()
        picker.addWidget(QtWidgets.QLabel("Curve"))
        self.metric_combo = QtWidgets.QComboBox()
        keys: list[str] = []
        for run_epochs in epochs.values():
            for ev in run_epochs:
                for k in ev.get("metrics", {}):
                    if k not in keys and k != "lr":
                        keys.append(k)
        self.metric_combo.addItems(keys)
        picker.addWidget(self.metric_combo)
        picker.addStretch(1)
        chart_box.addLayout(picker)
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.addLegend(offset=(8, 8))
        self.plot.setLabel("bottom", "epoch")
        chart_box.addWidget(self.plot, 1)
        split.addWidget(chart)
        layout.addWidget(split, 1)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.metric_combo.currentTextChanged.connect(self._plot)
        self._plot(self.metric_combo.currentText())

    def _plot(self, key: str) -> None:
        self.plot.clear()
        for i, rec in enumerate(self._records):
            points = [(ev.get("epoch"), ev.get("metrics", {}).get(key))
                      for ev in self._epochs.get(rec.run_id, [])]
            points = [(x, y) for x, y in points if isinstance(y, (int, float))]
            if not points:
                continue
            color = _SERIES[i % len(_SERIES)]
            self.plot.plot([p[0] for p in points], [p[1] for p in points],
                           pen=pg.mkPen(color, width=2), symbol="o", symbolSize=4,
                           symbolBrush=color, symbolPen=None,
                           name=f"{rec.name} ({rec.run_id[-4:]})")



class SweepDialog(QtWidgets.QDialog):
    """Pick parameters and ranges, the objective and the search strategy."""

    def __init__(self, parent, params: list[dict], metrics: list[str]):
        super().__init__(parent)
        self.setWindowTitle("New Hyperparameter Sweep")
        self.resize(900, 600)
        self._params = params
        layout = QtWidgets.QVBoxLayout(self)
        hint = QtWidgets.QLabel(
            "Tick the parameters to vary. Ranges use low/high (log scale for rates); "
            "choices take comma-separated values. Trials whose design fails validation "
            "are skipped.")
        hint.setObjectName("blockMeta")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.filter = QtWidgets.QLineEdit()
        self.filter.setPlaceholderText("Filter parameters…")
        self.filter.textChanged.connect(self._filter)
        layout.addWidget(self.filter)

        self.table = QtWidgets.QTableWidget(len(params), 7)
        self.table.setHorizontalHeaderLabels(["", "Block", "Parameter", "Current", "Low",
                                              "High / Values", "Log"])
        self.table.verticalHeader().setVisible(False)
        for row, p in enumerate(params):
            check = QtWidgets.QTableWidgetItem()
            check.setFlags(QtCore.Qt.ItemFlag.ItemIsUserCheckable
                           | QtCore.Qt.ItemFlag.ItemIsEnabled)
            check.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.table.setItem(row, 0, check)
            for col, text in ((1, f"{p['block']} ({p['node']})"), (2, p["param"]),
                              (3, _fmt(p["current"]))):
                item = QtWidgets.QTableWidgetItem(text)
                item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
                self.table.setItem(row, col, item)
            if p["kind"] == "choice":
                low = QtWidgets.QTableWidgetItem("")
                low.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
                self.table.setItem(row, 4, low)
                self.table.setItem(row, 5, QtWidgets.QTableWidgetItem(
                    ", ".join(str(v) for v in p["values"])))
                log = QtWidgets.QTableWidgetItem("")
                log.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
            else:
                self.table.setItem(row, 4, QtWidgets.QTableWidgetItem(_fmt(p["low"])))
                self.table.setItem(row, 5, QtWidgets.QTableWidgetItem(_fmt(p["high"])))
                log = QtWidgets.QTableWidgetItem()
                log.setFlags(QtCore.Qt.ItemFlag.ItemIsUserCheckable
                             | QtCore.Qt.ItemFlag.ItemIsEnabled)
                log.setCheckState(QtCore.Qt.CheckState.Checked if p.get("log")
                                  else QtCore.Qt.CheckState.Unchecked)
            self.table.setItem(row, 6, log)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table, 1)

        form = QtWidgets.QFormLayout()
        self.metric = QtWidgets.QComboBox()
        self.metric.setEditable(True)
        self.metric.addItems(metrics)
        self.direction = QtWidgets.QComboBox()
        self.direction.addItems(["Automatic", "Minimize", "Maximize"])
        self.strategy = QtWidgets.QComboBox()
        self.strategy.addItem("Bayesian (TPE)", "tpe")
        self.strategy.addItem("Random search", "random")
        self.strategy.addItem("Grid search", "grid")
        self.trials = QtWidgets.QSpinBox()
        self.trials.setRange(1, 500)
        self.trials.setValue(10)
        self.points = QtWidgets.QSpinBox()
        self.points.setRange(2, 20)
        self.points.setValue(3)
        self.points.setToolTip("Grid points per continuous range")
        self.seed = QtWidgets.QSpinBox()
        self.seed.setRange(0, 1_000_000)
        form.addRow("Objective metric", self.metric)
        form.addRow("Direction", self.direction)
        form.addRow("Strategy", self.strategy)
        form.addRow("Max trials", self.trials)
        form.addRow("Grid points", self.points)
        form.addRow("Sampler seed", self.seed)
        layout.addLayout(form)
        self.error = QtWidgets.QLabel("")
        self.error.setObjectName("errorText")
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Start Sweep")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._spec: dict | None = None

    def _filter(self, text: str) -> None:
        text = text.lower()
        for row, p in enumerate(self._params):
            hay = f"{p['block']} {p['node']} {p['param']}".lower()
            self.table.setRowHidden(row, bool(text) and text not in hay)

    def check_row(self, key: str, checked: bool = True) -> None:
        for row, p in enumerate(self._params):
            if p["key"] == key:
                self.table.item(row, 0).setCheckState(
                    QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked)

    def _parse_value(self, text: str, kind: str, template):  # noqa: ANN001
        text = text.strip()
        if isinstance(template, bool):
            return text.lower() in ("true", "1", "yes")
        if kind == "int" or isinstance(template, int):
            return int(float(text))
        try:
            return float(text) if kind == "float" or isinstance(template, float) else text
        except ValueError:
            return text

    def build_spec(self) -> dict:
        dims = []
        for row, p in enumerate(self._params):
            if self.table.item(row, 0).checkState() != QtCore.Qt.CheckState.Checked:
                continue
            high_text = self.table.item(row, 5).text()
            if p["kind"] == "choice":
                template = p["values"][0] if p["values"] else ""
                values = [self._parse_value(v, "choice", template)
                          for v in high_text.split(",") if v.strip()]
                dims.append({"node": p["node"], "param": p["param"], "kind": "choice",
                             "values": values})
                continue
            try:
                low = float(self.table.item(row, 4).text())
                high = float(high_text)
            except ValueError as exc:
                raise ValueError(f"{p['key']}: low and high must be numbers") from exc
            log = self.table.item(row, 6).checkState() == QtCore.Qt.CheckState.Checked
            dims.append({"node": p["node"], "param": p["param"], "kind": p["kind"],
                         "low": int(low) if p["kind"] == "int" else low,
                         "high": int(high) if p["kind"] == "int" else high,
                         "log": log, "points": self.points.value()})
        direction = {"Minimize": "min", "Maximize": "max"}.get(self.direction.currentText(), "")
        return {"dimensions": dims, "metric": self.metric.currentText().strip(),
                "direction": direction, "strategy": self.strategy.currentData(),
                "max_trials": self.trials.value(), "seed": self.seed.value()}

    def _accept(self) -> None:
        try:
            spec = self.build_spec()
        except ValueError as exc:
            self.error.setText(str(exc))
            return
        if not spec["dimensions"]:
            self.error.setText("Tick at least one parameter.")
            return
        if not spec["metric"]:
            self.error.setText("Choose the metric to optimise.")
            return
        self._spec = spec
        self.accept()

    def spec(self) -> dict | None:
        return self._spec
