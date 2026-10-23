"""Experiments dialogs: compare runs side by side, and set up a hyperparameter sweep."""
from __future__ import annotations

import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ai_made_easy.core.runs.history import RunRecord, compare
from ai_made_easy.ui.features.experiments import _SERIES, _duration, _fmt


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
        if any(data.get("design") or []):
            rows += [("Run", "design vs the first", ["—" if i == 0 else
                                                       (d["summary"] if d else "not saved")
                                                       for i, d in enumerate(data["design"])])]
        rows += [("Run", "status", data["status"]),
                 ("Run", "duration", [_duration(d) for d in data["duration"]]),
                 ("Run", "data" if data["same_data"] else "data (DIFFERENT)",
                  [d or "—" for d in data["data"]])]
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
        self.skip_budget = QtWidgets.QCheckBox("Skip trials over the project budget")
        self.skip_budget.setChecked(True)
        self.skip_budget.setToolTip("Trials whose estimated memory, latency or size break the "
                                    "budget set in the Summary are recorded but not trained")
        form.addRow("", self.skip_budget)
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
                "max_trials": self.trials.value(), "seed": self.seed.value(),
                "skip_over_budget": self.skip_budget.isChecked()}

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

    def prefill(self, spec: dict, note: str = "") -> None:
        """Fill the form from a spec (3.7: sweeping again near the best)."""
        if note:
            self.setWindowTitle("Sweep Again Near the Best")
            self.error.setObjectName("blockMeta")
            self.error.setText(note)
        dims = {f"{d['node']}.{d['param']}": d for d in spec.get("dimensions", [])}
        for row, p in enumerate(self._params):
            d = dims.get(p["key"])
            self.table.item(row, 0).setCheckState(
                QtCore.Qt.CheckState.Checked if d else QtCore.Qt.CheckState.Unchecked)
            if not d:
                continue
            if p["kind"] == "choice" or d.get("values"):
                self.table.item(row, 5).setText(", ".join(str(v) for v in d.get("values", [])))
            else:
                self.table.item(row, 4).setText(_fmt(d.get("low")))
                self.table.item(row, 5).setText(_fmt(d.get("high")))
                self.table.item(row, 6).setCheckState(
                    QtCore.Qt.CheckState.Checked if d.get("log")
                    else QtCore.Qt.CheckState.Unchecked)
        self.metric.setCurrentText(spec.get("metric", self.metric.currentText()))
        self.direction.setCurrentText({"min": "Minimize", "max": "Maximize"}.get(
            spec.get("direction", ""), "Automatic"))
        index = self.strategy.findData(spec.get("strategy"))
        if index >= 0:
            self.strategy.setCurrentIndex(index)
        self.trials.setValue(int(spec.get("max_trials", self.trials.value())))
        self.seed.setValue(int(spec.get("seed", 0)))
        self.skip_budget.setChecked(bool(spec.get("skip_over_budget", True)))
