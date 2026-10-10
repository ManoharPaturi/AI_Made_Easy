"""New from Task: the task-first wizard (task → data → budget → ranked recipes).

Detection and ranking run on a worker thread; the dialog ends with either a design to
open (``result_graph``) or an AutoML search to start (``automl_spec``).
"""
from __future__ import annotations

import threading
from html import escape
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from ai_made_easy.core import api

STEPS = ("Task", "Data", "Budget", "Recipe")


def _memory(gb: float) -> str:
    return f"{max(1, round(gb * 1024))} MB" if gb < 0.1 else f"{gb:.1f} GB"


def _count(n: int | None) -> str:
    if n is None:
        return "—"
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.1f}k" if n >= 1e3 else str(n)


class NewProjectWizard(QtWidgets.QDialog):
    _detected = QtCore.Signal(object, str)      # facts dict | None, error
    _ranked = QtCore.Signal(object, str)        # suggestions | None, error

    def __init__(self, parent, start_dir: Path | None = None, project: str = ""):
        super().__init__(parent)
        self.setWindowTitle("New from Task")
        self.resize(900, 620)
        self._start_dir = start_dir or Path.home()
        self._project = project
        self.result_graph: dict | None = None
        self.automl_spec: dict | None = None
        self._tasks = {t["id"]: t for t in api.wizard_tasks()["tasks"]}
        self._task: dict | None = None
        self._facts: dict | None = None
        self._rows: list[dict] = []
        self._knob_widgets: dict[str, QtWidgets.QWidget] = {}

        layout = QtWidgets.QVBoxLayout(self)
        self.steps = QtWidgets.QLabel()
        self.steps.setObjectName("blockMeta")
        layout.addWidget(self.steps)
        self.stack = QtWidgets.QStackedWidget()
        layout.addWidget(self.stack, 1)
        self.stack.addWidget(self._task_page())
        self.stack.addWidget(self._data_page())
        self.stack.addWidget(self._budget_page())
        self.stack.addWidget(self._recipe_page())

        buttons = QtWidgets.QHBoxLayout()
        self.back = QtWidgets.QPushButton("Back")
        self.back.clicked.connect(lambda: self._go(self.stack.currentIndex() - 1))
        self.next = QtWidgets.QPushButton("Next")
        self.next.setDefault(True)
        self.next.clicked.connect(lambda: self._go(self.stack.currentIndex() + 1))
        self.trials = QtWidgets.QSpinBox(minimum=1, maximum=500, value=12)
        self.epochs = QtWidgets.QSpinBox(minimum=0, maximum=1000, value=0)
        self.epochs.setSpecialValueText("default")
        self.automl = QtWidgets.QPushButton("Run AutoML")
        self.automl.setToolTip("Train every recipe once, then tune recipes and settings; "
                               "weak trials stop early")
        self.automl.clicked.connect(self._start_automl)
        self.create = QtWidgets.QPushButton("Create Design")
        self.create.clicked.connect(self._create)
        cancel = QtWidgets.QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        self._automl_box = QtWidgets.QWidget()
        box = QtWidgets.QHBoxLayout(self._automl_box)
        box.setContentsMargins(0, 0, 0, 0)
        for widget in (QtWidgets.QLabel("trials"), self.trials, QtWidgets.QLabel("epochs"),
                       self.epochs, self.automl):
            box.addWidget(widget)
        buttons.addWidget(self.back)
        buttons.addStretch(1)
        buttons.addWidget(self._automl_box)
        buttons.addWidget(self.create)
        buttons.addWidget(self.next)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)

        self._detected.connect(self._on_detected)
        self._ranked.connect(self._on_ranked)
        self._go(0)

    # ------------------------------------------------------------------ pages
    def _task_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search tasks (images, forecast, text, …)")
        self.search.textChanged.connect(self._fill_tasks)
        layout.addWidget(self.search)
        self.task_tree = QtWidgets.QTreeWidget()
        self.task_tree.setHeaderLabels(["Task", "Data", "AutoML"])
        self.task_tree.setColumnWidth(0, 520)
        self.task_tree.itemSelectionChanged.connect(self._pick_task)
        self.task_tree.itemDoubleClicked.connect(lambda *_: self._go(1))
        layout.addWidget(self.task_tree, 1)
        self.task_desc = QtWidgets.QLabel()
        self.task_desc.setWordWrap(True)
        self.task_desc.setObjectName("blockDesc")
        layout.addWidget(self.task_desc)
        self._fill_tasks()
        return page

    def _fill_tasks(self) -> None:
        query = self.search.text().lower() if hasattr(self, "search") else ""
        self.task_tree.clear()
        groups: dict[str, QtWidgets.QTreeWidgetItem] = {}
        for task in self._tasks.values():
            text = f"{task['label']} {task['description']} {' '.join(task['modalities'])}"
            if query and query not in text.lower():
                continue
            group = groups.get(task["family_label"])
            if group is None:
                group = QtWidgets.QTreeWidgetItem([task["family_label"]])
                group.setFlags(group.flags() & ~QtCore.Qt.ItemFlag.ItemIsSelectable)
                self.task_tree.addTopLevelItem(group)
                groups[task["family_label"]] = group
            item = QtWidgets.QTreeWidgetItem([task["label"], " · ".join(task["modalities"]),
                                              "yes" if task["automl"] else ""])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, task["id"])
            item.setToolTip(0, task["description"])
            group.addChild(item)
        self.task_tree.expandAll()

    def _pick_task(self) -> None:
        items = self.task_tree.selectedItems()
        task_id = items[0].data(0, QtCore.Qt.ItemDataRole.UserRole) if items else None
        task = self._tasks.get(task_id) if task_id else None
        if task is not None and task is not self._task:
            self._task, self._facts = task, None
            self.task_desc.setText(task["description"])
            self.modality.clear()
            self.modality.addItems(task["demo_modalities"])
            self.facts_label.setText("")
        self._update_buttons()

    def _data_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        self.use_demo = QtWidgets.QRadioButton("Use demo data (each recipe brings a small "
                                               "dataset, no files needed)")
        self.use_own = QtWidgets.QRadioButton("Use my data")
        self.use_demo.setChecked(True)
        self.use_demo.toggled.connect(self._data_mode)
        layout.addWidget(self.use_demo)
        demo_row = QtWidgets.QHBoxLayout()
        demo_row.addSpacing(24)
        demo_row.addWidget(QtWidgets.QLabel("Kind of data"))
        self.modality = QtWidgets.QComboBox()
        demo_row.addWidget(self.modality)
        demo_row.addStretch(1)
        layout.addLayout(demo_row)
        layout.addWidget(self.use_own)
        self.own_box = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(self.own_box)
        row = QtWidgets.QHBoxLayout()
        self.path = QtWidgets.QLineEdit()
        self.path.setPlaceholderText("a table file or a dataset folder")
        file_btn = QtWidgets.QPushButton("File…")
        file_btn.clicked.connect(self._browse_file)
        folder_btn = QtWidgets.QPushButton("Folder…")
        folder_btn.clicked.connect(self._browse_folder)
        row.addWidget(self.path, 1)
        row.addWidget(file_btn)
        row.addWidget(folder_btn)
        form.addRow("Data", row)
        self.target = QtWidgets.QLineEdit()
        self.target.setPlaceholderText("optional: the column to predict")
        form.addRow("Target column", self.target)
        self.detect_btn = QtWidgets.QPushButton("Detect")
        self.detect_btn.clicked.connect(self._detect)
        form.addRow("", self.detect_btn)
        layout.addWidget(self.own_box)
        hint = QtWidgets.QLabel("Tables (.csv, .tsv, .parquet, .xlsx), a folder with one "
                                "sub-folder per class, or COCO / YOLO / VOC / images + masks "
                                "folders.")
        hint.setObjectName("blockMeta")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.facts_label = QtWidgets.QLabel()
        self.facts_label.setWordWrap(True)
        self.facts_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        layout.addWidget(self.facts_label)
        layout.addStretch(1)
        self._data_mode()
        return page

    def _budget_page(self) -> QtWidgets.QWidget:
        from ai_made_easy.core import budget

        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        note = QtWidgets.QLabel("Optional. Recipes that would not fit are ranked lower and "
                                "flagged; AutoML skips them.")
        note.setObjectName("blockMeta")
        layout.addWidget(note)
        form = QtWidgets.QFormLayout()
        self.device = QtWidgets.QComboBox()
        self.device.addItem("No device", "")
        for device in budget.devices().values():
            self.device.addItem(f"{device.label} ({device.memory_gb:g} GB)", device.id)
        form.addRow("Target device", self.device)
        self.limits: dict[str, QtWidgets.QDoubleSpinBox] = {}
        for key, label, suffix in (("max_train_memory_gb", "Training memory", " GB"),
                                   ("max_latency_ms", "Inference latency", " ms"),
                                   ("max_params_m", "Parameters", " M")):
            spin = QtWidgets.QDoubleSpinBox(minimum=0, maximum=1e6, decimals=2)
            spin.setSpecialValueText("no limit")
            spin.setSuffix(suffix)
            self.limits[key] = spin
            form.addRow(label, spin)
        layout.addLayout(form)
        layout.addStretch(1)
        return page

    def _recipe_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(page)
        self.recipe_list = QtWidgets.QListWidget()
        self.recipe_list.setWordWrap(True)
        self.recipe_list.currentRowChanged.connect(self._pick_recipe)
        layout.addWidget(self.recipe_list, 3)
        side = QtWidgets.QWidget()
        self.side = QtWidgets.QVBoxLayout(side)
        self.recipe_detail = QtWidgets.QLabel()
        self.recipe_detail.setWordWrap(True)
        self.recipe_detail.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.side.addWidget(self.recipe_detail)
        self.knob_form = QtWidgets.QFormLayout()
        self.side.addLayout(self.knob_form)
        self.side.addStretch(1)
        layout.addWidget(side, 2)
        return page

    # ------------------------------------------------------------------ actions
    def _data_mode(self) -> None:
        own = self.use_own.isChecked()
        self.own_box.setEnabled(own)
        self.modality.setEnabled(not own)
        if not own:
            self._facts = None
            self.facts_label.setText("")
        self._update_buttons()

    def _browse_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose a table", str(self._start_dir),
            "Tables (*.csv *.tsv *.parquet *.xlsx *.json *.jsonl *.txt);;All files (*)")
        if path:
            self.path.setText(path)

    def _browse_folder(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose a dataset folder",
                                                          str(self._start_dir))
        if path:
            self.path.setText(path)

    def _detect(self) -> None:
        path, target = self.path.text().strip(), self.target.text().strip()
        if not path or self._task is None:
            return
        self.detect_btn.setEnabled(False)
        self.facts_label.setText("Reading the data…")
        task = self._task["id"]

        def work() -> None:
            try:
                self._detected.emit(api.detect_data(path, target, task), "")
            except Exception as exc:  # noqa: BLE001 — shown in the dialog
                self._detected.emit(None, str(exc))

        threading.Thread(target=work, daemon=True).start()

    def _on_detected(self, facts: dict | None, error: str) -> None:
        self.detect_btn.setEnabled(True)
        if facts is None or facts.get("error"):
            self._facts = None
            self.facts_label.setText(f"<span style='color:#e5484d'>"
                                     f"{escape(error or facts.get('error', ''))}</span>")
        else:
            self._facts = facts
            lines = [f"<b>{escape(facts['summary'])}</b>"]
            if facts.get("target"):
                classes = ", ".join(facts["classes"][:8])
                lines.append(f"target: {escape(facts['target'])}"
                             + (f" · classes: {escape(classes)}" if classes else ""))
            lines += [escape(d) for d in facts.get("details", [])]
            lines += [f"⚠ {escape(w)}" for w in facts.get("warnings", [])]
            if self._task and self._task["id"] not in facts.get("tasks", []):
                lines.append("⚠ this data looks like "
                             + escape(" / ".join(facts.get("tasks", [])) or "another task")
                             + f"; recipes for {escape(self._task['label'])} may not read it")
            self.facts_label.setText("<br>".join(lines))
        self._update_buttons()

    def _budget(self) -> dict:
        out: dict = {k: s.value() for k, s in self.limits.items() if s.value() > 0}
        if self.device.currentData():
            out["device"] = self.device.currentData()
        return out

    def _rank(self) -> None:
        self.recipe_list.clear()
        self.recipe_list.addItem("Ranking recipes…")
        self._rows = []
        task, facts, budget = self._task["id"], self._facts, self._budget()
        modality = "" if facts else self.modality.currentText()

        def work() -> None:
            try:
                out = api.recommend_recipes(task, facts=facts, budget=budget,
                                            modality=modality)
                self._ranked.emit(out["suggestions"], "")
            except Exception as exc:  # noqa: BLE001 — shown in the dialog
                self._ranked.emit(None, str(exc))

        threading.Thread(target=work, daemon=True).start()

    def _on_ranked(self, rows: list | None, error: str) -> None:
        self.recipe_list.clear()
        self._rows = rows or []
        if rows is None:
            self.recipe_list.addItem(f"Could not rank recipes: {error}")
        elif not rows:
            self.recipe_list.addItem("No recipe reads this data for this task.")
        for i, row in enumerate(self._rows):
            r = row["recipe"]
            state = "ready" if row["ready"] else "not ready"
            lines = [f"{i + 1}. {r['title']}   [{r['tier_label']} · {state}]", r["description"]]
            lines += [f"  ✓ {x}" for x in row["reasons"]]
            lines += [f"  ! {x}" for x in row["cautions"] + [f"over budget: {o}"
                                                             for o in row["over_budget"]]]
            lines += [f"  ✗ {x}" for x in row["errors"]]
            est = row["estimate"]
            if est.get("params") is not None:
                lines.append(f"  {_count(est['params'])} parameters"
                             + (f" · ~{_memory(est['train_memory_gb'])} to train"
                                if est.get("train_memory_gb") is not None else ""))
            item = QtWidgets.QListWidgetItem("\n".join(lines))
            if not row["ready"]:
                item.setForeground(self.palette().color(self.palette().ColorRole.PlaceholderText))
            self.recipe_list.addItem(item)
        if self._rows:
            best = next((i for i, r in enumerate(self._rows) if r["ready"]), 0)
            self.recipe_list.setCurrentRow(best)
        self._update_buttons()

    def _pick_recipe(self, index: int) -> None:
        while self.knob_form.rowCount():
            self.knob_form.removeRow(0)
        self._knob_widgets = {}
        if not 0 <= index < len(self._rows):
            self.recipe_detail.setText("")
            self._update_buttons()
            return
        recipe = self._rows[index]["recipe"]
        detail = f"<b>{escape(recipe['title'])}</b><br>{escape(recipe['description'])}"
        if recipe.get("strengths"):
            detail += f"<br><i>{escape(recipe['strengths'])}</i>"
        detail += "<br><br><b>Settings</b>" if recipe["knobs"] else \
            "<br><br>This recipe has no settings."
        self.recipe_detail.setText(detail)
        for knob in recipe["knobs"]:
            if knob["kind"] == "choice":
                widget = QtWidgets.QComboBox()
                for value in knob["values"]:
                    widget.addItem(str(value), value)
                widget.setCurrentIndex(max(0, widget.findData(knob["default"])))
            elif knob["kind"] == "int":
                widget = QtWidgets.QSpinBox(minimum=int(knob["low"] or 0),
                                            maximum=int(knob["high"] or 10**9),
                                            value=int(knob["default"]))
            else:
                low = float(knob["low"] or 0)
                widget = QtWidgets.QDoubleSpinBox(minimum=low, maximum=float(knob["high"] or 1e9),
                                                  decimals=6 if 0 < low < 1e-3 else 3,
                                                  value=float(knob["default"]))
                widget.setSingleStep(float(knob["default"]) / 2 or 0.1)
            widget.setToolTip(knob.get("help", ""))
            self.knob_form.addRow(knob["name"].replace("_", " "), widget)
            self._knob_widgets[knob["name"]] = widget
        self._update_buttons()

    def _knobs(self) -> dict:
        out = {}
        for name, widget in self._knob_widgets.items():
            out[name] = widget.currentData() if isinstance(widget, QtWidgets.QComboBox) \
                else widget.value()
        return out

    def _current(self) -> dict | None:
        index = self.recipe_list.currentRow()
        return self._rows[index] if 0 <= index < len(self._rows) else None

    def _create(self) -> None:
        row = self._current()
        if row is None or self._task is None:
            return
        try:
            self.result_graph = api.build_recipe(row["recipe"]["id"], self._task["id"],
                                                 facts=self._facts, knobs=self._knobs(),
                                                 budget=self._budget())
        except api.ApiError as exc:
            QtWidgets.QMessageBox.warning(self, "New from Task", str(exc))
            return
        self.accept()

    def _start_automl(self) -> None:
        if self._task is None:
            return
        self.automl_spec = {"task": self._task["id"], "facts": self._facts or {},
                            "modality": "" if self._facts else self.modality.currentText(),
                            "budget": self._budget(), "max_trials": self.trials.value(),
                            "epochs": self.epochs.value()}
        self.accept()

    # ------------------------------------------------------------------ navigation
    def _go(self, index: int) -> None:
        index = max(0, min(index, len(STEPS) - 1))
        if index > 0 and self._task is None:
            return
        if index == 3:
            self._rank()
        self.stack.setCurrentIndex(index)
        title = f": {self._task['label']}" if self._task and index else ""
        self.steps.setText("&nbsp;&nbsp;›&nbsp;&nbsp;".join(
            f"<b>{i + 1}. {s}</b>" if i == index else f"{i + 1}. {s}"
            for i, s in enumerate(STEPS)) + escape(title))
        self._update_buttons()

    def _update_buttons(self) -> None:
        if not hasattr(self, "automl"):          # still building the pages
            return
        index = self.stack.currentIndex()
        ready_data = self.use_demo.isChecked() or self._facts is not None
        self.back.setEnabled(index > 0)
        self.next.setVisible(index < 3)
        self.next.setEnabled(self._task is not None and (index != 1 or ready_data))
        row = self._current()
        self.create.setVisible(index == 3)
        self.create.setEnabled(row is not None and not row["errors"])
        self._automl_box.setVisible(index == 3 and bool(self._task and self._task["automl"]))
        self.automl.setEnabled(any(r["ready"] for r in self._rows))
