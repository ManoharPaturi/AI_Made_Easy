"""Data workspace: profile, browse and preview the design's dataset.

One dock page with a dataset picker and tabs: Overview (summary, findings,
class distribution), Columns (per-column statistics), Rows (a lazy table view
over the file), Samples (image thumbnails / text snippets by class), Splits
(samples per class in train / val / test) and Augmentation (what the model
sees in training vs evaluation).
"""
from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

_SEVERITY = {"error": "✕", "warning": "⚠", "info": "ℹ"}


def _fmt(value) -> str:  # noqa: ANN001
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


class FrameModel(QtCore.QAbstractTableModel):
    """Read-only Qt model over a pandas DataFrame (renders only visible cells)."""

    def __init__(self, frame=None, parent=None):  # noqa: ANN001
        super().__init__(parent)
        self._frame = frame

    def set_frame(self, frame) -> None:  # noqa: ANN001
        self.beginResetModel()
        self._frame = frame
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()) -> int:  # noqa: B008, N802
        return 0 if self._frame is None or parent.isValid() else len(self._frame)

    def columnCount(self, parent=QtCore.QModelIndex()) -> int:  # noqa: B008, N802
        return 0 if self._frame is None or parent.isValid() else len(self._frame.columns)

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):  # noqa: ANN001
        if self._frame is None or not index.isValid():
            return None
        if role in (QtCore.Qt.ItemDataRole.DisplayRole, QtCore.Qt.ItemDataRole.ToolTipRole):
            value = self._frame.iat[index.row(), index.column()]
            try:
                import pandas as pd

                if pd.isna(value):
                    return "" if role == QtCore.Qt.ItemDataRole.DisplayRole else "missing"
            except (TypeError, ValueError):
                pass
            return _fmt(value)
        return None

    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole):  # noqa: ANN001, N802
        if role != QtCore.Qt.ItemDataRole.DisplayRole or self._frame is None:
            return None
        if orientation == QtCore.Qt.Orientation.Horizontal:
            return str(self._frame.columns[section])
        return str(section + 1)


def _table(headers: list[str]) -> QtWidgets.QTableWidget:
    table = QtWidgets.QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
    table.horizontalHeader().setStretchLastSection(True)
    return table


def _fill(table: QtWidgets.QTableWidget, rows: list[list]) -> None:
    table.setSortingEnabled(False)
    table.setRowCount(len(rows))
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            if isinstance(value, QtWidgets.QWidget):
                table.setCellWidget(r, c, value)
                continue
            item = QtWidgets.QTableWidgetItem(_fmt(value))
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                item.setData(QtCore.Qt.ItemDataRole.UserRole, value)
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight
                                      | QtCore.Qt.AlignmentFlag.AlignVCenter)
            table.setItem(r, c, item)
    header = table.horizontalHeader()
    modes = {header.sectionResizeMode(c) for c in range(table.columnCount())}
    if QtWidgets.QHeaderView.ResizeMode.Stretch in modes:
        header.resizeSections()  # keep per-column modes (resizeColumnsToContents overrides them)
    else:
        table.resizeColumnsToContents()


def _bar(value: int, total: int) -> QtWidgets.QProgressBar:
    bar = QtWidgets.QProgressBar()
    bar.setRange(0, max(total, 1))
    bar.setValue(value)
    bar.setFormat(f"{value / max(total, 1):.1%}")
    bar.setMaximumHeight(16)
    return bar


class DataPage(QtWidgets.QWidget):
    dataset_selected = QtCore.Signal(str)        # node id ("" = none)
    refresh_requested = QtCore.Signal()
    open_file_requested = QtCore.Signal()
    rows_requested = QtCore.Signal()
    split_requested = QtCore.Signal()
    augment_requested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.profile = None
        self._rows_loaded = False
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Dataset"))
        self.dataset = QtWidgets.QComboBox()
        self.dataset.setMinimumContentsLength(24)
        self.dataset.currentIndexChanged.connect(
            lambda *_: self.dataset_selected.emit(self.dataset.currentData() or ""))
        bar.addWidget(self.dataset, 1)
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_btn.setToolTip("Re-read the data from disk")
        self.refresh_btn.clicked.connect(self.refresh_requested)
        open_btn = QtWidgets.QPushButton("Profile a File…")
        open_btn.setToolTip("Profile any table or class folder without adding it to the design")
        open_btn.clicked.connect(self.open_file_requested)
        bar.addWidget(self.refresh_btn)
        bar.addWidget(open_btn)
        layout.addLayout(bar)

        self.summary = QtWidgets.QLabel("Add a dataset block to the canvas to profile it.")
        self.summary.setObjectName("blockMeta")
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.summary)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        layout.addWidget(self.tabs, 1)

        # ---- overview
        overview = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.findings = QtWidgets.QListWidget()
        self.findings.setWordWrap(True)
        self.findings.setAlternatingRowColors(True)
        self.classes = _table(["Class", "Samples", "Share"])
        header = self.classes.horizontalHeader()
        for col, mode in ((0, "ResizeToContents"), (1, "ResizeToContents"), (2, "Stretch")):
            header.setSectionResizeMode(col, getattr(QtWidgets.QHeaderView.ResizeMode, mode))
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(QtWidgets.QLabel("Findings"))
        lv.addWidget(self.findings, 1)
        self.details = QtWidgets.QLabel()
        self.details.setObjectName("blockMeta")
        self.details.setWordWrap(True)
        lv.addWidget(self.details)
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(QtWidgets.QLabel("Class distribution"))
        rv.addWidget(self.classes, 1)
        overview.addWidget(left)
        overview.addWidget(right)
        overview.setSizes([520, 360])
        self.tabs.addTab(overview, "Overview")

        # ---- columns
        self.columns = _table(["Column", "Role", "Type", "Missing", "Unique", "Mean", "Std",
                               "Min", "Median", "Max", "Top values"])
        self.columns.setSortingEnabled(True)
        self.tabs.addTab(self.columns, "Columns")

        # ---- rows
        rows = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(rows)
        rl.setContentsMargins(0, 0, 0, 0)
        self.rows_info = QtWidgets.QLabel("")
        self.rows_info.setObjectName("blockMeta")
        rl.addWidget(self.rows_info)
        self.row_model = FrameModel()
        self.rows = QtWidgets.QTableView()
        self.rows.setModel(self.row_model)
        self.rows.setAlternatingRowColors(True)
        rl.addWidget(self.rows, 1)
        self.tabs.addTab(rows, "Rows")

        # ---- samples
        samples = QtWidgets.QWidget()
        sl = QtWidgets.QVBoxLayout(samples)
        sl.setContentsMargins(0, 0, 0, 0)
        sbar = QtWidgets.QHBoxLayout()
        sbar.addWidget(QtWidgets.QLabel("Class"))
        self.sample_class = QtWidgets.QComboBox()
        self.sample_class.currentIndexChanged.connect(lambda *_: self._show_samples())
        sbar.addWidget(self.sample_class, 1)
        sl.addLayout(sbar)
        self.sample_grid = QtWidgets.QListWidget()
        self.sample_grid.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.sample_grid.setIconSize(QtCore.QSize(96, 96))
        self.sample_grid.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.sample_grid.setMovement(QtWidgets.QListView.Movement.Static)
        self.sample_grid.setSpacing(6)
        self.sample_text = QtWidgets.QPlainTextEdit()
        self.sample_text.setReadOnly(True)
        self.sample_text.setObjectName("codeView")
        sl.addWidget(self.sample_grid, 1)
        sl.addWidget(self.sample_text, 1)
        self.tabs.addTab(samples, "Samples")

        # ---- splits
        splits = QtWidgets.QWidget()
        spl = QtWidgets.QVBoxLayout(splits)
        spl.setContentsMargins(0, 0, 0, 0)
        sp_bar = QtWidgets.QHBoxLayout()
        self.split_btn = QtWidgets.QPushButton("Preview Split")
        self.split_btn.setToolTip("Samples per class in each split, exactly as training splits")
        self.split_btn.clicked.connect(self.split_requested)
        self.split_info = QtWidgets.QLabel("Shows how the Train / Val / Test Split block "
                                           "divides the data.")
        self.split_info.setObjectName("blockMeta")
        self.split_info.setWordWrap(True)
        sp_bar.addWidget(self.split_btn)
        sp_bar.addWidget(self.split_info, 1)
        spl.addLayout(sp_bar)
        self.split_table = _table(["Class", "Train", "Validation", "Test"])
        spl.addWidget(self.split_table, 1)
        self.tabs.addTab(splits, "Splits")

        # ---- augmentation
        aug = QtWidgets.QWidget()
        al = QtWidgets.QVBoxLayout(aug)
        al.setContentsMargins(0, 0, 0, 0)
        a_bar = QtWidgets.QHBoxLayout()
        self.augment_btn = QtWidgets.QPushButton("Preview Augmentation")
        self.augment_btn.setToolTip("Run the design's image transforms on a few dataset images")
        self.augment_btn.clicked.connect(self.augment_requested)
        self.augment_info = QtWidgets.QLabel("Shows each image as evaluation sees it and "
                                             "several random training views.")
        self.augment_info.setObjectName("blockMeta")
        self.augment_info.setWordWrap(True)
        a_bar.addWidget(self.augment_btn)
        a_bar.addWidget(self.augment_info, 1)
        al.addLayout(a_bar)
        self.augment_grid = QtWidgets.QListWidget()
        self.augment_grid.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.augment_grid.setIconSize(QtCore.QSize(112, 112))
        self.augment_grid.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.augment_grid.setMovement(QtWidgets.QListView.Movement.Static)
        self.augment_grid.setSpacing(6)
        al.addWidget(self.augment_grid, 1)
        self.tabs.addTab(aug, "Augmentation")
        self.tabs.currentChanged.connect(self._tab_changed)

    # ------------------------------------------------------------ datasets
    def set_datasets(self, items: list[tuple[str, str]], keep: str | None = None) -> None:
        """(node id, label) pairs; keeps the current selection when possible."""
        current = keep if keep is not None else (self.dataset.currentData() or "")
        if [(self.dataset.itemData(i), self.dataset.itemText(i))
                for i in range(self.dataset.count())] == items:
            if current and self.dataset.currentData() != current:
                self.select(current)
            return
        self.dataset.blockSignals(True)
        self.dataset.clear()
        for node_id, label in items:
            self.dataset.addItem(label, node_id)
        index = max(0, self.dataset.findData(current)) if items else -1
        self.dataset.setCurrentIndex(index)
        self.dataset.blockSignals(False)
        self.dataset_selected.emit(self.dataset.currentData() or "")

    def select(self, node_id: str) -> None:
        index = self.dataset.findData(node_id)
        if index >= 0:
            self.dataset.setCurrentIndex(index)

    def current_dataset(self) -> str:
        return self.dataset.currentData() or ""

    # ------------------------------------------------------------ profile
    def set_loading(self, text: str = "Profiling…") -> None:
        self.summary.setText(text)

    def clear(self, message: str = "Add a dataset block to the canvas to profile it.") -> None:
        self.profile = None
        self.summary.setText(message)
        self.findings.clear()
        self.details.clear()
        for table in (self.classes, self.columns, self.split_table):
            table.setRowCount(0)
        self.row_model.set_frame(None)
        self.rows_info.clear()
        self.sample_class.clear()
        self.sample_grid.clear()
        self.sample_text.clear()
        self.augment_grid.clear()

    def set_profile(self, profile) -> None:  # noqa: ANN001
        self.profile = profile
        self._rows_loaded = False
        self.row_model.set_frame(None)
        head = profile.summary or profile.source
        if profile.error:
            head = f"{head}\n{profile.error}" if profile.summary else profile.error
        if profile.truncated:
            head += "  (profiled on the first rows)"
        self.summary.setText(head)
        self.findings.clear()
        for f in profile.findings:
            item = QtWidgets.QListWidgetItem(
                f"{_SEVERITY.get(f.severity, '•')}  {f.message}" + (f"\n     {f.hint}"
                                                                    if f.hint else ""))
            item.setData(QtCore.Qt.ItemDataRole.UserRole, f.severity)
            self.findings.addItem(item)
        if not profile.findings and not profile.error and profile.kind in (
                "table", "folder", "arrays", "records"):
            self.findings.addItem("✓  No problems found")
        self.details.setText("\n".join(profile.details))
        total = sum(c.count for c in profile.classes)
        _fill(self.classes, [[c.name, c.count, _bar(c.count, total)] for c in profile.classes])
        _fill(self.columns, [[c.name, c.role, c.kind, f"{c.missing_fraction:.1%}", c.unique,
                              c.mean, c.std, c.min, c.median, c.max,
                              ", ".join(f"{v} ({n})" for v, n in c.top[:4])]
                             for c in profile.columns])
        tabular = profile.kind == "table"
        self.tabs.setTabEnabled(1, bool(profile.columns))
        self.tabs.setTabEnabled(2, tabular and not profile.error)
        self.tabs.setTabEnabled(3, bool(profile.samples))
        self.rows_info.setText("")
        self.sample_class.blockSignals(True)
        self.sample_class.clear()
        self.sample_class.addItems(list(profile.samples))
        self.sample_class.blockSignals(False)
        self._show_samples()
        if self.tabs.currentIndex() == 2 and tabular:
            self.rows_requested.emit()

    def set_rows(self, frame) -> None:  # noqa: ANN001
        self._rows_loaded = True
        self.row_model.set_frame(frame)
        if frame is None:
            self.rows_info.setText("The rows could not be read.")
            return
        total = self.profile.rows if self.profile else len(frame)
        self.rows_info.setText(f"showing {len(frame):,} of {total:,} rows"
                               if len(frame) < total else f"{len(frame):,} rows")

    def _tab_changed(self, index: int) -> None:
        if index == 2 and self.profile is not None and not self._rows_loaded \
                and self.profile.kind == "table" and not self.profile.error:
            self.rows_info.setText("Loading rows…")
            self.rows_requested.emit()

    def _show_samples(self) -> None:
        self.sample_grid.clear()
        self.sample_text.clear()
        files = (self.profile.samples.get(self.sample_class.currentText(), [])
                 if self.profile else [])
        images = self.profile is not None and self.profile.type_id == "data.image_folder"
        self.sample_grid.setVisible(images)
        self.sample_text.setVisible(not images)
        if images:
            for f in files:
                pix = QtGui.QPixmap(f)
                if pix.isNull():
                    continue
                item = QtWidgets.QListWidgetItem(
                    QtGui.QIcon(pix.scaled(96, 96, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                           QtCore.Qt.TransformationMode.SmoothTransformation)),
                    Path(f).name)
                item.setToolTip(f"{f}\n{pix.width()}×{pix.height()}")
                self.sample_grid.addItem(item)
        elif files:
            parts = []
            for f in files[:12]:
                p = Path(f)
                if p.suffix.lower() == ".txt":
                    try:
                        parts.append(f"— {p.name}\n{p.read_text(errors='replace')[:400]}")
                    except OSError:
                        continue
                else:
                    parts.append(f"— {p.name}")
            self.sample_text.setPlainText("\n\n".join(parts))

    # ------------------------------------------------------------ previews
    def set_split(self, preview, error: str = "") -> None:  # noqa: ANN001
        self.split_btn.setEnabled(True)
        if preview is None:
            self.split_table.setRowCount(0)
            self.split_info.setText(error)
            return
        totals = preview.totals
        rows = [[cls, c.get("train", 0), c.get("val", 0), c.get("test", 0)]
                for cls, c in preview.per_class.items()]
        rows.append(["Total", totals.get("train", 0), totals.get("val", 0),
                     totals.get("test", 0)])
        _fill(self.split_table, rows)
        bold = self.split_table.font()
        bold.setBold(True)
        for c in range(4):
            item = self.split_table.item(len(rows) - 1, c)
            if item is not None:
                item.setFont(bold)
        self.split_info.setText(f"Split: {preview.method}." + "".join(
            f"  ⚠ {n}." for n in preview.notes))

    def set_augmentation(self, result, error: str = "") -> None:  # noqa: ANN001
        self.augment_btn.setEnabled(True)
        self.augment_grid.clear()
        if result is None:
            self.augment_info.setText(error)
            return
        for image in result["images"]:
            name = Path(image["source"]).parent.name + "/" + Path(image["source"]).name
            views = [("original", image["original"]), ("evaluation", image["eval"])]
            views += [(f"training {i + 1}", p) for i, p in enumerate(image["train"])]
            for label, path in views:
                item = QtWidgets.QListWidgetItem(QtGui.QIcon(QtGui.QPixmap(path)), label)
                item.setToolTip(f"{name} — {label}")
                self.augment_grid.addItem(item)
        transforms = result.get("train_transforms") or []
        info = ("training transforms: " + ", ".join(t.split("(")[0].replace("v2.", "")
                                                    for t in transforms)
                if transforms else "no training augmentation in the design")
        if result.get("skipped"):
            info += " · not previewed: " + ", ".join(result["skipped"])
        self.augment_info.setText(info)
