"""Problems panel: every validation error and warning, IDE-style.

Double-click (or Enter) jumps to the block; Quick Fix applies an automatic
fix when one is known for the selected problem.
"""
from __future__ import annotations

from typing import Callable

from PySide6 import QtCore, QtWidgets

from ai_made_easy.ui import icons

ROLE_ISSUE = QtCore.Qt.ItemDataRole.UserRole


class ProblemsPanel(QtWidgets.QWidget):
    locate_requested = QtCore.Signal(str)       # node_id
    fix_requested = QtCore.Signal(object)       # ValidationIssue

    def __init__(self, title_of: Callable[[str], str], fixable: Callable[[object], bool],
                 parent=None):
        super().__init__(parent)
        self.setObjectName("dockBody")
        self._title_of = title_of
        self._fixable = fixable
        self._issues: list = []
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(8, 4, 8, 4)
        self.summary = QtWidgets.QLabel()
        self.summary.setObjectName("blockMeta")
        bar.addWidget(self.summary)
        bar.addStretch(1)
        self.show_errors = QtWidgets.QCheckBox("Errors")
        self.show_warnings = QtWidgets.QCheckBox("Warnings")
        for box in (self.show_errors, self.show_warnings):
            box.setChecked(True)
            box.toggled.connect(self._render)
            bar.addWidget(box)
        self.fix_button = QtWidgets.QPushButton("Quick Fix")
        self.fix_button.setIcon(icons.icon("wand"))
        self.fix_button.setEnabled(False)
        self.fix_button.clicked.connect(self._fix_current)
        bar.addWidget(self.fix_button)
        layout.addLayout(bar)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Problem", "Block"])
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        self.tree.setAlternatingRowColors(False)
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.tree, 1)
        self.tree.itemActivated.connect(self._locate)
        self.tree.itemDoubleClicked.connect(self._locate)
        self.tree.currentItemChanged.connect(self._on_current)

    # ------------------------------------------------------------- api
    def set_issues(self, issues: list) -> None:
        self._issues = list(issues)
        self._render()

    @property
    def counts(self) -> tuple[int, int]:
        errors = sum(1 for i in self._issues if i.severity == "error")
        return errors, len(self._issues) - errors

    # ---------------------------------------------------------- render
    def _render(self) -> None:
        self.tree.clear()
        errors, warnings = self.counts
        if not self._issues:
            self.summary.setText("No problems detected")
        else:
            self.summary.setText(f"{errors} error{'s' if errors != 1 else ''}, "
                                 f"{warnings} warning{'s' if warnings != 1 else ''}")
        ordered = sorted(self._issues, key=lambda i: (i.severity != "error", i.node_id or ""))
        for issue in ordered:
            if issue.severity == "error" and not self.show_errors.isChecked():
                continue
            if issue.severity != "error" and not self.show_warnings.isChecked():
                continue
            block = self._title_of(issue.node_id) if issue.node_id else "Project"
            item = QtWidgets.QTreeWidgetItem([issue.message, block])
            item.setIcon(0, icons.icon("error" if issue.severity == "error" else "warning",
                                       color="#f85149" if issue.severity == "error"
                                       else "#e3a008", size=16))
            item.setToolTip(0, issue.message)
            item.setData(0, ROLE_ISSUE, issue)
            self.tree.addTopLevelItem(item)
        self.fix_button.setEnabled(False)

    def _on_current(self, current, _previous) -> None:
        issue = current.data(0, ROLE_ISSUE) if current is not None else None
        self.fix_button.setEnabled(bool(issue is not None and self._fixable(issue)))

    def _locate(self, item) -> None:
        issue = item.data(0, ROLE_ISSUE)
        if issue is not None and issue.node_id:
            self.locate_requested.emit(issue.node_id)

    def _fix_current(self) -> None:
        item = self.tree.currentItem()
        if item is not None:
            self.fix_requested.emit(item.data(0, ROLE_ISSUE))
