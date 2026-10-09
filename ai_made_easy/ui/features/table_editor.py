"""Grid editor for table parameters (a random variable's conditional probability table).

Rows are the variable's states, columns the combinations of its parents' states; the
layout comes from ``core.pgm.network.layout`` so the desktop and web editors agree.
"""
from __future__ import annotations

import json

from PySide6 import QtCore, QtGui, QtWidgets


class TableEditorDialog(QtWidgets.QDialog):
    """Edit a table; ``result_text`` is the JSON to store ("" = learn from data)."""

    def __init__(self, parent, layout: dict) -> None:  # noqa: ANN001
        super().__init__(parent)
        self.setWindowTitle(f"Probability table — {layout['variable']}")
        self.result_text: str | None = None
        self._rows, self._cols = layout["rows"], layout["columns"]
        box = QtWidgets.QVBoxLayout(self)
        parents = ", ".join(layout["parents"])
        intro = QtWidgets.QLabel(
            f"P({layout['variable']}{' | ' + parents if parents else ''}): each column "
            "is one combination of parent states and must sum to 1.")
        intro.setWordWrap(True)
        box.addWidget(intro)
        self.table = QtWidgets.QTableWidget(len(self._rows), len(self._cols))
        self.table.setVerticalHeaderLabels([str(r) for r in self._rows])
        self.table.setHorizontalHeaderLabels([c.replace(", ", "\n") for c in self._cols])
        validator_delegate = _NumberDelegate(self.table)
        self.table.setItemDelegate(validator_delegate)
        for i, row in enumerate(layout["values"]):
            for j, value in enumerate(row):
                self.table.setItem(i, j, QtWidgets.QTableWidgetItem(f"{value:g}"))
        self.table.resizeColumnsToContents()
        self.table.itemChanged.connect(self._check)
        box.addWidget(self.table, 1)
        self.status = QtWidgets.QLabel()
        box.addWidget(self.status)
        buttons = QtWidgets.QHBoxLayout()
        for label, slot in (("Normalize columns", self._normalize), ("Uniform", self._uniform),
                            ("Learn from data", self._learn)):
            btn = QtWidgets.QPushButton(label)
            btn.clicked.connect(slot)
            buttons.addWidget(btn)
        buttons.addStretch(1)
        ok = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Save
                                        | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        ok.accepted.connect(self._save)
        ok.rejected.connect(self.reject)
        buttons.addWidget(ok)
        box.addLayout(buttons)
        self.resize(min(980, 160 + 110 * len(self._cols)), min(620, 220 + 34 * len(self._rows)))
        self._check()

    def values(self) -> list[list[float]]:
        out = []
        for i in range(len(self._rows)):
            row = []
            for j in range(len(self._cols)):
                item = self.table.item(i, j)
                try:
                    row.append(float(item.text()) if item else 0.0)
                except ValueError:
                    row.append(0.0)
            out.append(row)
        return out

    def _set(self, values: list[list[float]]) -> None:
        self.table.blockSignals(True)
        for i, row in enumerate(values):
            for j, value in enumerate(row):
                self.table.item(i, j).setText(f"{value:.8g}")
        self.table.blockSignals(False)
        self._check()

    def _check(self) -> None:
        values = self.values()
        bad = [self._cols[j] for j in range(len(self._cols))
               if abs(sum(values[i][j] for i in range(len(values))) - 1) > 1e-4
               or any(values[i][j] < 0 for i in range(len(values)))]
        if bad:
            self.status.setText(f"⚠ {len(bad)} column(s) do not sum to 1 (e.g. {bad[0]}) — "
                                "Normalize columns fixes them")
        else:
            self.status.setText("✓ every column sums to 1")

    def _normalize(self) -> None:
        values = self.values()
        for j in range(len(self._cols)):
            total = sum(max(values[i][j], 0.0) for i in range(len(values)))
            for i in range(len(values)):
                values[i][j] = max(values[i][j], 0.0) / total if total else 1 / len(values)
        self._set(values)

    def _uniform(self) -> None:
        self._set([[1 / len(self._rows)] * len(self._cols) for _ in self._rows])

    def _learn(self) -> None:
        self.result_text = ""
        self.accept()

    def _save(self) -> None:
        self.result_text = json.dumps([[round(v, 8) for v in row] for row in self.values()])
        self.accept()


class _NumberDelegate(QtWidgets.QStyledItemDelegate):
    def createEditor(self, parent, option, index):  # noqa: N802, ANN001, ANN201
        editor = QtWidgets.QLineEdit(parent)
        editor.setValidator(QtGui.QDoubleValidator(0.0, 1.0, 8, editor))
        editor.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        return editor
