"""Properties inspector: typed parameter editors generated from a block's spec.

Shows the selected block's identity (name, category, library, description),
its live output shape and parameter count, any problems attached to it, and
one editor per parameter honouring types, bounds and options. Edits are
emitted as ``param_changed(node_id, name, value)``; the context writes them
back to the canvas, which re-validates.
"""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

_MULTILINE = {"template", "param_grid", "expression", "target_modules", "system_prompt"}


class _FloatEdit(QtWidgets.QLineEdit):
    """Float editor that accepts scientific notation (1e-4) and honours bounds."""

    committed = QtCore.Signal(float)

    def __init__(self, value: float, lo, hi, parent=None):
        super().__init__(parent)
        validator = QtGui.QDoubleValidator(self)
        validator.setNotation(QtGui.QDoubleValidator.Notation.ScientificNotation)
        if lo is not None:
            validator.setBottom(float(lo))
        if hi is not None:
            validator.setTop(float(hi))
        self.setValidator(validator)
        self.setText(f"{float(value):g}")
        self.editingFinished.connect(self._commit)

    def _commit(self) -> None:
        try:
            self.committed.emit(float(self.text()))
        except ValueError:
            pass


class PropertyInspector(QtWidgets.QWidget):
    param_changed = QtCore.Signal(str, str, object)   # node_id, name, value
    reset_requested = QtCore.Signal(str)               # node_id
    locate_requested = QtCore.Signal(str)              # node_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("dockBody")
        self._node_id: str | None = None
        self._loading = False
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        outer.addWidget(self.scroll)
        self.empty = QtWidgets.QLabel("Select a block on the canvas to edit its parameters.")
        self.empty.setObjectName("emptyState")
        self.empty.setWordWrap(True)
        self.empty.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop
                                | QtCore.Qt.AlignmentFlag.AlignLeft)
        self.scroll.setWidget(self.empty)

    # --------------------------------------------------------------- api
    def clear(self) -> None:
        self._node_id = None
        self._status_layout = None
        self.empty = QtWidgets.QLabel("Select a block on the canvas to edit its parameters.")
        self.empty.setObjectName("emptyState")
        self.empty.setWordWrap(True)
        self.empty.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.scroll.setWidget(self.empty)

    @property
    def node_id(self) -> str | None:
        return self._node_id

    def show_node(self, node_id: str, block, params: dict, issues: list,
                  shape: list | None = None, param_count: int | None = None) -> None:
        """Render the inspector for one block (``block``: BlockDefinition)."""
        self._loading = True
        self._node_id = node_id
        body = QtWidgets.QWidget()
        body.setObjectName("dockBody")
        layout = QtWidgets.QVBoxLayout(body)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(6)

        title = QtWidgets.QLabel(block.display_name)
        title.setObjectName("blockTitle")
        layout.addWidget(title)
        meta = " · ".join(x for x in (block.category, block.library) if x)
        meta_label = QtWidgets.QLabel(meta)
        meta_label.setObjectName("blockMeta")
        layout.addWidget(meta_label)
        if block.description:
            desc = QtWidgets.QLabel(block.description)
            desc.setObjectName("blockDesc")
            desc.setWordWrap(True)
            layout.addWidget(desc)

        self._status_host = QtWidgets.QWidget()
        self._status_layout = QtWidgets.QVBoxLayout(self._status_host)
        self._status_layout.setContentsMargins(0, 4, 0, 4)
        self._status_layout.setSpacing(6)
        layout.addWidget(self._status_host)
        self.update_status(issues, shape, param_count)

        if block.params:
            section = QtWidgets.QLabel("PARAMETERS")
            section.setObjectName("sectionTitle")
            layout.addWidget(section)
            form = QtWidgets.QFormLayout()
            form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft
                                   | QtCore.Qt.AlignmentFlag.AlignVCenter)
            form.setFieldGrowthPolicy(
                QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            form.setHorizontalSpacing(12)
            form.setVerticalSpacing(8)
            for spec in block.params:
                value = params.get(spec.name, spec.default)
                label = QtWidgets.QLabel(spec.name.replace("_", " "))
                label.setObjectName("paramName")
                editor = self._editor(spec, value)
                tip = spec.help or ""
                if spec.minimum is not None or spec.maximum is not None:
                    bounds = f"range {spec.minimum if spec.minimum is not None else '−∞'} … " \
                             f"{spec.maximum if spec.maximum is not None else '∞'}"
                    tip = f"{tip}\n{bounds}".strip()
                if tip:
                    label.setToolTip(tip)
                    editor.setToolTip(tip)
                form.addRow(label, editor)
            layout.addLayout(form)
            reset = QtWidgets.QPushButton("Reset to defaults")
            reset.clicked.connect(lambda: self.reset_requested.emit(node_id))
            layout.addWidget(reset, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        else:
            note = QtWidgets.QLabel("This block has no parameters.")
            note.setObjectName("blockMeta")
            layout.addWidget(note)
        layout.addStretch(1)
        self.scroll.setWidget(body)
        self._loading = False

    def update_status(self, issues: list, shape: list | None = None,
                      param_count: int | None = None) -> None:
        """Refresh shape, parameter count and problems without rebuilding editors."""
        host = getattr(self, "_status_layout", None)
        if host is None:
            return
        while host.count():
            item = host.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
            elif item.layout() is not None:
                _clear_layout(item.layout())
        facts = []
        if shape is not None:
            facts.append(f"Output  [{', '.join(str(d) for d in shape)}]")
        if param_count:
            facts.append(f"Parameters  {param_count:,}")
        if facts:
            row = QtWidgets.QHBoxLayout()
            for fact in facts:
                chip = QtWidgets.QLabel(fact)
                chip.setObjectName("badge")
                row.addWidget(chip)
            row.addStretch(1)
            host.addLayout(row)
        for issue in issues:
            host.addWidget(self._issue_banner(issue))

    # ----------------------------------------------------------- editors
    def _emit(self, name: str, value) -> None:
        if not self._loading and self._node_id is not None:
            self.param_changed.emit(self._node_id, name, value)

    def _editor(self, spec, value) -> QtWidgets.QWidget:
        if spec.type == "bool":
            box = QtWidgets.QCheckBox()
            box.setChecked(bool(value))
            box.toggled.connect(lambda v, n=spec.name: self._emit(n, bool(v)))
            return box
        if spec.type == "enum":
            combo = QtWidgets.QComboBox()
            combo.addItems([str(o) for o in spec.options])
            idx = combo.findText(str(value))
            combo.setCurrentIndex(max(idx, 0))
            combo.currentTextChanged.connect(lambda v, n=spec.name: self._emit(n, v))
            return combo
        if spec.type == "int":
            spin = QtWidgets.QSpinBox()
            spin.setRange(int(spec.minimum) if spec.minimum is not None else -2_000_000_000,
                          int(spec.maximum) if spec.maximum is not None else 2_000_000_000)
            spin.setValue(int(value))
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(lambda v, n=spec.name: self._emit(n, int(v)))
            return spin
        if spec.type == "float":
            edit = _FloatEdit(value if value is not None else 0.0, spec.minimum, spec.maximum)
            edit.committed.connect(lambda v, n=spec.name: self._emit(n, v))
            return edit
        if spec.name in _MULTILINE:
            text = QtWidgets.QPlainTextEdit(str(value))
            text.setObjectName("paramText")
            text.setFixedHeight(84)
            timer = QtCore.QTimer(text, singleShot=True, interval=600)
            timer.timeout.connect(lambda n=spec.name, t=text: self._emit(n, t.toPlainText()))
            text.textChanged.connect(timer.start)
            return text
        line = QtWidgets.QLineEdit(str(value))
        line.editingFinished.connect(lambda n=spec.name, w=line: self._emit(n, w.text()))
        return line

    def _issue_banner(self, issue) -> QtWidgets.QWidget:
        from ai_made_easy.ui import icons
        from ai_made_easy.ui.theme import THEMES

        frame = QtWidgets.QFrame()
        frame.setObjectName("issueBanner")
        severity = issue.severity
        tokens = THEMES["dark"]
        color = tokens["ERROR"] if severity == "error" else tokens["WARNING"]
        frame.setStyleSheet(f"QFrame#issueBanner {{ border: 1px solid {color}; }}")
        row = QtWidgets.QHBoxLayout(frame)
        row.setContentsMargins(8, 6, 8, 6)
        glyph = QtWidgets.QLabel()
        glyph.setPixmap(icons.icon("error" if severity == "error" else "warning",
                                   color=color, size=16).pixmap(16, 16))
        row.addWidget(glyph, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        text = QtWidgets.QLabel(issue.message)
        text.setWordWrap(True)
        row.addWidget(text, 1)
        return frame


def _clear_layout(layout) -> None:  # noqa: ANN001
    while layout.count():
        item = layout.takeAt(0)
        if item.widget() is not None:
            item.widget().deleteLater()
