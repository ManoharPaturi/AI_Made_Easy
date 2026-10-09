"""Right-hand panels: Model Summary, Code preview and the AI Assistant.

Pages are presentation only: they render what the context hands them and
emit intents (target changed, copy, export, apply graph).
"""
from __future__ import annotations

import re
import threading

from PySide6 import QtCore, QtGui, QtWidgets

from ai_made_easy.core import assistant as assistant_core
from ai_made_easy.core.registry import get_registry
from ai_made_easy.ui import icons
from ai_made_easy.ui.services.export_service import PREVIEW_TARGETS, target_label

_SYNTAX = {
    "dark": ("#ff7b72", "#a5d6ff", "#8b949e", "#79c0ff", "#d2a8ff", "#ffa657"),
    "light": ("#cf222e", "#0a3069", "#6e7781", "#0550ae", "#8250df", "#953800"),
}


# ------------------------------------------------------------------ summary

class SummaryPage(QtWidgets.QWidget):
    """Per-layer shapes, parameters and FLOPs, plus the project's resource budget."""

    budget_changed = QtCore.Signal(dict)  # the new graph.meta["budget"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("dockBody")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        tiles = QtWidgets.QHBoxLayout()
        self.params_tile = self._tile("Trainable parameters")
        self.layers_tile = self._tile("Layers")
        self.kind_tile = self._tile("Project")
        for tile in (self.params_tile, self.layers_tile, self.kind_tile):
            tiles.addWidget(tile[0])
        layout.addLayout(tiles)
        cost = QtWidgets.QHBoxLayout()
        self.flops_tile = self._tile("Forward FLOPs / sample")
        self.memory_tile = self._tile("Training memory")
        self.latency_tile = self._tile("Latency / sample")
        for tile in (self.flops_tile, self.memory_tile, self.latency_tile):
            cost.addWidget(tile[0])
        layout.addLayout(cost)
        layout.addWidget(self._budget_box())
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Layer", "Output shape", "Parameters", "FLOPs"])
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for col in (1, 2, 3):
            header.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table, 1)
        self.note = QtWidgets.QLabel()
        self.note.setObjectName("blockMeta")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)

    # ------------------------------------------------------------ budget
    def _budget_box(self) -> QtWidgets.QWidget:
        from ai_made_easy.core import budget

        box = QtWidgets.QGroupBox("Budget")
        form = QtWidgets.QFormLayout(box)
        form.setContentsMargins(8, 6, 8, 6)
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.addItem("No target device", "")
        for device in budget.devices().values():
            self.device_combo.addItem(device.label, device.id)
        self.device_combo.setToolTip("Estimates memory and latency on this device")
        form.addRow("Device", self.device_combo)
        self.limit_spins: dict[str, QtWidgets.QDoubleSpinBox] = {}
        for key, label, suffix, top in (
                ("max_train_memory_gb", "Max training memory", " GB", 1024.0),
                ("max_latency_ms", "Max latency", " ms", 1e6),
                ("max_params_m", "Max parameters", " M", 1e5),
                ("max_model_mb", "Max model size", " MB", 1e6)):
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(0.0, top)
            spin.setDecimals(1)
            spin.setSuffix(suffix)
            spin.setSpecialValueText("no limit")
            spin.setKeyboardTracking(False)
            self.limit_spins[key] = spin
            form.addRow(label, spin)
        self.memory_bar = QtWidgets.QProgressBar()
        self.memory_bar.setRange(0, 100)
        self.memory_bar.setTextVisible(True)
        self.memory_bar.setVisible(False)
        form.addRow("Memory use", self.memory_bar)
        self._quiet = False
        self.device_combo.currentIndexChanged.connect(self._emit_budget)
        for spin in self.limit_spins.values():
            spin.valueChanged.connect(self._emit_budget)
        return box

    def set_budget(self, values: dict) -> None:
        """Show the project's budget without echoing a change."""
        self._quiet = True
        try:
            index = self.device_combo.findData(values.get("device") or "")
            self.device_combo.setCurrentIndex(max(index, 0))
            for key, spin in self.limit_spins.items():
                spin.setValue(float(values.get(key) or 0))
        finally:
            self._quiet = False

    def budget(self) -> dict:
        return {"device": self.device_combo.currentData() or "",
                **{k: spin.value() for k, spin in self.limit_spins.items()}}

    def _emit_budget(self, *_args) -> None:
        if not self._quiet:
            self.budget_changed.emit(self.budget())

    # ------------------------------------------------------------ tiles
    @staticmethod
    def _tile(name: str):
        frame = QtWidgets.QFrame()
        frame.setObjectName("metricTile")
        box = QtWidgets.QVBoxLayout(frame)
        box.setContentsMargins(10, 8, 10, 8)
        box.setSpacing(0)
        value = QtWidgets.QLabel("—")
        value.setObjectName("metricValue")
        label = QtWidgets.QLabel(name)
        label.setObjectName("metricName")
        box.addWidget(value)
        box.addWidget(label)
        return frame, value

    def set_kind(self, kind: str) -> None:
        self.kind_tile[1].setText(kind)

    def set_estimate(self, est, checks=()) -> None:  # noqa: ANN001 — core.budget.Estimate
        from ai_made_easy.core.budget import GB, human_bytes, human_flops

        for tile in (self.flops_tile, self.memory_tile, self.latency_tile):
            tile[1].setText("—")
            tile[1].setToolTip("")
        self.memory_bar.setVisible(False)
        if est is None:
            return
        self.flops_tile[1].setText(human_flops(est.flops))
        memory = est.train_memory_bytes()
        self.memory_tile[1].setText(human_bytes(memory))
        self.memory_tile[1].setToolTip(
            f"batch {est.batch_size}" + (", mixed precision" if est.mixed_precision else "")
            + f"; model {human_bytes(est.model_bytes())}")
        latency = est.latency_ms()
        self.latency_tile[1].setText("—" if latency is None else f"{latency:.2f} ms")
        if est.device is not None:
            self.latency_tile[1].setToolTip(f"batch 1 on {est.device.label}")
        for c in checks:
            if c.kind == "train_memory":
                share = int(min(100.0, 100.0 * c.used / c.limit))
                self.memory_bar.setValue(share)
                self.memory_bar.setFormat(f"{memory / GB:.2f} of {c.limit:g} GB")
                self.memory_bar.setProperty("over", c.over)
                self.memory_bar.setStyleSheet(
                    "QProgressBar::chunk { background: #cf222e; }" if c.over else "")
                self.memory_bar.setVisible(True)

    def set_summary(self, summary, note: str = "", flops: list[int] | None = None) -> None:
        if summary is None:
            self.params_tile[1].setText("—")
            self.layers_tile[1].setText("—")
            self.table.setRowCount(0)
            self.set_estimate(None)
            self.note.setText(note or "The model summary appears once the model is valid.")
            return
        from ai_made_easy.core.budget import human_flops

        self.note.setText(note)
        self.table.setRowCount(len(summary.layers))
        for row, layer in enumerate(summary.layers):
            shape = "[" + ", ".join(str(d) for d in layer.output_shape) + "]"
            params = f"{layer.params:,}" if layer.params else "—"
            cost = human_flops(flops[row]) if flops and row < len(flops) and flops[row] else "—"
            for col, text in enumerate((layer.name, shape, params, cost)):
                item = QtWidgets.QTableWidgetItem(text)
                if col:
                    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight
                                          | QtCore.Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(row, col, item)
        self.params_tile[1].setText(summary.total_params_display)
        self.params_tile[1].setToolTip(f"{summary.total_params:,}")
        self.layers_tile[1].setText(str(len(summary.layers)))


# ---------------------------------------------------------------- code view

class PythonHighlighter(QtGui.QSyntaxHighlighter):
    KEYWORDS = (r"\b(def|class|return|if|elif|else|for|while|in|not|and|or|is|None|True|"
                r"False|import|from|as|with|try|except|raise|lambda|pass|break|continue|"
                r"global|assert|yield|self)\b")

    def __init__(self, document, theme: str = "dark") -> None:
        super().__init__(document)
        self.set_theme(theme)

    def set_theme(self, theme: str) -> None:
        kw, string, comment, number, func, deco = _SYNTAX.get(theme, _SYNTAX["dark"])

        def fmt(color, italic=False):
            f = QtGui.QTextCharFormat()
            f.setForeground(QtGui.QColor(color))
            f.setFontItalic(italic)
            return f

        self.rules = [
            (re.compile(r"\b[A-Za-z_]\w*(?=\()"), fmt(func)),
            (re.compile(self.KEYWORDS), fmt(kw)),
            (re.compile(r"\b\d[\d._eE+-]*\b"), fmt(number)),
            (re.compile(r"@\w+"), fmt(deco)),
            (re.compile(r'"[^"\n]*"|\'[^\'\n]*\''), fmt(string)),
            (re.compile(r"#[^\n]*"), fmt(comment, italic=True)),
        ]
        self.rehighlight()

    def highlightBlock(self, text: str) -> None:  # noqa: N802
        for pattern, fmt in self.rules:
            for m in pattern.finditer(text):
                self.setFormat(m.start(), m.end() - m.start(), fmt)


class CodePage(QtWidgets.QWidget):
    target_changed = QtCore.Signal(str)
    export_requested = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("dockBody")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(8, 6, 8, 6)
        self.selector = QtWidgets.QComboBox()
        for target in PREVIEW_TARGETS:
            self.selector.addItem(target_label(target), target)
        bar.addWidget(self.selector, 1)
        self.copy_button = QtWidgets.QToolButton()
        self.copy_button.setIcon(icons.icon("copy"))
        self.copy_button.setToolTip("Copy code")
        self.copy_button.clicked.connect(self.copy)
        bar.addWidget(self.copy_button)
        self.export_button = QtWidgets.QToolButton()
        self.export_button.setIcon(icons.icon("download"))
        self.export_button.setToolTip("Export this file")
        self.export_button.clicked.connect(
            lambda: self.export_requested.emit(self.current_target()))
        bar.addWidget(self.export_button)
        layout.addLayout(bar)
        self.view = QtWidgets.QPlainTextEdit()
        self.view.setObjectName("codeView")
        self.view.setReadOnly(True)
        self.view.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self._highlighter = PythonHighlighter(self.view.document())
        layout.addWidget(self.view, 1)
        self.selector.currentIndexChanged.connect(
            lambda _: self.target_changed.emit(self.current_target()))

    def set_theme(self, theme: str) -> None:
        self._highlighter.set_theme(theme)

    def current_target(self) -> str:
        return self.selector.currentData() or PREVIEW_TARGETS[0]

    def select_target(self, target: str) -> None:
        idx = self.selector.findData(target)
        if idx >= 0:
            self.selector.setCurrentIndex(idx)

    def set_code(self, code: str) -> None:
        if self.view.toPlainText() != code:
            bar = self.view.verticalScrollBar()
            pos = bar.value()
            self.view.setPlainText(code)
            bar.setValue(pos)
        self.export_button.setEnabled(True)

    def set_error(self, message: str) -> None:
        self.view.setPlainText(f"# {target_label(self.current_target())} is not available:\n"
                               + "\n".join(f"#   {line}" for line in message.splitlines()))
        self.export_button.setEnabled(False)

    def copy(self) -> None:
        QtWidgets.QApplication.clipboard().setText(self.view.toPlainText())


# ---------------------------------------------------------------- assistant

class AssistantPage(QtWidgets.QWidget):
    """Chat about the current design with any OpenAI-compatible endpoint."""

    apply_requested = QtCore.Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("dockBody")
        self._history: list[dict] = []
        self._graph_json: dict = {"nodes": [], "edges": []}
        self._pending: dict | None = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self.view = QtWidgets.QTextBrowser()
        self.view.setOpenExternalLinks(True)
        layout.addWidget(self.view, 1)
        row = QtWidgets.QHBoxLayout()
        self.input = QtWidgets.QLineEdit()
        self.input.setPlaceholderText("Ask about the current design")
        self.input.returnPressed.connect(self.send)
        self.send_btn = QtWidgets.QPushButton("Send")
        self.send_btn.setObjectName("primaryButton")
        self.send_btn.clicked.connect(self.send)
        self.apply_btn = QtWidgets.QPushButton("Apply proposed graph")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self._apply)
        row.addWidget(self.input, 1)
        row.addWidget(self.send_btn)
        layout.addLayout(row)
        layout.addWidget(self.apply_btn)
        self._intro()

    def set_graph(self, graph_json: dict) -> None:
        self._graph_json = graph_json

    def _intro(self) -> None:
        if assistant_core.is_configured():
            cfg = assistant_core.assistant_config()
            self.view.setHtml(f"<p>Connected to <b>{cfg['model']}</b>. Ask about the design; "
                              "replies that contain a revised graph can be applied after "
                              "validation.</p>")
        else:
            self.view.setHtml(
                "<p><b>Assistant not configured.</b></p><p>Set these environment variables "
                "and restart to connect any OpenAI-compatible endpoint (including local "
                "servers such as Ollama or LM Studio):</p><p><code>AIME_ASSISTANT_BASE_URL"
                "</code><br><code>AIME_ASSISTANT_API_KEY</code><br><code>AIME_ASSISTANT_MODEL"
                "</code></p>")

    def send(self) -> None:
        text = self.input.text().strip()
        if not text:
            return
        self.input.clear()
        self.view.append(f"<p><b>You</b><br>{_escape(text)}</p>")
        if not assistant_core.is_configured():
            self.view.append("<p><i>The assistant is not configured.</i></p>")
            return
        self.send_btn.setEnabled(False)
        self._history.append({"role": "user", "content": text})
        messages = [{"role": "system", "content": assistant_core.build_system_prompt(
            self._graph_json, get_registry().list_blocks())}] + self._history[-12:]

        def worker() -> None:
            try:
                reply, error = assistant_core.chat(messages), ""
            except Exception as exc:  # noqa: BLE001 — surfaced to the user
                reply, error = "", str(exc)
            QtCore.QMetaObject.invokeMethod(
                self, "_reply", QtCore.Qt.ConnectionType.QueuedConnection,
                QtCore.Q_ARG(str, reply), QtCore.Q_ARG(str, error))

        threading.Thread(target=worker, daemon=True).start()

    @QtCore.Slot(str, str)
    def _reply(self, reply: str, error: str) -> None:
        self.send_btn.setEnabled(True)
        if error:
            self.view.append(f"<p style='color:#f85149'>Request failed: {_escape(error)}</p>")
            return
        self._history.append({"role": "assistant", "content": reply})
        self.view.append(f"<p><b>Assistant</b></p><pre>{_escape(reply)}</pre>")
        candidate = assistant_core.extract_graph(reply)
        if candidate is not None:
            self._pending = candidate
            self.apply_btn.setEnabled(True)

    def _apply(self) -> None:
        if self._pending is not None:
            self.apply_requested.emit(self._pending)
            self.apply_btn.setEnabled(False)
            self._pending = None

    def applied(self, ok: bool) -> None:
        self.view.append("<p><i>Graph applied.</i></p>" if ok else
                         "<p><i>The proposed graph has errors and was not applied.</i></p>")


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
