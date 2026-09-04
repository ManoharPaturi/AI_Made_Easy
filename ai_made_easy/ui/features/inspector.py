"""InspectorStack: the right-side pages + switching policy (Langflow
inspection-panel logic). Pages are dumb — they render what they're given.

- PropertiesPage wraps the OdenGraphQt properties bin (built by CanvasArea)
- SummaryPage renders rows (name, shape, params, total)
- PreviewPage renders a code string it's handed — NO codegen here
- AssistantPage is a chat view; transport lives in core/assistant.py
"""
from __future__ import annotations

import re
import threading

from PySide6 import QtCore, QtGui, QtWidgets

from ai_made_easy.core import assistant as assistant_core
from ai_made_easy.core.registry import get_registry
from ai_made_easy.ui.services.export_service import (
    PREVIEW_TARGETS,
    target_label,
)


# ------------------------------------------------------------- summary

class IssueRow(QtWidgets.QFrame):
    """One check row: glyph + message (+ optional 🔧 one-click fix)."""

    fix_requested = QtCore.Signal(object)   # the ValidationIssue
    locate_requested = QtCore.Signal(str)   # node_id

    def __init__(self, issue, fix_available: bool, parent=None):  # noqa: ANN001
        super().__init__(parent)
        self.issue = issue
        self.setObjectName("issueRow")
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        glyph = "✖" if issue.severity == "error" else "⚠"
        label = QtWidgets.QLabel(f"{glyph} {issue.message}")
        label.setWordWrap(True)
        layout.addWidget(label, 1)
        if fix_available:
            btn = QtWidgets.QPushButton("🔧 Fix")
            btn.setToolTip("Apply the suggested fix automatically")
            btn.clicked.connect(lambda: self.fix_requested.emit(self.issue))
            layout.addWidget(btn, 0)
        if issue.node_id:
            go = QtWidgets.QToolButton()
            go.setText("➤")
            go.setToolTip("Show this block on the canvas")
            go.clicked.connect(
                lambda: self.locate_requested.emit(issue.node_id))
            layout.addWidget(go, 0)


class SummaryPage(QtWidgets.QWidget):
    node_requested = QtCore.Signal(str)  # issue row -> jump to block
    fix_requested = QtCore.Signal(object)  # issue -> one-click fix

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # guardrail checklist (hidden when everything is fine)
        self.issues_label = QtWidgets.QLabel("🩺 Checks")
        self.issues_label.setObjectName("cardTitle")
        self.issues_host = QtWidgets.QWidget()
        self.issues_layout = QtWidgets.QVBoxLayout(self.issues_host)
        self.issues_layout.setContentsMargins(0, 0, 0, 0)
        self.issues_layout.setSpacing(4)
        self.issues_scroll = QtWidgets.QScrollArea()
        self.issues_scroll.setWidget(self.issues_host)
        self.issues_scroll.setWidgetResizable(True)
        self.issues_scroll.setMaximumHeight(150)
        self.issues_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        layout.addWidget(self.issues_label)
        layout.addWidget(self.issues_scroll)

        # model size chip — the headline number, visible without scrolling
        self.total = QtWidgets.QLabel("—")
        self.total.setProperty("chip", True)
        self.total.setToolTip("trainable parameters — how many little "
                              "settings the model adjusts while learning")
        layout.addWidget(self.total)

        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Layer", "Output Shape", "Params"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table)

    def set_issues(self, issues: list, graph=None) -> None:
        """Render the validation checklist: ✖/⚠ rows, 🔧 fix, ➤ locate."""
        while self.issues_layout.count():
            item = self.issues_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if not issues:
            self.issues_label.setText("🩺 Checks — all good ✓")
            self.issues_scroll.setVisible(False)
            return
        from ai_made_easy.core.fixes import fix_for_issue

        errors = sum(1 for i in issues if i.severity == "error")
        self.issues_label.setText(
            f"🩺 Checks — {errors} error(s), "
            f"{len(issues) - errors} warning(s)")
        self.issues_scroll.setVisible(True)
        for issue in issues:
            fixable = False
            if graph is not None:
                try:
                    fixable = fix_for_issue(graph, issue) is not None
                except Exception:
                    fixable = False
            row = IssueRow(issue, fixable)
            row.fix_requested.connect(self.fix_requested.emit)
            row.locate_requested.connect(self.node_requested.emit)
            self.issues_layout.addWidget(row)
        self.issues_layout.addStretch(1)

    def set_summary(self, summary) -> None:
        """summary: core.summary.ModelSummary (or None)."""
        if summary is None:
            self.total.setText("—")
            self.table.setRowCount(0)
            return
        self.table.setRowCount(len(summary.layers))
        for row, layer in enumerate(summary.layers):
            shape = "[" + ", ".join(str(d) for d in layer.output_shape) + "]"
            params = f"{layer.params:,}" if layer.params else "—"
            for col, text in enumerate((layer.name, shape, params)):
                item = QtWidgets.QTableWidgetItem(text)
                if col:
                    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
                self.table.setItem(row, col, item)
        self.total.setText(
            f"🧠 {summary.total_params:,} parameters "
            f"({summary.total_params_display})")
        self.table.resizeColumnsToContents()


# ------------------------------------------------------------- preview

class PythonHighlighter(QtGui.QSyntaxHighlighter):
    KEYWORDS = (
        r"\b(def|class|return|if|elif|else|for|while|in|not|and|or|is|None|"
        r"True|False|import|from|as|with|try|except|raise|lambda|pass|break|"
        r"continue|global|assert|yield)\b"
    )

    def __init__(self, document) -> None:
        super().__init__(document)
        # syntax colours follow the chrome: dark hexes on dark themes,
        # GitHub-light hexes on the light ones (white code surfaces)
        app = QtWidgets.QApplication.instance()
        light = app is not None and "#F6F4ED" in app.styleSheet()
        kw_c, str_c, com_c, num_c, fn_c = (
            ("#CF222E", "#0A3069", "#6E7781", "#0550AE", "#8250DF") if light
            else ("#ff7b72", "#a5d6ff", "#8b949e", "#79c0ff", "#d2a8ff"))
        self.rules: list[tuple[re.Pattern, QtGui.QTextCharFormat]] = []
        keyword = QtGui.QTextCharFormat()
        keyword.setForeground(QtGui.QColor(kw_c))
        self.rules.append((re.compile(self.KEYWORDS), keyword))
        string = QtGui.QTextCharFormat()
        string.setForeground(QtGui.QColor(str_c))
        self.rules.append((re.compile(r'"[^"\n]*"|\'[^\'\n]*\''), string))
        comment = QtGui.QTextCharFormat()
        comment.setForeground(QtGui.QColor(com_c))
        comment.setFontItalic(True)
        self.rules.append((re.compile(r"#[^\n]*"), comment))
        number = QtGui.QTextCharFormat()
        number.setForeground(QtGui.QColor(num_c))
        self.rules.append((re.compile(r"\b\d[\d._eE+-]*\b"), number))
        funcdef = QtGui.QTextCharFormat()
        funcdef.setForeground(QtGui.QColor(fn_c))
        self.rules.append((re.compile(r"\b[A-Za-z_]\w*(?=\()"), funcdef))

    def highlightBlock(self, text: str) -> None:  # noqa: N802
        for pattern, fmt in self.rules:
            for m in re.finditer(pattern, text):
                self.setFormat(m.start(), m.end() - m.start(), fmt)


class PreviewPage(QtWidgets.QWidget):
    target_changed = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.selector = QtWidgets.QComboBox()
        for target in PREVIEW_TARGETS:
            self.selector.addItem(target_label(target), target)
        layout.addWidget(self.selector)
        self.view = QtWidgets.QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        font = QtGui.QFont("Menlo")
        font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        self.view.setFont(font)
        self._highlighter = PythonHighlighter(self.view.document())
        layout.addWidget(self.view)
        self.selector.currentIndexChanged.connect(
            lambda _: self.target_changed.emit(self.current_target()))

    def current_target(self) -> str:
        return self.selector.currentData() or PREVIEW_TARGETS[0]

    def set_code(self, code: str) -> None:
        self.view.setPlainText(code)

    def set_error(self, message: str) -> None:
        self.view.setPlainText(
            f"# cannot generate {target_label(self.current_target())}:\n# {message}")


# ----------------------------------------------------------- assistant

class AssistantPage(QtWidgets.QWidget):
    apply_requested = QtCore.Signal(dict)  # graph JSON proposed by the model

    @staticmethod
    def _card_html(body: str) -> str:
        """Chat bubble in the active chrome: light card on light themes."""
        app = QtWidgets.QApplication.instance()
        light = app is not None and "#F6F4ED" in app.styleSheet()
        bg, border, fg = (("#EFECE2", "#E1DDCF", "#33312B") if light else
                          ("#242933", "#333a46", "#c9d1d9"))
        return (f'<div style="background-color:{bg}; border:1px solid {border};'
                f' border-radius:8px; padding:10px 14px; margin:6px 0;'
                f' color:{fg};">{body}</div>')

    def __init__(self, parent=None):
        super().__init__(parent)
        self._history: list[dict] = []
        self._graph_json: dict = {"nodes": [], "edges": []}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.view = QtWidgets.QTextBrowser()
        layout.addWidget(self.view, stretch=1)

        row = QtWidgets.QHBoxLayout()
        self.input = QtWidgets.QLineEdit()
        self.input.setPlaceholderText("Ask about your graph…")
        self.input.returnPressed.connect(self.send)
        self.send_btn = QtWidgets.QPushButton("Send")
        self.send_btn.clicked.connect(self.send)
        self.apply_btn = QtWidgets.QPushButton("✓ Apply graph")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self._apply)
        row.addWidget(self.input, stretch=1)
        row.addWidget(self.send_btn)
        row.addWidget(self.apply_btn)
        layout.addLayout(row)

        self._pending: dict | None = None
        self._append_system_note()

    def set_graph(self, graph_json: dict) -> None:
        self._graph_json = graph_json

    def _append_system_note(self) -> None:
        if assistant_core.is_configured():
            cfg = assistant_core.assistant_config()
            self.view.append(self._card_html(
                f"Assistant ready — model <b>{cfg['model']}</b>.<br>"
                "Ask about the graph; replies containing a corrected graph "
                "get an <b>Apply graph</b> button."))
        else:
            self.view.append(self._card_html(
                "Assistant is not configured. Set these environment variables "
                "and restart the app:<br>"
                "<code>AIME_ASSISTANT_BASE_URL</code> — OpenAI-compatible API<br>"
                "<code>AIME_ASSISTANT_API_KEY</code><br>"
                "<code>AIME_ASSISTANT_MODEL</code>"))

    def send(self) -> None:
        text = self.input.text().strip()
        if not text:
            return
        if not assistant_core.is_configured():
            self.view.append(f"<b>you:</b> {text}")
            self.view.append("<i>(not configured — see above)</i>")
            return
        self.input.clear()
        self.send_btn.setEnabled(False)
        self.view.append(f"<b>you:</b> {text}")
        self._history.append({"role": "user", "content": text})
        messages = [self._system_prompt()] + self._history[-12:]

        def worker() -> None:
            try:
                reply = assistant_core.chat(messages)
                payload = reply
            except Exception as exc:
                payload = f"<font color='#ff7b72'>{exc}</font>"
                reply = ""
            QtCore.QMetaObject.invokeMethod(
                self, "_reply", QtCore.Qt.ConnectionType.QueuedConnection,
                QtCore.Q_ARG(str, payload), QtCore.Q_ARG(str, reply))

        threading.Thread(target=worker, daemon=True).start()

    @QtCore.Slot(str, str)
    def _reply(self, display: str, reply: str) -> None:
        self.send_btn.setEnabled(True)
        if reply:
            self._history.append({"role": "assistant", "content": reply})
        escaped = display.replace("&", "&amp;").replace("<", "&lt;")
        self.view.append(f"<b>assistant:</b><br><pre>{escaped}</pre>")
        candidate = assistant_core.extract_graph(reply) if reply else None
        if candidate is not None:
            self._pending = candidate
            self.apply_btn.setEnabled(True)
            self.view.append("<i>reply contains a graph — review it, then "
                             "press ✓ Apply graph</i>")

    def _apply(self) -> None:
        if self._pending is None:
            return
        self.apply_requested.emit(self._pending)
        self.apply_btn.setEnabled(False)
        self._pending = None

    def applied(self, ok: bool) -> None:
        self.view.append(
            "<i>applied to the canvas ✓</i>" if ok
            else "<i>graph was not applied — see Console</i>")

    def _system_prompt(self) -> dict:
        return {"role": "system",
                "content": assistant_core.build_system_prompt(
                    self._graph_json, get_registry().list_blocks())}


# -------------------------------------------------------------- stack

class InspectorStack(QtWidgets.QTabWidget):
    """Switching POLICY lives here: selection -> Properties, else Summary."""

    def __init__(self, properties_page, summary_page, preview_page,
                 assistant_page, parent=None):
        super().__init__(parent)
        self.properties_page = properties_page
        self.summary_page = summary_page
        self.preview_page = preview_page
        self.assistant_page = assistant_page
        self.addTab(summary_page, "📋 Summary")
        self.addTab(properties_page, "⚙️ Block")
        self.addTab(preview_page, "👁️ Code")
        self.addTab(assistant_page, "🤖 Coach")
        self.setTabToolTip(0, "Summary — layers, shapes and health checks")
        self.setTabToolTip(1, "Properties of the selected block")
        self.setTabToolTip(2, "The Python code for these blocks")
        self.setTabToolTip(3, "AI coach — ask about your graph")

    def show_properties(self) -> None:
        self.setCurrentWidget(self.properties_page)

    def show_summary(self) -> None:
        self.setCurrentWidget(self.summary_page)
