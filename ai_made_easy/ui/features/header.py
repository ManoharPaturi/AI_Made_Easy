"""HeaderBar: identity + primary actions (Langflow flowbar logic).

Product-grade organisation: brand + project identity on the left, then
three purposeful clusters to the right — run (Train leads, primary),
tools, and export — separated by hairline dividers so the eye can parse
the row at a glance. The project-name field stays two-way bound to
ProjectStore with a loop guard — the stale-name bug class dies here.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ai_made_easy.ui.stores import ProjectStore


def _vline() -> QtWidgets.QFrame:
    """A 1px vertical hairline separating header clusters."""
    line = QtWidgets.QFrame()
    line.setProperty("vline", True)
    line.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    line.setFixedWidth(1)
    return line


class HeaderBar(QtWidgets.QWidget):
    train_clicked = QtCore.Signal()
    test_clicked = QtCore.Signal()
    validate_clicked = QtCore.Signal()
    llm_clicked = QtCore.Signal()
    expand_clicked = QtCore.Signal()
    save_selection_clicked = QtCore.Signal()
    export_requested = QtCore.Signal(str, str)  # framework, kind
    runtime_export_requested = QtCore.Signal(str)  # onnx | jit

    def __init__(self, project_store: ProjectStore, parent=None):
        super().__init__(parent)
        self._store = project_store
        self._guard = False

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(10)

        # ---- identity: brand · divider · project field
        title = QtWidgets.QLabel("🧩 AI Made Easy")
        title.setObjectName("appTitle")
        layout.addWidget(title)

        layout.addWidget(_vline())

        project = QtWidgets.QVBoxLayout()
        project.setSpacing(1)
        micro = QtWidgets.QLabel("PROJECT")
        micro.setObjectName("microLabel")
        project.addWidget(micro)
        self.name_edit = QtWidgets.QLineEdit(project_store.name)
        self.name_edit.setMinimumWidth(140)
        self.name_edit.setMaximumWidth(190)
        self.name_edit.setToolTip("Project name (used in exports)")
        project.addWidget(self.name_edit)
        layout.addLayout(project)

        layout.addStretch(1)

        def action(text, tip, signal, primary=False):
            btn = QtWidgets.QPushButton(text)
            btn.setToolTip(tip)
            btn.clicked.connect(signal.emit)
            if primary:
                btn.setObjectName("primaryBtn")
            layout.addWidget(btn)
            return btn

        # ---- run cluster: the one primary action leads
        self.train_btn = action("▶ Train", "Generate the training script and "
                                "run it in-app", self.train_clicked, primary=True)
        self.test_btn = action("⚡ Test Run", "Build the model and run one "
                               "forward pass (shape smoke test) in a subprocess",
                               self.test_clicked)
        action("✓ Validate", "Check the graph and report issues",
               self.validate_clicked)
        layout.addWidget(_vline())

        # ---- tools cluster
        action("⤓ LLM Script", "Generate a script from the LLM blocks on canvas "
               "(generation / LoRA fine-tune / RAG)", self.llm_clicked)
        action("⤢ Expand", "Expand selected architecture blocks into primitives",
               self.expand_clicked)
        action("💾 Save Selection", "Save the selected blocks as a reusable "
               "Custom block", self.save_selection_clicked)
        layout.addWidget(_vline())

        # ---- export cluster
        for label, framework in (("⬇ PyTorch", "pytorch"), ("⬇ Keras", "keras")):
            button = QtWidgets.QToolButton(self)
            button.setText(label)
            button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
            button.setPopupMode(
                QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
            menu = QtWidgets.QMenu(button)
            _menu_section(menu, "Python scripts")
            _menu_btn(menu, "Export model (.py)",
                      lambda f=framework: self.export_requested.emit(f, "model"))
            _menu_btn(menu, "Export training script (.py)",
                      lambda f=framework: self.export_requested.emit(f, "train"))
            if framework == "pytorch":
                menu.addSeparator()
                _menu_section(menu, "Share formats")
                _menu_btn(menu, "Export ONNX (.onnx)",
                          lambda: self.runtime_export_requested.emit("onnx"))
                _menu_btn(menu, "Export TorchScript (.pt)",
                          lambda: self.runtime_export_requested.emit("jit"))
                _menu_btn(menu, "Export web demo (.html)",
                          lambda: self.runtime_export_requested.emit("web"))
            button.setMenu(menu)
            layout.addWidget(button)

        # two-way store binding with loop guard
        self.name_edit.textChanged.connect(self._on_text_changed)
        project_store.name_changed.connect(self._on_store_name)

    def _on_text_changed(self, text: str) -> None:
        if not self._guard:
            self._guard = True
            self._store.set_name(text)
            self._guard = False

    def _on_store_name(self, name: str) -> None:
        if not self._guard and self.name_edit.text() != name:
            self._guard = True
            self.name_edit.setText(name)
            self._guard = False

    def set_running(self, running: bool) -> None:
        self.train_btn.setEnabled(not running)
        self.test_btn.setEnabled(not running)


def _menu_btn(menu, text, handler) -> None:
    act = menu.addAction(text)
    act.triggered.connect(handler)


def _menu_section(menu, text: str) -> None:
    """Small non-interactive title that groups menu entries."""
    menu.addAction(text).setEnabled(False)
