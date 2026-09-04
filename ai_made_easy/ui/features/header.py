"""HeaderBar: identity + primary actions (Langflow flowbar logic).

The project-name field is two-way bound to ProjectStore with a loop guard
— the stale-name bug class dies here.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ai_made_easy.ui.stores import ProjectStore


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
        layout.setContentsMargins(6, 2, 6, 2)

        title = QtWidgets.QLabel("🧩 AI Made Easy")
        title.setObjectName("appTitle")
        layout.addWidget(title)

        self.name_edit = QtWidgets.QLineEdit(project_store.name)
        self.name_edit.setMinimumWidth(160)
        self.name_edit.setMaximumWidth(220)
        self.name_edit.setToolTip("Project name (used in exports)")
        layout.addWidget(self.name_edit)

        layout.addStretch(1)

        def action(text, tip, signal, primary=False):
            btn = QtWidgets.QPushButton(text)
            btn.setToolTip(tip)
            btn.clicked.connect(signal.emit)
            if primary:
                btn.setObjectName("primaryBtn")
            layout.addWidget(btn)
            return btn

        self.train_btn = action("▶ Train", "Generate the training script and "
                                "run it in-app", self.train_clicked, primary=True)
        self.test_btn = action("⚡ Test Run", "Build the model and run one "
                               "forward pass (shape smoke test) in a subprocess",
                               self.test_clicked)
        action("✓ Validate", "Check the graph and report issues",
               self.validate_clicked)
        action("⤓ LLM Script", "Generate a script from the LLM blocks on canvas "
               "(generation / LoRA fine-tune / RAG)", self.llm_clicked)
        action("⤢ Expand", "Expand selected architecture blocks into primitives",
               self.expand_clicked)
        action("💾 Save Selection", "Save the selected blocks as a reusable "
               "Custom block", self.save_selection_clicked)

        for label, framework in (("⬇ PyTorch", "pytorch"), ("⬇ Keras", "keras")):
            button = QtWidgets.QToolButton(self)
            button.setText(label)
            button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
            button.setPopupMode(
                QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
            menu = QtWidgets.QMenu(button)
            _menu_btn(menu, "Export model (.py)",
                      lambda f=framework: self.export_requested.emit(f, "model"))
            _menu_btn(menu, "Export training script (.py)",
                      lambda f=framework: self.export_requested.emit(f, "train"))
            if framework == "pytorch":
                menu.addSeparator()
                _menu_btn(menu, "Export ONNX (.onnx)",
                          lambda: self.runtime_export_requested.emit("onnx"))
                _menu_btn(menu, "Export TorchScript (.pt)",
                          lambda: self.runtime_export_requested.emit("jit"))
                menu.addSeparator()
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
    btn = QtWidgets.QPushButton(text, menu)
    btn.setFlat(True)
    btn.setStyleSheet("text-align:left; padding:6px 20px; background:transparent;"
                      " border:none; font-weight:400;")
    btn.clicked.connect(handler)
    btn.clicked.connect(menu.close)
    action = QtWidgets.QWidgetAction(menu)
    action.setDefaultWidget(btn)
    menu.addAction(action)
