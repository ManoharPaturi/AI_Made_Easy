"""MissionsPanel: guided starter builds for young learners.

Three missions, each backed by a sample project and a live checklist of
plain-language steps. Steps are predicates over the IR (pure) plus one
"run training" step that completes when a training run finishes. The panel
sits at the top of the Blocks palette and can be collapsed.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ai_made_easy.core.registry import get_registry


def _types(ir) -> set[str]:
    return {n.type_id for n in ir.nodes.values()}


def _valid(ir) -> bool:
    return not [i for i in ir.validate() if i.severity == "error"]


MISSIONS: list[dict] = [
    {
        "id": "first",
        "title": "🚀 Train your first recognizer",
        "sample": "mnist_cnn.json",
        "blurb": "Build a network that reads handwritten digits.",
        "steps": [
            ("1. An Input block feeding an Output block",
             lambda ir: "core.input" in _types(ir) and "core.output" in _types(ir)),
            ("2. At least one Conv2D layer",
             lambda ir: "core.conv2d" in _types(ir)),
            ("3. No ✖ errors in 🩺 Checks", _valid),
            ("4. Press ▶ Train and watch it learn", "run"),
        ],
    },
    {
        "id": "memory",
        "title": "🧠 Remember sequences like a human",
        "sample": "lstm_classifier.json",
        "blurb": "Use an LSTM to understand order, not just values.",
        "steps": [
            ("1. An LSTM or GRU block on the canvas",
             lambda ir: bool(_types(ir) & {"core.lstm", "core.gru"})),
            ("2. A model wired Input → … → Output", _valid),
            ("3. Press ▶ Train", "run"),
        ],
    },
    {
        "id": "rag",
        "title": "💬 Ask an AI anything (RAG)",
        "sample": "llm_rag_assistant.json",
        "blurb": "Make the AI answer from documents you give it.",
        "steps": [
            ("1. An HF Model block", lambda ir: "llm.model" in _types(ir)),
            ("2. A RAG Pipeline block", lambda ir: "llm.rag" in _types(ir)),
            ("3. A Text Splitter to chunk the documents",
             lambda ir: "llm.text_splitter" in _types(ir)),
            ("4. No ✖ errors in 🩺 Checks",
             lambda ir: _valid(ir) and "llm.rag" in _types(ir)),
        ],
    },
]


class MissionsPanel(QtWidgets.QWidget):
    mission_selected = QtCore.Signal(str)  # sample filename
    open_requested = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._active: dict | None = None
        self._done: set[int] = set()

        self._root = QtWidgets.QVBoxLayout(self)
        self._root.setContentsMargins(6, 6, 6, 6)
        self._root.setSpacing(4)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("🚀 Start here — pick a mission")
        title.setObjectName("cardTitle")
        header.addWidget(title, 1)
        self._collapse = QtWidgets.QToolButton()
        self._collapse.setText("▾")
        self._collapse.clicked.connect(self._toggle)
        header.addWidget(self._collapse)
        self._root.addLayout(header)

        self._body = QtWidgets.QWidget()
        body = QtWidgets.QVBoxLayout(self._body)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(3)
        for mission in MISSIONS:
            btn = QtWidgets.QPushButton(mission["title"])
            btn.setToolTip(mission["blurb"])
            btn.clicked.connect(
                lambda _=False, m=mission: self._select(m))
            body.addWidget(btn)
        self._steps = QtWidgets.QLabel("")
        self._steps.setWordWrap(True)
        self._steps.setStyleSheet("padding: 2px 4px;")
        body.addWidget(self._steps)
        self._root.addWidget(self._body)

    # ------------------------------------------------------------ updates

    def _select(self, mission: dict) -> None:
        self._active = mission
        self._done = set()
        self.mission_selected.emit(mission["sample"])
        self._render()

    def _toggle(self) -> None:
        hidden = not self._body.isVisible()
        self._body.setVisible(hidden)
        self._collapse.setText("▾" if hidden else "▸")

    def check(self, ir) -> None:  # noqa: ANN001 — core Graph
        if not self._active:
            return
        for idx, (_label, predicate) in enumerate(self._active["steps"]):
            if predicate == "run":
                continue
            try:
                if predicate(ir):
                    self._done.add(idx)
            except Exception:
                pass
        self._render()

    def notify_run_finished(self) -> None:
        if not self._active:
            return
        for idx, (_label, predicate) in enumerate(self._active["steps"]):
            if predicate == "run":
                self._done.add(idx)
        self._render()

    def _render(self) -> None:
        if not self._active:
            self._steps.setText("")
            return
        lines = [f"<b>{self._active['blurb']}</b>"]
        total = done = 0
        for idx, (label, _p) in enumerate(self._active["steps"]):
            total += 1
            ok = idx in self._done
            done += ok
            lines.append(f"{'✅' if ok else '⬜'} {label}")
        if done == total:
            lines.append("🎉 <b>Mission complete — amazing work!</b>")
        self._steps.setText("<br>".join(lines))
