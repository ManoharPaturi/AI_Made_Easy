"""MissionsPanel: PRIMM-guided builds for young learners.

Every mission is Predict → Run → Investigate → Modify → Make (PRIMM).
Stages complete on events the context forwards (guess made, run finished,
museum/inspect opened, graph modified between runs, own project trained)
plus plain IR predicates where the graph alone can tell. Dual tracks:
🟢 ages 8–10 (guided, fewer knobs) and 🔵 11–14. Finishing a mission
unlocks a tiny checkpoint quiz — wrong answers point back at the mission.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

_PRIMM = [
    ("predict", "🔮 Predict — make your guess first"),
    ("run", "▶ Run — train it"),
    ("investigate", "🔍 Investigate — open the Mistake Museum or 👀"),
    ("modify", "🔧 Modify — change something and retrain"),
    ("make", "🎨 Make — train your own new project"),
]


def _types(ir) -> set[str]:
    return {n.type_id for n in ir.nodes.values()}


def _valid(ir) -> bool:
    return not [i for i in ir.validate() if i.severity == "error"]


MISSIONS: list[dict] = [
    {
        "id": "first",
        "band": "🟢",
        "title": "Train your first recognizer",
        "sample": "mnist_cnn.json",
        "blurb": "Build a network that reads handwritten digits.",
        "quiz": [
            ("What does the Conv2D layer look for in a picture?",
             ["whole objects at once", "small local patterns like edges",
              "the caption text", "colours only"], 1,
             "Conv filters slide over small patches — that's why they're "
             "great at edges and textures."),
            ("The model got an 8 wrong and guessed a 3. Best FIRST move?",
             ["add more layers", "look at it in the Mistake Museum",
              "train for 100 epochs", "delete everything"], 1,
             "Look before you tweak — the museum shows you WHY it messed up."),
            ("What fixes 'too few photos of cats' best?",
             ["more cat photos", "a bigger model", "less epochs",
              "a longer name"], 0,
             "Data problems need data solutions."),
        ],
    },
    {
        "id": "memory",
        "band": "🟢",
        "title": "Remember order, step by step",
        "sample": "lstm_classifier.json",
        "blurb": "Use an LSTM to follow the ORDER of words, not just the "
                 "words.",
        "quiz": [
            ("Why use an LSTM instead of plain Dense layers for text?",
             ["it's newer", "it remembers earlier words in the sequence",
              "it has more parameters", "it trains without data"], 1,
             "The recurrence carries context across the sequence."),
            ("In 'not good', why does 'not' matter to the model?",
             ["it doesn't — words are independent",
              "it flips the meaning of 'good' later in the sequence",
              "it's shorter", "it's a verb"], 1,
             "Exactly — order carries meaning; that's the whole point."),
            ("Your sequence model ignores early words. Likely cause?",
             ["too short a memory window", "too many epochs",
              "the name is bad", "black background"], 0,
             "Context window and architecture decide how far back it sees."),
        ],
    },
    {
        "id": "fair",
        "band": "🔵",
        "title": "Fair or fake? Beat the background trick",
        "sample": "bias_arc_biased.json",
        "blurb": "A model that cheats without lying — and how to fix it.",
        "quiz": [
            ("The biased model scores high in training but fails new data. "
             "What did it learn?",
             ["the pet shape", "the background shortcut",
              "nothing at all", "the labels by heart only"], 1,
             "Background perfectly matched the class in training — the "
             "easy shortcut won."),
            ("Which fix makes the model fair?",
             ["train longer", "add the fair sample with mixed backgrounds",
              "add layers", "rename the classes"], 1,
             "Breaking the shortcut in DATA is what breaks the shortcut in "
             "the model."),
            ("The Mistake Museum shows misses on unusual backgrounds. "
             "This is called…",
             ["overfitting to a shortcut", "undertraining",
              "bad luck", "a hardware bug"], 0,
             "Shortcut learning — the classic fairness trap."),
        ],
    },
    {
        "id": "rag",
        "band": "🔵",
        "title": "Ask an AI anything (RAG)",
        "sample": "llm_rag_assistant.json",
        "blurb": "Make the AI answer from documents you give it.",
        "quiz": [
            ("What stops a RAG assistant from making things up?",
             ["a longer prompt", "answers come only from your documents",
              "more epochs", "bigger fonts"], 1,
             "Retrieval grounds the answer in the loaded text."),
            ("Why split documents into chunks?",
             ["to confuse the model", "so each piece fits the context "
              "window and retrieval stays sharp",
              "chapters are traditional", "to save disk space"], 1,
             "Small focused chunks retrieve better than one huge blob."),
            ("The assistant cites the wrong chunk. Best first move?",
             ["shout at it", "tune chunk size / overlap and re-check",
              "delete the docs", "train a new model"], 1,
             "Chunking IS the tuning knob for retrieval quality."),
        ],
    },
]


class QuizDialog(QtWidgets.QDialog):
    """2–3 question checkpoint; wrong answers point back at the mission."""

    def __init__(self, parent, mission: dict):  # noqa: ANN001
        super().__init__(parent)
        self.setWindowTitle(f"🏅 Checkpoint — {mission['title']}")
        self.setModal(True)
        self.resize(520, 460)
        layout = QtWidgets.QVBoxLayout(self)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        host = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(host)
        col.setSpacing(6)
        self._questions = mission["quiz"]
        self._answers: list[int] = [-1] * len(self._questions)
        self._rows: list[list[QtWidgets.QRadioButton]] = []
        for qi, (q, options, _ans, _why) in enumerate(self._questions):
            label = QtWidgets.QLabel(f"<b>{qi + 1}. {q}</b>")
            label.setWordWrap(True)
            col.addWidget(label)
            group = QtWidgets.QButtonGroup(self)
            row = []
            for oi, opt in enumerate(options):
                rb = QtWidgets.QRadioButton(opt)
                group.addButton(rb, oi)
                rb.toggled.connect(lambda on, q=qi, o=oi:
                                   self._answers.__setitem__(q, o) if on
                                   else None)
                row.append(rb)
                col.addWidget(rb)
            self._rows.append(row)
        col.addStretch(1)
        scroll.setWidget(host)
        layout.addWidget(scroll, 1)

        self.feedback = QtWidgets.QLabel(" ")
        self.feedback.setWordWrap(True)
        layout.addWidget(self.feedback)
        check = QtWidgets.QPushButton("Check my answers ✓")
        check.clicked.connect(self._check)
        layout.addWidget(check)
        self._check_btn = check
        self._mission = mission

    def _check(self) -> None:
        correct_count = 0
        parts = []
        for qi, (_q, _options, ans, why) in enumerate(self._questions):
            got = self._answers[qi]
            ok = got == ans
            correct_count += ok
            parts.append(("✅" if ok else "❌") + " " + why)
            if not ok:
                for rb in self._rows[qi]:
                    rb.setStyleSheet("font-weight: 700;"
                                     if rb.text() == _options_of(
                                         self._questions[qi])[ans] else "")
        total = len(self._questions)
        if correct_count == total:
            self._check_btn.setEnabled(False)
            parts.append("🎉 Mission mastered!")
        else:
            parts.append("💡 re-open the mission — the steps hint at the "
                         "answers")
        self.feedback.setText(f"{correct_count}/{total} — " + " · ".join(parts))


def _options_of(question):
    return question[1]


class MissionsPanel(QtWidgets.QWidget):
    mission_selected = QtCore.Signal(str)  # sample filename

    def __init__(self, parent=None):
        super().__init__(parent)
        self._active: dict | None = None
        self._events: set[str] = set()
        self._quiz_done = False
        # hug the mission list — the block library below takes the rest
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                           QtWidgets.QSizePolicy.Policy.Maximum)

        self._root = QtWidgets.QVBoxLayout(self)
        self._root.setContentsMargins(6, 6, 6, 6)
        self._root.setSpacing(4)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("🚀 Missions")
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
        for band, label in (("🟢", "ages 8–10"), ("🔵", "ages 11–14")):
            tag = QtWidgets.QLabel(f"{band} <i>{label}</i>")
            tag.setObjectName("bandTag")
            body.addWidget(tag)
            for mission in MISSIONS:
                if mission["band"] != band:
                    continue
                btn = QtWidgets.QPushButton(f"{band} {mission['title']}")
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
        self._events = set()
        self._quiz_done = False
        self.mission_selected.emit(mission["sample"])
        self._render()

    def _toggle(self) -> None:
        hidden = not self._body.isVisible()
        self._body.setVisible(hidden)
        self._collapse.setText("▾" if hidden else "▸")

    def active_sample(self) -> str | None:
        return self._active["sample"] if self._active else None

    def mission_event(self, key: str) -> None:
        """Context forwards PRIMM events: predict/run/investigate/..."""
        if self._active and key in dict(_PRIMM):
            self._events.add(key)
            self._render()

    def check(self, ir) -> None:  # noqa: ANN001 — core Graph
        """IR predicates can pre-complete modify for RAG-style missions."""
        if not self._active:
            return
        if self._active["id"] == "rag":
            types = _types(ir)
            if {"llm.model", "llm.rag"} <= types:
                self._events.add("predict")  # loaded = prediction material
        self._render()

    def notify_run_finished(self) -> None:
        self.mission_event("run")

    def all_done(self) -> bool:
        return (self._active is not None
                and set(dict(_PRIMM)) <= self._events)

    def take_quiz(self) -> dict | None:
        """Return the quiz payload once, when the mission completes."""
        if self.all_done() and not self._quiz_done:
            self._quiz_done = True
            return self._active
        return None

    def _render(self) -> None:
        if not self._active:
            self._steps.setText("")
            return
        band = self._active["band"]
        lines = [f"<b>{band} {self._active['blurb']}</b>"]
        for key, label in _PRIMM:
            ok = key in self._events
            lines.append(f"{'✅' if ok else '⬜'} {label}")
        if self.all_done():
            lines.append("🎉 <b>All stages done — take the checkpoint "
                         "quiz!</b>" if not self._quiz_done
                         else "🏅 <b>Mission mastered!</b>")
        self._steps.setText("<br>".join(lines))
