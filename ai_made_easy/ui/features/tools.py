"""The tools you already use, from the desktop app: experiment tracking, Kaggle / Hub
datasets, and sharing registered models (MLflow, W&B, the Hugging Face Hub)."""
from __future__ import annotations

import threading
from html import escape

from PySide6 import QtCore, QtWidgets

from ai_made_easy.core import api, tracking

SHARE_TARGETS = (
    ("mlflow", "Register in MLflow"),
    ("wandb", "Register in Weights && Biases"),
    ("hub", "Push to Hugging Face Hub…"),
    ("space", "Deploy as a Hugging Face Space…"),
)


class Background(QtCore.QObject):
    """Run ``fn`` on a thread; ``done(result, error)`` arrives on the UI thread."""

    done = QtCore.Signal(object, str)

    def start(self, fn) -> None:  # noqa: ANN001
        def work() -> None:
            try:
                self.done.emit(fn(), "")
            except Exception as exc:  # noqa: BLE001 — shown to the user
                self.done.emit(None, str(exc))

        threading.Thread(target=work, daemon=True).start()


class TrackingDialog(QtWidgets.QDialog):
    """Per-project experiment tracking: ``meta["tracking"]``."""

    def __init__(self, parent, chosen: list[str]) -> None:  # noqa: ANN001
        super().__init__(parent)
        self.setWindowTitle("Experiment Tracking")
        status = api.tracking_status()
        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "Mirror this project's runs as they train, on every backend: parameters, "
            "per-epoch metrics, the design and the run's files. Sign in with the tool "
            "itself (<code>wandb login</code>, <code>MLFLOW_TRACKING_URI</code>); "
            "AI Made Easy never stores a token.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.checks: dict[str, QtWidgets.QCheckBox] = {}
        for row in status["trackers"]:
            box = QtWidgets.QCheckBox(row["label"] if row["installed"] else
                                      f"{row['label']} (not installed: {tracking.EXTRA})")
            box.setChecked(row["id"] in chosen)
            box.setEnabled(row["installed"] or row["id"] in chosen)
            box.setObjectName(f"track_{row['id']}")
            self.checks[row["id"]] = box
            layout.addWidget(box)
        note = QtWidgets.QLabel(
            f"MLflow store: <code>{escape(status['mlflow_uri'])}</code>"
            + (f"<br>Every run also uses <code>AIME_TRACKING={escape(status['env'])}</code>"
               if status["env"] else ""))
        note.setObjectName("blockMeta")
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                             | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected(self) -> list[str]:
        return [k for k, box in self.checks.items() if box.isChecked()]


class FetchDatasetDialog(QtWidgets.QDialog):
    """Download a Kaggle / Hub dataset into the cache and detect it."""

    def __init__(self, parent=None) -> None:  # noqa: ANN001
        super().__init__(parent)
        self.setWindowTitle("Download a Dataset")
        self.resize(560, 260)
        self.path = ""
        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "<code>kaggle:owner/name</code>, <code>kaggle-competition:name</code> or "
            "<code>hf:owner/repo</code>. Kaggle uses your <code>~/.kaggle/kaggle.json</code>; "
            "the download is detected like a local folder.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        row = QtWidgets.QHBoxLayout()
        self.source = QtWidgets.QLineEdit()
        self.source.setPlaceholderText("kaggle:owner/name")
        self.fetch_btn = QtWidgets.QPushButton("Download")
        row.addWidget(self.source, 1)
        row.addWidget(self.fetch_btn)
        layout.addLayout(row)
        self.result = QtWidgets.QLabel("")
        self.result.setWordWrap(True)
        self.result.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.result, 1)
        buttons = QtWidgets.QDialogButtonBox()
        self.use_btn = buttons.addButton("New Project from It…",
                                         QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        self.use_btn.setEnabled(False)
        buttons.addButton(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._worker = Background(self)
        self._worker.done.connect(self._on_done)
        self.fetch_btn.clicked.connect(self._fetch)
        self.source.returnPressed.connect(self._fetch)

    def _fetch(self) -> None:
        source = self.source.text().strip()
        if not source:
            return
        self.fetch_btn.setEnabled(False)
        self.result.setText("Downloading…")
        self._worker.start(lambda: api.fetch_dataset(source))

    def _on_done(self, result: dict | None, error: str) -> None:
        self.fetch_btn.setEnabled(True)
        if result is None:
            self.result.setText(f"<span style='color:#e5484d'>{escape(error)}</span>")
            return
        self.path = result["path"]
        facts = result["facts"]
        tasks = ", ".join(facts.get("tasks") or []) or "no task detected"
        self.result.setText(f"<b>{escape(self.path)}</b><br>{escape(facts.get('kind', ''))}: "
                            f"{facts.get('rows', 0):,} rows · fits {escape(tasks)}")
        self.use_btn.setEnabled(True)


def ask_repo(parent, title: str, label: str, default: str) -> str:  # noqa: ANN001
    repo, ok = QtWidgets.QInputDialog.getText(parent, title, label, text=default)
    return repo.strip() if ok else ""


def confirm_publish(parent, what: str, where: str) -> bool:  # noqa: ANN001
    answer = QtWidgets.QMessageBox.question(
        parent, "Publish to the Hugging Face Hub",
        f"Upload {what} to {where} as a private repo? It leaves this computer; you can "
        "make it public on the Hub later.")
    return answer == QtWidgets.QMessageBox.StandardButton.Yes
