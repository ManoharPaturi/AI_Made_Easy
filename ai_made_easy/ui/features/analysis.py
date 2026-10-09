"""Post-training analysis dialogs: Error Analysis and Model Card."""
from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from ai_made_easy.core.model_card import (
    build_card,
    per_class_accuracy,
    read_artifact,
    top_confusions,
)


class ErrorAnalysisDialog(QtWidgets.QDialog):
    """Per-class accuracy, top confusions and the misclassified test samples."""

    def __init__(self, parent, workdir: Path):
        super().__init__(parent)
        self.setWindowTitle("Error Analysis")
        self.resize(980, 660)
        predictions = read_artifact(workdir, "predictions.json", []) or []
        mistakes = read_artifact(workdir, "mistakes.json", []) or []
        classes = read_artifact(workdir, "classes.json", None)
        self._label = (lambda c: str(classes[c]) if classes and isinstance(c, int)
                       and c < len(classes) else f"class {c}")
        self._workdir = Path(workdir)

        layout = QtWidgets.QVBoxLayout(self)
        scored = [p for p in predictions if len(p.get("probs", [])) > 1]
        correct = sum(1 for p in scored if _argmax(p["probs"]) == p["true"])
        headline = QtWidgets.QLabel(
            f"{correct}/{len(scored)} correct ({correct / max(len(scored), 1):.1%}) on the "
            f"first {len(predictions)} test samples · {len(mistakes)} misclassified shown")
        headline.setObjectName("blockMeta")
        layout.addWidget(headline)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        title = QtWidgets.QLabel("PER-CLASS ACCURACY")
        title.setObjectName("sectionTitle")
        left_layout.addWidget(title)
        stats = per_class_accuracy(predictions)
        table = QtWidgets.QTableWidget(len(stats), 3)
        table.setHorizontalHeaderLabels(["Class", "Correct", "Accuracy"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for row, (cls, (ok, n)) in enumerate(stats.items()):
            table.setItem(row, 0, QtWidgets.QTableWidgetItem(self._label(cls)))
            table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{ok}/{n}"))
            bar = QtWidgets.QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(int(100 * ok / max(n, 1)))
            bar.setToolTip(f"{ok / max(n, 1):.1%}")
            table.setCellWidget(row, 2, bar)
        left_layout.addWidget(table, 1)
        title2 = QtWidgets.QLabel("MOST FREQUENT CONFUSIONS")
        title2.setObjectName("sectionTitle")
        left_layout.addWidget(title2)
        conf = QtWidgets.QListWidget()
        for a, b, n in top_confusions(predictions, 10):
            conf.addItem(f"{self._label(a)} → {self._label(b)}   ×{n}")
        if not conf.count():
            conf.addItem("No misclassifications in the sampled predictions.")
        left_layout.addWidget(conf, 1)
        split.addWidget(left)

        samples = QtWidgets.QListWidget()
        samples.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        samples.setIconSize(QtCore.QSize(96, 96))
        samples.setGridSize(QtCore.QSize(210, 150))
        samples.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        samples.setWordWrap(True)
        samples.setMovement(QtWidgets.QListView.Movement.Static)
        for item in mistakes:
            guess = _argmax(item["probs"])
            top = sorted(range(len(item["probs"])), key=lambda i: -item["probs"][i])[:3]
            text = (f"#{item['index']}  actual {self._label(item['true'])}\n"
                    f"predicted {self._label(guess)} ({item['probs'][guess]:.0%})")
            entry = QtWidgets.QListWidgetItem(text)
            entry.setToolTip("\n".join(f"{self._label(c)}: {item['probs'][c]:.1%}" for c in top))
            path = self._resolve(item.get("file"))
            if path is not None:
                entry.setIcon(QtGui.QIcon(QtGui.QPixmap(str(path))))
            samples.addItem(entry)
        if not mistakes:
            samples.addItem("No misclassified samples recorded.")
        split.addWidget(samples)
        split.setSizes([320, 640])
        layout.addWidget(split, 1)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _resolve(self, file) -> Path | None:  # noqa: ANN001
        if not file:
            return None
        path = Path(file)
        if not path.is_absolute():
            path = self._workdir / path
        return path if path.exists() else None


class VisionAnalysisDialog(QtWidgets.QDialog):
    """Detection / segmentation results: per-class scores and test-image overlays."""

    def __init__(self, parent, workdir: Path):
        super().__init__(parent)
        self.setWindowTitle("Error Analysis")
        self.resize(1040, 680)
        workdir = Path(workdir)
        metrics = read_artifact(workdir, "metrics.json", {}) or {}
        classes = read_artifact(workdir, "classes.json", []) or []
        layout = QtWidgets.QVBoxLayout(self)
        shown = [f"{k} {v:.3f}" for k, v in metrics.items() if isinstance(v, (int, float))]
        headline = QtWidgets.QLabel("Test set: " + (" · ".join(shown) or "no metrics"))
        headline.setObjectName("blockMeta")
        layout.addWidget(headline)
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        per_class = metrics.get("per_class_ap") or metrics.get("per_class_iou") or []
        offset = 0 if "per_class_ap" in metrics or len(classes) == len(per_class) else 1
        label = "AP" if "per_class_ap" in metrics else "IoU"
        table = QtWidgets.QTableWidget(len(per_class), 2)
        table.setHorizontalHeaderLabels(["Class", label])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for row, value in enumerate(per_class):
            name = classes[row - offset] if 0 <= row - offset < len(classes) else str(row)
            if offset and row == 0:
                name = "background"
            table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(name)))
            table.setItem(row, 1, QtWidgets.QTableWidgetItem(
                "—" if value is None else f"{value:.3f}"))
        split.addWidget(table)
        gallery = QtWidgets.QListWidget()
        gallery.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        gallery.setIconSize(QtCore.QSize(300, 220))
        gallery.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        gallery.setMovement(QtWidgets.QListView.Movement.Static)
        for path in sorted((workdir / "eval_samples").glob("*.png")):
            item = QtWidgets.QListWidgetItem(QtGui.QIcon(QtGui.QPixmap(str(path))), path.stem)
            item.setToolTip("green: ground truth · red: prediction" if label == "AP"
                            else "left: ground truth · right: prediction")
            gallery.addItem(item)
        split.addWidget(gallery)
        split.setSizes([260, 780])
        layout.addWidget(split, 1)
        note = QtWidgets.QLabel("Boxes: green = ground truth, red = prediction (with score)."
                                if label == "AP" else
                                "Each sample shows the true mask (left) and the prediction "
                                "(right).")
        note.setObjectName("blockMeta")
        layout.addWidget(note)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class ModelCardDialog(QtWidgets.QDialog):
    """Generated model card with editable intended-use / limitations sections."""

    def __init__(self, parent, name: str, dataset_comment: str, trainer: dict,
                 workdir: Path | None, details: dict):
        super().__init__(parent)
        self.setWindowTitle(f"Model Card — {name}")
        self.resize(860, 680)
        self._args = (name, dataset_comment, trainer, workdir, details)
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.use = QtWidgets.QPlainTextEdit()
        self.use.setPlaceholderText("Intended uses and users")
        self.use.setFixedHeight(64)
        self.limits = QtWidgets.QPlainTextEdit()
        self.limits.setPlaceholderText("Known limitations, failure modes, out-of-scope uses")
        self.limits.setFixedHeight(64)
        form.addRow("Intended use", self.use)
        form.addRow("Limitations", self.limits)
        layout.addLayout(form)
        self.preview = QtWidgets.QTextBrowser()
        layout.addWidget(self.preview, 1)
        buttons = QtWidgets.QDialogButtonBox()
        save = buttons.addButton("Save as Markdown…",
                                 QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(QtWidgets.QDialogButtonBox.StandardButton.Close)
        save.clicked.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.use.textChanged.connect(self._render)
        self.limits.textChanged.connect(self._render)
        self._render()

    def markdown(self) -> str:
        name, comment, trainer, workdir, details = self._args
        return build_card(name, comment, trainer, workdir, self.use.toPlainText(),
                          self.limits.toPlainText(), details)

    def _render(self) -> None:
        self.preview.setMarkdown(self.markdown().split("---", 2)[-1])

    def _save(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Model Card", f"{self._args[0]}_model_card.md", "Markdown (*.md)")
        if path:
            Path(path).write_text(self.markdown())


def _argmax(values: list) -> int:
    return max(range(len(values)), key=lambda i: values[i])
