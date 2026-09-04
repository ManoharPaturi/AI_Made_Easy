"""DataPreviewDialog: peek at the data a block will feed the model.

Double-click a Data block → a friendly look at what's inside (first rows,
class names, file peek). Reads are small and local; nothing heavy imports
unless the dataset kind needs it.
"""
from __future__ import annotations

from pathlib import Path

from PySide6 import QtWidgets

_TV_CLASSES = {
    "mnist": "digits 0-9 (60,000 train images, 28×28 grayscale)",
    "fashion_mnist": "T-shirt, Trouser, Pullover, Dress, Coat, Sandal, "
                     "Shirt, Sneaker, Bag, Ankle boot (28×28 grayscale)",
    "cifar10": "airplane, automobile, bird, cat, deer, dog, frog, horse, "
               "ship, truck (32×32 color)",
    "cifar100": "100 fine-grained classes in 20 groups (32×32 color)",
}


def _synthetic_rows(params: dict) -> str:
    """Numpy-only preview mirroring how the training script builds data."""
    try:
        import numpy as np

        kind = params.get("kind", "classification")
        n = min(int(params.get("n_samples", 1000)), 100_000)
        d = min(int(params.get("n_features", 20)), 2000)
        k = max(int(params.get("n_classes", 2)), 2)
        noise = float(params.get("noise", 0.3))
        rng = np.random.default_rng(int(params.get("seed", 42)))
        n_show = min(n, 200)

        if kind == "regression":
            x = rng.normal(size=(n_show, d))
            w = rng.normal(size=d)
            y = x @ w + rng.normal(scale=noise, size=n_show)
        else:
            labels = rng.integers(0, k, size=n_show)
            centers = rng.normal(scale=2.5 + noise * 5, size=(k, d))
            x = centers[labels] + rng.normal(scale=0.6 + noise, size=(n_show, d))
            y = labels

        lines = [f"{n} rows × {d} features (previewing {n_show}, "
                 f"showing first 6)", ""]
        for i in range(min(6, len(x))):
            cells = ", ".join(f"{v:6.2f}" for v in x[i][:6])
            more = " …" if d > 6 else ""
            lines.append(f"  [{cells}{more}]  → {y[i]}")
        import collections
        counts = ", ".join(f"{c}×{cnt}" for c, cnt in
                           sorted(collections.Counter(y.tolist()).items()))
        lines.append(f"\nclass mix: {counts}")
        return "\n".join(lines)
    except Exception as exc:
        return f"(could not generate a preview: {exc})"


def _csv_rows(path: str, target: str) -> str:
    import csv as _csv
    p = Path(path)
    if not p.exists():
        return (f"📄 {path}\n\n(file not found — it will need to exist "
                "before training runs)")
    with p.open() as fh:
        reader = _csv.reader(fh)
        header = next(reader)
        rows = [row for _, row in zip(range(5), reader)]
    out = [f"📄 {path} — {len(header)} columns", "",
           "  " + " | ".join(header)]
    for row in rows:
        out.append("  " + " | ".join(row))
    mark = f"   → target column: {target}" if target else ""
    return "\n".join(out) + mark


def _peek(params: dict, type_id: str) -> str:
    path = params.get("path", "")
    if type_id == "data.torchvision":
        name = params.get("dataset", "mnist")
        return (f"🖼️ torchvision '{name}'\n\n"
                + _TV_CLASSES.get(name, "images + labels (downloaded on "
                                         "first training run)"))
    if type_id == "data.numpy":
        import numpy as np
        p = Path(path)
        if not p.exists():
            return f"🗄️ {path}\n\n(file not found yet)"
        with np.load(p) as z:
            keys = ", ".join(f"{k} {list(z[k].shape)}" for k in z.files)
        return f"🗄️ {path}\n\narrays: {keys}"
    if type_id == "data.json":
        p = Path(path)
        if not p.exists():
            return f"🧾 {path}\n\n(file not found yet)"
        import json
        text = p.read_text().strip()
        records = (json.loads(text) if text.startswith("[") else
                   [json.loads(l) for l in text.splitlines() if l.strip()][:3])
        return (f"🧾 {path} — {len(records)} record(s) shown\n\n"
                + "\n".join(str(r) for r in records[:3]))
    if type_id == "data.image_folder":
        root = Path(params.get("root", "data/images"))
        if not root.exists():
            return (f"🖼️ {root}\n\n(folder not found yet — one subfolder "
                    "per class, images inside)")
        classes = [d.name for d in sorted(root.iterdir()) if d.is_dir()]
        return f"🖼️ {root}\n\nclasses ({len(classes)}): {', '.join(classes[:10])}"
    if type_id == "data.huggingface":
        return (f"🤗 HuggingFace dataset '{params.get('repo_id', '')}' "
                f"(split: {params.get('split', 'train')})\n\n"
                "downloaded from the Hub on first training run")
    return ""


class DataPreviewDialog(QtWidgets.QDialog):
    def __init__(self, parent, definition, params: dict):  # noqa: ANN001
        super().__init__(parent)
        self.setWindowTitle(f"👀 {definition.display_name} — data preview")
        self.setModal(True)
        self.resize(560, 420)
        layout = QtWidgets.QVBoxLayout(self)

        type_id = definition.type_id
        if type_id == "data.synthetic":
            text = _synthetic_rows(params)
        elif type_id == "data.csv":
            text = _csv_rows(params.get("path", ""),
                             params.get("target_column", ""))
        else:
            text = _peek(params, type_id) or "\n".join(
                f"{k}: {v}" for k, v in params.items() if v)

        view = QtWidgets.QPlainTextEdit(text)
        view.setReadOnly(True)
        layout.addWidget(view)
        close = QtWidgets.QPushButton("Got it 👍")
        close.clicked.connect(self.accept)
        layout.addWidget(close)
