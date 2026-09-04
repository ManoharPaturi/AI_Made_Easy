"""The .aime bundle: one shareable file for a whole project.

zip layout:
    manifest.json        {format, version, app_version, created, name, ...}
    graph.json           the canvas (Graph.to_dict)
    model_card.md        the report card (optional)
    thumbnail.png        canvas picture (optional)
    dataset/<class>/…    image-folder photos (optional, size-capped)
    run/checkpoint, predictions.json, train script (optional)

`test_on_a_friend` swaps the bundle's weights onto the friend's own
dataset (same architecture, different photos) and reports both scores.
"""
from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

FORMAT = "aime"
BUNDLE_VERSION = 1
MAX_DATASET_FILES = 600
MAX_DATASET_BYTES = 40 * 1024 * 1024


def _manifest(name: str, entries: dict) -> dict:
    try:
        from importlib.metadata import version

        app_version = version("ai-made-easy")
    except Exception:
        app_version = "dev"
    return {
        "format": FORMAT,
        "version": BUNDLE_VERSION,
        "app": "AI Made Easy",
        "app_version": app_version,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "name": name,
        "entries": entries,
    }


def write_bundle(path: Path, graph_dict: dict, *, name: str = "project",
                 card_md: str | None = None, thumbnail_png: bytes | None = None,
                 dataset_dir: Path | None = None,
                 workdir: Path | None = None) -> Path:
    """Assemble the .aime zip. Missing pieces are simply omitted."""
    entries: dict[str, bool] = {}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        payload = {"graph.json": json.dumps(graph_dict, indent=1)}
        entries["graph"] = True
        if card_md:
            payload["model_card.md"] = card_md
            entries["model_card"] = True
        if thumbnail_png:
            payload["thumbnail.png"] = None  # binary, written separately
        for member, text in payload.items():
            if text is not None:
                zf.writestr(member, text)
        if thumbnail_png:
            zf.writestr("thumbnail.png", thumbnail_png)
            entries["thumbnail"] = True
        if dataset_dir and Path(dataset_dir).exists():
            count = size = 0
            for f in sorted(Path(dataset_dir).rglob("*")):
                if not f.is_file() or f.suffix.lower() not in (
                        ".png", ".jpg", ".jpeg", ".bmp"):
                    continue
                if count >= MAX_DATASET_FILES or size > MAX_DATASET_BYTES:
                    break
                zf.write(f, f"dataset/{f.relative_to(dataset_dir)}")
                count += 1
                size += f.stat().st_size
            entries["dataset_files"] = count
        if workdir and Path(workdir).exists():
            for pattern, member in (("*_best.pt", "run/checkpoint.pt"),
                                    ("*_train_pytorch.py", "run/train.py"),
                                    ("predictions.json",
                                     "run/predictions.json"),
                                    ("mistakes.json", "run/mistakes.json")):
                hits = sorted(Path(workdir).glob(pattern))
                if hits:
                    zf.write(hits[0], member)
                    entries[member.split("/")[0]] = True
        zf.writestr("manifest.json", json.dumps(
            _manifest(name, entries), indent=1))
    return Path(path)


def read_bundle(path: Path) -> dict:
    """Open + validate; returns manifest + members (dataset extracted to a
    temp dir the caller owns)."""
    import tempfile

    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        if "manifest.json" not in names or "graph.json" not in names:
            raise ValueError(f"{path.name} isn't an .aime bundle "
                             "(missing manifest/graph)")
        manifest = json.loads(zf.read("manifest.json"))
        if manifest.get("format") != FORMAT:
            raise ValueError(f"unknown bundle format "
                             f"{manifest.get('format')!r}")
        if int(manifest.get("version", 0)) > BUNDLE_VERSION:
            raise ValueError("this bundle is from a newer AI Made Easy — "
                             "please update the app")
        out = {
            "manifest": manifest,
            "graph": json.loads(zf.read("graph.json")),
            "card": (zf.read("model_card.md").decode()
                     if "model_card.md" in names else None),
            "has_dataset": any(n.startswith("dataset/") for n in names),
            "has_run": "run/checkpoint.pt" in names,
            "run_dir": None,
            "dataset_dir": None,
        }
        if out["has_run"]:
            run_dir = Path(tempfile.mkdtemp(prefix="aime_bundle_run_"))
            for n in names:
                if n.startswith("run/"):
                    zf.extract(n, run_dir)
            out["run_dir"] = run_dir / "run"
        if out["has_dataset"]:
            ds_dir = Path(tempfile.mkdtemp(prefix="aime_bundle_ds_"))
            for n in names:
                if n.startswith("dataset/"):
                    zf.extract(n, ds_dir)
            out["dataset_dir"] = ds_dir / "dataset"
    return out


SWAP_EVAL_TEMPLATE = '''\
"""Swap-test: the friend's checkpoint on MY dataset (generated)."""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = r"{root}"
MODE = "L" if {grayscale} else "RGB"
INPUT_SHAPE = {input_shape}
NORM = {norm}
BATCH = 64

import importlib.util as _ilu

spec = _ilu.spec_from_file_location("friend_model", r"{model_script}")
mod = _ilu.module_from_spec(spec)
spec.loader.exec_module(mod)
model_cls = next(o for o in vars(mod).values()
                 if isinstance(o, type) and issubclass(o, torch.nn.Module)
                 and o.__module__ == "friend_model")
model = model_cls()
state = torch.load(r"{checkpoint}", map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.eval()

classes = sorted(d.name for d in Path(ROOT).iterdir() if d.is_dir())
c, h, w = INPUT_SHAPE
xs, ys = [], []
for ci, cls in enumerate(classes):
    for f in sorted((Path(ROOT) / cls).iterdir()):
        if f.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
            continue
        img = Image.open(f).convert(MODE)
        if img.size != (w, h):
            img = img.resize((w, h))
        xs.append(np.asarray(img, dtype="float32") / 255.0)
        ys.append(ci)
x = torch.tensor(np.stack(xs))
if x.ndim == 3:
    x = x.unsqueeze(1)
else:
    x = x.permute(0, 3, 1, 2)
if NORM:
    mean = torch.tensor(NORM[0]).view(1, -1, 1, 1)
    std = torch.tensor(NORM[1]).view(1, -1, 1, 1)
    x = (x - mean) / std
correct = n = 0
preds = []
with torch.no_grad():
    for i in range(0, len(x), BATCH):
        out = torch.softmax(model(x[i:i + BATCH]), dim=1)
        top = out.argmax(dim=1)
        correct += int((top == torch.tensor(ys[i:i + BATCH])).sum())
        for j, p in enumerate(out.tolist()):
            preds.append({{"index": i + j, "true": ys[i + j], "probs": p}})
acc = correct / max(len(x), 1)
json.dump({{"accuracy": acc, "n": len(x), "classes": classes,
            "predictions": preds[:300]}},
          open("swap_predictions.json", "w"), indent=1)
print(f"swap-test accuracy: {{acc:.3%}} on {{len(x)}} photos")
'''
