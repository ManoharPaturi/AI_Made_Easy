"""Augmentation preview: run the design's image transforms on real samples.

Builds the same torchvision ``transforms.v2`` pipeline the generated PyTorch
script uses (always-applied transforms, training augmentation, after-tensor
augmentation) and runs it in a subprocess of the training environment on a few
images from the dataset, writing PNGs: the original, the evaluation view and
several random training views per image.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from ai_made_easy.core.data.profile import IMAGE_SUFFIXES, resolve_path
from ai_made_easy.core.graph import Graph

MARKER = "PREVIEW-RESULT "

_SCRIPT = r'''
import json
import random
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F

cfg = json.loads(Path(sys.argv[1]).read_text())
torch.manual_seed(cfg["seed"])
random.seed(cfg["seed"])
out = Path(cfg["out"])
out.mkdir(parents=True, exist_ok=True)
always = [eval(e, {"v2": v2, "torch": torch}) for e in cfg["always"]]
train = [eval(e, {"v2": v2, "torch": torch}) for e in cfg["train"]]
after = [eval(e, {"v2": v2, "torch": torch}) for e in cfg["after"]]
to_float = v2.ToDtype(torch.float32, scale=True)
eval_tf = v2.Compose([v2.ToImage(), *always, to_float])
train_tf = v2.Compose([v2.ToImage(), *always, *train, to_float, *after])


def save(tensor, name):
    path = out / name
    F.to_pil_image(tensor.clamp(0, 1)).save(path)
    return str(path)


result = []
for i, file in enumerate(cfg["files"]):
    image = Image.open(file).convert("L" if cfg["grayscale"] else "RGB")
    item = {"source": file, "size": list(image.size)}
    original = image.copy()
    original.thumbnail((256, 256))
    item["original"] = str(out / f"{i}_original.png")
    original.save(item["original"])
    view = eval_tf(image)
    item["eval"] = save(view, f"{i}_eval.png")
    item["eval_shape"] = list(view.shape)
    item["train"] = [save(train_tf(image), f"{i}_train_{k}.png")
                     for k in range(cfg["variants"])]
    result.append(item)
print("PREVIEW-RESULT " + json.dumps({"images": result}))
'''


class PreviewError(RuntimeError):
    """The preview could not be produced (message is user-facing)."""


# previewers for other dataset kinds: fn(graph, out_dir, **options) -> result or None
PREVIEWERS: list = []


def register_previewer(fn) -> None:  # noqa: ANN001
    if fn not in PREVIEWERS:
        PREVIEWERS.append(fn)


def run_preview(script: str, cfg: dict, out: Path, python: str | None, timeout: float) -> dict:
    """Run a preview script on ``cfg`` in the training environment; parse its result."""
    (out / "preview.json").write_text(json.dumps(cfg))
    (out / "preview.py").write_text(script)
    env = {**os.environ, "PYTHONWARNINGS": "ignore"}
    try:
        proc = subprocess.run([python or sys.executable, str(out / "preview.py"),
                               str(out / "preview.json")], capture_output=True, text=True,
                              timeout=timeout, env=env)
    except subprocess.TimeoutExpired as exc:
        raise PreviewError(f"the preview took longer than {timeout:.0f}s") from exc
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith(MARKER)), None)
    if line is None:
        tail = (proc.stderr or proc.stdout)[-1500:]
        if "No module named" in tail:
            missing = tail.split("No module named", 1)[1].strip().splitlines()[0]
            raise PreviewError(f"the training environment lacks {missing}")
        raise PreviewError("the preview failed:\n" + tail)
    return json.loads(line[len(MARKER):])


def image_pipeline(graph: Graph) -> dict:
    """Transform expressions in the order the training script applies them."""
    from ai_made_easy.core.training import data_catalog as dcat

    order = {pid: i for i, pid in enumerate(dcat.PREP_IDS)}
    nodes = sorted((n for n in graph.nodes.values() if n.type_id in order),
                   key=lambda n: order[n.type_id])
    pipe: dict[str, list] = {"always": [], "train": [], "after": [], "skipped": []}
    for node in nodes:
        blk = dcat.BLOCKS[node.type_id]
        if "image" not in blk.applies_to or blk.stage not in ("always", "train"):
            continue
        if blk.torch is None:
            pipe["skipped"].append(blk.name)
            continue
        expr = blk.torch(dict(node.resolved_params()))
        bucket = ("always" if blk.stage == "always" else
                  "after" if blk.meta.get("after_tensor") else "train")
        pipe[bucket].append(expr)
    explicit = any(n.type_id in ("prep.resize", "prep.center_crop") for n in nodes)
    if not explicit:  # the script resizes folder images to the model input
        try:
            shape = graph.infer_shapes()[graph.model_nodes()[0].instance_id]
            if len(shape) == 3:
                pipe["always"].append(f"v2.Resize(({shape[1]}, {shape[2]}), antialias=True)")
        except Exception:  # noqa: BLE001 — incomplete designs keep the image size
            pass
    if any(n.type_id == "prep.mixup_cutmix" for n in nodes):
        pipe["skipped"].append("MixUp / CutMix (mixes whole batches)")
    # expressions like "A(), B()" (Random Flip "both") expand to several transforms
    for bucket in ("always", "train", "after"):
        pipe[bucket] = [part for expr in pipe[bucket] for part in _split_exprs(expr)]
    return pipe


def _split_exprs(expr: str) -> list[str]:
    import ast

    tree = ast.parse(expr, mode="eval").body
    items = tree.elts if isinstance(tree, ast.Tuple) else [tree]
    return [ast.unparse(item) for item in items]


def sample_images(root: Path, count: int) -> list[Path]:
    """One image per class (round-robin) up to ``count``."""
    classes = sorted(d for d in root.iterdir() if d.is_dir())
    files = [sorted(f for f in d.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
             for d in classes]
    picked: list[Path] = []
    depth = 0
    while len(picked) < count and any(len(f) > depth for f in files):
        picked += [f[depth] for f in files if len(f) > depth][: count - len(picked)]
        depth += 1
    return picked


def augmentation_preview(graph: Graph, out_dir: str | Path, *, base=None,
                         python: str | None = None, images: int = 4, variants: int = 6,
                         seed: int = 0, files: list[str] | None = None,
                         timeout: float = 180) -> dict:
    """Render evaluation and training views of a few dataset images."""
    for previewer in PREVIEWERS:
        result = previewer(graph, out_dir, base=base, python=python, images=images,
                           variants=variants, seed=seed, timeout=timeout)
        if result is not None:
            return result
    data = next((n for n in graph.nodes.values() if n.type_id == "data.image_folder"), None)
    if data is None and not files:
        raise PreviewError("augmentation preview needs an Image Folder dataset")
    params = dict(data.resolved_params()) if data is not None else {}
    if not files:
        root = resolve_path(params.get("root", ""), base)
        if not root.is_dir():
            raise PreviewError(f"dataset folder {root} was not found")
        files = [str(f) for f in sample_images(root, images)]
        if not files:
            raise PreviewError(f"no images found in {root}")
    pipe = image_pipeline(graph)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"files": files, "out": str(out), "seed": seed, "variants": variants,
           "grayscale": bool(params.get("grayscale")), "always": pipe["always"],
           "train": pipe["train"], "after": pipe["after"]}
    result = run_preview(_SCRIPT, cfg, out, python, timeout)
    result.update(always=pipe["always"], train_transforms=pipe["train"] + pipe["after"],
                  skipped=pipe["skipped"])
    return result
