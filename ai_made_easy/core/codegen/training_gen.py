"""Training-script generation: collect the canvas config blocks (data,
preprocessing, training, evaluation) into a TrainingSpec and render a
runnable, self-contained training script.

The generated PyTorch script depends only on torch + numpy (torchvision is
imported only when a Torchvision Dataset block is used). Phase 3's in-app
runner executes exactly this output, so progress lines follow a stable
``epoch e/E key=value ...`` format.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from jinja2 import Environment

from ai_made_easy.core.codegen import (
    CodegenError,
    class_name_for,
    emit_dag,
    sanitize_identifier,
)
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.spec import shape_volume

_env = Environment(trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)

_LOSSES = ("train.loss_cross_entropy", "train.loss_mse", "train.loss_bce_logits",
           "train.loss_l1", "train.loss_smooth_l1", "train.loss_poisson",
           "train.loss_kl_div", "train.loss_focal")
_OPTIMIZERS = ("train.sgd", "train.adam", "train.adamw", "train.adagrad",
               "train.rmsprop", "train.adadelta", "train.adamax", "train.nadam",
               "train.radam")
_SCHEDULERS = ("train.step_lr", "train.cosine_annealing_lr", "train.one_cycle_lr",
               "train.multistep_lr", "train.exponential_lr",
               "train.warm_restarts_lr", "train.plateau_lr", "train.linear_lr")
_DATASETS = ("data.torchvision", "data.csv", "data.synthetic", "data.numpy",
             "data.json", "data.image_folder", "data.huggingface")
_REGRESSION_LOSSES = ("train.loss_mse", "train.loss_l1", "train.loss_smooth_l1",
                      "train.loss_poisson")

METRIC_BLOCKS = {
    "eval.accuracy": "accuracy",
    "eval.precision": "precision",
    "eval.recall": "recall",
    "eval.f1": "f1",
    "eval.confusion_matrix": "confusion",
    "eval.mse": "mse",
    "eval.mae": "mae",
    "eval.roc_auc": "roc_auc",
}


@dataclass
class TrainingSpec:
    """Normalized view of every config block on the canvas."""

    name: str = "untitled"
    input_volume: int = 0
    output_units: int = 0
    is_regression: bool = False
    dataset: dict = field(default_factory=dict)
    normalize: dict | None = None
    split: dict = field(default_factory=lambda: {
        "val_fraction": 0.1, "test_fraction": 0.1, "shuffle": True, "seed": 42})
    loader: dict = field(default_factory=lambda: {
        "batch_size": None, "num_workers": 0, "pin_memory": True})
    optimizer: dict = field(default_factory=lambda: {"kind": "train.adam", "lr": 1e-3, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8})
    loss: dict = field(default_factory=lambda: {"kind": "train.loss_cross_entropy", "label_smoothing": 0.0})
    scheduler: dict | None = None
    trainer: dict = field(default_factory=lambda: {
        "epochs": 10, "batch_size": 32, "device": "auto", "seed": 42,
        "early_stopping_patience": 0})
    metrics: list[str] = field(default_factory=list)
    kfold: dict | None = None
    impute: dict | None = None
    predict: dict | None = None
    metric_average: str = "macro"
    minmax: dict | None = None
    augmentations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _parse_floats(value: str):
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None
    return nums[0] if len(nums) == 1 else nums


def _collect_augmentations(graph: Graph, spec: TrainingSpec) -> None:
    """Fold augmentation blocks into a canonical torchvision transform order:
    resize -> crop -> flip -> rotation -> jitter. Warns when the dataset
    cannot use them."""
    found = {n.type_id: n.resolved_params() for n in graph.nodes.values()
             if n.type_id.startswith("prep.") and n.type_id in _AUG_BLOCKS}
    if not found:
        return
    if spec.dataset.get("block") != "data.torchvision":
        spec.warnings.append(
            "augmentation blocks apply to the Torchvision Dataset pipeline; "
            f"dataset '{spec.dataset.get('block')}' loads arrays, so they are skipped"
        )
        return
    if "prep.resize" in found:
        p = found["prep.resize"]
        spec.augmentations.append(
            f"tv_transforms.Resize(({int(p['height'])}, {int(p['width'])}))")
    if "prep.center_crop" in found:
        p = found["prep.center_crop"]
        spec.augmentations.append(f"tv_transforms.CenterCrop({int(p['size'])})")
    if "prep.random_flip" in found:
        mode = found["prep.random_flip"]["mode"]
        if mode in ("horizontal", "both"):
            spec.augmentations.append("tv_transforms.RandomHorizontalFlip()")
        if mode in ("vertical", "both"):
            spec.augmentations.append("tv_transforms.RandomVerticalFlip()")
    if "prep.random_rotation" in found:
        p = found["prep.random_rotation"]
        spec.augmentations.append(f"tv_transforms.RandomRotation({p['degrees']})")
    if "prep.color_jitter" in found:
        p = found["prep.color_jitter"]
        spec.augmentations.append(
            "tv_transforms.ColorJitter("
            f"brightness={p['brightness']}, contrast={p['contrast']}, "
            f"saturation={p['saturation']}, hue={p['hue']})")


_AUG_BLOCKS = (
    "prep.resize", "prep.center_crop", "prep.random_flip",
    "prep.random_rotation", "prep.color_jitter",
)


def collect_spec(graph: Graph) -> TrainingSpec:
    """Fold config blocks into a TrainingSpec; raise on contradictions."""
    shapes = graph.infer_shapes()
    chain = graph.model_nodes()
    input_volume = shape_volume(shapes[chain[0].instance_id])
    output_units = shapes[chain[-1].instance_id][0]

    spec = TrainingSpec(name=graph.name, input_volume=input_volume,
                        output_units=output_units)

    def pick(candidates: tuple[str, ...], label: str) -> list:
        found = [n for n in graph.nodes.values() if n.type_id in candidates]
        if len(found) > 1:
            raise CodegenError(
                f"expected at most one {label} block, found {len(found)}: "
                + ", ".join(n.instance_id for n in found)
            )
        return found

    ds = pick(_DATASETS, "dataset")
    if ds:
        spec.dataset = {"block": ds[0].type_id, **ds[0].resolved_params()}
    else:
        # smart default: synthetic data shaped exactly like the model's IO
        spec.dataset = {
            "block": "data.synthetic", "kind": "classification",
            "n_samples": 1000, "n_features": input_volume,
            "n_classes": output_units, "noise": 0.3, "seed": 42,
        }

    norm = [n for n in graph.nodes.values() if n.type_id == "prep.normalize"]
    if norm:
        p = norm[0].resolved_params()
        spec.normalize = {"mean": _parse_floats(p["mean"]),
                          "std": _parse_floats(p["std"])}

    mm = [n for n in graph.nodes.values() if n.type_id == "prep.minmax"]
    if mm:
        p = mm[0].resolved_params()
        spec.minmax = {"range_min": float(p["range_min"]),
                       "range_max": float(p["range_max"])}
        if spec.normalize:
            spec.warnings.append(
                "Normalize and MinMax are both present: z-score is applied "
                "first, then MinMax rescaling"
            )

    _collect_augmentations(graph, spec)

    sp = [n for n in graph.nodes.values() if n.type_id == "prep.split"]
    if sp:
        spec.split = dict(sp[0].resolved_params())

    ld = [n for n in graph.nodes.values() if n.type_id == "prep.dataloader"]
    if ld:
        spec.loader = dict(ld[0].resolved_params())

    opt = pick(_OPTIMIZERS, "optimizer")
    if opt:
        spec.optimizer = dict(opt[0].resolved_params()) | {"kind": opt[0].type_id}

    loss = pick(_LOSSES, "loss")
    if loss:
        spec.loss = dict(loss[0].resolved_params()) | {"kind": loss[0].type_id}
    spec.is_regression = spec.loss["kind"] in _REGRESSION_LOSSES

    sch = pick(_SCHEDULERS, "scheduler")
    if sch:
        spec.scheduler = dict(sch[0].resolved_params()) | {"kind": sch[0].type_id}

    tr = pick(("train.trainer",), "trainer")
    if tr:
        spec.trainer = dict(tr[0].resolved_params())

    kf = [n for n in graph.nodes.values() if n.type_id == "train.kfold"]
    if kf:
        spec.kfold = dict(kf[0].resolved_params())
    im = [n for n in graph.nodes.values() if n.type_id == "prep.impute"]
    if im:
        spec.impute = dict(im[0].resolved_params())
    pr = [n for n in graph.nodes.values() if n.type_id == "eval.predict"]
    if pr:
        spec.predict = dict(pr[0].resolved_params())

    averages = set()
    for n in graph.nodes.values():
        short = METRIC_BLOCKS.get(n.type_id)
        if short:
            if short not in spec.metrics:
                spec.metrics.append(short)
            avg = n.resolved_params().get("average")
            if avg:
                averages.add(avg)
    if len(averages) == 1:
        spec.metric_average = averages.pop()
    if spec.is_regression:
        kept = [m for m in spec.metrics if m in ("mse", "mae")]
        spec.metrics = list(dict.fromkeys(kept + ["mse", "mae"]))
    else:
        if "accuracy" not in spec.metrics:
            spec.metrics.insert(0, "accuracy")
    return spec


# ---------------------------------------------------------------- templates

TRAIN_PYTORCH_TEMPLATE = '''\
"""{{ spec.name }} - training script generated by AI Made Easy."""
{% if spec.warnings %}
# NOTE (from your block setup):
{% for w in spec.warnings %}
# - {{ w }}
{% endfor %}
{% endif %}
import time
{% if dataset_is_json %}
import json as _json
{% endif %}

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ------------------------------------------------------------------ config
SEED = {{ spec.trainer.seed }}
DEVICE = "{{ spec.trainer.device }}"
EPOCHS = {{ spec.trainer.epochs }}
{% if spec.kfold %}
K_FOLDS = {{ spec.kfold.k }}
STRATIFIED = {{ "True" if spec.kfold.stratified else "False" }}
KFOLD_SEED = {{ spec.kfold.seed }}
{% endif %}
BATCH_SIZE = {{ loader_batch }}
EARLY_STOP = {{ spec.trainer.early_stopping_patience }}
VAL_FRACTION = {{ spec.split.val_fraction }}
TEST_FRACTION = {{ spec.split.test_fraction }}
NUM_WORKERS = {{ spec.loader.num_workers }}
PIN_MEMORY = {{ pin_memory_str }}
CHECKPOINT = "{{ spec.name }}_best.pt"
INPUT_SHAPE = {{ input_shape_literal }}
{% if spec.normalize %}
NORM_MEAN = {{ spec.normalize.mean }}
NORM_STD = {{ spec.normalize.std }}
{% endif %}

def pick_device() -> str:
    if DEVICE != "auto":
        return DEVICE
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ------------------------------------------------------------------- model
class {{ class_name }}(nn.Module):
    def __init__(self):
        super().__init__()
{% for m in modules %}
        {{ m.torch_module }}
{% endfor %}

    def forward(self, x):
{% for n in nodes %}
        {{ n.torch_expr }}
{% endfor %}
        return {{ output_var }}


# ----------------------------------------------------------------- dataset
{% if dataset_is_torchvision %}
from torchvision import datasets, transforms as tv_transforms


def make_loaders(device_hint: str):
    tf = []
{% for aug in spec.augmentations %}
    tf.append({{ aug }})
{% endfor %}
    tf.append(tv_transforms.ToTensor())
{% if spec.normalize %}
    tf.append(tv_transforms.Normalize(NORM_MEAN, NORM_STD))
{% endif %}
    transform = tv_transforms.Compose(tf)
    train_full = datasets.{{ dataset_class }}(root="{{ spec.dataset.data_dir }}", train=True,
                                               download={{ download_str }}, transform=transform)
    test = datasets.{{ dataset_class }}(root="{{ spec.dataset.data_dir }}", train=False,
                                        download={{ download_str }}, transform=transform)
    n_val = int(len(train_full) * VAL_FRACTION)
    n_train = len(train_full) - n_val
    gen = torch.Generator().manual_seed(SEED)
    train, val = torch.utils.data.random_split(train_full, [n_train, n_val], generator=gen)
    mk = lambda ds: DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    return mk(train), mk(val), mk(test)

{% else %}

{% if spec.impute %}
def impute_missing(data):
    """Fill Missing Values block (strategy={{ spec.impute.strategy }})."""
    strategy = "{{ spec.impute.strategy }}"
    if strategy == "drop rows":
        return data[~np.isnan(data).any(axis=1)]
    for c in range(data.shape[1]):
        col_vals = data[:, c]
        known = col_vals[~np.isnan(col_vals)]
        if np.isnan(col_vals).any() and known.size:
            if strategy == "constant":
                fill = {{ spec.impute.constant }}
            elif strategy == "mode":
                vals, counts = np.unique(known, return_counts=True)
                fill = float(vals[counts.argmax()])
            else:
                fill = float(np.{{ "nanmean" if spec.impute.strategy == "mean" else "nanmedian" }}(known))
            col_vals[np.isnan(col_vals)] = fill
    return data


{% endif %}
def make_arrays():
    """{{ dataset_comment }}"""
    rng = np.random.default_rng(SEED)
{% if dataset_is_csv %}
    import csv
    with open("{{ spec.dataset.path }}") as fh:
        header = next(csv.reader(fh))
    target = "{{ spec.dataset.target_column }}"
    col = header.index(target) if target in header else int(target)
    data = np.genfromtxt("{{ spec.dataset.path }}", delimiter=",", skip_header=1)
{% if spec.impute %}
    data = impute_missing(data)
{% endif %}
    x = np.delete(data, col, axis=1)
    y = data[:, col]
{% elif dataset_is_numpy %}
    archive = np.load("{{ spec.dataset.path }}")
    x = archive["{{ spec.dataset.x_key }}"]
    y = archive["{{ spec.dataset.y_key }}"]
{% elif dataset_is_json %}
    with open("{{ spec.dataset.path }}") as fh:
        _text = fh.read().strip()
    records = (_json.loads(_text) if _text.startswith("[") else
               [_json.loads(line) for line in _text.splitlines() if line.strip()])
    x = np.asarray([r["{{ spec.dataset.x_field }}"] for r in records], dtype=np.float32)
    y = np.asarray([r["{{ spec.dataset.y_field }}"] for r in records])
{% elif dataset_is_huggingface %}
    from datasets import load_dataset
    ds = load_dataset("{{ spec.dataset.repo_id }}", split="{{ spec.dataset.split }}")
    x = np.asarray(ds["{{ spec.dataset.x_field }}"], dtype=np.float32)
    y = np.asarray(ds["{{ spec.dataset.y_field }}"])
{% if input_rank > 1 %}
    x = x.reshape(len(x), *INPUT_SHAPE)
{% endif %}
{% elif dataset_is_image_folder %}
    import os
    from PIL import Image
    root = "{{ spec.dataset.root }}"
    mode = "L" if {{ grayscale_lit }} else "RGB"
    classes = sorted(d for d in os.listdir(root)
                     if os.path.isdir(os.path.join(root, d)))
    if not classes:
        raise SystemExit(f"no class subfolders under {root}")
    th, tw = INPUT_SHAPE[-2], INPUT_SHAPE[-1]

    def _to_chw(arr):
        return arr[None, :, :] if arr.ndim == 2 else arr.transpose(2, 0, 1)

    xs, ys = [], []
    for ci, cls in enumerate(classes):
        folder = os.path.join(root, cls)
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue
            img = Image.open(os.path.join(folder, fname)).convert(mode)
            if img.size != (tw, th):
                img = img.resize((tw, th))
            xs.append(_to_chw(np.asarray(img, dtype=np.float32)) / 255.0)
            ys.append(ci)
    x = np.stack(xs)
    y = np.asarray(ys)
{% elif dataset_is_synthetic %}
{% if synthetic_kind == "moons" %}
    n = {{ spec.dataset.n_samples }}
    t = rng.uniform(0, np.pi, size=(n // 2, 1))
    x0 = np.hstack([np.cos(t), np.sin(t)]) + rng.normal(0, {{ spec.dataset.noise }}, (n // 2, 2))
    x1 = np.hstack([1 - np.cos(t), 0.5 - np.sin(t)]) + rng.normal(0, {{ spec.dataset.noise }}, (n // 2, 2))
    x = np.vstack([x0, x1]).astype(np.float32)
    y = np.repeat([0, 1], n // 2)
{% elif synthetic_kind == "circles" %}
    n = {{ spec.dataset.n_samples }}
    r0, r1 = 0.6, 1.4
    a0 = rng.uniform(0, 2 * np.pi, (n // 2, 1)); a1 = rng.uniform(0, 2 * np.pi, (n // 2, 1))
    x0 = np.hstack([r0 * np.cos(a0), r0 * np.sin(a0)]) + rng.normal(0, {{ spec.dataset.noise }}, (n // 2, 2))
    x1 = np.hstack([r1 * np.cos(a1), r1 * np.sin(a1)]) + rng.normal(0, {{ spec.dataset.noise }}, (n // 2, 2))
    x = np.vstack([x0, x1]).astype(np.float32)
    y = np.repeat([0, 1], n // 2)
{% elif synthetic_kind == "regression" %}
    n, d = {{ spec.dataset.n_samples }}, {{ spec.dataset.n_features }}
    w = rng.normal(size=d)
    x = rng.normal(size=(n, d)).astype(np.float32)
    y = (x @ w + rng.normal(0, {{ spec.dataset.noise }}, n)).astype(np.float32)
{% if input_rank > 1 %}
    x = x.reshape(n, *INPUT_SHAPE)
{% endif %}
{% else %}
    n, d, k = {{ spec.dataset.n_samples }}, {{ spec.dataset.n_features }}, {{ spec.dataset.n_classes }}
    centers = rng.normal(scale=2.0, size=(k, d))
    y = rng.integers(0, k, size=n)
    x = (centers[y] + rng.normal(scale=0.5, size=(n, d))).astype(np.float32)
{% if input_rank > 1 %}
    x = x.reshape(n, *INPUT_SHAPE)
{% endif %}
{% endif %}
{% endif %}
    return x, y


def make_loaders(device_hint: str):
    x, y = make_arrays()
{% if spec.normalize %}
    x = (x - {{ norm_mean_expr }}) / {{ norm_std_expr }}
{% endif %}
{% if spec.minmax %}
    lo, hi = x.min(axis=0), x.max(axis=0)
    x = (x - lo) / (hi - lo + 1e-12) * ({{ spec.minmax.range_max }} - {{ spec.minmax.range_min }}) + {{ spec.minmax.range_min }}
{% endif %}
    x = x.astype(np.float32)
    y = y.astype({{ y_dtype }})
    n = len(x)
    idx = {{ permute_expr }}
    n_val = int(n * VAL_FRACTION)
    n_test = int(n * TEST_FRACTION)
    val_i = idx[:n_val]
    test_i = idx[n_val:n_val + n_test]
    train_i = idx[n_val + n_test:]
    mk = lambda ids: DataLoader(
        TensorDataset(torch.from_numpy(x[ids]), torch.from_numpy(y[ids])),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    return mk(train_i), mk(val_i), mk(test_i)

{% endif %}

{% if not is_regression %}
N_CLASSES = {{ spec.output_units }}


def class_metrics(preds: torch.Tensor, targets: torch.Tensor):
    """Confusion-matrix based {{ spec.metric_average }} precision/recall/F1."""
    cm = torch.bincount(targets * N_CLASSES + preds, minlength=N_CLASSES ** 2)
    cm = cm.reshape(N_CLASSES, N_CLASSES).float()
    tp = cm.diag()
    precision = tp / (cm.sum(0) + 1e-12)
    recall = tp / (cm.sum(1) + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return precision.mean().item(), recall.mean().item(), f1.mean().item(), cm.long()

{% endif %}


{% if needs_auc %}
def _binary_auc(scores: torch.Tensor, pos: torch.Tensor) -> float:
    """Rank-based ROC-AUC (Mann-Whitney U) for one positive class."""
    order = scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float64)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if not n_pos or not n_neg:
        return float("nan")
    return float((ranks[pos].sum().item() - n_pos * (n_pos + 1) / 2)
                 / (n_pos * n_neg))


{% endif %}
@torch.no_grad()
def evaluate(model, loss_fn, loader, device):
    model.eval()
    total, n_seen = 0.0, 0
{% if is_regression %}
    abs_sum = sq_sum = 0.0
{% else %}
    all_p, all_t = [], []
{% if needs_auc %}
    all_o = []
{% endif %}
{% endif %}
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
{% if is_regression %}
        err = out.squeeze(-1) - yb
        total += err.pow(2).sum().item()
        abs_sum += err.abs().sum().item()
        sq_sum += err.pow(2).sum().item()
        n_seen += len(yb)
{% else %}
        total += loss_fn(out, yb).item() * len(yb)
        n_seen += len(yb)
        all_p.append(out.argmax(1).cpu())
        all_t.append(yb.cpu())
{% if needs_auc %}
        all_o.append(out.detach().cpu())
{% endif %}
{% endif %}
    metrics = {}
{% if is_regression %}
    metrics["mse"] = sq_sum / max(n_seen, 1)
    metrics["mae"] = abs_sum / max(n_seen, 1)
    return total / max(n_seen, 1), metrics
{% else %}
    p, t = torch.cat(all_p), torch.cat(all_t)
    metrics["accuracy"] = (p == t).float().mean().item()
{% if needs_auc %}
    logits = torch.cat(all_o)
    probs = torch.softmax(logits, dim=1)
    if probs.shape[1] == 2:
        metrics["roc_auc"] = _binary_auc(probs[:, 1], t == 1)
    else:
        per_class = [_binary_auc(probs[:, c], t == c)
                     for c in range(probs.shape[1])
                     if 0 < int((t == c).sum()) < len(t)]
        metrics["roc_auc"] = (float(np.mean(per_class))
                              if per_class else float("nan"))
{% endif %}
{% if needs_prf %}
    prec, rec, f1, cm = class_metrics(p, t)
    metrics["precision"] = prec
    metrics["recall"] = rec
    metrics["f1"] = f1
{% endif %}
{% if needs_cm %}
    _, _, _, cm = class_metrics(p, t)
    metrics["confusion"] = cm
{% endif %}
    return total / max(n_seen, 1), metrics
{% endif %}


{% if loss_is_focal %}
class FocalLoss(nn.Module):
    """Focal cross-entropy (Lin et al., 2017)."""

    def __init__(self, gamma: float, alpha: float):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, logits, target):
        ce = torch.nn.functional.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.alpha:
            loss = self.alpha * loss
        return loss.mean()


{% endif %}
{% if spec.predict %}
PREDICT_SAMPLES = {{ spec.predict.n_samples }}


def predict_demo(model, loader, device):
    """Predictions block: use the trained model on fresh examples."""
    model.eval()
    shown = 0
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(device)).cpu()
            single = out.shape[1] == 1
            probs = out if single else torch.softmax(out, dim=1)
            for i in range(len(yb)):
                p, actual = probs[i], int(yb[i])
                if single:
                    print(f"example {shown + 1}: predicted "
                          f"{p[0].item():.3f} | actual {actual}")
                else:
                    if {{ "True" if spec.predict.show_probabilities else "False" }}:
                        top = torch.topk(p, k=min(3, len(p)))
                        picks = ", ".join(
                            f"class {int(c)} ({v:.0%})"
                            for c, v in zip(top.indices, top.values))
                        print(f"example {shown + 1}: actual class {actual}"
                              f" -> {picks}")
                    else:
                        print(f"example {shown + 1}: predicted class "
                              f"{int(p.argmax())} | actual {actual}")
                shown += 1
                if shown >= PREDICT_SAMPLES:
                    return


{% endif %}
{% if spec.kfold %}
def kfold_cross_validate(train_loader, val_loader, device, loss_fn):
    """K-Fold Cross-Validation block: retrain K times, report mean +- std."""
    full = torch.utils.data.ConcatDataset(
        [train_loader.dataset, val_loader.dataset])
    n = len(full)
    ys = torch.tensor([int(y) for _, y in full])
    gen = torch.Generator().manual_seed(KFOLD_SEED)
    fold_of = torch.zeros(n, dtype=torch.long)
    if STRATIFIED and ys.unique().numel() > 1:
        for c in ys.unique():
            idx_c = torch.where(ys == c)[0]
            idx_c = idx_c[torch.randperm(len(idx_c), generator=gen)]
            fold_of[idx_c] = torch.arange(len(idx_c)) % K_FOLDS
    else:
        fold_of[torch.randperm(n, generator=gen)] = torch.arange(n) % K_FOLDS

    scores = []
    for k in range(K_FOLDS):
        val_idx = torch.where(fold_of == k)[0]
        tr_idx = torch.where(fold_of != k)[0]
        tr_dl = torch.utils.data.DataLoader(
            torch.utils.data.Subset(full, tr_idx.tolist()),
            batch_size=BATCH_SIZE, shuffle=True)
        va_dl = torch.utils.data.DataLoader(
            torch.utils.data.Subset(full, val_idx.tolist()),
            batch_size=BATCH_SIZE)
        model = {{ class_name }}().to(device)
        optimizer = {{ optimizer_expr }}
        model.train()
        for epoch in range(1, EPOCHS + 1):
            running = 0.0
            for xb, yb in tr_dl:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()
                running += loss.item() * len(xb)
        val_loss, val_metrics = evaluate(model, loss_fn, va_dl, device)
        score = val_metrics.get("accuracy", -val_loss)
        scores.append(score)
        extra = " ".join(f"{k2}={v:.4f}" if isinstance(v, float) else f"{k2}={v}"
                         for k2, v in val_metrics.items())
        print(f"fold {k + 1}/{K_FOLDS}: loss={val_loss:.4f} {extra}")
    mean = sum(scores) / len(scores)
    spread = (sum((s - mean) ** 2 for s in scores) / max(len(scores) - 1, 1)) ** 0.5
    unit = "accuracy" if not isinstance(scores[0], float) or True else ""
    print(f"{K_FOLDS}-fold cross-validation: score = {mean:.4f} "
          f"+- {spread:.4f} (higher is better)")
{% endif %}


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(pick_device())
    print(f"device: {device}")
    train_loader, val_loader, test_loader = make_loaders(str(device))
{% if spec.kfold %}
    loss_fn = {{ loss_expr }}
    print(f"parameters: {sum(p.numel() for p in {{ class_name }}().parameters()):,}")
    kfold_cross_validate(train_loader, val_loader, device, loss_fn)
    return
{% endif %}
    model = {{ class_name }}().to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    loss_fn = {{ loss_expr }}
    optimizer = {{ optimizer_expr }}
{% if scheduler_expr %}
    scheduler = {{ scheduler_expr }}
{% endif %}
    best_val, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0, running = time.time(), 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
{% if scheduler_steps_per_batch %}
            scheduler.step()
{% endif %}
            running += loss.item() * len(xb)
        train_loss = running / len(train_loader.dataset)
        val_loss, val_metrics = evaluate(model, loss_fn, val_loader, device)
{% if scheduler_expr and not scheduler_steps_per_batch %}
        scheduler.step({{ scheduler_step_arg }})
{% endif %}
        extra = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in val_metrics.items())
        print(f"epoch {epoch}/{EPOCHS} train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} {extra} ({time.time() - t0:.1f}s)")
        if val_loss < best_val:
            best_val, bad_epochs = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if EARLY_STOP and bad_epochs >= EARLY_STOP:
                print(f"early stopping after {epoch} epochs without improvement")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_metrics = evaluate(model, loss_fn, test_loader, device)
    print("test:", {**test_metrics, "loss": round(test_loss, 4)})
{% if spec.predict %}
    print("\\nsample predictions (Predict block):")
    predict_demo(model, test_loader, device)
{% endif %}
    torch.save(model.state_dict(), CHECKPOINT)
    print(f"saved best weights to {CHECKPOINT}")


if __name__ == "__main__":
    main()
'''

TRAIN_KERAS_TEMPLATE = '''\
"""{{ spec.name }} - Keras training script generated by AI Made Easy."""
import numpy as np
import keras
from keras import layers

# ------------------------------------------------------------------ config
SEED = {{ spec.trainer.seed }}
EPOCHS = {{ spec.trainer.epochs }}
BATCH_SIZE = {{ loader_batch }}
EARLY_STOP = {{ spec.trainer.early_stopping_patience }}
VAL_FRACTION = {{ spec.split.val_fraction }}
TEST_FRACTION = {{ spec.split.test_fraction }}


# ------------------------------------------------------------------- model
def build_model() -> keras.Model:
    inputs = keras.Input(shape={{ keras_input_shape }})
{% for n in nodes %}
    {{ n.keras_expr }}
{% endfor %}
    return keras.Model(inputs=inputs, outputs={{ keras_output_var }})


# ----------------------------------------------------------------- dataset
{% if spec.impute %}
def impute_missing(data):
    """Fill Missing Values block (strategy={{ spec.impute.strategy }})."""
    strategy = "{{ spec.impute.strategy }}"
    if strategy == "drop rows":
        return data[~np.isnan(data).any(axis=1)]
    for c in range(data.shape[1]):
        col_vals = data[:, c]
        known = col_vals[~np.isnan(col_vals)]
        if np.isnan(col_vals).any() and known.size:
            if strategy == "constant":
                fill = {{ spec.impute.constant }}
            elif strategy == "mode":
                vals, counts = np.unique(known, return_counts=True)
                fill = float(vals[counts.argmax()])
            else:
                fill = float(np.{{ "nanmean" if spec.impute.strategy == "mean" else "nanmedian" }}(known))
            col_vals[np.isnan(col_vals)] = fill
    return data


{% endif %}
def make_arrays():
    """{{ dataset_comment }}"""
    rng = np.random.default_rng(SEED)
{% if dataset_is_synthetic %}
    n, d, k = {{ spec.dataset.n_samples }}, {{ spec.dataset.n_features }}, {{ spec.dataset.n_classes }}
    centers = rng.normal(scale=2.0, size=(k, d))
    y = rng.integers(0, k, size=n)
    x = (centers[y] + rng.normal(scale=0.5, size=(n, d))).astype(np.float32)
{% if input_rank > 1 %}
    x = x.reshape(n, *{{ keras_input_shape_literal }})
{% endif %}
    return x, y.astype(np.int64)


{% elif dataset_is_csv %}
    import csv
    with open("{{ spec.dataset.path }}") as fh:
        header = next(csv.reader(fh))
    target = "{{ spec.dataset.target_column }}"
    col = header.index(target) if target in header else int(target)
    data = np.genfromtxt("{{ spec.dataset.path }}", delimiter=",", skip_header=1)
{% if spec.impute %}
    data = impute_missing(data)
{% endif %}
    x = np.delete(data, col, axis=1).astype(np.float32)
{% if is_regression %}
    y = data[:, col].astype(np.float32)
{% else %}
    y = data[:, col].astype(np.int64)
{% endif %}
    return x, y
{% elif dataset_is_numpy %}
    archive = np.load("{{ spec.dataset.path }}")
    return archive["{{ spec.dataset.x_key }}"].astype(np.float32), archive["{{ spec.dataset.y_key }}"]
{% elif dataset_is_json %}
    import json as _json
    with open("{{ spec.dataset.path }}") as fh:
        _text = fh.read().strip()
    records = (_json.loads(_text) if _text.startswith("[") else
               [_json.loads(line) for line in _text.splitlines() if line.strip()])
    x = np.asarray([r["{{ spec.dataset.x_field }}"] for r in records], dtype=np.float32)
    y = np.asarray([r["{{ spec.dataset.y_field }}"] for r in records])
    return x, y
{% else %}
    raise SystemExit("this dataset kind is not supported for Keras export")
{% endif %}
def main():
    keras.utils.set_random_seed(SEED)
    x, y = make_arrays()
{% if spec.normalize %}
    x = (x - {{ norm_mean_expr }}) / {{ norm_std_expr }}
{% endif %}
    n = len(x)
    idx = np.random.default_rng(SEED).permutation(n)
    n_val = int(n * VAL_FRACTION)
    n_test = int(n * TEST_FRACTION)
    x_train, y_train = x[idx[n_val + n_test:]], y[idx[n_val + n_test:]]
    x_val, y_val = x[idx[:n_val]], y[idx[:n_val]]
    x_test, y_test = x[idx[n_val:n_val + n_test]], y[idx[n_val:n_val + n_test]]

    model = build_model()
    model.compile(
        optimizer={{ keras_optimizer }},
        loss={{ keras_loss }},
        metrics={{ keras_metrics }},
    )
    callbacks = []
    if EARLY_STOP:
        callbacks.append(keras.callbacks.EarlyStopping(
            patience=EARLY_STOP, restore_best_weights=True))
{% if keras_lr_scheduler %}
    callbacks.append(keras.callbacks.LearningRateScheduler({{ keras_lr_scheduler }}))
{% endif %}
{% if scheduler_is_plateau %}
    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        factor={{ spec.scheduler.factor }}, patience={{ spec.scheduler.patience }}))
{% endif %}
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=EPOCHS,
              batch_size=BATCH_SIZE, callbacks=callbacks, verbose=2)
    results = model.evaluate(x_test, y_test, verbose=0)
    print("test:", dict(zip(model.metrics_names, results)))
{% if spec.predict %}
    print("\\nsample predictions (Predict block):")
    probs = model.predict(x_test[:{{ spec.predict.n_samples }}], verbose=0)
    for i, p in enumerate(probs):
        actual = int(y_test[i])
        if p.ndim == 1 and len(p) == 1:
            print(f"example {i + 1}: predicted {float(p[0]):.3f} | actual {actual}")
        elif {{ "True" if spec.predict.show_probabilities else "False" }}:
            top = np.argsort(p)[::-1][:3]
            picks = ", ".join(f"class {int(c)} ({p[c]:.0%})" for c in top)
            print(f"example {i + 1}: actual class {actual} -> {picks}")
        else:
            print(f"example {i + 1}: predicted class {int(np.argmax(p))} | actual {actual}")
{% endif %}
    model.save("{{ spec.name }}_best.keras")
    print("saved model to {{ spec.name }}_best.keras")


if __name__ == "__main__":
    main()
'''


# ------------------------------------------------------------------ helpers

_TORCHVISION_CLASSES = {
    "mnist": "MNIST", "fashion_mnist": "FashionMNIST", "cifar10": "CIFAR10",
    "cifar100": "CIFAR100", "stl10": "STL10",
}

_LOSS_EXPRS = {
    "train.loss_cross_entropy": "nn.CrossEntropyLoss(label_smoothing={label_smoothing})",
    "train.loss_mse": "nn.MSELoss()",
    "train.loss_bce_logits": "nn.BCEWithLogitsLoss()",
    "train.loss_l1": "nn.L1Loss()",
    "train.loss_smooth_l1": "nn.SmoothL1Loss(beta={beta})",
    "train.loss_poisson": "nn.PoissonNLLLoss()",
    "train.loss_kl_div": "nn.KLDivLoss()",
    "train.loss_focal": "FocalLoss(gamma={gamma}, alpha={alpha})",
}

_OPT_EXPRS = {
    "train.sgd": "torch.optim.SGD(model.parameters(), lr={lr}, momentum={momentum}, nesterov={nesterov})",
    "train.adam": "torch.optim.Adam(model.parameters(), lr={lr}, betas=({beta1}, {beta2}), eps={eps})",
    "train.adamw": "torch.optim.AdamW(model.parameters(), lr={lr}, weight_decay={weight_decay})",
    "train.adagrad": "torch.optim.Adagrad(model.parameters(), lr={lr}, eps={eps})",
    "train.rmsprop": "torch.optim.RMSprop(model.parameters(), lr={lr}, alpha={alpha}, eps={eps})",
    "train.adadelta": "torch.optim.Adadelta(model.parameters(), lr={lr}, rho={rho})",
    "train.adamax": "torch.optim.Adamax(model.parameters(), lr={lr}, betas=({beta1}, {beta2}))",
    "train.nadam": "torch.optim.NAdam(model.parameters(), lr={lr}, betas=({beta1}, {beta2}))",
    "train.radam": "torch.optim.RAdam(model.parameters(), lr={lr}, betas=({beta1}, {beta2}))",
}

_SCHED_EXPRS = {
    "train.step_lr": "torch.optim.lr_scheduler.StepLR(optimizer, step_size={step_size}, gamma={gamma})",
    "train.cosine_annealing_lr": "torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max={T_max}, eta_min={eta_min})",
    "train.one_cycle_lr": (
        "torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr={max_lr}, "
        "total_steps=EPOCHS * len(train_loader), pct_start={pct_start})"
    ),
    "train.multistep_lr": "torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones={milestones}, gamma={gamma})",
    "train.exponential_lr": "torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma={gamma})",
    "train.warm_restarts_lr": "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0={T_0}, T_mult={T_mult})",
    "train.plateau_lr": 'torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor={factor}, patience={patience})',
    "train.linear_lr": "torch.optim.lr_scheduler.LinearLR(optimizer, start_factor={start_factor}, end_factor=1.0, total_iters={total_iters})",
}


def _render_loss(spec: TrainingSpec) -> str:
    template = _LOSS_EXPRS[spec.loss["kind"]]
    return template.format(**{k: repr(v) for k, v in spec.loss.items()})


def _render_optimizer(spec: TrainingSpec) -> str:
    template = _OPT_EXPRS[spec.optimizer["kind"]]
    return template.format(**{k: repr(v) for k, v in spec.optimizer.items()})


def _render_scheduler(spec: TrainingSpec) -> str | None:
    if not spec.scheduler:
        return None
    values = dict(spec.scheduler)
    template = _SCHED_EXPRS[values["kind"]]
    if values["kind"] == "train.multistep_lr":
        literal = "[" + ", ".join(
            p.strip() for p in str(values.pop("milestones", "")).split(",") if p.strip()
        ) + "]"
        template = template.replace("{milestones}", literal)
    return template.format(**{k: repr(v) for k, v in values.items()})


def _keras_optimizer(spec: TrainingSpec) -> str:
    o = spec.optimizer
    lr = f"learning_rate={o['lr']!r}"
    if o["kind"] == "train.sgd":
        return (f"keras.optimizers.SGD({lr}, momentum={o['momentum']!r}, "
                f"nesterov={o['nesterov']!r})")
    if o["kind"] == "train.adamw":
        return (f"keras.optimizers.AdamW({lr}, "
                f"weight_decay={o.get('weight_decay', 0.01)!r})")
    if o["kind"] == "train.adagrad":
        return f"keras.optimizers.Adagrad({lr})"
    if o["kind"] == "train.rmsprop":
        return (f"keras.optimizers.RMSprop({lr}, alpha={o['alpha']!r}, "
                f"epsilon={o['eps']!r})")
    if o["kind"] == "train.adadelta":
        return f"keras.optimizers.Adadelta({lr})"
    if o["kind"] == "train.adamax":
        return f"keras.optimizers.Adamax({lr})"
    if o["kind"] == "train.nadam":
        return f"keras.optimizers.Nadam({lr})"
    if o["kind"] == "train.radam":
        raise CodegenError("RAdam optimizer cannot be exported to Keras yet")
    return f"keras.optimizers.Adam({lr})"


def _keras_loss(spec: TrainingSpec) -> str:
    kind = spec.loss["kind"]
    if kind == "train.loss_cross_entropy":
        return '"sparse_categorical_crossentropy"'
    if kind == "train.loss_mse":
        return '"mse"'
    if kind == "train.loss_l1":
        return '"mae"'
    if kind == "train.loss_smooth_l1":
        return '"huber"'
    if kind == "train.loss_poisson":
        return '"poisson"'
    if kind == "train.loss_kl_div":
        return '"kl_divergence"'
    if kind == "train.loss_focal":
        raise CodegenError("Focal loss cannot be exported to Keras yet")
    return "keras.losses.BinaryCrossentropy(from_logits=True)"


def _keras_lr_scheduler(spec: TrainingSpec) -> str | None:
    if not spec.scheduler:
        return None
    s = spec.scheduler
    lr0 = spec.optimizer["lr"]
    if s["kind"] == "train.step_lr":
        return f"lambda epoch, lr: lr * {s['gamma']!r} ** epoch if epoch else lr"
    if s["kind"] == "train.cosine_annealing_lr":
        return ("lambda epoch, lr: "
                f"{s['eta_min']!r} + ({lr0!r} - {s['eta_min']!r}) * "
                f"(1 + np.cos(np.pi * epoch / {s['T_max']})) / 2")
    if s["kind"] == "train.multistep_lr":
        milestones = ", ".join(
            p.strip() for p in str(s.get("milestones", "")).split(",") if p.strip()
        )
        return (f"lambda epoch, lr: {lr0!r} * {s['gamma']!r} ** "
                f"sum(1 for m in ({milestones},) if epoch >= m)")
    if s["kind"] == "train.exponential_lr":
        return f"lambda epoch, lr: {lr0!r} * {s['gamma']!r} ** epoch"
    if s["kind"] == "train.linear_lr":
        start, iters = s["start_factor"], int(s["total_iters"])
        return (f"lambda epoch, lr: {lr0!r} * ({start!r} + (1 - {start!r}) * "
                f"min(epoch / {iters}, 1.0))")
    if s["kind"] == "train.plateau_lr":
        return None  # rendered as a ReduceLROnPlateau callback instead
    raise CodegenError(
        f"{s['kind'].split('.')[-1]} scheduler cannot be exported to Keras yet"
    )


def _keras_input_shape_str(ir_shape: list[int]) -> str:
    dims = ", ".join(str(d) for d in reversed(ir_shape))
    return f"({dims},)"


def _py_literal(value) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_py_literal(v) for v in value) + "]"
    return repr(value)


def _norm_expr(value):
    """Render mean/std as a python literal (scalar or per-feature list)."""
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(repr(float(v)) for v in value) + "]"
    return repr(float(value))


def _dataset_comment(spec: TrainingSpec) -> str:
    block = spec.dataset.get("block")
    if block == "data.torchvision":
        return f"Torchvision {spec.dataset.get('dataset', 'mnist')}"
    if block == "data.csv":
        return f"CSV {spec.dataset.get('path')}"
    return f"Synthetic {spec.dataset.get('kind', 'classification')} data"


def _base_ctx(graph: Graph, spec: TrainingSpec) -> dict:
    nodes, input_shape, (out_t, out_k) = emit_dag(graph)
    batch = spec.loader["batch_size"] or spec.trainer["batch_size"]
    ds_kind = spec.dataset.get("block")
    return {
        "spec": spec,
        "class_name": class_name_for(graph.name),
        "nodes": nodes,
        "modules": [n for n in nodes if n["torch_module"]],
        "output_var": out_t,
        "keras_output_var": out_k,
        "keras_input_shape": _keras_input_shape_str(input_shape),
        "input_shape": input_shape,
        "input_rank": len(input_shape),
        "input_shape_literal": tuple(input_shape),
        "keras_input_shape_literal": tuple(reversed(input_shape)),
        "loader_batch": batch,
        "is_regression": spec.is_regression,
        "needs_prf": any(m in spec.metrics for m in ("precision", "recall", "f1")),
        "needs_auc": "roc_auc" in spec.metrics,
        "needs_cm": "confusion" in spec.metrics,
        "dataset_is_torchvision": ds_kind == "data.torchvision",
        "dataset_is_csv": ds_kind == "data.csv",
        "dataset_is_synthetic": ds_kind == "data.synthetic",
        "dataset_is_numpy": ds_kind == "data.numpy",
        "dataset_is_json": ds_kind == "data.json",
        "dataset_is_image_folder": ds_kind == "data.image_folder",
        "dataset_is_huggingface": ds_kind == "data.huggingface",
        "grayscale_lit": "True" if spec.dataset.get("grayscale", False) else "False",
        "loss_is_focal": spec.loss["kind"] == "train.loss_focal",
        "scheduler_is_plateau": bool(spec.scheduler and spec.scheduler["kind"] == "train.plateau_lr"),
        "dataset_class": _TORCHVISION_CLASSES.get(spec.dataset.get("dataset", "mnist"), "MNIST"),
        "dataset_comment": _dataset_comment(spec),
        "synthetic_kind": spec.dataset.get("kind", "classification"),
        "permute_expr": (
            "np.random.default_rng(SEED).permutation(n)" if spec.split["shuffle"]
            else "np.arange(n)"
        ),
        "y_dtype": "np.float32" if spec.is_regression else "np.int64",
        "pin_memory_str": "True" if spec.loader["pin_memory"] else "False",
        "download_str": "True" if spec.dataset.get("download", True) else "False",
        "norm_mean_expr": _norm_expr(spec.normalize["mean"]) if spec.normalize else "0.0",
        "norm_std_expr": _norm_expr(spec.normalize["std"]) if spec.normalize else "1.0",
        "scheduler_expr": _render_scheduler(spec),
        "scheduler_steps_per_batch": bool(spec.scheduler and spec.scheduler["kind"] == "train.one_cycle_lr"),
        "scheduler_step_arg": "val_loss" if (spec.scheduler and spec.scheduler["kind"] == "train.plateau_lr") else "",
    }


def generate_training(graph: Graph, framework: str) -> str:
    if errors := [i for i in graph.validate() if i.severity == "error"]:
        raise CodegenError(
            "graph has validation errors:\n" + "\n".join(f"  - {e}" for e in errors)
        )
    spec = collect_spec(graph)
    ctx = _base_ctx(graph, spec)
    if framework == "pytorch":
        ctx["loss_expr"] = _render_loss(spec)
        ctx["optimizer_expr"] = _render_optimizer(spec)
        return _env.from_string(TRAIN_PYTORCH_TEMPLATE).render(**ctx)
    if framework == "keras":
        if spec.dataset.get("block") in ("data.torchvision", "data.image_folder",
                                         "data.huggingface"):
            raise CodegenError(
                f"{spec.dataset['block']} datasets cannot be exported to Keras yet"
            )
        ctx["keras_optimizer"] = _keras_optimizer(spec)
        ctx["keras_loss"] = _keras_loss(spec)
        ctx["keras_lr_scheduler"] = _keras_lr_scheduler(spec)
        metrics = [m for m in spec.metrics if m in ("accuracy", "mse", "mae")]
        ctx["keras_metrics"] = _py_literal(metrics)
        return _env.from_string(TRAIN_KERAS_TEMPLATE).render(**ctx)
    raise ValueError(f"unknown framework {framework!r}")
