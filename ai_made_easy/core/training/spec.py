"""TrainingSpec: a normalized, framework-neutral view of the training setup.

``collect_spec`` folds every configuration block on the canvas (dataset,
preprocessing, loss, optimizer, scheduler, trainer, metrics) into one object
the script builders render from. Contradictions raise ``CodegenError``;
questionable-but-valid setups add ``warnings`` that are written into the
generated script header.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_made_easy.core.codegen import CodegenError, class_name_for, sanitize_identifier
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.tasks import get_task, resolve_task
from ai_made_easy.core.training import catalog as cat
from ai_made_easy.core.training import data_catalog as dcat


@dataclass
class TrainingSpec:
    name: str = "untitled"
    class_name: str = "Model"
    task: str = "multiclass"
    modality: str = "tabular"
    input_shape: list[int] = field(default_factory=list)
    input_dtype: str = "float32"
    output_shape: list[int] = field(default_factory=list)
    dataset: dict[str, Any] = field(default_factory=dict)     # {"block": id, **params}
    steps: dict[str, dict] = field(default_factory=dict)      # prep id -> params
    image_always: list[tuple[str, dict]] = field(default_factory=list)
    image_train: list[tuple[str, dict]] = field(default_factory=list)
    loss: dict[str, Any] = field(default_factory=dict)        # {"kind": id, **params}
    optimizer: dict[str, Any] = field(default_factory=dict)
    scheduler: dict[str, Any] | None = None
    trainer: dict[str, Any] = field(default_factory=dict)
    split: dict[str, Any] = field(default_factory=dict)
    loader: dict[str, Any] = field(default_factory=dict)
    metrics: list[tuple[str, dict]] = field(default_factory=list)
    kfold: dict[str, Any] | None = None
    predict: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)

    # ---------------------------------------------------------- derived
    @property
    def is_classification(self) -> bool:
        return get_task(self.task).classification

    @property
    def is_regression(self) -> bool:
        return self.task == "regression"

    @property
    def num_outputs(self) -> int:
        return self.output_shape[0] if self.output_shape else 1

    @property
    def metric_keys(self) -> list[str]:
        return [cat.COMPONENTS[m].meta["key"] for m, _ in self.metrics]

    @property
    def batch_size(self) -> int:
        return int(self.loader.get("batch_size") or 0) or int(self.trainer["batch_size"])

    @property
    def normalize(self) -> dict | None:
        """Back-compat view used by the UI: fixed per-channel stats or None."""
        p = self.steps.get("prep.normalize")
        if not p or p.get("mode") != "fixed":
            return None
        return {"mean": _floats(p["mean"]), "std": _floats(p["std"])}

    @property
    def output_units(self) -> int:
        return self.num_outputs


def _floats(value) -> list[float]:
    return [float(v) for v in str(value).split(",") if v.strip()]


def _resolved(node) -> dict:
    return dict(node.resolved_params())


def collect_spec(graph: Graph) -> TrainingSpec:
    """Fold config blocks into a TrainingSpec; raise on contradictions."""
    shapes = graph.infer_shapes()
    chain = graph.model_nodes()
    head = chain[0]
    out_node = chain[-1]
    spec = TrainingSpec(
        name=sanitize_identifier(graph.name),
        class_name=class_name_for(graph.name),
        input_shape=list(shapes[head.instance_id]),
        input_dtype=str(head.resolved_params().get("dtype", "float32")),
        output_shape=list(shapes[out_node.instance_id]),
    )

    def pick(ids: tuple[str, ...], label: str):
        found = [n for n in graph.nodes.values() if n.type_id in ids]
        if len(found) > 1:
            names = ", ".join(n.definition().display_name for n in found)
            raise CodegenError(f"expected at most one {label} block, found {len(found)}: {names}")
        return found[0] if found else None

    # ------------------------------------------------------------ dataset
    ds = pick(dcat.DATASET_IDS, "dataset")
    if ds is not None:
        spec.dataset = {"block": ds.type_id, **_resolved(ds)}
    # ------------------------------------------------------------ loss/task
    loss = pick(cat.LOSS_IDS, "loss")
    spec.loss = ({"kind": loss.type_id, **cat.resolved(loss.type_id, loss.params)}
                 if loss else {"kind": "train.loss_cross_entropy",
                               **cat.resolved("train.loss_cross_entropy", {})})
    meta = cat.COMPONENTS[spec.loss["kind"]].meta
    spec.task = resolve_task(meta["task"], spec.num_outputs).id
    if ds is None:
        spec.dataset = _default_dataset(spec)
        spec.warnings.append("no dataset block: training on synthetic data shaped like "
                             "the model's input and output")
    spec.modality = dcat.BLOCKS[spec.dataset["block"]].modality
    for key in ("root", "path"):  # training runs in a temp workdir
        raw = spec.dataset.get(key)
        if raw and not Path(str(raw)).expanduser().is_absolute():
            candidate = Path.cwd() / str(raw)
            if candidate.exists():
                spec.dataset[key] = str(candidate)

    # ------------------------------------------------------------ preprocessing
    order = {pid: i for i, pid in enumerate(dcat.PREP_IDS)}
    preps = sorted((n for n in graph.nodes.values() if n.type_id in order),
                   key=lambda n: order[n.type_id])
    seen: set[str] = set()
    for node in preps:
        blk = dcat.BLOCKS[node.type_id]
        if node.type_id in seen:
            raise CodegenError(f"{blk.name} appears more than once")
        seen.add(node.type_id)
        if spec.modality not in blk.applies_to:
            spec.warnings.append(f"{blk.name} does not apply to {spec.modality} data "
                                 "and is ignored")
            continue
        params = _resolved(node)
        if blk.stage == "always" and blk.torch is not None:
            spec.image_always.append((node.type_id, params))
        elif blk.stage == "train" and blk.torch is not None:
            spec.image_train.append((node.type_id, params))
        else:
            spec.steps[node.type_id] = params
    spec.split = spec.steps.pop("prep.split", None) or cat_defaults("prep.split")
    spec.loader = spec.steps.pop("prep.dataloader", None) or cat_defaults("prep.dataloader")
    if spec.modality == "timeseries" and spec.split.get("shuffle"):
        spec.split = {**spec.split, "shuffle": False}
    scalers = [s for s in ("prep.normalize", "prep.robust_scale", "prep.minmax")
               if s in spec.steps]
    if len(scalers) > 1:
        spec.warnings.append("several scaling blocks are applied in sequence: "
                             + " → ".join(dcat.BLOCKS[s].name for s in scalers))
    if "prep.mixup_cutmix" in spec.steps and spec.task != "multiclass":
        raise CodegenError("MixUp / CutMix needs a multi-class classification loss")
    if "prep.class_balance" in spec.steps and not spec.is_classification:
        raise CodegenError("Class Balancing applies to classification tasks only")

    # ------------------------------------------------------------ optimization
    opt = pick(cat.OPTIMIZER_IDS, "optimizer")
    spec.optimizer = ({"kind": opt.type_id, **cat.resolved(opt.type_id, opt.params)}
                      if opt else {"kind": "train.adam", **cat.resolved("train.adam", {})})
    sch = pick(cat.SCHEDULER_IDS, "scheduler")
    if sch is not None:
        spec.scheduler = {"kind": sch.type_id, **cat.resolved(sch.type_id, sch.params)}
    tr = pick(("train.trainer",), "Trainer")
    spec.trainer = cat.resolved("train.trainer", tr.params if tr else {})
    kf = pick(("train.kfold",), "K-Fold")
    if kf is not None:
        if spec.modality in ("image", "timeseries"):
            raise CodegenError("K-Fold cross-validation supports tabular, text, audio and "
                               "array datasets")
        spec.kfold = cat.resolved("train.kfold", kf.params)
    pr = pick(("eval.predict",), "Prediction Preview")
    if pr is not None:
        spec.predict = cat.resolved("eval.predict", pr.params)

    # ------------------------------------------------------------ metrics
    for node in graph.nodes.values():
        if node.type_id in cat.METRIC_IDS:
            comp = cat.COMPONENTS[node.type_id]
            if spec.task in comp.meta["tasks"]:
                spec.metrics.append((node.type_id, cat.resolved(node.type_id, node.params)))
            else:
                spec.warnings.append(f"{comp.name} does not apply to {spec.task} tasks "
                                     "and is skipped")
    spec.metrics.sort(key=lambda m: cat.METRIC_IDS.index(m[0]))
    if not spec.metrics:
        default = get_task(spec.task).default_metrics
        spec.metrics = [(m, cat.resolved(m, {})) for m in default]
    return spec


def cat_defaults(block_id: str) -> dict:
    return {p.name: p.default for p in dcat.BLOCKS[block_id].params}


def _default_dataset(spec: TrainingSpec) -> dict:
    from ai_made_easy.core.spec import shape_volume

    regression = spec.task in ("regression", "distribution")
    return {
        "block": "data.synthetic",
        "kind": "regression" if regression else "classification",
        "n_samples": 1000, "n_features": shape_volume(spec.input_shape),
        "n_classes": max(spec.num_outputs, 2), "noise": 0.3, "seed": 42,
    }


def dataset_comment(spec: TrainingSpec) -> str:
    d = spec.dataset
    block = d["block"]
    if block == "data.torchvision":
        return f"Torchvision {d['dataset']}"
    if block == "data.sklearn":
        return f"scikit-learn {d['dataset']}"
    if block in ("data.csv", "data.text_csv", "data.timeseries_csv", "data.numpy", "data.json"):
        return f"{dcat.BLOCKS[block].name}: {d.get('path')}"
    if block in ("data.image_folder", "data.text_folder", "data.audio_folder"):
        return f"{dcat.BLOCKS[block].name}: {d.get('root')}"
    if block == "data.huggingface":
        return f"Hugging Face {d['repo_id']} [{d['split']}]"
    return f"Synthetic {d.get('kind', 'classification')} data"


def expected_input_shape(spec: TrainingSpec) -> list[int] | None:
    """Sample shape the data pipeline produces, when known at design time."""
    d, block = spec.dataset, spec.dataset.get("block")
    if block == "data.torchvision":
        c, h, w = dcat.TORCHVISION[d["dataset"]][1]
        for tid, p in spec.image_always + spec.image_train:
            if tid == "prep.grayscale":
                c = int(p["channels"])
            elif tid == "prep.resize":
                h, w = int(p["height"]), int(p["width"])
            elif tid in ("prep.center_crop", "prep.random_crop", "prep.random_resized_crop"):
                h = w = int(p["size"])
        return [c, h, w]
    if block == "data.synthetic":
        return None  # reshaped to the Input automatically
    if block == "data.sklearn":
        return [dcat.SKLEARN[d["dataset"]][1]]
    if block in ("data.text_csv", "data.text_folder"):
        tok = spec.steps.get("prep.tokenize")
        return [int(tok["max_length"])] if tok else None
    if block == "data.audio_folder":
        return audio_feature_shape(d, spec.steps.get("prep.audio_features"))
    if block == "data.timeseries_csv":
        return None  # feature count depends on the file
    return None


def audio_feature_shape(dataset: dict, feat: dict | None) -> list[int]:
    samples = int(round(float(dataset["duration"]) * int(dataset["sample_rate"])))
    if not feat or feat["kind"] == "waveform":
        return [1, samples]
    frames = samples // int(feat["hop_length"]) + 1
    if feat["kind"] == "spectrogram":
        return [1, int(feat["n_fft"]) // 2 + 1, frames]
    if feat["kind"] == "mel_spectrogram":
        return [1, int(feat["n_mels"]), frames]
    return [1, int(feat["n_mfcc"]), frames]
