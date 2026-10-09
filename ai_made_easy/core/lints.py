"""Architecture lints: designs that are valid but very likely wrong.

Runs after structural validation and shape inference succeeded. Every rule is
a pure function ``(LintContext) -> list[ValidationIssue]``; messages state
what is wrong, why it matters, and the usual fix.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from ai_made_easy.core.spec import parse_shape, shape_volume

if TYPE_CHECKING:  # pragma: no cover
    from ai_made_easy.core.graph import Graph, NodeInstance, ValidationIssue

CLASSIFICATION_LOSSES = {"train.loss_cross_entropy", "train.loss_focal", "train.loss_nll"}
LOGIT_LOSSES = {"train.loss_cross_entropy", "train.loss_focal", "train.loss_bce_logits"}
REGRESSION_LOSSES = {"train.loss_mse", "train.loss_l1", "train.loss_smooth_l1",
                     "train.loss_huber", "train.loss_poisson"}
SOFTMAX_LIKE = {"core.softmax", "core.log_softmax", "core.sigmoid", "core.softmin"}
BOUNDED_ACTIVATIONS = {"core.relu", "core.relu6", "core.sigmoid", "core.softmax",
                       "core.tanh", "core.hardsigmoid", "core.hardtanh", "core.softplus"}
ACTIVATIONS_PREFIX = "core."
DTYPE_PRESERVING = {"core.flatten", "core.reshape", "core.permute", "core.transpose",
                    "core.squeeze", "core.unsqueeze", "core.slice", "core.identity",
                    "core.repeat_vector", "core.select_timestep"}
DATASET_SHAPES = {"mnist": [1, 28, 28], "fashion_mnist": [1, 28, 28],
                  "cifar10": [3, 32, 32], "cifar100": [3, 32, 32], "stl10": [3, 96, 96],
                  "svhn": [3, 32, 32], "kmnist": [1, 28, 28], "emnist": [1, 28, 28]}
DATASET_CLASSES = {"mnist": 10, "fashion_mnist": 10, "cifar10": 10, "cifar100": 100,
                   "stl10": 10, "svhn": 10, "kmnist": 10, "emnist": 47}
DENSE_PARAM_WARNING = 50_000_000


@dataclass
class LintContext:
    graph: "Graph"
    chain: list["NodeInstance"]
    shapes: dict[str, list[int]]

    def consumers(self, node_id: str) -> list["NodeInstance"]:
        return [self.graph.nodes[e.target_id] for e in self.graph.outgoing(node_id)
                if e.target_id in self.graph.nodes]

    def producers(self, node_id: str) -> list["NodeInstance"]:
        return [self.graph.nodes[e.source_id] for e in self.graph.incoming(node_id)
                if e.source_id in self.graph.nodes]

    def nodes_of(self, *type_ids: str) -> list["NodeInstance"]:
        return [n for n in self.graph.nodes.values() if n.type_id in type_ids]

    def loss_type(self) -> str | None:
        from ai_made_easy.core.codegen.training_gen import _LOSSES

        losses = [n.type_id for n in self.graph.nodes.values() if n.type_id in _LOSSES]
        if losses:
            return losses[0]
        if self.nodes_of("train.trainer"):
            return "train.loss_cross_entropy"  # the trainer's default
        return None

    def output_node(self) -> "NodeInstance | None":
        return next((n for n in self.chain if n.type_id == "core.output"), None)

    def last_compute(self) -> "NodeInstance | None":
        out = self.output_node()
        if out is None:
            return None
        prods = self.producers(out.instance_id)
        return prods[0] if prods else None


def _issue(severity: str, message: str, node_id: str | None = None) -> "ValidationIssue":
    from ai_made_easy.core.graph import ValidationIssue

    return ValidationIssue(severity, message, node_id)


def _name(node: "NodeInstance") -> str:
    return node.definition().display_name


# --------------------------------------------------------------------- rules

def dense_on_spatial(ctx: LintContext):
    out = []
    for n in ctx.nodes_of("core.dense"):
        prods = ctx.producers(n.instance_id)
        if not prods or prods[0].instance_id not in ctx.shapes:
            continue
        s = ctx.shapes[prods[0].instance_id]
        if len(s) >= 3:
            out.append(_issue(
                "warning",
                f"Dense applies to the last axis only, but its input is {s}: the "
                f"output becomes {[*s[:-1], n.resolved_params()['units']]}. Insert a "
                "Flatten or a global pooling layer before it to use the whole sample.",
                n.instance_id))
    return out


def dtype_flow(ctx: LintContext):
    """Integer index tensors may only reach dtype-preserving ops or int consumers."""
    out = []
    int_nodes: set[str] = set()
    for n in ctx.chain:
        if n.type_id == "core.input":
            if str(n.resolved_params().get("dtype", "float32")).startswith("int"):
                int_nodes.add(n.instance_id)
            continue
        if n.type_id == "core.output":
            continue
        defn = n.definition()
        prods = ctx.producers(n.instance_id)
        any_int = any(p.instance_id in int_nodes for p in prods)
        if defn.input_dtype == "int":
            if not any_int:
                out.append(_issue(
                    "error",
                    f"{_name(n)} expects integer indices but receives floating-point "
                    "values. Set the Input dtype to int64 (token or category ids).",
                    n.instance_id))
            continue
        if any_int:
            if n.type_id in DTYPE_PRESERVING:
                int_nodes.add(n.instance_id)
            else:
                src = next(p for p in prods if p.instance_id in int_nodes)
                out.append(_issue(
                    "error",
                    f"{_name(n)} needs floating-point input but receives integer "
                    f"indices from {_name(src)}. Add an Embedding before it, or set "
                    "the Input dtype to float32.",
                    n.instance_id))
    return out


def softmax_before_logit_loss(ctx: LintContext):
    loss, last = ctx.loss_type(), ctx.last_compute()
    if loss not in LOGIT_LOSSES or last is None or last.type_id not in SOFTMAX_LIKE:
        return []
    loss_name = {"train.loss_bce_logits": "BCEWithLogitsLoss"}.get(loss, "CrossEntropyLoss")
    inner = "sigmoid" if loss == "train.loss_bce_logits" else "log-softmax"
    return [_issue(
        "warning",
        f"{loss_name} applies {inner} internally, so the final {_name(last)} squashes "
        "the logits twice and slows learning. Remove it and output raw logits "
        "(apply it only at inference time).",
        last.instance_id)]


def classification_output_size(ctx: LintContext):
    loss, last = ctx.loss_type(), ctx.last_compute()
    out_node = ctx.output_node()
    if loss not in CLASSIFICATION_LOSSES or out_node is None or last is None:
        return []
    shape = ctx.shapes.get(last.instance_id)
    if not shape:
        return []
    classes = shape[0]
    issues = []
    if classes < 2:
        issues.append(_issue(
            "error",
            f"CrossEntropyLoss needs at least 2 output units (one per class); the model "
            f"outputs {shape}. For binary targets use BCEWithLogits with 1 unit.",
            last.instance_id))
    for ds in ctx.nodes_of("data.torchvision", "data.synthetic"):
        p = ds.resolved_params()
        want = (DATASET_CLASSES.get(p.get("dataset")) if ds.type_id == "data.torchvision"
                else (int(p["n_classes"]) if p.get("kind") != "regression" else None))
        if want and classes != want:
            issues.append(_issue(
                "error",
                f"{_name(ds)} has {want} classes but the model outputs {classes} units. "
                f"Set the last layer to {want} outputs.",
                last.instance_id))
    return issues


def regression_output_activation(ctx: LintContext):
    loss, last = ctx.loss_type(), ctx.last_compute()
    if loss not in REGRESSION_LOSSES or last is None or last.type_id not in BOUNDED_ACTIVATIONS:
        return []
    return [_issue(
        "warning",
        f"Regression output passes through {_name(last)}, which restricts predictions "
        "to a bounded range. Use a linear output unless the target range matches.",
        last.instance_id)]


def consecutive_linear(ctx: LintContext):
    out = []
    for n in ctx.nodes_of("core.dense"):
        for c in ctx.consumers(n.instance_id):
            if c.type_id == "core.dense":
                out.append(_issue(
                    "warning",
                    "Two Dense layers with no activation between them compose into a "
                    "single linear map. Insert an activation (e.g. ReLU/GELU).",
                    c.instance_id))
    return out


def stacked_activations(ctx: LintContext):
    acts = {b.type_id for b in _registry().all() if b.category == "Activations"}
    out = []
    for n in ctx.chain:
        if n.type_id not in acts:
            continue
        for c in ctx.consumers(n.instance_id):
            if c.type_id == n.type_id and n.type_id in ("core.relu", "core.sigmoid",
                                                         "core.tanh", "core.softmax"):
                out.append(_issue("warning",
                                  f"{_name(c)} directly after {_name(n)} has no effect.",
                                  c.instance_id))
    return out


def dropout_before_output(ctx: LintContext):
    last = ctx.last_compute()
    if last is None or not last.type_id.startswith("core.dropout"):
        return []
    return [_issue(
        "warning",
        "Dropout directly before the Output perturbs the logits themselves. Move it "
        "before the final linear layer.",
        last.instance_id)]


def huge_dense(ctx: LintContext):
    out = []
    for n in ctx.nodes_of("core.dense"):
        prods = ctx.producers(n.instance_id)
        if not prods or prods[0].instance_id not in ctx.shapes:
            continue
        fan_in = ctx.shapes[prods[0].instance_id][-1]
        count = fan_in * int(n.resolved_params()["units"])
        if count > DENSE_PARAM_WARNING:
            out.append(_issue(
                "warning",
                f"This Dense layer has {count:,} weights ({fan_in:,} → "
                f"{n.resolved_params()['units']:,}). Reduce the input with pooling or a "
                "smaller layer to save memory.",
                n.instance_id))
    return out


def stride_skips_input(ctx: LintContext):
    out = []
    for n in ctx.chain:
        p = n.resolved_params()
        if "stride" in p and "kernel_size" in p and n.definition().category in (
                "Convolution", "Pooling"):
            try:
                # 1x1 strided projections are the standard downsampling shortcut
                if int(p["kernel_size"]) > 1 and int(p["stride"]) > int(p["kernel_size"]):
                    out.append(_issue(
                        "warning",
                        f"stride {p['stride']} is larger than kernel_size "
                        f"{p['kernel_size']}: some input positions are never read.",
                        n.instance_id))
            except (TypeError, ValueError):
                continue
    return out


def attention_without_positions(ctx: LintContext):
    tf = {"core.transformer_encoder", "core.multihead_attention", "core.transformer_decoder"}
    pos = {"core.positional_encoding", "core.learned_positional", "core.lstm", "core.gru",
           "core.rnn", "core.conv1d"}
    out = []
    for emb in ctx.nodes_of("core.embedding"):
        frontier, seen = [emb.instance_id], set()
        while frontier:
            nid = frontier.pop()
            for c in ctx.consumers(nid):
                if c.instance_id in seen or c.type_id in pos:
                    continue
                seen.add(c.instance_id)
                if c.type_id in tf:
                    out.append(_issue(
                        "warning",
                        "Self-attention is permutation-invariant and this sequence carries "
                        "no position information. Add a Positional Encoding after the "
                        "Embedding.",
                        c.instance_id))
                    frontier = []
                    break
                frontier.append(c.instance_id)
    return out


def batchnorm_tiny_batch(ctx: LintContext):
    bns = [n for n in ctx.chain if n.type_id.startswith("core.batch_norm")]
    if not bns:
        return []
    sizes = [int(n.resolved_params().get("batch_size", 32))
             for n in ctx.nodes_of("train.trainer", "prep.dataloader")]
    if sizes and min(sizes) < 2:
        return [_issue("error", "BatchNorm needs a batch size of at least 2 in training; "
                                "increase batch_size or use LayerNorm/GroupNorm.",
                       bns[0].instance_id)]
    if sizes and min(sizes) < 8:
        return [_issue("warning", f"BatchNorm statistics are noisy with batch size "
                                  f"{min(sizes)}; consider GroupNorm or a larger batch.",
                       bns[0].instance_id)]
    return []


def dataset_matches_input(ctx: LintContext):
    head = ctx.chain[0] if ctx.chain else None
    if head is None or head.type_id != "core.input":
        return []
    try:
        want = parse_shape(head.resolved_params()["shape"])
    except ValueError:
        return []
    resize = ctx.nodes_of("prep.resize")
    out = []
    for ds in ctx.nodes_of("data.synthetic"):
        n_feat = int(ds.resolved_params().get("n_features", 0))
        if n_feat and n_feat != shape_volume(want):
            out.append(_issue(
                "error",
                f"{_name(ds)} produces {n_feat} features per sample but the Input holds "
                f"{shape_volume(want)} values ({want}). Make them equal.",
                head.instance_id))
    for ds in ():  # torchvision shapes are checked by training_setup
        sample = DATASET_SHAPES.get(ds.resolved_params().get("dataset"))
        if not sample:
            continue
        expect = list(sample)
        if resize:
            size = int(resize[0].resolved_params().get("size", expect[1]))
            expect = [expect[0], size, size]
        if want != expect and shape_volume(want) != shape_volume(expect):
            out.append(_issue(
                "error",
                f"{ds.resolved_params()['dataset']} samples are {expect} but the Input "
                f"expects {want}. Set the Input shape to "
                f"'{', '.join(map(str, expect))}'.",
                head.instance_id))
        elif want != expect:
            out.append(_issue(
                "warning",
                f"{ds.resolved_params()['dataset']} samples are {expect}; the Input "
                f"{want} has the same size, so samples will be reshaped, not resized.",
                head.instance_id))
    return out


def loss_output_pairing(ctx: LintContext):
    """Losses that expect probabilities / log-probabilities need the matching output."""
    from ai_made_easy.core.training import catalog as cat

    loss = ctx.loss_type()
    last = ctx.last_compute()
    if loss is None or last is None or loss not in cat.COMPONENTS:
        return []
    want = cat.COMPONENTS[loss].meta.get("input")
    name = cat.COMPONENTS[loss].name
    if want == "log_probs" and last.type_id != "core.log_softmax":
        return [_issue("error", f"{name} expects log-probabilities; end the model with a "
                                "LogSoftmax layer", last.instance_id)]
    if want == "probs" and last.type_id != "core.sigmoid":
        return [_issue("error", f"{name} expects probabilities in [0, 1]; end the model with "
                                "a Sigmoid layer, or use BCE (logits) without it",
                       last.instance_id)]
    return []


def training_setup(ctx: LintContext):
    """Dataset, preprocessing and training blocks must form a consistent pipeline."""
    from ai_made_easy.core.codegen import CodegenError
    from ai_made_easy.core.training import data_catalog as dcat
    from ai_made_easy.core.training.spec import collect_spec, expected_input_shape

    if not any(n.type_id.startswith(("data.", "prep.", "train.", "eval."))
               for n in ctx.graph.nodes.values()):
        return []
    anchor = next((n.instance_id for n in ctx.graph.nodes.values()
                   if n.type_id.startswith("data.")), None)
    try:
        spec = collect_spec(ctx.graph)
    except CodegenError as exc:
        return [_issue("error", str(exc), anchor)]
    except Exception:  # noqa: BLE001 — structural problems are reported elsewhere
        return []
    out = [_issue("warning", w, anchor) for w in spec.warnings
           if not w.startswith("no dataset block")]
    head = ctx.chain[0]
    if spec.modality == "text" and spec.input_dtype != "int64":
        out.append(_issue("error", "text datasets produce token ids: set the Input dtype "
                                   "to int64 and start the model with an Embedding",
                          head.instance_id))
    if spec.modality != "text" and spec.input_dtype == "int64":
        out.append(_issue("error", f"{spec.modality} datasets produce floating-point "
                                   "features: set the Input dtype to float32",
                          head.instance_id))
    tok = spec.steps.get("prep.tokenize")
    if tok and tok["method"] != "huggingface":
        for emb in ctx.nodes_of("core.embedding"):
            if int(emb.resolved_params()["num_embeddings"]) < int(tok["vocab_size"]):
                out.append(_issue(
                    "error",
                    f"Embedding num_embeddings ({emb.resolved_params()['num_embeddings']}) "
                    f"is smaller than the tokenizer vocabulary ({tok['vocab_size']}); "
                    "token ids would be out of range", emb.instance_id))
    expected = expected_input_shape(spec)
    if expected is not None and list(expected) != list(spec.input_shape):
        name = dcat.BLOCKS[spec.dataset["block"]].name
        out.append(_issue(
            "error",
            f"{name} produces samples of shape {expected} but the Input expects "
            f"{spec.input_shape}. Set the Input shape to '{', '.join(map(str, expected))}'.",
            head.instance_id))
    if spec.dataset["block"] == "data.timeseries_csv":
        d = spec.dataset
        want = int(d["window"])
        dims = spec.input_shape
        axis = 1 if str(d["layout"]).startswith("channels") else 0
        if len(dims) != 2 or dims[axis] != want:
            layout = "[C, L]" if axis else "[L, C]"
            out.append(_issue(
                "error", f"time-series windows are {layout} with L = window = {want}; the "
                         f"Input shape {dims} does not match", head.instance_id))
        targets = [t for t in str(d["target_columns"]).split(",") if t.strip()]
        n_out = len(targets) * int(d["horizon"])
        last = ctx.last_compute()
        if last is not None and ctx.shapes.get(last.instance_id) not in ([n_out], None):
            out.append(_issue(
                "error", f"forecasting {len(targets)} target(s) × horizon {d['horizon']} needs "
                         f"{n_out} outputs; the model outputs {ctx.shapes[last.instance_id]}",
                last.instance_id))
    return out


IMAGENET_MEAN = (0.485, 0.456, 0.406)


def pretrained_needs_imagenet_norm(ctx: LintContext):
    out = []
    for n in ctx.nodes_of("core.pretrained_backbone"):
        if n.resolved_params()["weights"] != "imagenet":
            continue
        norm = ctx.nodes_of("prep.normalize")
        ok = False
        if norm:
            p = norm[0].resolved_params()
            try:
                mean = tuple(round(float(v), 3) for v in str(p["mean"]).split(",") if v.strip())
            except ValueError:
                mean = ()
            ok = p["mode"] == "fixed" and mean == IMAGENET_MEAN
        if not ok and any(d.type_id.startswith("data.") for d in ctx.graph.nodes.values()):
            out.append(_issue(
                "warning",
                "ImageNet weights expect inputs normalized with mean 0.485, 0.456, 0.406 and "
                "std 0.229, 0.224, 0.225: add Standardize in fixed mode with these values",
                n.instance_id))
    return out


def resource_budget(ctx: LintContext):
    """Designs that will not fit the project's device / latency / size budget."""
    from ai_made_easy.core import budget

    checks = budget.check(ctx.graph)
    if not checks:
        return []
    est = budget.estimate(ctx.graph)
    where = f" on {est.device.label}" if est.device else ""
    trainer = next(iter(ctx.nodes_of("train.trainer")), None)
    heaviest = max(est.layers, key=lambda layer: layer.flops, default=None)
    biggest = max(est.layers, key=lambda layer: layer.params, default=None)
    out = []
    for c in checks:
        if not c.over:
            continue
        if c.kind == "train_memory":
            out.append(_issue(
                "warning",
                f"Training needs about {c.used:.1f} GB at batch {est.batch_size} but the budget "
                f"is {c.limit:g} GB{where}: lower the batch size (gradient accumulation keeps "
                "the effective batch) or turn on mixed precision",
                trainer.instance_id if trainer else None))
        elif c.kind == "latency":
            out.append(_issue(
                "warning",
                f"Inference takes about {c.used:.1f} ms per sample{where}, over the "
                f"{c.limit:g} ms budget: use a smaller backbone, fewer or narrower layers, "
                "or a lower input resolution",
                heaviest.node_id if heaviest else None))
        elif c.kind == "params":
            out.append(_issue(
                "warning",
                f"The model has {c.used:.2f}M parameters, over the {c.limit:g}M budget: "
                "shrink the widest layers or pool before the dense head",
                biggest.node_id if biggest else None))
        elif c.kind == "model_size":
            out.append(_issue(
                "warning",
                f"The saved model is about {c.used:.1f} MB, over the {c.limit:g} MB budget: "
                "reduce parameters or quantize it when exporting",
                biggest.node_id if biggest else None))
    return out


def _registry():
    from ai_made_easy.core.registry import get_registry

    return get_registry()


RULES: list[Callable[[LintContext], list]] = [
    dtype_flow,
    dense_on_spatial,
    softmax_before_logit_loss,
    classification_output_size,
    regression_output_activation,
    consecutive_linear,
    stacked_activations,
    dropout_before_output,
    huge_dense,
    stride_skips_input,
    attention_without_positions,
    batchnorm_tiny_batch,
    dataset_matches_input,
    loss_output_pairing,
    training_setup,
    pretrained_needs_imagenet_norm,
    resource_budget,
]


# rules about the classification / regression data pipeline; tasks with their own
# training loop (detection, segmentation, ...) skip them and bring their own rules
PIPELINE_RULES = {classification_output_size, regression_output_activation,
                  dataset_matches_input, loss_output_pairing, training_setup,
                  pretrained_needs_imagenet_norm}


def register_rule(rule: Callable[[LintContext], list]) -> None:
    """Add a lint (task families register their design rules here)."""
    if rule not in RULES:
        RULES.append(rule)


def run_lints(graph: "Graph", chain: list["NodeInstance"],
              shapes: dict[str, list[int]]) -> list["ValidationIssue"]:
    from ai_made_easy.core.tasks import task_of

    ctx = LintContext(graph, chain, shapes)
    try:
        task = task_of(graph)
    except Exception:  # noqa: BLE001 — incomplete designs: no task
        task = None
    own_loop = task is not None and task.trainer_kind != "supervised"
    issues: list = []
    for rule in RULES:
        if own_loop and rule in PIPELINE_RULES:
            continue
        try:
            issues += rule(ctx)
        except Exception:  # noqa: BLE001 — a lint must never break validation
            continue
    return issues
