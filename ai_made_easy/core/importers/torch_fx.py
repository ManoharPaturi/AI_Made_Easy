"""PyTorch → block graph via ``torch.fx`` (runs where torch is installed).

Traces the module, propagates shapes from an example input, and maps every
operation to a block. Dimensions are converted from batch-including PyTorch
dims to the designer's per-sample dims. Anything without a faithful block
equivalent is reported instead of guessed.

Run as a script: ``python -m ai_made_easy.core.importers.torch_fx REQUEST.json``
(prints ``IMPORT-RESULT {json}``), so the desktop app never needs torch in-process.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import operator
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import fx
from torch.fx.passes.shape_prop import ShapeProp


class Unsupported(Exception):
    pass


# ----------------------------------------------------------------- helpers

def _uniform(value, what: str) -> int:
    """(3, 3) -> 3; (3, 5) -> Unsupported."""
    if isinstance(value, (tuple, list)):
        if len(set(value)) != 1:
            raise Unsupported(f"non-square {what} {tuple(value)}")
        value = value[0]
    if value is None:
        raise Unsupported(f"{what} None")
    return int(value)


def _sample_dim(dim: int, rank: int) -> int:
    """Batch-including dim -> per-sample dim (negative dims stay negative)."""
    if dim < 0:
        return dim
    if dim == 0:
        raise Unsupported("operation over the batch dimension")
    return dim - 1


def _meta(node: fx.Node):
    return node.meta.get("tensor_meta")


def _shape(node) -> list[int] | None:  # noqa: ANN001
    meta = _meta(node) if isinstance(node, fx.Node) else None
    shape = getattr(meta, "shape", None)  # TensorMetadata; tuples of them for multi-outputs
    return list(shape) if shape is not None else None


def _is_tensor(arg) -> bool:  # noqa: ANN001
    return isinstance(arg, fx.Node) and _shape(arg) is not None


# ------------------------------------------------------------ module table

def _conv(m: nn.Module, kind: str) -> tuple[str, dict]:
    if m.padding_mode != "zeros":
        raise Unsupported(f"padding_mode={m.padding_mode!r}")
    k = _uniform(m.kernel_size, "kernel_size")
    padding = m.padding
    if isinstance(padding, str):
        if padding == "valid":
            padding = 0
        elif padding == "same" and _uniform(m.stride, "stride") == 1:
            padding = (k - 1) * _uniform(m.dilation, "dilation") // 2
        else:
            raise Unsupported(f"padding={padding!r}")
    params = {"out_channels": m.out_channels, "kernel_size": k,
              "stride": _uniform(m.stride, "stride"),
              "padding": _uniform(padding, "padding"),
              "dilation": _uniform(m.dilation, "dilation"), "groups": m.groups,
              "bias": m.bias is not None}
    if kind.startswith("conv_transpose"):
        params["output_padding"] = _uniform(m.output_padding, "output_padding")
    return f"core.{kind}", params


def _pool(m: nn.Module, kind: str) -> tuple[str, dict]:
    if getattr(m, "ceil_mode", False):
        raise Unsupported("ceil_mode=True")
    if _uniform(getattr(m, "dilation", 1), "dilation") != 1:
        raise Unsupported("dilated pooling")
    k = _uniform(m.kernel_size, "kernel_size")
    stride = m.stride if m.stride not in (None, (), []) else k
    return f"core.{kind}", {"kernel_size": k, "stride": _uniform(stride, "stride"),
                            "padding": _uniform(m.padding, "padding")}


def _norm_bn(m: nn.Module, kind: str) -> tuple[str, dict]:
    return f"core.{kind}", {"epsilon": float(m.eps),
                            "momentum": float(m.momentum if m.momentum is not None else 0.1),
                            "affine": bool(m.affine)}


def _recurrent(m: nn.RNNBase, kind: str) -> tuple[str, dict]:
    if not m.batch_first:
        raise Unsupported(f"{type(m).__name__} with batch_first=False")
    if getattr(m, "proj_size", 0):
        raise Unsupported("LSTM proj_size")
    params = {"hidden_size": m.hidden_size, "num_layers": m.num_layers, "bias": m.bias,
              "bidirectional": m.bidirectional, "dropout": float(m.dropout),
              "return_sequences": True}
    if kind == "rnn":
        params["nonlinearity"] = m.nonlinearity
    return f"core.{kind}", params


def _act(type_id: str, **params) -> Any:
    return lambda m: (type_id, {k: (v(m) if callable(v) else v) for k, v in params.items()})


MODULES: dict[type, Any] = {
    nn.Linear: lambda m: ("core.dense", {"units": m.out_features, "bias": m.bias is not None}),
    nn.Conv1d: lambda m: _conv(m, "conv1d"),
    nn.Conv2d: lambda m: _conv(m, "conv2d"),
    nn.Conv3d: lambda m: _conv(m, "conv3d"),
    nn.ConvTranspose1d: lambda m: _conv(m, "conv_transpose1d"),
    nn.ConvTranspose2d: lambda m: _conv(m, "conv_transpose2d"),
    nn.ConvTranspose3d: lambda m: _conv(m, "conv_transpose3d"),
    nn.MaxPool1d: lambda m: _pool(m, "maxpool1d"),
    nn.MaxPool2d: lambda m: _pool(m, "maxpool2d"),
    nn.MaxPool3d: lambda m: _pool(m, "maxpool3d"),
    nn.AvgPool1d: lambda m: _pool(m, "avgpool1d"),
    nn.AvgPool2d: lambda m: _pool(m, "avgpool2d"),
    nn.AvgPool3d: lambda m: _pool(m, "avgpool3d"),
    nn.BatchNorm1d: lambda m: _norm_bn(m, "batch_norm1d"),
    nn.BatchNorm2d: lambda m: _norm_bn(m, "batch_norm2d"),
    nn.BatchNorm3d: lambda m: _norm_bn(m, "batch_norm3d"),
    nn.GroupNorm: lambda m: ("core.group_norm", {"num_groups": m.num_groups,
                                                 "epsilon": float(m.eps),
                                                 "affine": bool(m.affine)}),
    nn.InstanceNorm1d: lambda m: ("core.instance_norm1d", {"epsilon": float(m.eps),
                                                           "affine": bool(m.affine)}),
    nn.InstanceNorm2d: lambda m: ("core.instance_norm2d", {"epsilon": float(m.eps),
                                                           "affine": bool(m.affine)}),
    nn.InstanceNorm3d: lambda m: ("core.instance_norm3d", {"epsilon": float(m.eps),
                                                           "affine": bool(m.affine)}),
    nn.LocalResponseNorm: lambda m: ("core.local_response_norm", {
        "size": m.size, "alpha": m.alpha, "beta": m.beta, "k": m.k}),
    nn.Dropout: _act("core.dropout", p=lambda m: float(m.p)),
    nn.Dropout1d: _act("core.dropout1d", p=lambda m: float(m.p)),
    nn.Dropout2d: _act("core.dropout2d", p=lambda m: float(m.p)),
    nn.Dropout3d: _act("core.dropout3d", p=lambda m: float(m.p)),
    nn.AlphaDropout: _act("core.alpha_dropout", p=lambda m: float(m.p)),
    nn.ReLU: _act("core.relu"), nn.ReLU6: _act("core.relu6"),
    nn.LeakyReLU: _act("core.leaky_relu", negative_slope=lambda m: float(m.negative_slope)),
    nn.GELU: _act("core.gelu", approximate=lambda m: m.approximate),
    nn.SiLU: _act("core.silu"), nn.Sigmoid: _act("core.sigmoid"), nn.Tanh: _act("core.tanh"),
    nn.ELU: _act("core.elu", alpha=lambda m: float(m.alpha)),
    nn.CELU: _act("core.celu", alpha=lambda m: float(m.alpha)),
    nn.SELU: _act("core.selu"), nn.Mish: _act("core.mish"),
    nn.Hardswish: _act("core.hardswish"), nn.Hardsigmoid: _act("core.hardsigmoid"),
    nn.Softsign: _act("core.softsign"), nn.Tanhshrink: _act("core.tanhshrink"),
    nn.LogSigmoid: _act("core.logsigmoid"), nn.Identity: _act("core.identity"),
    nn.Softplus: _act("core.softplus", beta=lambda m: float(m.beta)),
    nn.Softshrink: _act("core.softshrink", lambd=lambda m: float(m.lambd)),
    nn.Hardshrink: _act("core.hardshrink", lambd=lambda m: float(m.lambd)),
    nn.Threshold: _act("core.threshold", threshold=lambda m: float(m.threshold),
                       value=lambda m: float(m.value)),
    nn.RReLU: _act("core.rrelu", lower=lambda m: float(m.lower), upper=lambda m: float(m.upper)),
    nn.PixelShuffle: _act("core.pixel_shuffle", upscale_factor=lambda m: m.upscale_factor),
    nn.PixelUnshuffle: _act("core.pixel_unshuffle",
                            downscale_factor=lambda m: m.downscale_factor),
    nn.LSTM: lambda m: _recurrent(m, "lstm"),
    nn.GRU: lambda m: _recurrent(m, "gru"),
    nn.RNN: lambda m: _recurrent(m, "rnn"),
}
_ADAPTIVE = {nn.AdaptiveAvgPool1d: "adaptive_avgpool1d", nn.AdaptiveAvgPool2d: "adaptive_avgpool2d",
             nn.AdaptiveAvgPool3d: "adaptive_avgpool3d", nn.AdaptiveMaxPool1d: "adaptive_maxpool1d",
             nn.AdaptiveMaxPool2d: "adaptive_maxpool2d", nn.AdaptiveMaxPool3d: "adaptive_maxpool3d"}
_PADS = {nn.ZeroPad1d: "zero_pad1d", nn.ZeroPad2d: "zero_pad2d", nn.ZeroPad3d: "zero_pad3d",
         nn.ReflectionPad1d: "reflection_pad1d", nn.ReflectionPad2d: "reflection_pad2d",
         nn.ReflectionPad3d: "reflection_pad3d", nn.ReplicationPad1d: "replication_pad1d",
         nn.ReplicationPad2d: "replication_pad2d", nn.ReplicationPad3d: "replication_pad3d",
         nn.CircularPad1d: "circular_pad1d", nn.CircularPad2d: "circular_pad2d",
         nn.CircularPad3d: "circular_pad3d"}
_DIM_ACTS = {nn.Softmax: "core.softmax", nn.LogSoftmax: "core.log_softmax",
             nn.Softmin: "core.softmin", nn.GLU: "core.glu"}

# functions applied to one tensor with no parameters
_UNARY_FUNCS = {
    torch.relu: "core.relu", F.relu: "core.relu", F.relu6: "core.relu6",
    torch.sigmoid: "core.sigmoid", F.sigmoid: "core.sigmoid", torch.tanh: "core.tanh",
    F.tanh: "core.tanh", F.silu: "core.silu", F.mish: "core.mish", F.selu: "core.selu",
    F.hardswish: "core.hardswish", F.hardsigmoid: "core.hardsigmoid", F.softsign: "core.softsign",
    F.logsigmoid: "core.logsigmoid", F.tanhshrink: "core.tanhshrink",
}
_UNARY_METHODS = {"relu": "core.relu", "sigmoid": "core.sigmoid", "tanh": "core.tanh"}
_MATH = {torch.abs: "abs", torch.exp: "exp", torch.log: "log", torch.log1p: "log1p",
         torch.sqrt: "sqrt", torch.square: "square", torch.neg: "negative",
         torch.reciprocal: "reciprocal", torch.sin: "sin", torch.cos: "cos", torch.sign: "sign"}
_MATH_METHODS = {"abs": "abs", "exp": "exp", "log": "log", "sqrt": "sqrt", "neg": "negative"}
_PASSTHROUGH_METHODS = {"contiguous", "float", "clone", "detach"}
_REDUCE = {torch.mean: "mean", torch.sum: "sum", torch.amax: "max", torch.amin: "min",
           torch.prod: "prod", torch.std: "std", torch.var: "var"}
_REDUCE_METHODS = {"mean": "mean", "sum": "sum", "amax": "max", "amin": "min", "prod": "prod",
                   "std": "std", "var": "var"}


# ---------------------------------------------------------------- importer

class Importer:
    def __init__(self, model: nn.Module, example: torch.Tensor):
        self.model = model.eval()
        self.example = example
        self.nodes: list[dict] = []
        self.edges: list[tuple[str, str, str]] = []  # (source_id, target_id, target_port)
        self.value_of: dict[str, Any] = {}  # fx node name -> block id | ("seq", id) …
        self.unsupported: list[str] = []
        self.warnings: list[str] = []
        self._ids: dict[str, int] = {}
        self._rnn_use: dict[str, str] = {}  # recurrent block id -> "seq" | "final"
        self.origin: dict[str, str] = {}  # block id -> source module path (for weights)
        self._modules = dict(model.named_modules())

    # ---------------------------------------------------------------- utils
    def _new_id(self, base: str) -> str:
        base = base.replace(".", "_") or "block"
        n = self._ids.get(base, 0)
        self._ids[base] = n + 1
        return base if n == 0 else f"{base}_{n}"

    def _add(self, base: str, type_id: str, params: dict, inputs: list) -> str:
        bid = self._new_id(base)
        self.nodes.append({"id": bid, "type": type_id, "params": params})
        from ai_made_easy.core.registry import get_registry

        ports = [p.name for p in get_registry().get(type_id).inputs]
        if len(ports) != len(inputs):
            raise Unsupported(f"{type_id} takes {len(ports)} input(s), got {len(inputs)}")
        for port, src in zip(ports, inputs, strict=True):
            self.edges.append((self._source(src), bid, port))
        return bid

    def _source(self, arg) -> str:  # noqa: ANN001
        if isinstance(arg, _Ref):
            return arg.bid
        value = self.value_of.get(arg.name) if isinstance(arg, fx.Node) else None
        if isinstance(value, str):
            return value
        if isinstance(value, tuple) and value[0] in ("seq", "final"):
            self._use_rnn(value[1], value[0])
            return value[1]
        if isinstance(value, tuple) and value[0] == "unusable":
            raise Unsupported(value[1])
        raise Unsupported(f"cannot use {getattr(arg, 'name', arg)!r} as a tensor input")

    def _use_rnn(self, bid: str, mode: str) -> None:
        """A recurrent block feeds either its sequence or its final state, not both."""
        if self._rnn_use.setdefault(bid, mode) != mode:
            raise Unsupported("both the output sequence and the final hidden state are used")
        block = next(n for n in self.nodes if n["id"] == bid)
        block["params"]["return_sequences"] = mode == "seq"

    def _fail(self, node: fx.Node, why: str) -> None:
        target = node.target if isinstance(node.target, str) else getattr(
            node.target, "__name__", str(node.target))
        self.unsupported.append(f"{node.name} ({node.op} {target}): {why}")
        self.value_of[node.name] = None

    # ----------------------------------------------------------------- main
    def run(self) -> dict:
        try:
            traced = fx.symbolic_trace(self.model)
        except Exception as exc:  # noqa: BLE001 — Python control flow etc.
            raise Unsupported(
                f"the model could not be traced ({type(exc).__name__}: {exc}). Models with "
                "data-dependent Python control flow cannot be imported.") from exc
        ShapeProp(traced).propagate(self.example)
        self._modules = dict(traced.named_modules())
        out_node = None
        for node in traced.graph.nodes:
            try:
                if node.op == "placeholder":
                    self._placeholder(node)
                elif node.op == "call_module":
                    self._call_module(node)
                elif node.op == "call_function":
                    self._call_function(node)
                elif node.op == "call_method":
                    self._call_method(node)
                elif node.op == "get_attr":
                    self._fail(node, "direct parameter access in forward()")
                elif node.op == "output":
                    out_node = node
            except Unsupported as exc:
                if node.users:
                    self._fail(node, str(exc))
                else:  # an unused value (e.g. the LSTM cell state in `_, (h, _) = ...`)
                    self.value_of[node.name] = ("unusable", str(exc))
        if out_node is not None:
            result = out_node.args[0]
            if isinstance(result, (tuple, list, dict)):
                self.unsupported.append("forward() returns several outputs")
            elif self.value_of.get(getattr(result, "name", "")) is not None:
                oid = self._new_id("output")
                self.nodes.append({"id": oid, "type": "core.output", "params": {}})
                self.edges.append((self._source(result), oid, "in"))
        return self._graph()

    def _graph(self) -> dict:
        used = {s for s, _, _ in self.edges} | {t for _, t, _ in self.edges}
        nodes = [n for n in self.nodes if n["id"] in used or n["type"] == "core.input"]
        for i, n in enumerate(nodes):
            n["position"] = [i * 240.0, 0.0]
        return {"name": type(self.model).__name__, "nodes": nodes,
                "edges": [{"from": f"{s}/out", "to": f"{t}/{p}"} for s, t, p in self.edges],
                "meta": {"imported_from": "pytorch"}}

    # ------------------------------------------------------------ node kinds
    def _placeholder(self, node: fx.Node) -> None:
        if any(n["type"] == "core.input" for n in self.nodes):
            raise Unsupported("models with several inputs")
        shape = _shape(node)
        dtype = "int64" if not self.example.is_floating_point() else "float32"
        bid = self._new_id("input")
        self.nodes.append({"id": bid, "type": "core.input",
                           "params": {"shape": ", ".join(str(d) for d in shape[1:]),
                                      "dtype": dtype}})
        self.value_of[node.name] = bid

    def _single_input(self, node: fx.Node) -> fx.Node:
        tensors = [a for a in node.args if isinstance(a, fx.Node)]
        if len(tensors) != 1 or not _is_tensor(tensors[0]):
            raise Unsupported("expected one tensor input")
        return tensors[0]

    def _call_module(self, node: fx.Node) -> None:
        before = len(self.nodes)
        self._call_module_inner(node)
        added = self.nodes[before:]
        if len(added) == 1:
            self.origin[added[0]["id"]] = str(node.target)

    def _call_module_inner(self, node: fx.Node) -> None:
        module = self._modules[node.target]
        name = str(node.target)
        cls = type(module)
        if cls in MODULES:
            type_id, params = MODULES[cls](module)
            src = self._single_input(node)
            bid = self._add(name, type_id, params, [src])
            if cls in (nn.LSTM, nn.GRU, nn.RNN):
                self.value_of[node.name] = ("rnn", bid, module)
            else:
                self.value_of[node.name] = bid
            return
        if cls in _ADAPTIVE:
            size = _uniform(module.output_size, "output_size")
            self.value_of[node.name] = self._add(name, f"core.{_ADAPTIVE[cls]}",
                                                 {"output_size": size},
                                                 [self._single_input(node)])
            return
        if cls in _PADS:
            pad = _uniform(module.padding, "padding")
            self.value_of[node.name] = self._add(name, f"core.{_PADS[cls]}", {"padding": pad},
                                                 [self._single_input(node)])
            return
        if cls in _DIM_ACTS:
            src = self._single_input(node)
            dim = module.dim if module.dim is not None else -1
            dim = _sample_dim(dim, len(_shape(src)))
            self.value_of[node.name] = self._add(name, _DIM_ACTS[cls], {"dim": dim}, [src])
            return
        if cls is nn.Flatten:
            self._flatten(node, self._single_input(node), module.start_dim, module.end_dim, name)
            return
        if cls is nn.Embedding:
            if module.max_norm is not None:
                raise Unsupported("Embedding max_norm")
            self.value_of[node.name] = self._add(name, "core.embedding", {
                "num_embeddings": module.num_embeddings, "embedding_dim": module.embedding_dim,
                "padding_idx": -1 if module.padding_idx is None else module.padding_idx},
                [self._single_input(node)])
            return
        if cls is nn.LayerNorm:
            src = self._single_input(node)
            shape = _shape(src)[1:]
            normalized = list(module.normalized_shape)
            if normalized == shape[-len(normalized):] and len(normalized) == 1:
                dims = "last"
            elif normalized == shape:
                dims = "all"
            else:
                raise Unsupported(f"LayerNorm over {normalized} of {shape}")
            self.value_of[node.name] = self._add(name, "core.layer_norm", {
                "normalized_dims": dims, "epsilon": float(module.eps),
                "affine": bool(module.elementwise_affine)}, [src])
            return
        if cls is nn.RMSNorm:
            if len(module.normalized_shape) != 1:
                raise Unsupported("RMSNorm over several dims")
            self.value_of[node.name] = self._add(name, "core.rms_norm", {
                "epsilon": float(module.eps if module.eps is not None else 1e-6)},
                [self._single_input(node)])
            return
        if cls is nn.PReLU:
            if module.num_parameters != 1:
                raise Unsupported("PReLU with one slope per channel")
            self.value_of[node.name] = self._add(name, "core.prelu", {},
                                                 [self._single_input(node)])
            return
        if cls is nn.Hardtanh:
            if (module.min_val, module.max_val) != (-1.0, 1.0):
                raise Unsupported("Hardtanh with custom bounds")
            self.value_of[node.name] = self._add(name, "core.hardtanh", {},
                                                 [self._single_input(node)])
            return
        if cls in (nn.Upsample,):
            src = self._single_input(node)
            rank = len(_shape(src)) - 2
            if module.scale_factor is None:
                raise Unsupported("Upsample with an output size")
            mode = {"linear": "bilinear", "trilinear": "bilinear"}.get(module.mode, module.mode)
            if mode not in ("nearest", "bilinear", "bicubic"):
                raise Unsupported(f"Upsample mode {module.mode!r}")
            self.value_of[node.name] = self._add(name, f"core.upsample{rank}d", {
                "scale_factor": _uniform(module.scale_factor, "scale_factor"), "mode": mode},
                [src])
            return
        if cls is nn.MultiheadAttention:
            self._attention(node, module, name)
            return
        if cls is nn.TransformerEncoderLayer:
            self.value_of[node.name] = self._encoder_layer(module, self._single_input(node), name)
            return
        if cls is nn.TransformerEncoder:
            src = self._single_input(node)
            current: Any = src
            for i, layer in enumerate(module.layers):
                bid = self._encoder_layer(layer, current, f"{name}_{i}")
                self.origin[bid] = f"{node.target}.layers.{i}"
                self.value_of[f"__{node.name}_{i}"] = bid
                current = _Ref(bid)
            if module.norm is not None:
                if not isinstance(module.norm, nn.LayerNorm):
                    raise Unsupported("TransformerEncoder with a non-LayerNorm final norm")
                bid = self._add(f"{name}_norm", "core.layer_norm", {
                    "normalized_dims": "last", "epsilon": float(module.norm.eps),
                    "affine": bool(module.norm.elementwise_affine)}, [current])
                self.origin[bid] = f"{node.target}.norm"
            self.value_of[node.name] = bid
            return
        if cls is nn.Bilinear:
            a, b = [x for x in node.args if isinstance(x, fx.Node)]
            self.value_of[node.name] = self._add(name, "core.bilinear", {
                "units": module.out_features, "bias": module.bias is not None}, [a, b])
            return
        raise Unsupported(f"no block for {cls.__module__}.{cls.__name__}")

    def _encoder_layer(self, layer: nn.Module, src, name: str) -> str:  # noqa: ANN001
        if not getattr(layer.self_attn, "batch_first", False):
            raise Unsupported("TransformerEncoderLayer with batch_first=False")
        act = layer.activation
        act_name = ("gelu" if act is F.gelu or isinstance(act, nn.GELU) else
                    "relu" if act is F.relu or isinstance(act, nn.ReLU) else None)
        if act_name is None:
            raise Unsupported("TransformerEncoderLayer activation")
        return self._add(name, "core.transformer_encoder", {
            "nhead": layer.self_attn.num_heads, "dim_feedforward": layer.linear1.out_features,
            "dropout": float(layer.dropout.p), "activation": act_name,
            "norm_first": bool(layer.norm_first)}, [src])

    def _attention(self, node: fx.Node, module: nn.MultiheadAttention, name: str) -> None:
        if not module.batch_first:
            raise Unsupported("MultiheadAttention with batch_first=False")
        q, k, v = (list(node.args) + [node.kwargs.get(x) for x in ("key", "value")])[:3]
        if k is None:
            k = v = q
        if q is k and k is v:
            bid = self._add(name, "core.multihead_attention",
                            {"embed_dim": 0, "num_heads": module.num_heads,
                             "dropout": float(module.dropout)}, [q])
        elif k is v:
            bid = self._add(name, "core.cross_attention",
                            {"num_heads": module.num_heads, "dropout": float(module.dropout)},
                            [q, k])
        else:
            raise Unsupported("attention with different key and value tensors")
        self.value_of[node.name] = ("attn", bid)

    # --------------------------------------------------------- functions
    def _scalar_arith(self, node: fx.Node, op) -> bool:  # noqa: ANN001
        a, b = node.args[:2]
        tensor, scalar = (a, b) if _is_tensor(a) else (b, a)
        if not _is_tensor(tensor) or not isinstance(scalar, (int, float)):
            return False
        if op in (operator.add, torch.add, operator.iadd):
            params = {"scale": 1.0, "shift": float(scalar)}
        elif op in (operator.sub, torch.sub, operator.isub) and tensor is a:
            params = {"scale": 1.0, "shift": -float(scalar)}
        elif op in (operator.mul, torch.mul, operator.imul):
            params = {"scale": float(scalar), "shift": 0.0}
        elif op in (operator.truediv, torch.div) and tensor is a:
            params = {"scale": 1.0 / float(scalar), "shift": 0.0}
        else:
            return False
        self.value_of[node.name] = self._add(node.name, "core.scale", params, [tensor])
        return True

    def _call_function(self, node: fx.Node) -> None:
        fn = node.target
        args = node.args
        kwargs = node.kwargs
        binary = {operator.add: "core.add", torch.add: "core.add", operator.iadd: "core.add",
                  operator.sub: "core.subtract", torch.sub: "core.subtract",
                  operator.mul: "core.multiply", torch.mul: "core.multiply",
                  operator.imul: "core.multiply", torch.maximum: "core.maximum",
                  torch.minimum: "core.minimum"}
        if fn in binary or fn in (operator.truediv, torch.div):
            if self._scalar_arith(node, fn):
                return
            if fn in binary and all(_is_tensor(a) for a in args[:2]):
                if _shape(args[0]) != _shape(args[1]):
                    raise Unsupported("broadcasting between different shapes")
                self.value_of[node.name] = self._add(node.name, binary[fn], {}, list(args[:2]))
                return
            raise Unsupported("arithmetic that has no block")
        if fn is operator.getitem:
            self._getitem(node)
            return
        if fn in _UNARY_FUNCS:
            self.value_of[node.name] = self._add(node.name, _UNARY_FUNCS[fn], {}, [args[0]])
            return
        if fn in _MATH:
            self.value_of[node.name] = self._add(node.name, "core.math", {"op": _MATH[fn]},
                                                 [args[0]])
            return
        if fn is F.leaky_relu:
            slope = args[1] if len(args) > 1 else kwargs.get("negative_slope", 0.01)
            self.value_of[node.name] = self._add(node.name, "core.leaky_relu",
                                                 {"negative_slope": float(slope)}, [args[0]])
            return
        if fn is F.gelu:
            self.value_of[node.name] = self._add(node.name, "core.gelu", {
                "approximate": kwargs.get("approximate", "none")}, [args[0]])
            return
        if fn is F.elu:
            self.value_of[node.name] = self._add(node.name, "core.elu", {
                "alpha": float(args[1] if len(args) > 1 else kwargs.get("alpha", 1.0))},
                [args[0]])
            return
        if fn in (F.softmax, torch.softmax, F.log_softmax, torch.log_softmax):
            dim = args[1] if len(args) > 1 else kwargs.get("dim", -1)
            type_id = "core.softmax" if fn in (F.softmax, torch.softmax) else "core.log_softmax"
            self.value_of[node.name] = self._add(node.name, type_id, {
                "dim": _sample_dim(dim, len(_shape(args[0])))}, [args[0]])
            return
        if fn is F.dropout:
            p = args[1] if len(args) > 1 else kwargs.get("p", 0.5)
            self.value_of[node.name] = self._add(node.name, "core.dropout", {"p": float(p)},
                                                 [args[0]])
            return
        if fn in (torch.flatten,):
            start = args[1] if len(args) > 1 else kwargs.get("start_dim", 0)
            end = args[2] if len(args) > 2 else kwargs.get("end_dim", -1)
            self._flatten(node, args[0], start, end, node.name)
            return
        if fn in (torch.cat, torch.concat):
            tensors = list(args[0])
            dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
            self._concat(node, tensors, dim)
            return
        if fn in _REDUCE:
            self._reduce(node, args[0], _REDUCE[fn], args[1] if len(args) > 1 else
                         kwargs.get("dim"), kwargs.get("keepdim", False))
            return
        if fn in (torch.clamp, torch.clip):
            lo = args[1] if len(args) > 1 else kwargs.get("min")
            hi = args[2] if len(args) > 2 else kwargs.get("max")
            if lo is None or hi is None:
                raise Unsupported("one-sided clamp")
            self.value_of[node.name] = self._add(node.name, "core.clamp",
                                                 {"min": float(lo), "max": float(hi)}, [args[0]])
            return
        if fn is F.normalize:
            dim = kwargs.get("dim", args[2] if len(args) > 2 else 1)
            self.value_of[node.name] = self._add(node.name, "core.l2_normalize", {
                "dim": _sample_dim(dim, len(_shape(args[0])))}, [args[0]])
            return
        if fn in (F.max_pool1d, F.max_pool2d, F.max_pool3d, F.avg_pool1d, F.avg_pool2d,
                  F.avg_pool3d):
            rank = len(_shape(args[0])) - 2
            kind = ("maxpool" if "max" in fn.__name__ else "avgpool") + f"{rank}d"
            k = _uniform(args[1] if len(args) > 1 else kwargs["kernel_size"], "kernel_size")
            stride = args[2] if len(args) > 2 else kwargs.get("stride") or k
            self.value_of[node.name] = self._add(node.name, f"core.{kind}", {
                "kernel_size": k, "stride": _uniform(stride, "stride"),
                "padding": _uniform(kwargs.get("padding", 0), "padding")}, [args[0]])
            return
        if fn in (F.adaptive_avg_pool1d, F.adaptive_avg_pool2d, F.adaptive_avg_pool3d,
                  F.adaptive_max_pool1d, F.adaptive_max_pool2d, F.adaptive_max_pool3d):
            rank = len(_shape(args[0])) - 2
            kind = ("adaptive_maxpool" if "max" in fn.__name__ else "adaptive_avgpool")
            size = _uniform(args[1] if len(args) > 1 else kwargs["output_size"], "output_size")
            self.value_of[node.name] = self._add(node.name, f"core.{kind}{rank}d",
                                                 {"output_size": size}, [args[0]])
            return
        if fn is F.interpolate:
            scale = kwargs.get("scale_factor")
            if scale is None:
                raise Unsupported("interpolate to a fixed size")
            rank = len(_shape(args[0])) - 2
            mode = {"linear": "bilinear", "trilinear": "bilinear"}.get(
                kwargs.get("mode", "nearest"), kwargs.get("mode", "nearest"))
            self.value_of[node.name] = self._add(node.name, f"core.upsample{rank}d", {
                "scale_factor": _uniform(scale, "scale_factor"), "mode": mode}, [args[0]])
            return
        if fn in (torch.reshape,):
            self._reshape(node, args[0])
            return
        if fn in (torch.permute,):
            self._permute(node, args[0], args[1])
            return
        if fn in (torch.transpose,):
            self._transpose(node, args[0], args[1], args[2])
            return
        if fn in (torch.squeeze, torch.unsqueeze):
            self._squeeze(node, args[0], fn.__name__, args[1] if len(args) > 1 else None)
            return
        if fn is getattr:
            self.value_of[node.name] = ("meta", None)
            return
        name = getattr(fn, "__name__", str(fn))
        raise Unsupported(f"no block for {getattr(fn, '__module__', '')}.{name}")

    def _call_method(self, node: fx.Node) -> None:
        method = node.target
        args = node.args
        src = args[0]
        if method == "size" or method == "dim":
            self.value_of[node.name] = ("meta", None)
            return
        if method in _PASSTHROUGH_METHODS or (method == "to" and _is_tensor(src)):
            self.value_of[node.name] = self.value_of.get(src.name)
            return
        if method in _UNARY_METHODS:
            self.value_of[node.name] = self._add(node.name, _UNARY_METHODS[method], {}, [src])
            return
        if method in _MATH_METHODS:
            self.value_of[node.name] = self._add(node.name, "core.math",
                                                 {"op": _MATH_METHODS[method]}, [src])
            return
        if method in ("view", "reshape"):
            self._reshape(node, src)
            return
        if method == "flatten":
            start = args[1] if len(args) > 1 else node.kwargs.get("start_dim", 0)
            end = args[2] if len(args) > 2 else node.kwargs.get("end_dim", -1)
            self._flatten(node, src, start, end, node.name)
            return
        if method == "permute":
            dims = args[1:] if not isinstance(args[1], (tuple, list)) else args[1]
            self._permute(node, src, dims)
            return
        if method == "transpose":
            self._transpose(node, src, args[1], args[2])
            return
        if method in ("squeeze", "unsqueeze"):
            self._squeeze(node, src, method, args[1] if len(args) > 1 else
                          node.kwargs.get("dim"))
            return
        if method in _REDUCE_METHODS:
            self._reduce(node, src, _REDUCE_METHODS[method],
                         args[1] if len(args) > 1 else node.kwargs.get("dim"),
                         node.kwargs.get("keepdim", False))
            return
        if method in ("softmax", "log_softmax"):
            dim = args[1] if len(args) > 1 else node.kwargs.get("dim", -1)
            self.value_of[node.name] = self._add(node.name, f"core.{method}", {
                "dim": _sample_dim(dim, len(_shape(src)))}, [src])
            return
        if method in ("add", "mul", "sub"):
            op = {"add": operator.add, "mul": operator.mul, "sub": operator.sub}[method]
            fake = _FakeCall(node, op)
            self._call_function(fake)
            self.value_of[node.name] = self.value_of[fake.name]
            return
        if method == "clamp":
            lo = args[1] if len(args) > 1 else node.kwargs.get("min")
            hi = args[2] if len(args) > 2 else node.kwargs.get("max")
            if lo is None or hi is None:
                raise Unsupported("one-sided clamp")
            self.value_of[node.name] = self._add(node.name, "core.clamp",
                                                 {"min": float(lo), "max": float(hi)}, [src])
            return
        raise Unsupported(f"no block for tensor.{method}()")

    # ------------------------------------------------------- shape ops
    def _flatten(self, node: fx.Node, src, start: int, end: int, name: str) -> None:  # noqa: ANN001
        rank = len(_shape(src))
        end = end if end >= 0 else rank + end
        if start in (1, -rank + 1) and end == rank - 1:
            self.value_of[node.name] = self._add(name, "core.flatten", {}, [src])
        else:
            self._reshape(node, src)

    def _reshape(self, node: fx.Node, src) -> None:  # noqa: ANN001
        out = _shape(node)
        if out is None or out[0] != _shape(src)[0]:
            raise Unsupported("reshape that changes the batch size")
        if len(out) == 2:
            self.value_of[node.name] = self._add(node.name, "core.flatten", {}, [src])
            return
        self.value_of[node.name] = self._add(node.name, "core.reshape", {
            "target": ", ".join(str(d) for d in out[1:])}, [src])

    def _permute(self, node: fx.Node, src, dims) -> None:  # noqa: ANN001
        rank = len(_shape(src))
        dims = [d % rank for d in dims]
        if dims[0] != 0:
            raise Unsupported("permute that moves the batch dimension")
        self.value_of[node.name] = self._add(node.name, "core.permute", {
            "order": ", ".join(str(d - 1) for d in dims[1:])}, [src])

    def _transpose(self, node: fx.Node, src, a: int, b: int) -> None:  # noqa: ANN001
        rank = len(_shape(src))
        a, b = a % rank, b % rank
        if 0 in (a, b):
            raise Unsupported("transpose with the batch dimension")
        self.value_of[node.name] = self._add(node.name, "core.transpose",
                                             {"dim0": a - 1, "dim1": b - 1}, [src])

    def _squeeze(self, node: fx.Node, src, kind: str, dim) -> None:  # noqa: ANN001
        if dim is None:
            raise Unsupported(f"{kind}() without a dim")
        rank = len(_shape(src)) + (1 if kind == "unsqueeze" else 0)
        dim = dim % rank
        if dim == 0:
            raise Unsupported(f"{kind} of the batch dimension")
        self.value_of[node.name] = self._add(node.name, f"core.{kind}", {"dim": dim - 1}, [src])

    def _concat(self, node: fx.Node, tensors: list, dim: int) -> None:
        if len(tensors) == 1:
            self.value_of[node.name] = self._source(tensors[0])
            return
        rank = len(_shape(tensors[0]))
        dim = dim % rank
        if dim == 0:
            raise Unsupported("concatenation along the batch dimension")
        axis = dim - 1
        current = tensors[0]
        for i, other in enumerate(tensors[1:]):
            bid = self._add(f"{node.name}_{i}" if len(tensors) > 2 else node.name,
                            "core.concatenate", {"axis": axis}, [current, other])
            current = _Ref(bid)
        self.value_of[node.name] = bid

    def _reduce(self, node: fx.Node, src, op: str, dim, keepdim: bool) -> None:  # noqa: ANN001
        shape = _shape(src)
        rank = len(shape)
        if dim is None:
            raise Unsupported("reduction over every dimension")
        dims = sorted(d % rank for d in (dim if isinstance(dim, (tuple, list)) else [dim]))
        if 0 in dims:
            raise Unsupported("reduction over the batch dimension")
        if not keepdim and op in ("mean", "max") and dims == list(range(2, rank)) and rank >= 3:
            kind = "avg" if op == "mean" else "max"
            if rank in (3, 4, 5) and len(dims) == rank - 2:
                # [B, C, *spatial] global pooling ... unless this is a sequence [B, L, C]
                self.value_of[node.name] = self._add(
                    node.name, f"core.global_{kind}pool{rank - 2}d", {}, [src])
                return
        if not keepdim and op in ("mean", "max") and dims == [1] and rank == 3:
            self.value_of[node.name] = self._add(
                node.name, "core.mean_over_time" if op == "mean" else "core.max_over_time",
                {}, [src])
            return
        if len(dims) != 1:
            raise Unsupported("reduction over several dimensions")
        self.value_of[node.name] = self._add(node.name, "core.reduce", {
            "op": op, "dim": dims[0] - 1, "keepdim": bool(keepdim)}, [src])

    def _getitem(self, node: fx.Node) -> None:
        src, index = node.args
        value = self.value_of.get(src.name) if isinstance(src, fx.Node) else None
        if isinstance(value, tuple) and value[0] == "rnn":
            _, bid, module = value
            if index == 0:
                self.value_of[node.name] = ("seq", bid)
            elif index == 1:  # LSTM: (h_n, c_n); GRU / RNN: h_n
                kind = "lstm_state" if isinstance(module, nn.LSTM) else "h_n"
                self.value_of[node.name] = (kind, bid, module)
            else:
                raise Unsupported(f"recurrent output index {index}")
            return
        if isinstance(value, tuple) and value[0] == "lstm_state":
            self.value_of[node.name] = (("h_n", value[1], value[2]) if index == 0
                                        else ("unusable", "the LSTM cell state has no block"))
            return
        if isinstance(value, tuple) and value[0] == "h_n":
            _, bid, module = value
            if index == -1 or index == module.num_layers * (2 if module.bidirectional else 1) - 1:
                if module.bidirectional:
                    raise Unsupported("final hidden state of a bidirectional layer")
                self.value_of[node.name] = ("final", bid)
                return
            raise Unsupported("indexing the recurrent hidden state")
        if isinstance(value, tuple) and value[0] == "attn":
            if index == 0:
                self.value_of[node.name] = value[1]
                return
            raise Unsupported("attention weights output")
        if isinstance(value, tuple) and value[0] == "meta":
            self.value_of[node.name] = ("meta", None)
            return
        if not _is_tensor(src):
            raise Unsupported("indexing a non-tensor value")
        rank = len(_shape(src))
        if isinstance(index, tuple) and len(index) == 2 and index[0] == slice(None) \
                and isinstance(index[1], int) and rank == 3:
            self.value_of[node.name] = self._add(node.name, "core.select_timestep",
                                                 {"index": index[1]}, [src])
            return
        if isinstance(index, tuple) and all(isinstance(i, slice) for i in index) \
                and index[0] == slice(None):
            cut = [(d, i) for d, i in enumerate(index) if i != slice(None)]
            if len(cut) == 1 and cut[0][1].step in (None, 1):
                d, s = cut[0]
                size = _shape(src)[d]
                end = s.stop if s.stop is not None else size
                self.value_of[node.name] = self._add(node.name, "core.slice", {
                    "dim": d - 1, "start": s.start or 0, "end": end}, [src])
                return
        raise Unsupported(f"indexing {index!r}")


class _Ref:
    """A block id posing as an fx node (for chained internal blocks)."""

    def __init__(self, bid: str):
        self.name = f"__ref_{bid}"
        self.bid = bid


class _FakeCall:
    """tensor.add(y) etc. re-dispatched as operator calls."""

    def __init__(self, node: fx.Node, op):  # noqa: ANN001
        self.name = node.name + "__fn"
        self.target = op
        self.args = node.args
        self.kwargs = node.kwargs
        self.meta = node.meta


# ------------------------------------------------------------- entrypoint

def load_model(source: str, attr: str, kwargs: dict) -> nn.Module:
    """``source`` is a .py file path or an importable module name."""
    if source.endswith(".py"):
        path = Path(source).expanduser().resolve()
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(path.parent))
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(source)
    factory = getattr(module, attr, None)
    if factory is None:
        raise Unsupported(f"{source} has no {attr!r}")
    model = factory(**kwargs)
    if not isinstance(model, nn.Module):
        raise Unsupported(f"{attr} did not produce a torch.nn.Module")
    return model


def import_module(model: nn.Module, input_shape: list[int], dtype: str = "float32",
                  weights_out: str | None = None) -> dict:
    example = (torch.zeros((2, *input_shape), dtype=torch.long) if dtype == "int64"
               else torch.randn(2, *input_shape))
    with torch.no_grad():
        reference = model.eval()(example)
    if not isinstance(reference, torch.Tensor):
        raise Unsupported("forward() must return one tensor")
    imp = Importer(model, example)
    graph = imp.run()
    params = sum(p.numel() for p in model.parameters())
    result = {"graph": graph, "unsupported": imp.unsupported, "warnings": imp.warnings,
              "original": {"parameters": int(params),
                           "output_shape": list(reference.shape[1:]),
                           "class": f"{type(model).__module__}.{type(model).__name__}"}}
    if not imp.unsupported:
        result["verification"] = verify(graph, model, example, reference, imp.origin,
                                        weights_out)
    return result


def verify(graph_dict: dict, model: nn.Module, example: torch.Tensor,
           reference: torch.Tensor, origin: dict[str, str],
           weights_out: str | None = None) -> dict:
    """Regenerate PyTorch code from the imported graph, copy the original weights
    into it and compare outputs."""
    import re

    from ai_made_easy.core.codegen import emit_graph, generate
    from ai_made_easy.core.graph import Graph

    graph = Graph.from_dict(graph_dict)
    errors = [i.message for i in graph.validate() if i.severity == "error"]
    if errors:
        return {"ok": False, "errors": errors}
    namespace: dict = {"__name__": "aime_imported"}
    exec(compile(generate(graph, "pytorch"), "<imported>", "exec"), namespace)  # noqa: S102
    rebuilt = namespace["build_model"]().eval()
    report = {"ok": True, "parameters": int(sum(p.numel() for p in rebuilt.parameters()))}
    attrs = {}
    for entry in emit_graph(graph).nodes:
        m = re.match(r"self\.(\w+) =", entry["torch_module"] or "")
        if m:
            attrs[entry["id"]] = m.group(1)
    copied = missing = 0
    for bid, attr in attrs.items():
        target = rebuilt.get_submodule(attr)
        if not any(True for _ in target.parameters()) and not list(target.buffers()):
            continue
        path = origin.get(bid)
        try:
            target.load_state_dict(model.get_submodule(path).state_dict())
            copied += 1
        except Exception:  # noqa: BLE001 — reported as missing
            missing += 1
    with torch.no_grad():
        out = rebuilt(example)
    report["output_shape"] = list(out.shape[1:])
    report["weights_copied"] = copied
    report["weights_missing"] = missing
    if not missing and out.shape == reference.shape:
        report["max_abs_diff"] = float((out.float() - reference.float()).abs().max())
        report["outputs_match"] = report["max_abs_diff"] <= 1e-4 * max(
            1.0, float(reference.abs().max()))
        if weights_out:
            torch.save(rebuilt.state_dict(), weights_out)
            report["weights_file"] = weights_out
    return report


def main() -> int:
    request = json.loads(Path(sys.argv[1]).read_text())
    try:
        model = load_model(request["source"], request["attr"], request.get("kwargs") or {})
        result = import_module(model, request["input_shape"], request.get("dtype", "float32"),
                               request.get("weights_out"))
    except Unsupported as exc:
        result = {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001 — reported to the caller
        result = {"error": f"{type(exc).__name__}: {exc}"}
    print("IMPORT-RESULT " + json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
