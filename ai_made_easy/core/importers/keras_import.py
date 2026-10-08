"""Keras 3 (.keras / .h5) → block graph.

Keras tensors are channels-last; the designer's IR is channels-first for
convolutional tensors and ``[L, C]`` for sequences. Every tensor carries a
layout tag — ``same`` (IR order == Keras order) or ``cl`` (IR is the
channels-first view of a channels-last Keras tensor) — and ``Permute`` blocks
are inserted where a layer needs the other view. Weights are converted to
PyTorch conventions and the regenerated model is compared with Keras.

Run as a script: ``python -m ai_made_easy.core.importers.keras_import REQUEST.json``.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras  # noqa: E402
import numpy as np  # noqa: E402


class Unsupported(Exception):
    pass


SAME, CL = "same", "cl"
_ACTIVATIONS = {"relu": "core.relu", "sigmoid": "core.sigmoid", "tanh": "core.tanh",
                "gelu": "core.gelu", "silu": "core.silu", "swish": "core.silu",
                "elu": "core.elu", "selu": "core.selu", "softplus": "core.softplus",
                "softsign": "core.softsign", "relu6": "core.relu6", "mish": "core.mish",
                "hard_sigmoid": "core.hardsigmoid", "hard_silu": "core.hardswish",
                "hard_swish": "core.hardswish", "leaky_relu": "core.leaky_relu",
                "log_softmax": "core.log_softmax", "softmax": "core.softmax"}
_CF_LAYERS = {"Conv1D", "Conv2D", "Conv3D", "Conv1DTranspose", "Conv2DTranspose",
              "Conv3DTranspose", "MaxPooling1D", "MaxPooling2D", "MaxPooling3D",
              "AveragePooling1D", "AveragePooling2D", "AveragePooling3D",
              "GlobalAveragePooling1D", "GlobalAveragePooling2D", "GlobalAveragePooling3D",
              "GlobalMaxPooling1D", "GlobalMaxPooling2D", "GlobalMaxPooling3D",
              "UpSampling1D", "UpSampling2D", "UpSampling3D", "ZeroPadding1D",
              "ZeroPadding2D", "ZeroPadding3D", "Cropping1D", "Cropping2D", "Cropping3D",
              "SpatialDropout1D", "SpatialDropout2D", "SpatialDropout3D",
              "DepthwiseConv2D", "SeparableConv2D", "DepthwiseConv1D", "SeparableConv1D"}
_SEQ_LAYERS = {"LSTM", "GRU", "SimpleRNN", "Bidirectional", "MultiHeadAttention",
               "RepeatVector", "Dense", "LayerNormalization"}


def _name(cls: str) -> str:
    return cls.lower()


def _int(value, what: str) -> int:  # noqa: ANN001
    values = list(value) if isinstance(value, (tuple, list)) else [value]
    if len(set(values)) != 1:
        raise Unsupported(f"non-uniform {what} {tuple(values)}")
    return int(values[0])


class KerasImporter:
    def __init__(self, model: keras.Model):
        self.model = model
        self.nodes: list[dict] = []
        self.edges: list[tuple[str, str, str]] = []
        self.tensors: dict[int, tuple[str, str, tuple]] = {}  # id(tensor) -> (bid, layout, kshape)
        self.weights: dict[str, dict[str, np.ndarray]] = {}
        self.unsupported: list[str] = []
        self.warnings: list[str] = []
        self.exact = True  # whether outputs can be compared numerically
        self._ids: dict[str, int] = {}

    # --------------------------------------------------------------- utils
    def _new_id(self, base: str) -> str:
        base = "".join(c if c.isalnum() else "_" for c in base.lower()).strip("_") or "block"
        n = self._ids.get(base, 0)
        self._ids[base] = n + 1
        return base if n == 0 else f"{base}_{n}"

    def _block(self, base: str, type_id: str, params: dict, sources: list[str]) -> str:
        from ai_made_easy.core.registry import get_registry

        ports = [p.name for p in get_registry().get(type_id).inputs]
        if len(ports) != len(sources):
            raise Unsupported(f"{type_id} takes {len(ports)} input(s)")
        bid = self._new_id(base)
        self.nodes.append({"id": bid, "type": type_id, "params": params})
        for port, src in zip(ports, sources, strict=True):
            self.edges.append((src, bid, port))
        return bid

    def _view(self, tensor, want: str) -> str:  # noqa: ANN001
        """Block id producing ``tensor`` in the requested IR layout."""
        bid, layout, kshape = self.tensors[id(tensor)]
        rank = len(kshape)
        if rank < 2 or layout == want:
            return bid
        if rank > 2:
            raise Unsupported("this layer needs a channels-last view of an image tensor")
        key = (bid, want)
        cached = getattr(self, "_views", {}).get(key)
        if cached:
            return cached
        pid = self._block(f"{bid}_permute", "core.permute", {"order": "1, 0"}, [bid])
        self.__dict__.setdefault("_views", {})[key] = pid
        return pid

    def _ir_axis(self, axis: int, layout: str, rank: int) -> int:
        """Keras per-sample axis -> IR per-sample axis."""
        axis = axis % rank
        if layout == SAME or rank < 2:
            return axis
        return 0 if axis == rank - 1 else axis + 1

    def _activation(self, base: str, name, src: str, axis_layout=None) -> str:  # noqa: ANN001
        name = name if isinstance(name, str) else getattr(name, "__name__", str(name))
        if name in ("linear", None, "None"):
            return src
        type_id = _ACTIVATIONS.get(name)
        if type_id is None:
            raise Unsupported(f"activation {name!r}")
        params = {}
        if type_id in ("core.softmax", "core.log_softmax"):
            rank, layout = axis_layout or (1, SAME)
            params["dim"] = self._ir_axis(-1, layout, rank) if rank > 1 else -1
        if type_id == "core.leaky_relu":
            params["negative_slope"] = 0.2  # keras.activations.leaky_relu default
        return self._block(f"{base}_{name}", type_id, params, [src])

    # ---------------------------------------------------------------- walk
    def _order(self) -> list:
        """(layer, node) pairs in dependency order from the model inputs."""
        seen, order = set(), []

        def visit(tensor) -> None:  # noqa: ANN001
            op, node_index, _ = tensor._keras_history
            key = (id(op), node_index)
            if key in seen:
                return
            node = op._inbound_nodes[node_index]
            for t in node.input_tensors:
                visit(t)
            seen.add(key)
            order.append((op, node))

        for out in self.model.outputs:
            visit(out)
        return order

    def run(self) -> dict:
        if len(self.model.inputs) != 1:
            raise Unsupported(f"the model has {len(self.model.inputs)} inputs; one is supported")
        if len(self.model.outputs) != 1:
            raise Unsupported(f"the model has {len(self.model.outputs)} outputs; one is supported")
        for layer, node in self._order():
            cls = type(layer).__name__
            try:
                if cls == "InputLayer":
                    self._input(node.output_tensors[0])
                else:
                    self._layer(layer, node, cls)
            except Unsupported as exc:
                self.unsupported.append(f"{layer.name} ({cls}): {exc}")
                for t in node.output_tensors:
                    self.tensors.pop(id(t), None)
            except KeyError:
                self.unsupported.append(f"{layer.name} ({cls}): its input comes from an "
                                        "unsupported layer")
        out = self.model.outputs[0]
        if id(out) in self.tensors:
            bid, layout, kshape = self.tensors[id(out)]
            if layout == CL and len(kshape) >= 2:
                self.warnings.append("the output is channels-first in the designer "
                                     "(channels-last in Keras)")
            oid = self._new_id("output")
            self.nodes.append({"id": oid, "type": "core.output", "params": {}})
            self.edges.append((bid, oid, "in"))
        used = {s for s, _, _ in self.edges} | {t for _, t, _ in self.edges}
        nodes = [n for n in self.nodes if n["id"] in used or n["type"] == "core.input"]
        for i, n in enumerate(nodes):
            n["position"] = [i * 240.0, 0.0]
        return {"name": self.model.name, "nodes": nodes,
                "edges": [{"from": f"{s}/out", "to": f"{t}/{p}"} for s, t, p in self.edges],
                "meta": {"imported_from": "keras"}}

    def _input(self, tensor) -> None:  # noqa: ANN001
        kshape = tuple(tensor.shape[1:])
        if any(d is None for d in kshape):
            raise Unsupported(f"input shape {kshape} has unknown dimensions")
        dtype = "int64" if "int" in str(tensor.dtype) else "float32"
        # rank-2 inputs stay [L, C] unless the first consumer is convolutional
        layout = CL if len(kshape) >= 3 else SAME
        if len(kshape) == 2:
            consumers = [type(n.operation).__name__ for n in tensor._keras_history[0]
                         ._outbound_nodes] if hasattr(tensor._keras_history[0],
                                                      "_outbound_nodes") else []
            if consumers and consumers[0] in _CF_LAYERS:
                layout = CL
        ir = (kshape[-1], *kshape[:-1]) if layout == CL else kshape
        bid = self._new_id("input")
        self.nodes.append({"id": bid, "type": "core.input",
                           "params": {"shape": ", ".join(str(d) for d in ir), "dtype": dtype}})
        self.tensors[id(tensor)] = (bid, layout, kshape)
        self.input_layout = layout
        self.input_kshape = kshape
        self.dtype = dtype

    # -------------------------------------------------------------- layers
    def _layer(self, layer, node, cls: str) -> None:  # noqa: ANN001, C901
        cfg = layer.get_config()
        inputs = list(node.input_tensors)
        out = node.output_tensors[0]
        okshape = tuple(out.shape[1:])
        base = layer.name

        def emit(type_id: str, params: dict, layout: str, sources: list[str] | None = None,
                 tensor=None) -> str:  # noqa: ANN001
            src = sources or [self._view(inputs[0], layout)]
            bid = self._block(base, type_id, params, src)
            act = cfg.get("activation")
            last = bid
            if act not in (None, "linear"):
                last = self._activation(base, act, bid, (len(okshape), layout))
            self.tensors[id(tensor or out)] = (last, layout, okshape)
            return bid

        if cls in _CF_LAYERS and cfg.get("data_format", "channels_last") != "channels_last":
            raise Unsupported("channels_first Keras layers")

        if cls == "Dense":
            in_shape = tuple(inputs[0].shape[1:])
            if len(in_shape) > 2:
                raise Unsupported("Dense on a tensor of rank > 2")
            bid = emit("core.dense", {"units": cfg["units"], "bias": cfg["use_bias"]}, SAME)
            w = layer.get_weights()
            kernel = w[0]
            flat = self._flatten_of(inputs[0])
            if flat is not None:  # undo Keras' channels-last flatten order
                kshape, layout = flat
                if layout == CL and len(kshape) >= 2:
                    kernel = np.moveaxis(kernel.reshape(*kshape, -1), len(kshape) - 1, 0)
                    kernel = kernel.reshape(-1, kernel.shape[-1])
            sd = {"weight": kernel.T}
            if cfg["use_bias"]:
                sd["bias"] = w[1]
            self.weights[bid] = sd
        elif cls in ("Conv1D", "Conv2D", "Conv3D"):
            rank = int(cls[-2])
            k = _int(cfg["kernel_size"], "kernel_size")
            stride = _int(cfg["strides"], "strides")
            dilation = _int(cfg["dilation_rate"], "dilation_rate")
            padding = self._padding(cfg["padding"], k, stride, dilation)
            bid = emit(f"core.conv{rank}d", {
                "out_channels": cfg["filters"], "kernel_size": k, "stride": stride,
                "padding": padding, "dilation": dilation, "groups": cfg.get("groups", 1),
                "bias": cfg["use_bias"]}, CL)
            w = layer.get_weights()
            sd = {"weight": np.transpose(w[0], (rank + 1, rank, *range(rank)))}
            if cfg["use_bias"]:
                sd["bias"] = w[1]
            self.weights[bid] = sd
        elif cls in ("Conv1DTranspose", "Conv2DTranspose", "Conv3DTranspose"):
            rank = int(cls[4])
            k = _int(cfg["kernel_size"], "kernel_size")
            s = _int(cfg["strides"], "strides")
            if _int(cfg["dilation_rate"], "dilation_rate") != 1:
                raise Unsupported("dilated transposed convolution")
            if cfg["padding"] == "valid":
                p, op = 0, 0
            else:  # 'same': output = n * s
                p = (k - s + 1) // 2
                op = s + 2 * p - k
                if not 0 <= op < s:
                    raise Unsupported("this 'same' transposed convolution")
            if cfg.get("output_padding"):
                op = _int(cfg["output_padding"], "output_padding")
            bid = emit(f"core.conv_transpose{rank}d", {
                "out_channels": cfg["filters"], "kernel_size": k, "stride": s, "padding": p,
                "output_padding": op, "dilation": 1, "groups": 1, "bias": cfg["use_bias"]}, CL)
            self.exact = False  # kernel flipping conventions differ between backends
        elif cls in ("DepthwiseConv2D", "DepthwiseConv1D"):
            rank = int(cls[-2])
            k = _int(cfg["kernel_size"], "kernel_size")
            stride = _int(cfg["strides"], "strides")
            dilation = _int(cfg["dilation_rate"], "dilation_rate")
            bid = emit(f"core.depthwise_conv{rank}d", {
                "depth_multiplier": cfg["depth_multiplier"], "kernel_size": k,
                "stride": stride, "padding": self._padding(cfg["padding"], k, stride, dilation),
                "dilation": dilation, "bias": cfg["use_bias"]}, CL)
            self.exact = False
        elif cls in ("SeparableConv2D", "SeparableConv1D"):
            rank = int(cls[-2])
            k = _int(cfg["kernel_size"], "kernel_size")
            stride = _int(cfg["strides"], "strides")
            dilation = _int(cfg["dilation_rate"], "dilation_rate")
            emit(f"core.separable_conv{rank}d", {
                "out_channels": cfg["filters"], "depth_multiplier": cfg["depth_multiplier"],
                "kernel_size": k, "stride": stride,
                "padding": self._padding(cfg["padding"], k, stride, dilation),
                "dilation": dilation, "bias": cfg["use_bias"]}, CL)
            self.exact = False
        elif cls.startswith(("MaxPooling", "AveragePooling")):
            rank = int(cls[-2])
            k = _int(cfg["pool_size"], "pool_size")
            s = _int(cfg["strides"] or cfg["pool_size"], "strides")
            if cfg["padding"] != "valid":
                raise Unsupported("'same' pooling")
            kind = "maxpool" if cls.startswith("Max") else "avgpool"
            emit(f"core.{kind}{rank}d", {"kernel_size": k, "stride": s, "padding": 0}, CL)
        elif cls.startswith(("GlobalAveragePooling", "GlobalMaxPooling")):
            rank = int(cls[-2])
            if cfg.get("keepdims"):
                kind = "adaptive_avgpool" if "Average" in cls else "adaptive_maxpool"
                emit(f"core.{kind}{rank}d", {"output_size": 1}, CL)
            else:
                kind = "avg" if "Average" in cls else "max"
                emit(f"core.global_{kind}pool{rank}d", {}, CL)
                self.tensors[id(out)] = (self.tensors[id(out)][0], SAME, okshape)
        elif cls == "Flatten":
            src_bid, layout, kshape = self.tensors[id(inputs[0])]
            bid = self._block(base, "core.flatten", {}, [src_bid])
            self.tensors[id(out)] = (bid, SAME, okshape)
            self.__dict__.setdefault("_flattened", {})[bid] = (kshape, layout)
        elif cls in ("Dropout", "SpatialDropout1D", "SpatialDropout2D", "SpatialDropout3D",
                     "GaussianNoise", "GaussianDropout", "AlphaDropout"):
            layout = self.tensors[id(inputs[0])][1]
            if cls.startswith("Spatial"):
                type_id, layout = f"core.dropout{cls[-2]}d", CL
                params = {"p": float(cfg["rate"])}
            elif cls == "GaussianNoise":
                type_id, params = "core.gaussian_noise", {"stddev": float(cfg["stddev"])}
            elif cls == "GaussianDropout":
                type_id, params = "core.gaussian_dropout", {"p": float(cfg["rate"])}
            elif cls == "AlphaDropout":
                type_id, params = "core.alpha_dropout", {"p": float(cfg["rate"])}
            else:
                type_id, params = "core.dropout", {"p": float(cfg["rate"])}
            emit(type_id, params, layout)
        elif cls == "BatchNormalization":
            rank = len(okshape)
            axis = cfg["axis"][0] if isinstance(cfg["axis"], (list, tuple)) else cfg["axis"]
            if axis not in (-1, rank):
                raise Unsupported("BatchNormalization over a non-channel axis")
            if not (cfg["center"] and cfg["scale"]):
                raise Unsupported("BatchNormalization without center/scale")
            kind = {1: "batch_norm1d", 2: "batch_norm1d", 3: "batch_norm2d",
                    4: "batch_norm3d"}[rank]
            bid = emit(f"core.{kind}", {"epsilon": float(cfg["epsilon"]),
                                        "momentum": round(1 - float(cfg["momentum"]), 6),
                                        "affine": True}, CL)
            gamma, beta, mean, var = layer.get_weights()
            self.weights[bid] = {"weight": gamma, "bias": beta, "running_mean": mean,
                                 "running_var": var}
        elif cls == "LayerNormalization":
            axis = cfg["axis"]
            axis = axis if isinstance(axis, int) else (axis[0] if len(axis) == 1 else None)
            if axis not in (-1, len(okshape)):
                raise Unsupported("LayerNormalization over several axes")
            if not (cfg["center"] and cfg["scale"]):
                raise Unsupported("LayerNormalization without center/scale")
            if len(okshape) > 2:
                raise Unsupported("LayerNormalization on an image tensor")
            bid = emit("core.layer_norm", {"normalized_dims": "last",
                                           "epsilon": float(cfg["epsilon"]), "affine": True},
                       SAME)
            gamma, beta = layer.get_weights()
            self.weights[bid] = {"weight": gamma, "bias": beta}
        elif cls == "Activation":
            src, layout, _ = self.tensors[id(inputs[0])]
            last = self._activation(base, cfg["activation"], src, (len(okshape), layout))
            self.tensors[id(out)] = (last, layout, okshape)
        elif cls in ("ReLU", "LeakyReLU", "ELU", "Softmax", "PReLU"):
            src, layout, _ = self.tensors[id(inputs[0])]
            if cls == "ReLU":
                if cfg.get("max_value") == 6.0 and not cfg.get("negative_slope"):
                    type_id, params = "core.relu6", {}
                elif cfg.get("max_value") is None and not cfg.get("negative_slope") \
                        and not cfg.get("threshold"):
                    type_id, params = "core.relu", {}
                else:
                    raise Unsupported("ReLU with custom max_value / threshold")
            elif cls == "LeakyReLU":
                type_id = "core.leaky_relu"
                params = {"negative_slope": float(cfg.get("negative_slope",
                                                          cfg.get("alpha", 0.3)))}
            elif cls == "ELU":
                type_id, params = "core.elu", {"alpha": float(cfg.get("alpha", 1.0))}
            elif cls == "PReLU":
                if np.asarray(layer.get_weights()[0]).size != 1:
                    raise Unsupported("PReLU with one slope per channel")
                type_id, params = "core.prelu", {}
            else:
                axis = cfg.get("axis", -1)
                type_id = "core.softmax"
                params = {"dim": self._ir_axis(axis if isinstance(axis, int) else axis[0],
                                               layout, len(okshape))}
            bid = self._block(base, type_id, params, [src])
            if cls == "PReLU":
                self.weights[bid] = {"weight": np.asarray(layer.get_weights()[0]).reshape(1)}
            self.tensors[id(out)] = (bid, layout, okshape)
        elif cls == "Embedding":
            bid = self._block(base, "core.embedding", {
                "num_embeddings": cfg["input_dim"], "embedding_dim": cfg["output_dim"],
                "padding_idx": -1}, [self.tensors[id(inputs[0])][0]])
            self.weights[bid] = {"weight": layer.get_weights()[0]}
            self.tensors[id(out)] = (bid, SAME, okshape)
        elif cls in ("LSTM", "GRU", "SimpleRNN"):
            self._recurrent(layer, cfg, cls, inputs[0], out, okshape, base, False)
        elif cls == "Bidirectional":
            inner = layer.forward_layer
            if cfg.get("merge_mode", "concat") != "concat":
                raise Unsupported("Bidirectional merge_mode other than concat")
            self._recurrent(inner, inner.get_config(), type(inner).__name__, inputs[0], out,
                            okshape, base, True, layer)
        elif cls == "MultiHeadAttention":
            width = inputs[0].shape[-1]
            if cfg["num_heads"] * cfg["key_dim"] != width or cfg.get("value_dim") not in (
                    None, cfg["key_dim"]) or cfg.get("output_shape"):
                raise Unsupported("MultiHeadAttention whose head size differs from "
                                  "width / heads")
            q = inputs[0]
            kv = inputs[1] if len(inputs) > 1 else inputs[0]
            if kv is q:
                bid = self._block(base, "core.multihead_attention", {
                    "embed_dim": 0, "num_heads": cfg["num_heads"],
                    "dropout": float(cfg["dropout"])}, [self._view(q, SAME)])
            else:
                bid = self._block(base, "core.cross_attention", {
                    "num_heads": cfg["num_heads"], "dropout": float(cfg["dropout"])},
                    [self._view(q, SAME), self._view(kv, SAME)])
            self.tensors[id(out)] = (bid, SAME, okshape)
            self.exact = False
        elif cls in ("Add", "Subtract", "Multiply", "Average", "Maximum", "Minimum"):
            type_id = {"Add": "core.add", "Subtract": "core.subtract",
                       "Multiply": "core.multiply", "Average": "core.average",
                       "Maximum": "core.maximum", "Minimum": "core.minimum"}[cls]
            if len(inputs) != 2 and cls not in ("Add", "Multiply", "Maximum", "Minimum"):
                raise Unsupported(f"{cls} of {len(inputs)} tensors")
            layout = self.tensors[id(inputs[0])][1]
            current = self._view(inputs[0], layout)
            for other in inputs[1:]:
                current = self._block(base, type_id, {}, [current, self._view(other, layout)])
            self.tensors[id(out)] = (current, layout, okshape)
        elif cls == "Concatenate":
            layout = self.tensors[id(inputs[0])][1]
            rank = len(okshape)
            axis = self._ir_axis(cfg["axis"] if cfg["axis"] < 0 else cfg["axis"] - 1,
                                 layout, rank)
            current = self._view(inputs[0], layout)
            for other in inputs[1:]:
                current = self._block(base, "core.concatenate", {"axis": axis},
                                      [current, self._view(other, layout)])
            self.tensors[id(out)] = (current, layout, okshape)
        elif cls == "Reshape":
            target = tuple(okshape)
            src, layout, kshape = self.tensors[id(inputs[0])]
            if len(target) == 1:
                bid = self._block(base, "core.flatten", {}, [src])
                self.tensors[id(out)] = (bid, SAME, okshape)
                return
            if layout == CL and len(kshape) >= 2 or len(target) >= 3:
                self.warnings.append(f"{base}: Keras reshapes in channels-last order; the "
                                     "imported Reshape keeps shapes but not element order")
                self.exact = False
            new_layout = CL if len(target) >= 3 else SAME
            ir = (target[-1], *target[:-1]) if new_layout == CL else target
            bid = self._block(base, "core.reshape",
                              {"target": ", ".join(str(d) for d in ir)}, [src])
            self.tensors[id(out)] = (bid, new_layout, okshape)
        elif cls == "Permute":
            src, layout, kshape = self.tensors[id(inputs[0])]
            if len(kshape) != 2 or layout != SAME or tuple(cfg["dims"]) != (2, 1):
                raise Unsupported("Permute other than swapping the two axes of a sequence")
            bid = self._block(base, "core.permute", {"order": "1, 0"}, [src])
            self.tensors[id(out)] = (bid, SAME, okshape)
        elif cls == "RepeatVector":
            emit("core.repeat_vector", {"times": cfg["n"]}, SAME)
        elif cls.startswith("UpSampling"):
            rank = int(cls[-2])
            size = _int(cfg["size"], "size")
            mode = cfg.get("interpolation", "nearest")
            mode = {"bilinear": "bilinear", "nearest": "nearest", "bicubic": "bicubic"}.get(mode)
            if mode is None:
                raise Unsupported(f"interpolation {cfg.get('interpolation')!r}")
            emit(f"core.upsample{rank}d", {"scale_factor": size, "mode": mode}, CL)
        elif cls.startswith("ZeroPadding"):
            rank = int(cls[-2])
            pad = cfg["padding"]
            flat = [v for p in (pad if isinstance(pad, (list, tuple)) else [pad])
                    for v in (p if isinstance(p, (list, tuple)) else [p, p])]
            emit(f"core.zero_pad{rank}d", {"padding": _int(flat, "padding")}, CL)
        elif cls.startswith("Cropping"):
            rank = int(cls[-2])
            crop = cfg["cropping"]
            flat = [v for p in (crop if isinstance(crop, (list, tuple)) else [crop])
                    for v in (p if isinstance(p, (list, tuple)) else [p, p])]
            emit(f"core.crop{rank}d", {"crop": _int(flat, "cropping")}, CL)
        elif cls == "Rescaling":
            scale, offset = cfg["scale"], cfg["offset"]
            if isinstance(scale, (list, tuple)) or isinstance(offset, (list, tuple)):
                raise Unsupported("per-channel Rescaling")
            layout = self.tensors[id(inputs[0])][1]
            emit("core.scale", {"scale": float(scale), "shift": float(offset)}, layout)
        else:
            raise Unsupported("no block for this layer")

    def _flatten_of(self, tensor):  # noqa: ANN001, ANN202
        """(Keras shape, layout) of the Flatten feeding ``tensor`` through
        element-wise layers (dropout, noise, activations), if any."""
        flattened = getattr(self, "_flattened", {})
        bid = self.tensors[id(tensor)][0]
        types = {n["id"]: n["type"] for n in self.nodes}
        while bid not in flattened:
            if not (types.get(bid, "").startswith(("core.dropout", "core.gaussian",
                                                   "core.alpha_dropout"))
                    or types.get(bid) in set(_ACTIVATIONS.values())):
                return None
            parents = [s for s, t, _ in self.edges if t == bid]
            if len(parents) != 1:
                return None
            bid = parents[0]
        return flattened[bid]

    def _padding(self, padding: str, k: int, stride: int, dilation: int) -> int:
        if padding == "valid":
            return 0
        if padding == "same":
            total = (k - 1) * dilation
            if total % 2:
                raise Unsupported("'same' padding with an even kernel")
            if stride > 1:
                self.warnings.append("Keras pads strided 'same' convolutions asymmetrically; "
                                     "the import pads symmetrically (same shapes)")
                self.exact = False
            return total // 2
        raise Unsupported(f"padding {padding!r}")

    def _recurrent(self, layer, cfg: dict, cls: str, inp, out, okshape, base: str,  # noqa: ANN001
                   bidirectional: bool, wrapper=None) -> None:  # noqa: ANN001
        if cfg.get("go_backwards") or cfg.get("stateful") or cfg.get("return_state") \
                or cfg.get("unroll"):
            raise Unsupported("go_backwards / stateful / return_state / unroll")
        if cfg.get("activation", "tanh") != "tanh" and cls != "SimpleRNN":
            raise Unsupported(f"{cls} activation {cfg.get('activation')!r}")
        if cls in ("LSTM", "GRU") and cfg.get("recurrent_activation", "sigmoid") != "sigmoid":
            raise Unsupported("recurrent_activation other than sigmoid")
        if cls == "GRU" and not cfg.get("reset_after", True):
            raise Unsupported("GRU with reset_after=False")
        if cfg.get("dropout") or cfg.get("recurrent_dropout"):
            self.warnings.append(f"{base}: input / recurrent dropout is not imported")
        kind = {"LSTM": "lstm", "GRU": "gru", "SimpleRNN": "rnn"}[cls]
        params = {"hidden_size": cfg["units"], "num_layers": 1, "bias": cfg["use_bias"],
                  "bidirectional": bidirectional, "dropout": 0.0,
                  "return_sequences": bool(cfg["return_sequences"])}
        if kind == "rnn":
            act = cfg.get("activation", "tanh")
            if act not in ("tanh", "relu"):
                raise Unsupported(f"SimpleRNN activation {act!r}")
            params["nonlinearity"] = act
        bid = self._block(base, f"core.{kind}", params, [self._view(inp, SAME)])
        self.tensors[id(out)] = (bid, SAME, okshape)
        if not cfg["use_bias"]:
            self.exact = False
            return
        sd = {}
        layers = [(wrapper.forward_layer, ""), (wrapper.backward_layer, "_reverse")] \
            if wrapper is not None else [(layer, "")]
        for sub, suffix in layers:
            kernel, rec, bias = sub.get_weights()
            h = cfg["units"]
            if kind == "gru":  # keras gates z, r, h -> torch r, z, n
                def reorder(m):  # noqa: ANN001, ANN202
                    z, r, n = np.split(m, 3, axis=-1)
                    return np.concatenate([r, z, n], axis=-1)

                kernel, rec = reorder(kernel), reorder(rec)
                bias = reorder(bias.reshape(2, 3 * h))
                sd[f"bias_ih_l0{suffix}"] = bias[0]
                sd[f"bias_hh_l0{suffix}"] = bias[1]
            else:
                sd[f"bias_ih_l0{suffix}"] = bias
                sd[f"bias_hh_l0{suffix}"] = np.zeros_like(bias)
            sd[f"weight_ih_l0{suffix}"] = kernel.T
            sd[f"weight_hh_l0{suffix}"] = rec.T
        self.weights[bid] = sd


def verify(model: keras.Model, imp: KerasImporter, graph_dict: dict) -> dict:
    """Rebuild in PyTorch, copy the converted weights, compare with Keras."""
    import re

    import torch

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
        expected = target.state_dict()
        if not expected:
            continue
        given = imp.weights.get(bid)
        if not given:
            missing += 1
            continue
        state = {}
        for key, ref in expected.items():
            if key in given:
                state[key] = torch.as_tensor(np.ascontiguousarray(given[key])).reshape(ref.shape)
            elif key.endswith("num_batches_tracked"):
                state[key] = ref
        try:
            target.load_state_dict(state)
            copied += 1
        except Exception:  # noqa: BLE001
            missing += 1
    report["weights_copied"], report["weights_missing"] = copied, missing
    kshape = imp.input_kshape
    rng = np.random.default_rng(0)
    if imp.dtype == "int64":
        x = rng.integers(0, 2, (2, *kshape)).astype(np.int64)
    else:
        x = rng.normal(size=(2, *kshape)).astype(np.float32)
    x_ir = np.moveaxis(x, -1, 1) if imp.input_layout == CL else x
    with torch.no_grad():
        out = rebuilt(torch.from_numpy(np.ascontiguousarray(x_ir))).float().numpy()
    report["output_shape"] = list(out.shape[1:])
    if missing or not imp.exact:
        return report
    expected = np.asarray(model.predict(x, verbose=0))
    out_cmp = out
    if expected.ndim >= 3 and expected.shape != out.shape:
        out_cmp = np.moveaxis(out, 1, -1)
    if expected.shape == out_cmp.shape:
        report["max_abs_diff"] = float(np.abs(expected - out_cmp).max())
        report["outputs_match"] = report["max_abs_diff"] <= 1e-4 * max(
            1.0, float(np.abs(expected).max()))
    return report


def import_keras(path: str) -> dict:
    model = keras.models.load_model(path, compile=False)
    if not model.built or not getattr(model, "inputs", None):
        raise Unsupported("the model has no defined input shape (build it with keras.Input)")
    imp = KerasImporter(model)
    graph = imp.run()
    graph["name"] = Path(path).stem
    result = {"graph": graph, "unsupported": imp.unsupported, "warnings": imp.warnings,
              "original": {"parameters": int(model.count_params()),
                           "output_shape": list(model.outputs[0].shape[1:]),
                           "class": type(model).__name__}}
    if not imp.unsupported:
        result["verification"] = verify(model, imp, graph)
    return result


def main() -> int:
    request = json.loads(Path(sys.argv[1]).read_text())
    try:
        result = import_keras(request["source"])
    except Unsupported as exc:
        result = {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        result = {"error": f"{type(exc).__name__}: {exc}"}
    print("IMPORT-RESULT " + json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
