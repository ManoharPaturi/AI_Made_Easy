"""ONNX → block graph (needs ``onnx``; verification also uses torch + onnxruntime).

ONNX tensors are batch-first and channels-first, exactly like the designer's
IR, so operators map onto blocks directly. Weights stored as initializers are
copied into the regenerated PyTorch model and its outputs are compared with
onnxruntime.

Run as a script: ``python -m ai_made_easy.core.importers.onnx_import REQUEST.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference


class Unsupported(Exception):
    pass


def _attrs(node) -> dict:  # noqa: ANN001
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


def _uniform(values, what: str) -> int:  # noqa: ANN001
    values = list(values)
    if len(set(values)) != 1:
        raise Unsupported(f"non-uniform {what} {values}")
    return int(values[0])


def _sym_pads(pads, what: str = "pads") -> int:  # noqa: ANN001
    pads = list(pads or [])
    if not pads:
        return 0
    half = len(pads) // 2
    if pads[:half] != pads[half:]:
        raise Unsupported(f"asymmetric {what} {pads}")
    return _uniform(pads[:half], what)


_ACTS = {"Relu": "core.relu", "Sigmoid": "core.sigmoid", "Tanh": "core.tanh",
         "Selu": "core.selu", "Softsign": "core.softsign", "HardSwish": "core.hardswish",
         "Mish": "core.mish", "Identity": "core.identity", "Softplus": "core.softplus"}
_MATH = {"Abs": "abs", "Exp": "exp", "Log": "log", "Sqrt": "sqrt", "Neg": "negative",
         "Reciprocal": "reciprocal", "Sin": "sin", "Cos": "cos", "Sign": "sign"}
_BINARY = {"Add": "core.add", "Sub": "core.subtract", "Mul": "core.multiply",
           "Max": "core.maximum", "Min": "core.minimum"}


class OnnxImporter:
    def __init__(self, model: onnx.ModelProto, input_shape: list[int] | None):
        model = shape_inference.infer_shapes(model)
        self.model = model
        self.graph = model.graph
        self.init = {i.name: numpy_helper.to_array(i) for i in self.graph.initializer}
        for node in self.graph.node:  # constants and their aliases behave like initializers
            if node.op_type == "Constant":
                self.init[node.output[0]] = numpy_helper.to_array(_attrs(node)["value"])
            elif node.op_type == "Identity" and node.input[0] in self.init:
                self.init[node.output[0]] = self.init[node.input[0]]
        self.shapes: dict[str, list] = {}
        for vi in list(self.graph.value_info) + list(self.graph.input) + list(self.graph.output):
            dims = [d.dim_value if d.dim_value > 0 else None
                    for d in vi.type.tensor_type.shape.dim]
            self.shapes[vi.name] = dims
        self.input_shape = input_shape
        self.nodes: list[dict] = []
        self.edges: list[tuple[str, str, str]] = []
        self.value_of: dict[str, str] = {}
        self.weights: dict[str, dict[str, np.ndarray]] = {}  # block id -> torch state dict
        self.unsupported: list[str] = []
        self.warnings: list[str] = []
        self._ids: dict[str, int] = {}
        self._consumers: dict[str, int] = {}
        for node in self.graph.node:
            for name in node.input:
                self._consumers[name] = self._consumers.get(name, 0) + 1

    # --------------------------------------------------------------- utils
    def _new_id(self, base: str) -> str:
        base = "".join(c if c.isalnum() else "_" for c in base.lower()).strip("_") or "block"
        n = self._ids.get(base, 0)
        self._ids[base] = n + 1
        return base if n == 0 else f"{base}_{n}"

    def _add(self, base: str, type_id: str, params: dict, inputs: list[str]) -> str:
        from ai_made_easy.core.registry import get_registry

        ports = [p.name for p in get_registry().get(type_id).inputs]
        if len(ports) != len(inputs):
            raise Unsupported(f"{type_id} takes {len(ports)} input(s)")
        sources = []
        for name in inputs:
            if name not in self.value_of:
                raise Unsupported(f"input {name!r} comes from an unsupported operation")
            sources.append(self.value_of[name])
        bid = self._new_id(base)
        self.nodes.append({"id": bid, "type": type_id, "params": params})
        for port, src in zip(ports, sources, strict=True):
            self.edges.append((src, bid, port))
        return bid

    def _rank(self, name: str) -> int:
        shape = self.shapes.get(name)
        if shape is None:
            raise Unsupported(f"unknown rank for {name!r}")
        return len(shape)

    def _dynamic(self, node) -> list[str]:  # noqa: ANN001
        return [i for i in node.input if i and i not in self.init]

    # ---------------------------------------------------------------- main
    def run(self) -> dict:
        inputs = [i for i in self.graph.input if i.name not in self.init]
        if len(inputs) != 1:
            raise Unsupported(f"the model has {len(inputs)} inputs; one is supported")
        inp = inputs[0]
        dims = self.shapes[inp.name]
        per_sample = self.input_shape or dims[1:]
        if any(d is None for d in per_sample):
            raise Unsupported(f"the input shape {dims} has dynamic dimensions — give the "
                              "input shape explicitly")
        elem = inp.type.tensor_type.elem_type
        dtype = "int64" if elem in (onnx.TensorProto.INT64, onnx.TensorProto.INT32) else "float32"
        bid = self._new_id("input")
        self.nodes.append({"id": bid, "type": "core.input",
                           "params": {"shape": ", ".join(str(d) for d in per_sample),
                                      "dtype": dtype}})
        self.value_of[inp.name] = bid
        self.dtype = dtype
        self.per_sample = per_sample
        skip, fused = self._fuse_gelu()
        for node in self.graph.node:
            if node.op_type == "Constant" or node.output[0] in self.init:
                continue
            if node.output[0] in skip:
                if node.output[0] in fused:
                    try:
                        self.value_of[node.output[0]] = self._add(
                            "gelu", "core.gelu", {"approximate": "none"},
                            [fused[node.output[0]]])
                    except Unsupported as exc:
                        self.unsupported.append(f"GELU: {exc}")
                continue
            try:
                self._convert(node)
            except Unsupported as exc:
                self.unsupported.append(f"{node.name or node.op_type} ({node.op_type}): {exc}")
        outputs = list(self.graph.output)
        if len(outputs) != 1:
            self.unsupported.append(f"the model has {len(outputs)} outputs; one is supported")
        elif outputs[0].name in self.value_of:
            oid = self._new_id("output")
            self.nodes.append({"id": oid, "type": "core.output", "params": {}})
            self.edges.append((self.value_of[outputs[0].name], oid, "in"))
        used = {s for s, _, _ in self.edges} | {t for _, t, _ in self.edges}
        nodes = [n for n in self.nodes if n["id"] in used or n["type"] == "core.input"]
        for i, n in enumerate(nodes):
            n["position"] = [i * 240.0, 0.0]
        name = Path(self.graph.name or "imported_onnx").stem or "imported_onnx"
        return {"name": name, "nodes": nodes,
                "edges": [{"from": f"{s}/out", "to": f"{t}/{p}"} for s, t, p in self.edges],
                "meta": {"imported_from": "onnx"}}

    # ----------------------------------------------------------- operators
    def _convert(self, node) -> None:  # noqa: ANN001, C901
        op = node.op_type
        a = _attrs(node)
        out = node.output[0]
        dyn = self._dynamic(node)
        name = node.name or op

        def single(type_id: str, params: dict | None = None) -> None:
            self.value_of[out] = self._add(name, type_id, params or {}, dyn[:1])

        if op in _ACTS:
            single(_ACTS[op])
        elif op in _MATH:
            single("core.math", {"op": _MATH[op]})
        elif op == "LeakyRelu":
            single("core.leaky_relu", {"negative_slope": float(a.get("alpha", 0.01))})
        elif op == "Elu":
            single("core.elu", {"alpha": float(a.get("alpha", 1.0))})
        elif op == "Celu":
            single("core.celu", {"alpha": float(a.get("alpha", 1.0))})
        elif op == "HardSigmoid":
            if abs(a.get("alpha", 0.2) - 1 / 6) > 1e-6 or a.get("beta", 0.5) != 0.5:
                raise Unsupported("HardSigmoid with non-PyTorch alpha/beta")
            single("core.hardsigmoid")
        elif op == "Gelu":
            single("core.gelu", {"approximate": a.get("approximate", "none")})
        elif op in ("Softmax", "LogSoftmax"):
            axis = int(a.get("axis", -1))
            single("core.softmax" if op == "Softmax" else "core.log_softmax",
                   {"dim": axis - 1 if axis > 0 else axis})
        elif op == "PRelu":
            slope = self.init.get(node.input[1])
            if slope is None or slope.size != 1:
                raise Unsupported("PReLU with one slope per channel")
            single("core.prelu")
            self.weights[self.value_of[out]] = {"weight": slope.reshape(1)}
        elif op == "Clip":
            lo = self._scalar(node.input[1]) if len(node.input) > 1 and node.input[1] else \
                a.get("min")
            hi = self._scalar(node.input[2]) if len(node.input) > 2 and node.input[2] else \
                a.get("max")
            if lo == 0.0 and hi == 6.0:
                single("core.relu6")
            elif lo is not None and hi is not None:
                single("core.clamp", {"min": float(lo), "max": float(hi)})
            else:
                raise Unsupported("one-sided Clip")
        elif op == "Dropout":
            ratio = self._scalar(node.input[1]) if len(node.input) > 1 and node.input[1] \
                else a.get("ratio", 0.5)
            single("core.dropout", {"p": float(ratio or 0.5)})
        elif op in ("Conv", "ConvTranspose"):
            self._conv(node, a, op == "ConvTranspose")
        elif op in ("Gemm", "MatMul"):
            self._dense(node, a)
        elif op in ("MaxPool", "AveragePool", "LpPool"):
            self._pool(node, a)
        elif op in ("GlobalAveragePool", "GlobalMaxPool"):
            rank = self._rank(dyn[0]) - 2
            kind = "adaptive_avgpool" if op == "GlobalAveragePool" else "adaptive_maxpool"
            single(f"core.{kind}{rank}d", {"output_size": 1})
        elif op == "Flatten":
            if int(a.get("axis", 1)) != 1:
                raise Unsupported("Flatten with axis != 1")
            single("core.flatten")
        elif op == "BatchNormalization":
            rank = self._rank(dyn[0]) - 2
            kind = {0: "batch_norm1d", 1: "batch_norm1d", 2: "batch_norm2d",
                    3: "batch_norm3d"}[rank]
            single(f"core.{kind}", {"epsilon": float(a.get("epsilon", 1e-5)),
                                    "momentum": round(1.0 - float(a.get("momentum", 0.9)), 6),
                                    "affine": True})
            scale, bias, mean, var = (self.init[n] for n in node.input[1:5])
            self.weights[self.value_of[out]] = {"weight": scale, "bias": bias,
                                                "running_mean": mean, "running_var": var}
        elif op == "LayerNormalization":
            axis = int(a.get("axis", -1))
            rank = self._rank(dyn[0])
            if axis not in (-1, rank - 1):
                raise Unsupported("LayerNormalization over several axes")
            single("core.layer_norm", {"normalized_dims": "last",
                                       "epsilon": float(a.get("epsilon", 1e-5)),
                                       "affine": True})
            sd = {"weight": self.init[node.input[1]]}
            if len(node.input) > 2 and node.input[2]:
                sd["bias"] = self.init[node.input[2]]
            self.weights[self.value_of[out]] = sd
        elif op in _BINARY or op == "Div":
            self._binary(node, op)
        elif op == "Concat":
            axis = int(a["axis"])
            rank = self._rank(dyn[0])
            axis = axis % rank
            if axis == 0:
                raise Unsupported("concatenation along the batch axis")
            current = dyn[0]
            for i, other in enumerate(dyn[1:]):
                tmp = f"{out}__concat{i}"
                self.value_of[tmp] = self._add(name, "core.concatenate", {"axis": axis - 1},
                                               [current, other])
                current = tmp
            self.value_of[out] = self.value_of[current]
        elif op == "Reshape":
            shape = self.shapes.get(out)
            if not shape or any(d is None for d in shape[1:]):
                raise Unsupported("Reshape to a dynamic shape")
            if len(shape) == 2:
                single("core.flatten")
            else:
                single("core.reshape", {"target": ", ".join(str(d) for d in shape[1:])})
        elif op == "Transpose":
            perm = list(a["perm"])
            if perm[0] != 0:
                raise Unsupported("Transpose that moves the batch axis")
            single("core.permute", {"order": ", ".join(str(p - 1) for p in perm[1:])})
        elif op in ("Squeeze", "Unsqueeze"):
            axes = (list(self.init[node.input[1]]) if len(node.input) > 1 and node.input[1]
                    else list(a.get("axes", [])))
            if len(axes) != 1:
                raise Unsupported(f"{op} with {len(axes)} axes")
            rank = self._rank(dyn[0]) + (1 if op == "Unsqueeze" else 0)
            axis = int(axes[0]) % rank
            if axis == 0:
                raise Unsupported(f"{op} of the batch axis")
            single(f"core.{op.lower()}", {"dim": axis - 1})
        elif op.startswith("Reduce") and op[6:] in ("Mean", "Sum", "Max", "Min", "Prod"):
            self._reduce(node, a, op[6:].lower())
        elif op in ("Resize", "Upsample"):
            scales = None
            for idx in (2, 1):
                if len(node.input) > idx and node.input[idx] in self.init:
                    arr = self.init[node.input[idx]]
                    if arr.size and arr.dtype.kind == "f":
                        scales = list(arr)
                        break
            if not scales or scales[:2] != [1.0, 1.0]:
                raise Unsupported("Resize without per-axis scales")
            factor = _uniform([int(s) for s in scales[2:]], "scale")
            mode = a.get("mode", b"nearest")
            mode = mode.decode() if isinstance(mode, bytes) else mode
            mode = {"linear": "bilinear", "cubic": "bicubic"}.get(mode, mode)
            single(f"core.upsample{len(scales) - 2}d", {"scale_factor": factor, "mode": mode})
        elif op == "Pad":
            pads = list(self.init[node.input[1]]) if len(node.input) > 1 else list(a["pads"])
            rank = len(pads) // 2
            mode = a.get("mode", b"constant")
            mode = mode.decode() if isinstance(mode, bytes) else mode
            if pads[0] or pads[1] or pads[rank] or pads[rank + 1]:
                raise Unsupported("padding of batch or channel axes")
            amount = _sym_pads(pads[2:rank] + pads[rank + 2:])
            kind = {"constant": "zero_pad", "reflect": "reflection_pad",
                    "edge": "replication_pad"}.get(mode)
            if kind is None:
                raise Unsupported(f"Pad mode {mode}")
            single(f"core.{kind}{rank - 2}d", {"padding": amount})
        elif op == "Gather" and node.input[0] in self.init and self.dtype == "int64":
            table = self.init[node.input[0]]
            self.value_of[out] = self._add(name, "core.embedding", {
                "num_embeddings": int(table.shape[0]), "embedding_dim": int(table.shape[1]),
                "padding_idx": -1}, [node.input[1]])
            self.weights[self.value_of[out]] = {"weight": table}
        elif op in ("Cast", "Identity"):
            self.value_of[out] = self.value_of[dyn[0]]
        else:
            raise Unsupported("no block for this operator")

    def _fuse_gelu(self) -> tuple[set, dict]:
        """Find exact-GELU subgraphs x * 0.5 * (1 + erf(x / sqrt(2))) (older opsets)."""
        producer = {o: n for n in self.graph.node for o in n.output}
        consumers: dict[str, list] = {}
        for n in self.graph.node:
            for i in n.input:
                consumers.setdefault(i, []).append(n)

        def const(name: str):  # noqa: ANN202
            return self._scalar(name) if name in self.init else None

        def only(name: str, op: str):  # noqa: ANN202
            users = consumers.get(name, [])
            return users[0] if len(users) == 1 and users[0].op_type == op else None

        skip, fused = set(), {}
        for erf in (n for n in self.graph.node if n.op_type == "Erf"):
            pre = producer.get(erf.input[0])
            if pre is None:
                continue
            if pre.op_type == "Div" and abs((const(pre.input[1]) or 0) - 2 ** 0.5) < 1e-3:
                x = pre.input[0]
            elif pre.op_type == "Mul" and abs((const(pre.input[1]) or 0) - 2 ** -0.5) < 1e-3:
                x = pre.input[0]
            else:
                continue
            add = only(erf.output[0], "Add")
            if add is None or 1.0 not in [const(i) for i in add.input]:
                continue
            mul = only(add.output[0], "Mul")
            if mul is None:
                continue
            other = mul.input[0] if mul.input[1] == add.output[0] else mul.input[1]
            if other == x:
                last = only(mul.output[0], "Mul")
                if last is None or 0.5 not in [const(i) for i in last.input]:
                    continue
                chain = [pre, erf, add, mul, last]
            else:
                half = producer.get(other)
                if half is None or half.op_type != "Mul" or x not in half.input \
                        or 0.5 not in [const(i) for i in half.input] \
                        or len(consumers.get(other, [])) != 1:
                    continue
                chain, last = [pre, erf, add, half, mul], mul
            skip.update(n.output[0] for n in chain)  # protobuf wrappers are not stable ids
            fused[last.output[0]] = x
        return skip, fused

    def _scalar(self, name: str):  # noqa: ANN202
        if name not in self.init:
            return None
        arr = self.init[name]
        return float(arr.reshape(-1)[0]) if arr.size == 1 else None

    def _conv(self, node, a: dict, transpose: bool) -> None:  # noqa: ANN001
        weight = self.init.get(node.input[1])
        if weight is None:
            raise Unsupported("convolution weights computed at run time")
        rank = weight.ndim - 2
        group = int(a.get("group", 1))
        k = _uniform(a.get("kernel_shape", weight.shape[2:]), "kernel")
        auto_pad = a.get("auto_pad", b"NOTSET")
        auto_pad = auto_pad.decode() if isinstance(auto_pad, bytes) else auto_pad
        pads = a.get("pads", [0] * 2 * rank)
        if auto_pad == "VALID":
            pads = [0] * 2 * rank
        elif auto_pad in ("SAME_UPPER", "SAME_LOWER"):
            dilation = _uniform(a.get("dilations", [1] * rank), "dilations")
            stride = _uniform(a.get("strides", [1] * rank), "strides")
            total = (k - 1) * dilation
            if stride != 1 or total % 2 or transpose:
                raise Unsupported(f"auto_pad {auto_pad} that needs asymmetric padding")
            pads = [total // 2] * 2 * rank
        elif auto_pad != "NOTSET":
            raise Unsupported(f"auto_pad {auto_pad}")
        params = {"out_channels": int(weight.shape[1] * group if transpose else weight.shape[0]),
                  "kernel_size": k, "stride": _uniform(a.get("strides", [1] * rank), "strides"),
                  "padding": _sym_pads(pads),
                  "dilation": _uniform(a.get("dilations", [1] * rank), "dilations"),
                  "groups": group, "bias": len(node.input) > 2 and bool(node.input[2])}
        kind = f"conv_transpose{rank}d" if transpose else f"conv{rank}d"
        if transpose:
            params["output_padding"] = _uniform(a.get("output_padding", [0] * rank),
                                                "output_padding")
        bid = self._add(node.name or kind, f"core.{kind}", params, [node.input[0]])
        sd = {"weight": weight}
        if params["bias"]:
            sd["bias"] = self.init[node.input[2]]
        self.weights[bid] = sd
        self.value_of[node.output[0]] = bid

    def _dense(self, node, a: dict) -> None:  # noqa: ANN001
        x, w = node.input[0], node.input[1]
        weight = self.init.get(w)
        if weight is None or weight.ndim != 2:
            raise Unsupported("matrix product of two run-time tensors")
        if node.op_type == "Gemm":
            if a.get("transA", 0) or float(a.get("alpha", 1.0)) != 1.0 \
                    or float(a.get("beta", 1.0)) != 1.0:
                raise Unsupported("Gemm with transA/alpha/beta")
            torch_w = weight if a.get("transB", 0) else weight.T
            bias = self.init.get(node.input[2]) if len(node.input) > 2 and node.input[2] else None
        else:
            torch_w = weight.T
            bias = None
        bid = self._add(node.name or "dense", "core.dense",
                        {"units": int(torch_w.shape[0]), "bias": bias is not None}, [x])
        sd = {"weight": torch_w}
        if bias is not None:
            sd["bias"] = np.broadcast_to(bias, (torch_w.shape[0],)).copy()
        self.weights[bid] = sd
        self.value_of[node.output[0]] = bid

    def _pool(self, node, a: dict) -> None:  # noqa: ANN001
        if int(a.get("ceil_mode", 0)):
            raise Unsupported("ceil_mode")
        if node.op_type == "AveragePool" and int(a.get("count_include_pad", 0)) == 0 \
                and any(a.get("pads", [])):
            self.warnings.append(f"{node.name}: padded average pooling counts padding")
        kernel = list(a["kernel_shape"])
        rank = len(kernel)
        k = _uniform(kernel, "kernel")
        params = {"kernel_size": k, "stride": _uniform(a.get("strides", kernel), "strides"),
                  "padding": _sym_pads(a.get("pads", [0] * 2 * rank))}
        if node.op_type == "LpPool":
            if rank != 2 or params["padding"]:
                raise Unsupported("LpPool other than 2-D unpadded")
            params = {"norm_type": float(a.get("p", 2)), "kernel_size": k,
                      "stride": params["stride"]}
            kind = "lppool2d"
        else:
            kind = ("maxpool" if node.op_type == "MaxPool" else "avgpool") + f"{rank}d"
        self.value_of[node.output[0]] = self._add(node.name or kind, f"core.{kind}", params,
                                                  [node.input[0]])

    def _binary(self, node, op: str) -> None:  # noqa: ANN001
        a_name, b_name = node.input[0], node.input[1]
        out = node.output[0]
        consts = [n for n in (a_name, b_name) if n in self.init]
        if consts:
            tensor = a_name if b_name in self.init else b_name
            value = self.init[consts[0]]
            # MatMul + Add(bias) -> fold into the dense block
            src = self.value_of.get(tensor)
            block = next((n for n in self.nodes if n["id"] == src), None)
            if op == "Add" and block and block["type"] == "core.dense" \
                    and not block["params"]["bias"] and value.ndim == 1 \
                    and value.shape[0] == block["params"]["units"] \
                    and self._consumers.get(tensor, 0) == 1:
                block["params"]["bias"] = True
                self.weights[src]["bias"] = value
                self.value_of[out] = src
                return
            if value.size != 1:
                raise Unsupported(f"{op} with a constant tensor")
            c = float(value.reshape(-1)[0])
            if op == "Add":
                params = {"scale": 1.0, "shift": c}
            elif op == "Sub" and tensor == a_name:
                params = {"scale": 1.0, "shift": -c}
            elif op == "Mul":
                params = {"scale": c, "shift": 0.0}
            elif op == "Div" and tensor == a_name:
                params = {"scale": 1.0 / c, "shift": 0.0}
            else:
                raise Unsupported(f"{op} with a constant")
            self.value_of[out] = self._add(node.name or op, "core.scale", params, [tensor])
            return
        if op == "Div":
            raise Unsupported("division of two tensors")
        if self.shapes.get(a_name) != self.shapes.get(b_name):
            raise Unsupported("broadcasting between different shapes")
        self.value_of[out] = self._add(node.name or op, _BINARY[op], {}, [a_name, b_name])

    def _reduce(self, node, a: dict, op: str) -> None:  # noqa: ANN001
        x = node.input[0]
        rank = self._rank(x)
        axes = (list(self.init[node.input[1]]) if len(node.input) > 1 and node.input[1]
                else list(a.get("axes", [])))
        axes = sorted(int(ax) % rank for ax in axes)
        keep = bool(a.get("keepdims", 1))
        if not axes or 0 in axes:
            raise Unsupported("reduction over the batch axis")
        if not keep and op in ("mean", "max") and axes == list(range(2, rank)) and rank >= 3:
            kind = "avg" if op == "mean" else "max"
            self.value_of[node.output[0]] = self._add(node.name or op,
                                                      f"core.global_{kind}pool{rank - 2}d",
                                                      {}, [x])
            return
        if keep and op in ("mean", "max") and axes == list(range(2, rank)) and rank >= 3:
            kind = "adaptive_avgpool" if op == "mean" else "adaptive_maxpool"
            self.value_of[node.output[0]] = self._add(node.name or op,
                                                      f"core.{kind}{rank - 2}d",
                                                      {"output_size": 1}, [x])
            return
        if len(axes) != 1:
            raise Unsupported("reduction over several axes")
        self.value_of[node.output[0]] = self._add(node.name or op, "core.reduce", {
            "op": op, "dim": axes[0] - 1, "keepdim": keep}, [x])


def verify(path: str, imp: OnnxImporter, graph_dict: dict) -> dict:
    """Rebuild in PyTorch, copy the ONNX weights, compare with onnxruntime."""
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
        state = {k: torch.as_tensor(np.ascontiguousarray(v)).reshape(expected[k].shape)
                 for k, v in given.items() if k in expected}
        for k in expected:
            if k not in state and k.endswith("num_batches_tracked"):
                state[k] = expected[k]
        try:
            target.load_state_dict(state)
            copied += 1
        except Exception:  # noqa: BLE001
            missing += 1
    report = {"ok": True, "parameters": int(sum(p.numel() for p in rebuilt.parameters())),
              "weights_copied": copied, "weights_missing": missing}
    if imp.dtype == "int64":
        x = np.random.default_rng(0).integers(0, 2, (2, *imp.per_sample)).astype(np.int64)
    else:
        x = np.random.default_rng(0).normal(size=(2, *imp.per_sample)).astype(np.float32)
    with torch.no_grad():
        out = rebuilt(torch.from_numpy(x)).float().numpy()
    report["output_shape"] = list(out.shape[1:])
    if missing:
        return report
    try:
        import onnxruntime as ort
    except ImportError:
        report["note"] = "install onnxruntime to compare outputs numerically"
        return report
    session = ort.InferenceSession(path)
    feed = {session.get_inputs()[0].name: x}
    expected = session.run(None, feed)[0]
    if expected.shape == out.shape:
        report["max_abs_diff"] = float(np.abs(expected - out).max())
        report["outputs_match"] = report["max_abs_diff"] <= 1e-4 * max(
            1.0, float(np.abs(expected).max()))
    return report


def import_onnx(path: str, input_shape: list[int] | None = None) -> dict:
    model = onnx.load(path)
    imp = OnnxImporter(model, input_shape)
    graph = imp.run()
    graph["name"] = Path(path).stem
    params = int(sum(int(np.prod(i.dims)) for i in model.graph.initializer
                     if i.data_type in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16,
                                        onnx.TensorProto.DOUBLE)))
    result = {"graph": graph, "unsupported": imp.unsupported, "warnings": imp.warnings,
              "original": {"parameters_stored": params, "opset": int(
                  model.opset_import[0].version) if model.opset_import else None,
                  "producer": model.producer_name}}
    if not imp.unsupported:
        try:
            result["verification"] = verify(path, imp, graph)
        except ImportError as exc:
            result["verification"] = {"ok": True, "note": f"verification needs {exc.name}"}
    return result


def main() -> int:
    request = json.loads(Path(sys.argv[1]).read_text())
    try:
        result = import_onnx(request["source"], request.get("input_shape"))
    except Unsupported as exc:
        result = {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        result = {"error": f"{type(exc).__name__}: {exc}"}
    print("IMPORT-RESULT " + json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
