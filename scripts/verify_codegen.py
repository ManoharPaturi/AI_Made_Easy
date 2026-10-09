"""Execute generated model code for every NN block and compare real output
shapes against the designer's inferred shapes (PyTorch + Keras 3).

Usage: KERAS_BACKEND=torch python scripts/verify_codegen.py [--keras] [-v]
Also importable: ``block_cases()`` / ``run_case()`` power the test suite.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_made_easy.core.graph import Edge, Graph, NodeInstance  # noqa: E402
from ai_made_easy.core.registry import get_registry  # noqa: E402
from ai_made_easy.core.spec import missing_requirements  # noqa: E402

VARIANTS: dict[str, list[dict]] = {
    "core.conv2d": [{"padding": 2, "stride": 2}, {"dilation": 2, "padding": 2},
                    {"groups": 2, "out_channels": 8, "bias": False},
                    {"kernel_size": 5, "padding": 0, "stride": 3}],
    "core.conv1d": [{"padding": 0, "stride": 2}, {"padding": 3, "kernel_size": 3}],
    "core.conv3d": [{"stride": 2}],
    "core.conv_transpose2d": [{"stride": 2, "padding": 1, "output_padding": 1},
                              {"stride": 2, "kernel_size": 2, "padding": 0},
                              {"padding": 1, "dilation": 2}],
    "core.conv_transpose1d": [{"stride": 3, "padding": 2, "output_padding": 2}],
    "core.depthwise_conv2d": [{"depth_multiplier": 2, "stride": 2}],
    "core.separable_conv2d": [{"padding": 0, "bias": False}],
    "core.maxpool2d": [{"padding": 1, "kernel_size": 3, "stride": 2}],
    "core.avgpool1d": [{"padding": 1, "kernel_size": 3, "stride": 1}],
    "core.adaptive_avgpool2d": [{"output_size": 4}, {"output_size": 3}],
    "core.adaptive_maxpool1d": [{"output_size": 8}],
    "core.lstm": [{"bidirectional": True}, {"num_layers": 2, "dropout": 0.2},
                  {"return_sequences": False}, {"bidirectional": True,
                                                "return_sequences": False}],
    "core.gru": [{"bidirectional": True, "num_layers": 2, "return_sequences": False}],
    "core.rnn": [{"nonlinearity": "relu", "bias": False}],
    "core.layer_norm": [{"normalized_dims": "all"}, {"affine": False}],
    "core.batch_norm1d": [{"affine": False}],
    "core.gelu": [{"approximate": "tanh"}],
    "core.softplus": [{"beta": 2.0}],
    "core.softmax": [{"dim": 0}],
    "core.upsample2d": [{"mode": "bilinear"}, {"mode": "bicubic", "scale_factor": 3}],
    "core.upsample1d": [{"mode": "linear"}],
    "core.reduce": [{"op": "sum", "dim": 0, "keepdim": True}, {"op": "std"},
                    {"op": "max", "dim": 0}],
    "core.slice": [{"dim": -1, "start": 2, "end": -2}],
    "core.transpose": [{"dim0": -1, "dim1": 0}],
    "core.unsqueeze": [{"dim": -1}],
    "core.transformer_encoder": [{"activation": "gelu", "norm_first": True}],
    "core.dot": [{"normalize": True}],
    "core.embedding": [{"padding_idx": 0}],
    "core.math": [{"op": "square"}, {"op": "sin"}],
    "core.reshape": [{"target": "4, 8"}, {"target": "8, 2, 16"}],
    "core.pretrained_backbone": [{"architecture": a, "weights": "none"} for a in (
        "mobilenet_v3_small", "efficientnet_b0", "densenet121", "convnext_tiny", "vgg16")],
    "core.squeeze": [{"dim": 0}],
    "vision.unet": [{"variant": v, "depth": 3} for v in (
        "unet_plus_plus", "attention_unet", "resunet")] + [{"upsample": "bilinear",
                                                             "norm": "group"}],
    "vision.segmenter": [{"arch": "lraspp_mobilenet_v3_large", "weights": "none"}],
    "vision.detector": [{"arch": a, "weights": "none"} for a in (
        "ssdlite320_mobilenet_v3_large", "fcos_resnet50_fpn")],
}

# defaults that would download weights are replaced for offline verification
DEFAULT_OVERRIDES = {"core.pretrained_backbone": {"weights": "none"},
                     "core.hf_text_encoder": None,
                     **{t: {"weights": "none"} for t in (
                         "vision.detector", "vision.instance_segmenter",
                         "vision.keypoint_detector", "vision.hf_detector", "vision.segmenter",
                         "vision.hf_segmenter", "vision.timm_backbone",
                         "vision.hf_image_encoder")}}

CANDIDATE_SHAPES = ([32], [8, 16], [4, 16, 16], [4, 8, 8, 8], [8, 16, 16], [1, 16],
                    [3, 64, 64], [3, 224, 224])


def _graph_for(type_id: str, shape: list[int], params: dict | None = None) -> Graph:
    defn = get_registry().get(type_id)
    g = Graph(name=f"case_{type_id.replace('.', '_')}")
    dtype = "int64" if defn.input_dtype == "int" else "float32"
    g.add_node(NodeInstance("inp", "core.input",
                            {"shape": ",".join(map(str, shape)), "dtype": dtype}))
    g.add_node(NodeInstance("blk", type_id, dict(params or {})))
    g.add_node(NodeInstance("out", "core.output", {}))
    for port in defn.inputs:
        g.add_edge(Edge("inp", "out", "blk", port.name))
    g.add_edge(Edge("blk", defn.outputs[0].name, "out", "in"))
    return g


def block_cases(extra_params: dict | None = None):
    """Yield (type_id, input_shape, params, graph) for every NN block that
    accepts at least one candidate shape."""
    reg = get_registry()
    for defn in reg.all():
        if defn.shape_fn is None or defn.builder is not None:
            continue
        if defn.type_id in ("core.input", "core.output"):
            continue
        if DEFAULT_OVERRIDES.get(defn.type_id, {}) is None:
            continue  # needs network downloads (covered by tests that opt in)
        if missing_requirements(defn):
            continue  # optional package not installed (the extra's CI job covers it)
        variants = [DEFAULT_OVERRIDES.get(defn.type_id, {})] + list(
            (extra_params or {}).get(defn.type_id, []))
        for params in variants:
            for shape in CANDIDATE_SHAPES:
                g = _graph_for(defn.type_id, shape, params)
                if not [i for i in g.validate() if i.severity == "error"]:
                    yield defn.type_id, shape, params, g
                    break


def _sample(shape, int_input: bool, vocab: int = 10):
    import torch

    if int_input:
        return torch.randint(0, vocab, (2, *shape))
    return torch.randn(2, *shape)


def run_torch(graph: Graph) -> tuple[int, ...]:
    import torch

    from ai_made_easy.core.codegen import generate

    code = generate(graph, "pytorch")
    ns: dict = {"__name__": "aime_generated"}
    exec(compile(code, f"<{graph.name}_pytorch>", "exec"), ns)
    from ai_made_easy.core.codegen import class_name_for

    model = ns[class_name_for(graph.name)]()
    model.eval()
    shapes = graph.infer_shapes()
    in_shape = shapes["inp"]
    int_input = graph.nodes["inp"].params.get("dtype") == "int64"
    with torch.no_grad():
        out = model(_sample(in_shape, int_input))
    return tuple(out.shape[1:])


def run_keras(graph: Graph) -> tuple[int, ...]:
    import numpy as np

    from ai_made_easy.core.codegen import generate, keras_output_shape

    code = generate(graph, "keras")
    ns: dict = {"__name__": "aime_generated"}
    exec(compile(code, f"<{graph.name}_keras>", "exec"), ns)
    model = ns["build_model"]()
    in_shape = tuple(int(d) for d in model.inputs[0].shape[1:])
    if graph.nodes["inp"].params.get("dtype") == "int64":
        x = np.random.randint(0, 10, size=(2, *in_shape))
    else:
        x = np.random.randn(2, *in_shape).astype("float32")
    out = model(x)
    got = tuple(int(d) for d in out.shape[1:])
    return got, tuple(keras_output_shape(graph))


def main() -> int:
    verbose = "-v" in sys.argv
    do_keras = "--keras" in sys.argv
    fails = 0
    total = 0
    for type_id, shape, params, g in block_cases(VARIANTS):
        total += 1
        want = tuple(g.infer_shapes()["out"])
        try:
            got = run_torch(g)
            ok = got == want
            msg = "" if ok else f"shape {got} != designer {want}"
        except Exception as exc:  # noqa: BLE001
            ok, msg = False, f"{type(exc).__name__}: {str(exc)[:160]}"
        line = f"torch  {type_id:32} in={shape} {params or ''} {msg}"
        if not ok:
            fails += 1
            print("FAIL", line)
        elif verbose:
            print("ok  ", line)
        if do_keras and get_registry().get(type_id).supports("keras"):
            try:
                got_k, want_k = run_keras(g)
                ok = got_k == want_k
                msg = "" if ok else f"shape {got_k} != expected {want_k}"
            except Exception as exc:  # noqa: BLE001
                ok, msg = False, f"{type(exc).__name__}: {str(exc)[:160]}"
            if not ok and "no Keras equivalent" in msg:
                ok = True  # a configuration Keras cannot express, reported cleanly
            line = f"keras  {type_id:32} in={shape} {params or ''} {msg}"
            if not ok:
                fails += 1
                print("FAIL", line)
            elif verbose:
                print("ok  ", line)
    print(f"{total} block cases, {fails} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
