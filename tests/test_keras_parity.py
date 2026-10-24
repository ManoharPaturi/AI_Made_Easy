"""Keras forms of the 3.7 / 3.8 layers compute what the PyTorch ones do (3.9).

The sequence and attention layers keep PyTorch's weight layout and names in Keras
(``TorchLayout``), so a PyTorch layer's weights copy over one-to-one: the outputs must
then agree. The vision blocks are built from Keras' own layers (channels-last): their
parameter counts must match, and involution's values too, with its weights converted.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
keras = pytest.importorskip("keras")

from keras import layers, ops  # noqa: E402

import ai_made_easy.core.blocks  # noqa: E402,F401  (registers the helpers)
from ai_made_easy.core.codegen.helpers import KERAS_HELPERS, PYTORCH_HELPERS  # noqa: E402

pytestmark = pytest.mark.skipif(keras.backend.backend() != "torch",
                                reason="compared on the torch backend")


def _torch(name: str):
    ns = {"torch": torch, "nn": torch.nn, "math": math}
    for helper in ("rms_norm", "chunk_scan", name):
        exec(PYTORCH_HELPERS[helper], ns)  # noqa: S102 — our own helper sources
    return ns[name]


def _keras(*names: str) -> dict:
    ns = {"keras": keras, "layers": layers, "ops": ops}
    for helper in ("TorchLayout", "keras_chunk_scan", *names):
        exec(KERAS_HELPERS[helper], ns)  # noqa: S102
    return ns


SEQUENCE = [  # (helper, PyTorch args, Keras args, Keras kwargs)
    ("LatentAttention", (8, 2, 3, True), (2, 3, True), {}),
    ("QKNormAttention", (8, 2, True), (2, True), {}),
    ("GatedAttention", (8, 2, False), (2, False), {}),
    ("Mamba2Block", (8, 4, 2, 2, 3), (4, 2, 2, 3), {"chunk": 8}),
    ("GatedDeltaNet", (8, 2), (2,), {"chunk": 8}),
    ("GatedLinearAttention", (8, 2), (2,), {"chunk": 4}),
    ("GriffinBlock", (8, 6, 3), (6, 3), {}),
    ("DiffAttention", (8, 2, True), (2, True), {}),
    ("LinformerAttention", (8, 21, 5, 2), (5, 2), {}),
    ("Retention", (8, 2), (2,), {}),
    ("RWKVBlock", (8, 12), (12,), {}),
    ("HyenaOperator", (8, 6), (6,), {}),
]


@pytest.mark.parametrize("name,t_args,k_args,k_kwargs", SEQUENCE, ids=[s[0] for s in SEQUENCE])
def test_sequence_layers_compute_the_same_function(name, t_args, k_args, k_kwargs):
    torch.manual_seed(0)
    reference = _torch(name)(*t_args).eval()
    layer = _keras(name)[name](*k_args, **k_kwargs)
    x = np.random.default_rng(0).standard_normal((2, 21, 8)).astype("float32")
    layer(ops.convert_to_tensor(x))                                  # builds the weights
    weights = {w.name: w for w in layer.weights}
    params = {n.replace(".", "_"): p for n, p in reference.named_parameters()}
    assert set(weights) == set(params)
    for n, p in params.items():
        weights[n].assign(p.detach().numpy())
    with torch.no_grad():
        want = reference(torch.from_numpy(x)).numpy()
    got = ops.convert_to_numpy(layer(ops.convert_to_tensor(x)))
    assert np.abs(got - want).max() <= 1e-4 * (np.abs(want).max() + 1e-9)


VISION = [  # (helper, PyTorch class, PyTorch args, Keras args)
    ("uib", "UIB", (8, 8, 4, 3, 3, 1), (8, 4, 3, 3, 1)),
    ("uib", "UIB", (8, 16, 4, 0, 3, 2), (16, 4, 0, 3, 2)),
    ("uib", "UIB", (8, 16, 4, 5, 0, 2), (16, 4, 5, 0, 2)),
    ("uib", "UIB", (8, 4, 4, 0, 0, 2), (4, 4, 0, 0, 2)),
    ("star_block", "StarBlock", (8, 3), (3,)),
    ("involution", "Involution", (8, 3, 2, 4), (3, 2, 4)),
]


@pytest.mark.parametrize("helper,cls,t_args,k_args", VISION,
                         ids=[f"{v[0]}-{i}" for i, v in enumerate(VISION)])
def test_vision_blocks_match_parameters_and_shapes(helper, cls, t_args, k_args):
    reference = _torch(cls)(*t_args).eval()
    inp = keras.Input((8, 8, 8))
    out = _keras(helper)[helper](inp, *k_args)
    model = keras.Model(inp, out)
    assert sum(p.numel() for p in reference.parameters()) == sum(
        int(np.prod(w.shape)) for w in model.trainable_weights)
    with torch.no_grad():
        c, h, w = reference(torch.zeros(1, 8, 8, 8)).shape[1:]
    assert tuple(out.shape[1:]) == (h, w, c)


def test_involution_values_match_with_converted_weights():
    torch.manual_seed(0)
    reference = _torch("Involution")(8, 3, 2, 4).eval()
    bn = reference.reduce[1]
    with torch.no_grad():
        bn.running_mean.uniform_(-0.5, 0.5)
        bn.running_var.uniform_(0.5, 1.5)
    inp = keras.Input((6, 7, 8))
    model = keras.Model(inp, _keras("involution")["involution"](inp, 3, 2, 4))
    convs = [lay for lay in model.layers if isinstance(lay, layers.Conv2D)]
    norm = next(lay for lay in model.layers if isinstance(lay, layers.BatchNormalization))
    convs[0].set_weights([reference.reduce[0].weight.detach().permute(2, 3, 1, 0).numpy()])
    norm.set_weights([t.detach().numpy() for t in (bn.weight, bn.bias, bn.running_mean,
                                                   bn.running_var)])
    convs[1].set_weights([reference.span.weight.detach().permute(2, 3, 1, 0).numpy(),
                          reference.span.bias.detach().numpy()])
    x = np.random.default_rng(1).standard_normal((2, 8, 6, 7)).astype("float32")
    with torch.no_grad():
        want = reference(torch.from_numpy(x)).numpy()
    got = ops.convert_to_numpy(model(np.transpose(x, (0, 2, 3, 1)), training=False))
    assert np.abs(np.transpose(got, (0, 3, 1, 2)) - want).max() < 1e-5
