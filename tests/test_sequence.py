"""Sequence blocks (TCN, Mamba, S4D, sLSTM) and the in-model audio front-end."""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from ai_made_easy.core import budget
from ai_made_easy.core.graph import Graph

torch = pytest.importorskip("torch")


def single(type_id: str, shape: str, params: dict | None = None) -> Graph:
    return Graph.from_dict({"name": "seq_case", "nodes": [
        {"id": "in", "type": "core.input", "params": {"shape": shape}, "position": [0, 0]},
        {"id": "b", "type": type_id, "params": params or {}, "position": [0, 0]},
        {"id": "out", "type": "core.output", "params": {}, "position": [0, 0]}],
        "edges": [{"from": "in/out", "to": "b/in"}, {"from": "b/out", "to": "out/in"}]})


def build(graph: Graph):
    from ai_made_easy.core.codegen import class_name_for, generate

    ns: dict = {"__name__": "aime_generated"}
    exec(compile(generate(graph, "pytorch"), "<seq>", "exec"), ns)  # noqa: S102
    return ns[class_name_for(graph.name)]().eval()


@pytest.mark.parametrize("type_id, params", [
    ("seq.tcn", {"channels": 8, "levels": 3, "kernel_size": 3}),
    ("seq.tcn", {"channels": 6, "levels": 2, "kernel_size": 5}),
    ("seq.mamba", {"d_state": 8, "d_conv": 4, "expand": 2}),
    ("seq.s4d", {"d_state": 16}),
    ("seq.xlstm", {"hidden_size": 12, "num_layers": 2}),
    ("seq.xlstm", {"hidden_size": 12, "return_sequences": False}),
])
def test_sequence_blocks_match_torch(type_id, params):
    graph = single(type_id, "20, 5", params)
    assert not [i for i in graph.validate() if i.severity == "error"]
    net = build(graph)
    with torch.no_grad():
        out = net(torch.randn(2, 20, 5))
    assert list(out.shape[1:]) == list(graph.infer_shapes()["b"])
    assert budget.estimate(graph).params == sum(p.numel() for p in net.parameters())


@pytest.mark.parametrize("type_id, params", [
    ("seq.tcn", {"channels": 8, "levels": 3}), ("seq.mamba", {}), ("seq.xlstm", {}),
])
def test_sequence_blocks_are_causal(type_id, params):
    net = build(single(type_id, "16, 3", params))
    x = torch.randn(1, 16, 3)
    y = x.clone()
    y[:, 10:] = torch.randn(1, 6, 3)   # change only the future
    with torch.no_grad():
        a, b = net(x), net(y)
    assert torch.allclose(a[:, :10], b[:, :10], atol=1e-5)
    assert not torch.allclose(a[:, 10:], b[:, 10:])


def test_tcn_receptive_field():
    from ai_made_easy.core.sequence.helpers import tcn_receptive_field

    net = build(single("seq.tcn", "64, 1", {"channels": 4, "levels": 3, "kernel_size": 3,
                                           "dropout": 0.0}))
    torch.manual_seed(0)
    seen = torch.zeros(64, dtype=torch.bool)
    for _ in range(8):   # union over inputs: a ReLU can block a path for one input
        x = torch.randn(1, 64, 1, requires_grad=True)
        net(x)[0, -1].sum().backward()
        seen |= x.grad[0, :, 0] != 0
    assert int(seen.sum()) == tcn_receptive_field(3, 3) == 29
    assert seen[-29:].all()   # exactly the last 29 steps


def test_mel_filterbank_matches_reference():
    from ai_made_easy.core.sequence import helpers as h

    ns = {"torch": torch, "nn": torch.nn}
    exec(h.AUDIO_FRONTEND, ns)  # noqa: S102
    fb = ns["mel_filterbank"](400, 40, 16000).numpy()
    assert fb.shape == (40, 201) and np.all(fb >= 0)
    assert (fb.sum(1) > 0).all()     # no empty filters
    peaks = fb.argmax(1)
    assert np.all(np.diff(peaks) >= 0)   # filters move up in frequency
    # HTK triangles built independently (the formula torchaudio.melscale_fbanks uses)
    mel = np.linspace(0, 2595 * np.log10(1 + 8000 / 700), 42)
    hz = 700 * (10 ** (mel / 2595) - 1)
    freqs = np.linspace(0, 8000, 201)
    down = (freqs[None] - hz[:-2, None]) / (hz[1:-1] - hz[:-2])[:, None]
    up = (hz[2:, None] - freqs[None]) / (hz[2:] - hz[1:-1])[:, None]
    np.testing.assert_allclose(fb, np.maximum(0, np.minimum(down, up)), atol=1e-5)


def test_mel_layouts_and_pitch():
    image = build(single("audio.mel_spectrogram", "1, 16000", {"n_mels": 40}))
    seq = build(single("audio.mel_spectrogram", "1, 16000",
                       {"n_mels": 40, "layout": "sequence [frames, bins]"}))
    t = torch.arange(16000) / 16000
    low, high = torch.sin(2 * np.pi * 300 * t), torch.sin(2 * np.pi * 3000 * t)
    with torch.no_grad():
        a = image(low.reshape(1, 1, -1))
        b = seq(low.reshape(1, 1, -1))
        c = image(high.reshape(1, 1, -1))
    assert tuple(a.shape) == (1, 1, 40, 101) and tuple(b.shape) == (1, 101, 40)
    torch.testing.assert_close(a[0, 0].T, b[0])
    assert a[0, 0].mean(1).argmax() < c[0, 0].mean(1).argmax()   # higher tone, higher band


def test_mfcc_shape_and_rule():
    graph = single("audio.mfcc", "1, 8000", {"n_mfcc": 13})
    net = build(graph)
    with torch.no_grad():
        assert tuple(net(torch.randn(2, 1, 8000)).shape[1:]) == (1, 13, 51)
    bad = single("audio.mfcc", "1, 8000", {"n_mfcc": 80, "n_mels": 40})
    assert any("n_mfcc cannot exceed n_mels" in i.message for i in bad.validate())


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None,
                    reason="needs transformers (audio extra)")
def test_speech_encoder_frames_without_downloads():
    graph = single("audio.hf_encoder", "1, 16000",
                   {"model_id": "facebook/wav2vec2-base", "weights": "none", "pooling": "none"})
    net = build(graph)
    with torch.no_grad():
        out = net(torch.randn(1, 1, 16000))
    assert list(out.shape[1:]) == list(graph.infer_shapes()["b"]) == [49, 768]
