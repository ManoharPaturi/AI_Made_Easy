"""Composite architecture blocks — macros that expand into primitive blocks.

Each builder receives the composite's resolved params and returns a Fragment
(nodes/edges/entry/exit). Positions are laid out left-to-right; branches
get vertical offsets. Expansion is a canvas operation (splice); codegen
requires expanded graphs.
"""
from __future__ import annotations

from ai_made_easy.core.composites import Fragment
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import BlockDefinition, ParamSpec, PortSpec, parse_int_list
from ai_made_easy.core.blocks._palette import family_color

reg = get_registry()

_ARCH_COLOR = family_color("model")

_X_STEP = 200.0


def _node(nid: str, type_id: str, params: dict, x: float, y: float) -> dict:
    return {"id": nid, "type": type_id, "params": params, "position": (x, y)}


def _chain(frag: Fragment, specs: list[tuple[str, str, dict]], start_x=0.0, y=0.0) -> None:
    """Append a linear chain; entry stays as-is, exit becomes the last node."""
    x = start_x
    prev = frag.exit or frag.entry
    for nid, type_id, params in specs:
        frag.nodes.append(_node(nid, type_id, params, x, y))
        if prev != nid:  # first node may already be the entry
            frag.edges.append((prev, "out", nid, "in"))
        prev = nid
        x += _X_STEP
    frag.exit = prev


def _dense(u: int) -> tuple[str, str, dict]:
    return "", "core.dense", {"units": u, "bias": True}


def _conv(cout: int, k: int, s = 1, p = None) -> dict:
    return {"out_channels": cout, "kernel_size": k, "stride": s,
            "padding": (k // 2) if p is None else p, "dilation": 1}


# ------------------------------------------------------------------- MLP

def _build_mlp(params: dict) -> Fragment:
    hidden = parse_int_list(params["hidden"])
    frag = Fragment(entry="d0")
    chain: list[tuple[str, str, dict]] = []
    for i, u in enumerate(hidden):
        chain.append((f"d{i}", "core.dense", {"units": u, "bias": True}))
        chain.append((f"r{i}", "core.relu", {}))
    chain.append(("dout", "core.dense", {"units": int(params["num_classes"]), "bias": True}))
    _chain(frag, chain)
    frag.entry = "d0"
    return frag


# ------------------------------------------------------------------- CNN

def _build_cnn(params: dict) -> Fragment:
    base = int(params["base_channels"])
    frag = Fragment(entry="c1")
    _chain(frag, [
        ("c1", "core.conv2d", _conv(base, 3)),
        ("r1", "core.relu", {}),
        ("p1", "core.maxpool2d", {"kernel_size": 2, "stride": 2, "padding": 0}),
        ("c2", "core.conv2d", _conv(base * 2, 3)),
        ("r2", "core.relu", {}),
        ("p2", "core.maxpool2d", {"kernel_size": 2, "stride": 2, "padding": 0}),
        ("gap", "core.global_avgpool2d", {}),
        ("dout", "core.dense", {"units": int(params["num_classes"]), "bias": True}),
    ])
    return frag


# ------------------------------------------------------------------- VGG

def _build_vgg(params: dict) -> Fragment:
    base, blocks = int(params["base_channels"]), int(params["blocks"])
    frag = Fragment(entry="c0_0")
    x, y, prev = 0.0, 0.0, None
    for b in range(blocks):
        ch = base * (2 ** b)
        for rep in range(2):
            nid = f"c{b}_{rep}"
            frag.nodes.append(_node(nid, "core.conv2d", _conv(ch, 3), x, y))
            if prev is not None:
                frag.edges.append((prev, "out", nid, "in"))
            else:
                frag.entry = nid
            prev = nid
            x += _X_STEP
            rid = f"r{b}_{rep}"
            frag.nodes.append(_node(rid, "core.relu", {}, x, y))
            frag.edges.append((prev, "out", rid, "in"))
            prev = rid
            x += _X_STEP
        pid = f"p{b}"
        frag.nodes.append(_node(pid, "core.maxpool2d",
                                {"kernel_size": 2, "stride": 2, "padding": 0}, x, y))
        frag.edges.append((prev, "out", pid, "in"))
        prev = pid
        x += _X_STEP
    for nid, t, p in (("gap", "core.global_avgpool2d", {}),
                      ("dout", "core.dense",
                       {"units": int(params["num_classes"]), "bias": True})):
        frag.nodes.append(_node(nid, t, p, x, y))
        frag.edges.append((prev, "out", nid, "in"))
        prev = nid
        x += _X_STEP
    frag.exit = prev
    return frag


# ----------------------------------------------------------------- ResNet

def _basic_block(frag: Fragment, tag: str, cin: int, cout: int, stride: int,
                 x: float, y: float) -> float:
    """One ResNet basic block: [conv-bn-relu-conv-bn] + skip -> Add -> relu.
    Projection (1x1 conv) on the skip when the shape changes."""
    nodes = [
        (f"{tag}_c1", "core.conv2d", _conv(cout, 3, stride)),
        (f"{tag}_bn1", "core.batch_norm2d", {"epsilon": 1e-5, "momentum": 0.1}),
        (f"{tag}_r1", "core.relu", {}),
        (f"{tag}_c2", "core.conv2d", _conv(cout, 3)),
        (f"{tag}_bn2", "core.batch_norm2d", {"epsilon": 1e-5, "momentum": 0.1}),
    ]
    xi = x
    prev = frag.exit
    for nid, t, p in nodes:
        frag.nodes.append(_node(nid, t, p, xi, y))
        frag.edges.append((prev, "out", nid, "in"))
        prev = nid
        xi += _X_STEP
    if stride != 1 or cin != cout:
        frag.nodes.append(_node(f"{tag}_proj", "core.conv2d",
                                _conv(cout, 1, stride, p=0), xi, y + 150.0))
        frag.edges.append((frag.exit, "out", f"{tag}_proj", "in"))
        skip_src = f"{tag}_proj"
    else:
        skip_src = frag.exit
    frag.nodes.append(_node(f"{tag}_add", "core.add", {}, xi, y))
    frag.edges.append((prev, "out", f"{tag}_add", "in1"))
    frag.edges.append((skip_src, "out", f"{tag}_add", "in2"))
    frag.nodes.append(_node(f"{tag}_r2", "core.relu", {}, xi + _X_STEP, y))
    frag.edges.append((f"{tag}_add", "out", f"{tag}_r2", "in"))
    frag.exit = f"{tag}_r2"
    return xi + 2 * _X_STEP


def _build_resnet18(params: dict) -> Fragment:
    base, num_classes = int(params["base_channels"]), int(params["num_classes"])
    frag = Fragment(entry="stem_conv")
    frag.nodes.append(_node("stem_conv", "core.conv2d", _conv(base, 7, 2, 3), 0.0, 0.0))
    prev = "stem_conv"
    for nid, t, p in (
        ("stem_bn", "core.batch_norm2d", {"epsilon": 1e-5, "momentum": 0.1}),
        ("stem_relu", "core.relu", {}),
        ("stem_pool", "core.maxpool2d", {"kernel_size": 3, "stride": 2, "padding": 1}),
    ):
        frag.nodes.append(_node(nid, t, p, _x(frag, nid), 0.0))
        frag.edges.append((prev, "out", nid, "in"))
        prev = nid
    frag.exit = prev
    x = _X_STEP * 4
    channels = base
    for stage in range(4):
        for block in range(2):
            out_ch = channels * (2 if (stage > 0 and block == 0) else 1)
            stride = 2 if (stage > 0 and block == 0) else 1
            x = _basic_block(frag, f"s{stage}b{block}", channels, out_ch, stride, x, 0.0)
            channels = out_ch
    frag.nodes.append(_node("gap", "core.global_avgpool2d", {}, x, 0.0))
    frag.edges.append((frag.exit, "out", "gap", "in"))
    frag.nodes.append(_node("dout", "core.dense",
                            {"units": num_classes, "bias": True}, x + _X_STEP, 0.0))
    frag.edges.append(("gap", "out", "dout", "in"))
    frag.exit = "dout"
    return frag


def _x(frag: Fragment, _: str) -> float:
    return _X_STEP * len(frag.nodes)


# ------------------------------------------------------------------ UNet

def _build_unet(params: dict) -> Fragment:
    base, out_ch = int(params["base_channels"]), int(params["out_channels"])
    frag = Fragment(entry="e0_c0")
    x, prev = 0.0, None
    skips: list[str] = []
    # encoder: [conv-relu x2] -> pool, doubling channels
    for level in range(3):
        ch = base * (2 ** level)
        for rep in range(2):
            nid = f"e{level}_c{rep}"
            frag.nodes.append(_node(nid, "core.conv2d", _conv(ch, 3), x, 0.0))
            if prev is None:
                frag.entry = nid
            else:
                frag.edges.append((prev, "out", nid, "in"))
            prev = nid
            x += _X_STEP
            rid = f"e{level}_r{rep}"
            frag.nodes.append(_node(rid, "core.relu", {}, x, 0.0))
            frag.edges.append((prev, "out", rid, "in"))
            prev = rid
            x += _X_STEP
        skips.append(prev)
        if level < 2:
            pid = f"pool{level}"
            frag.nodes.append(_node(pid, "core.maxpool2d",
                                    {"kernel_size": 2, "stride": 2, "padding": 0}, x, 0.0))
            frag.edges.append((prev, "out", pid, "in"))
            prev = pid
            x += _X_STEP
    # decoder: upsample -> concat(skip) -> conv-relu x2 (one up per encoder pool)
    for level in (1, 0):
        ch = base * (2 ** level)
        up = f"up{level}"
        frag.nodes.append(_node(up, "core.conv_transpose2d",
                                {"out_channels": ch, "kernel_size": 2, "stride": 2,
                                 "padding": 0, "dilation": 1, "output_padding": 0},
                                x, 0.0))
        frag.edges.append((prev, "out", up, "in"))
        prev = up
        x += _X_STEP
        cat = f"cat{level}"
        frag.nodes.append(_node(cat, "core.concatenate", {"axis": 0}, x, 0.0))
        frag.edges.append((prev, "out", cat, "in1"))
        frag.edges.append((skips[level], "out", cat, "in2"))
        prev = cat
        x += _X_STEP
        for rep in range(2):
            nid = f"d{level}_c{rep}"
            frag.nodes.append(_node(nid, "core.conv2d", _conv(ch, 3), x, 0.0))
            frag.edges.append((prev, "out", nid, "in"))
            prev = nid
            x += _X_STEP
            rid = f"d{level}_r{rep}"
            frag.nodes.append(_node(rid, "core.relu", {}, x, 0.0))
            frag.edges.append((prev, "out", rid, "in"))
            prev = rid
            x += _X_STEP
    frag.nodes.append(_node("dout", "core.conv2d", _conv(out_ch, 1, 1, 0), x, 0.0))
    frag.edges.append((prev, "out", "dout", "in"))
    frag.exit = "dout"
    return frag


# ---------------------------------------------------------- Autoencoder

def _build_autoencoder(params: dict) -> Fragment:
    hidden = parse_int_list(params["hidden"])
    latent, input_dim = int(params["latent_dim"]), int(params["input_dim"])
    frag = Fragment(entry="e0")
    enc: list[tuple[str, str, dict]] = []
    for i, u in enumerate(hidden):
        enc += [(f"e{i}", "core.dense", {"units": u, "bias": True}),
                (f"er{i}", "core.relu", {})]
    enc.append(("lat", "core.dense", {"units": latent, "bias": True}))
    dec: list[tuple[str, str, dict]] = [("ldr", "core.relu", {})]
    for i, u in enumerate(reversed(hidden)):
        dec += [(f"d{i}", "core.dense", {"units": u, "bias": True}),
                (f"dr{i}", "core.relu", {})]
    dec.append(("dout", "core.dense", {"units": input_dim, "bias": True}))
    _chain(frag, enc + dec)
    frag.entry = "e0"
    return frag


# ------------------------------------------------------ LSTM classifier

def _build_lstm_classifier(params: dict) -> Fragment:
    hidden, layers = int(params["hidden_size"]), int(params["num_layers"])
    frag = Fragment(entry="lstm")
    _chain(frag, [
        ("lstm", "core.lstm", {"hidden_size": hidden, "num_layers": layers,
                                "bias": True, "bidirectional": False}),
        ("drop", "core.dropout", {"p": float(params["dropout"])}),
        ("mot", "core.mean_over_time", {}),
        ("dout", "core.dense", {"units": int(params["num_classes"]), "bias": True}),
    ])
    return frag


# ------------------------------------------------ Transformer classifier

def _build_transformer_classifier(params: dict) -> Fragment:
    """[L, C] -> encoder stack (d_model auto = input channels) -> pool -> head."""
    nhead = int(params["nhead"])
    ffn, layers = int(params["dim_feedforward"]), int(params["num_layers"])
    frag = Fragment(entry="te0")
    for i in range(layers):
        frag.nodes.append(_node(f"te{i}", "core.transformer_encoder",
                                {"nhead": nhead, "dim_feedforward": ffn,
                                 "dropout": 0.1}, _X_STEP * i, 0.0))
        if i:
            frag.edges.append((frag.exit, "out", f"te{i}", "in"))
        frag.exit = f"te{i}"
    _chain(frag, [
        ("mot", "core.mean_over_time", {}),
        ("dout", "core.dense", {"units": int(params["num_classes"]), "bias": True}),
    ], start_x=_X_STEP * layers)
    return frag


# ------------------------------------------------------------- registry

def _arch(type_id: str, name: str, params: tuple, builder) -> None:
    reg.register(BlockDefinition(
        type_id=type_id,
        display_name=name,
        category="Architectures",
        color=_ARCH_COLOR,
        params=params,
        inputs=(PortSpec("in"),),
        outputs=(PortSpec("out"),),
        builder=builder,
    ))


_num_classes = ParamSpec(name="num_classes", type="int", default=10, minimum=1)

_arch("arch.mlp", "MLP", (
    ParamSpec(name="hidden", type="str", default="128, 64",
              help="Comma-separated hidden layer sizes"),
    _num_classes,
), _build_mlp)

_arch("arch.cnn", "CNN", (
    ParamSpec(name="base_channels", type="int", default=16, minimum=1),
    _num_classes,
), _build_cnn)

_arch("arch.vgg", "VGG-style CNN", (
    ParamSpec(name="base_channels", type="int", default=32, minimum=1),
    ParamSpec(name="blocks", type="int", default=3, minimum=1, maximum=5),
    _num_classes,
), _build_vgg)

_arch("arch.resnet18", "ResNet-18", (
    ParamSpec(name="base_channels", type="int", default=64, minimum=1),
    _num_classes,
), _build_resnet18)

_arch("arch.unet", "UNet", (
    ParamSpec(name="base_channels", type="int", default=16, minimum=1),
    ParamSpec(name="out_channels", type="int", default=1, minimum=1,
              help="Segmentation classes"),
), _build_unet)

_arch("arch.autoencoder", "Autoencoder", (
    ParamSpec(name="input_dim", type="int", default=784, minimum=1),
    ParamSpec(name="hidden", type="str", default="256, 128"),
    ParamSpec(name="latent_dim", type="int", default=32, minimum=1),
), _build_autoencoder)

_arch("arch.lstm_classifier", "LSTM Classifier", (
    ParamSpec(name="hidden_size", type="int", default=64, minimum=1),
    ParamSpec(name="num_layers", type="int", default=1, minimum=1),
    ParamSpec(name="dropout", type="float", default=0.2,
              minimum=0.0, maximum=1.0),
    _num_classes,
), _build_lstm_classifier)

_arch("arch.transformer_classifier", "Transformer Classifier", (
    ParamSpec(name="nhead", type="int", default=4, minimum=1,
              help="Must divide the input channels"),
    ParamSpec(name="dim_feedforward", type="int", default=128, minimum=1),
    ParamSpec(name="num_layers", type="int", default=2, minimum=1),
    _num_classes,
), _build_transformer_classifier)
