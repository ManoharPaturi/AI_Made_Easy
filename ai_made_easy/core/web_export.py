"""Single-file HTML web demo export.

Converts a trained torch model (our block set: Conv/Dense/ReLU/Flatten/
pool/BatchNorm) into a tfjs-layers model whose weights ride along as
base64, plus a drag-drop page a kid can double-click — fully offline, no
server, no fetch (that's why the vendored tf.min.js is inlined).

Pure string assembly; torch is imported lazily by layers_from_torch only.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

_ASSETS = Path(__file__).parent.parent / "assets"


def _b64(tensor) -> str:
    import numpy as np

    return base64.b64encode(
        np.ascontiguousarray(tensor.detach().cpu().numpy(),
                             dtype="float32").tobytes()).decode("ascii")


def layers_from_torch(model) -> list[dict]:
    """nn.Module (our block set) → tfjs layer descriptors with weights."""
    import torch.nn as nn

    out: list[dict] = []

    def conv_pool_padding(pad):
        return list(pad) if pad else "valid"

    def pair(v):
        return [v, v] if isinstance(v, int) else list(v)

    for m in model.children():
        if isinstance(m, nn.Conv2d):
            cfg = {"filters": m.out_channels,
                   "kernelSize": pair(m.kernel_size),
                   "strides": pair(m.stride),
                   "padding": conv_pool_padding(m.padding),
                   "activation": "linear",
                   "useBias": m.bias is not None}
            weights = [{"shape": [m.kernel_size[0], m.kernel_size[1],
                                  m.in_channels, m.out_channels],
                        "b64": _b64(m.weight.permute(2, 3, 1, 0))}]
            if m.bias is not None:
                weights.append({"shape": [m.out_channels],
                                "b64": _b64(m.bias)})
            out.append({"type": "conv2d", "config": cfg, "weights": weights})
        elif isinstance(m, nn.Linear):
            cfg = {"units": m.out_features, "activation": "linear",
                   "useBias": m.bias is not None}
            weights = [{"shape": [m.in_features, m.out_features],
                        "b64": _b64(m.weight.T)}]
            if m.bias is not None:
                weights.append({"shape": [m.out_features],
                                "b64": _b64(m.bias)})
            out.append({"type": "dense", "config": cfg, "weights": weights})
        elif isinstance(m, (nn.ReLU, nn.GELU, nn.Sigmoid, nn.Tanh)):
            act = {nn.ReLU: "relu", nn.GELU: "gelu", nn.Sigmoid: "sigmoid",
                   nn.Tanh: "tanh"}[type(m)]
            out.append({"type": "activation",
                        "config": {"activation": act}, "weights": []})
        elif isinstance(m, nn.Flatten):
            out.append({"type": "flatten", "config": {}, "weights": []})
        elif isinstance(m, nn.MaxPool2d):
            out.append({"type": "maxpool2d", "config": {
                "poolSize": pair(m.kernel_size),
                "strides": pair(m.stride),
                "padding": conv_pool_padding(m.padding)}, "weights": []})
        elif isinstance(m, nn.AvgPool2d):
            out.append({"type": "avgpool2d", "config": {
                "poolSize": pair(m.kernel_size),
                "strides": pair(m.stride),
                "padding": conv_pool_padding(m.padding)}, "weights": []})
        elif isinstance(m, nn.BatchNorm2d):
            cfg = {"axis": 3, "epsilon": m.eps,
                   "scale": m.weight is not None,
                   "center": m.bias is not None}
            weights = [
                {"shape": [m.num_features], "b64": _b64(m.weight)},
                {"shape": [m.num_features], "b64": _b64(m.bias)},
                {"shape": [m.num_features], "b64": _b64(m.running_mean)},
                {"shape": [m.num_features], "b64": _b64(m.running_var)},
            ]
            out.append({"type": "batchnorm2d", "config": cfg,
                        "weights": weights})
        elif isinstance(m, nn.Dropout):
            out.append({"type": "dropout", "config": {"rate": m.p},
                        "weights": []})
        elif isinstance(m, nn.AdaptiveAvgPool2d):
            out.append({"type": "globalavgpool2d", "config": {},
                        "weights": []})
        else:
            raise RuntimeError(
                f"{m.__class__.__name__} isn't supported in the web demo "
                "yet — try a model built from Conv, Dense, ReLU, Pool, "
                "Flatten, BatchNorm")
    return out


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>@@TITLE@@ — AI Made Easy web demo</title>
<style>
 body { font-family: -apple-system, "Comic Sans MS", sans-serif;
        background: #EDEEE8; color: #33312B; margin: 0;
        display: flex; flex-direction: column; align-items: center; }
 h1 { margin: 18px 0 4px; }
 #drop { width: 420px; height: 240px; border: 4px dashed #74C0FC;
         border-radius: 20px; display: flex; align-items: center;
         justify-content: center; text-align: center; color: #7A7565;
         font-size: 20px; margin: 14px; background: #FFFDF8; }
 #drop.ready { border-color: #63E6BE; background: #F0FBF5; }
 #out { width: 440px; background: #FFFDF8; border-radius: 16px;
        padding: 12px 18px; margin-bottom: 24px; }
 .bar { height: 22px; border-radius: 8px; background: #EFECE2;
        margin: 5px 0; position: relative; overflow: hidden; }
 .bar > div { height: 100%; background: #63E6BE; border-radius: 8px;
              transition: width .3s; }
 .bar > span { position: absolute; left: 8px; top: 2px; font-size: 13px;
              font-weight: 700; }
 #status { color: #7A7565; min-height: 22px; }
 img#thumb { max-width: 200px; border-radius: 12px; margin-top: 8px; }
</style>
</head>
<body>
<h1>🧩 @@TITLE@@</h1>
<div id="status">loading the model…</div>
<div id="drop">📥 drop a picture here<br>(or click to choose)</div>
<img id="thumb" hidden>
<div id="out" hidden></div>
<input type="file" id="file" accept="image/*" hidden>
<script>@@TFJS@@</script>
<script>
const LAYERS = @@LAYERS@@;
const CLASSES = @@CLASSES@@;
const INPUT = @@INPUT@@;
const NORM = @@NORM@@;
function b64bytes(b64) {
  const bin = atob(b64), n = bin.length, buf = new ArrayBuffer(n),
        view = new Uint8Array(buf);
  for (let i = 0; i < n; i++) view[i] = bin.charCodeAt(i);
  return buf;
}
async function buildModel() {
  const model = tf.sequential();
  let first = true;
  for (const layer of LAYERS) {
    const cfg = Object.assign({}, layer.config);
    if (first && layer.type !== "activation" && layer.type !== "flatten"
        && layer.type !== "dropout" && layer.type !== "globalavgpool2d") {
      cfg.inputShape = INPUT;
    }
    let l;
    switch (layer.type) {
      case "conv2d":       l = tf.layers.conv2d(cfg); break;
      case "dense":        l = tf.layers.dense(cfg); break;
      case "activation":   l = tf.layers.activation(cfg); break;
      case "flatten":      l = tf.layers.flatten(); break;
      case "maxpool2d":    l = tf.layers.maxPooling2d(cfg); break;
      case "avgpool2d":    l = tf.layers.averagePooling2d(cfg); break;
      case "batchnorm2d":  l = tf.layers.batchNormalization(cfg); break;
      case "dropout":      l = tf.layers.dropout(cfg); break;
      case "globalavgpool2d": l = tf.layers.globalAveragePooling2d({}); break;
    }
    model.add(l);
    if (layer.weights.length) {
      const tensors = layer.weights.map(w => tf.tensor(
        new Float32Array(b64bytes(w.b64)), w.shape));
      l.setWeights(tensors);
    }
    first = false;
  }
  return model;
}
let MODEL = null;
async function ready() {
  try {
    MODEL = await buildModel();
    const probe = MODEL.predict(tf.zeros([1].concat(INPUT)));
    probe.dispose();
    document.getElementById("status").textContent =
      "✅ model ready — drop a picture!";
    if (new URLSearchParams(location.search).has("selftest")) {
      document.getElementById("status").textContent =
        "SELFTEST OK — model built and ran";
    }
  } catch (err) {
    document.getElementById("status").textContent = "😵 " + err;
  }
}
ready();
function predict(img) {
  const c = document.createElement("canvas");
  c.width = INPUT[2]; c.height = INPUT[1];
  const g = c.getContext("2d");
  g.drawImage(img, 0, 0, c.width, c.height);
  let x;
  if (INPUT[0] === 1) x = tf.tensor(grayscale(g, c), [c.height, c.width, 1]);
  else x = tf.browser.fromPixels(c).toFloat();
  x = x.div(255.0);
  if (NORM) {
    const mean = tf.tensor(NORM[0]).reshape([INPUT[0], 1, 1]);
    const std = tf.tensor(NORM[1]).reshape([INPUT[0], 1, 1]);
    x = x.sub(mean).div(std);
  }
  const batched = x.reshape([1, c.height, c.width, INPUT[0]]);
  const probs = MODEL.predict(batched).dataSync();
  renderBars(probs);
  batched.dispose();
}
function grayscale(g, c) {
  const d = g.getImageData(0, 0, c.width, c.height);
  const out = new Float32Array(c.width * c.height);
  for (let i = 0, j = 0; i < d.data.length; i += 4, j++) {
    out[j] = (d.data[i] + d.data[i + 1] + d.data[i + 2]) / (3 * 255);
  }
  return out;
}
function renderBars(probs) {
  const host = document.getElementById("out");
  host.hidden = false;
  host.innerHTML = "";
  const order = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]);
  for (const i of order.slice(0, 6)) {
    const name = CLASSES[i] || ("class " + i);
    const row = document.createElement("div");
    row.className = "bar";
    const fill = document.createElement("div");
    fill.style.width = (probs[i] * 100).toFixed(1) + "%";
    const label = document.createElement("span");
    label.textContent = name + " — " + (probs[i] * 100).toFixed(0) + "%";
    if (i === order[0]) fill.style.background = "#F5D547";
    row.appendChild(fill); row.appendChild(label);
    host.appendChild(row);
  }
}
const drop = document.getElementById("drop");
const file = document.getElementById("file");
drop.onclick = () => file.click();
drop.ondragover = e => { e.preventDefault(); drop.classList.add("ready"); };
drop.ondragleave = () => drop.classList.remove("ready");
drop.ondrop = e => { e.preventDefault(); handle(e.dataTransfer.files[0]); };
file.onchange = () => handle(file.files[0]);
function handle(f) {
  if (!f || !MODEL) return;
  const img = new Image();
  img.onload = () => {
    const thumb = document.getElementById("thumb");
    thumb.src = img.src; thumb.hidden = false;
    predict(img);
  };
  img.src = URL.createObjectURL(f);
}
</script>
</body>
</html>
"""


def build_web_demo(layers: list[dict], classes: list[str],
                   input_shape: list[int], norm=None, title="My model",
                   tfjs_path: Path | None = None) -> str:
    """Assemble the single-file HTML demo (token replacement — the vendored
    tfjs bundle contains braces, so str.format is off limits)."""
    tfjs = (tfjs_path or _ASSETS / "tf.min.js").read_text()
    html = _HTML_TEMPLATE
    for token, value in (
            ("@@TITLE@@", title),
            ("@@TFJS@@", tfjs),
            ("@@LAYERS@@", json.dumps(layers)),
            ("@@CLASSES@@", json.dumps(classes)),
            ("@@INPUT@@", json.dumps(list(input_shape))),
            ("@@NORM@@", json.dumps(norm) if norm else "null")):
        html = html.replace(token, value, 1)
    return html
