"""End-to-end check: build datasets on disk, generate training scripts for every
modality / task, run them, and assert they train and report metrics.

Usage: python scripts/verify_training.py [-k name-substring] [--keras] [-v]
Importable: ``SCENARIOS`` and ``run_scenario`` power tests/test_training_e2e.py.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_made_easy.core.codegen.training_gen import generate_training  # noqa: E402
from ai_made_easy.core.graph import Edge, Graph, NodeInstance  # noqa: E402


@dataclass
class Scenario:
    name: str
    build: Callable[[Path], Graph]
    frameworks: tuple[str, ...] = ("pytorch", "keras")
    expect: tuple[str, ...] = ()           # metric keys that must appear in the test line
    tags: set[str] = field(default_factory=set)


def chain(name: str, input_shape: str, layers: list[tuple[str, dict]],
          config: list[tuple[str, dict]], dtype: str = "float32") -> Graph:
    g = Graph(name=name)
    g.add_node(NodeInstance("in", "core.input", {"shape": input_shape, "dtype": dtype}))
    prev = "in"
    for i, (tid, params) in enumerate(layers):
        nid = f"l{i}"
        g.add_node(NodeInstance(nid, tid, params))
        g.add_edge(Edge(prev, "out", nid, g.nodes[nid].definition().inputs[0].name))
        prev = nid
    g.add_node(NodeInstance("out", "core.output", {}))
    g.add_edge(Edge(prev, "out", "out", "in"))
    for i, (tid, params) in enumerate(config):
        g.add_node(NodeInstance(f"c{i}", tid, params))
    return g


TRAINER = ("train.trainer", {"epochs": 2, "batch_size": 16, "device": "cpu", "seed": 0})
MLP = lambda k: [("core.dense", {"units": 16}), ("core.relu", {}), ("core.dense", {"units": k})]  # noqa: E731


# ------------------------------------------------------------------ datasets

def write_table(tmp: Path) -> Path:
    rng = np.random.default_rng(0)
    n = 160
    color = rng.choice(["red", "green", "blue"], n)
    size = rng.normal(10, 3, n)
    size[rng.random(n) < 0.1] = np.nan
    weight = np.abs(rng.normal(50, 20, n))
    label = np.where((color == "red") | (size > 11), "yes", "no")
    lines = ["color,size,weight,id,label"]
    for i in range(n):
        s = "" if np.isnan(size[i]) else f"{size[i]:.3f}"
        lines.append(f"{color[i]},{s},{weight[i]:.3f},{i},{label[i]}")
    path = tmp / "table.csv"
    path.write_text("\n".join(lines))
    return path


def write_texts(tmp: Path) -> Path:
    rng = np.random.default_rng(1)
    pos = ["great movie", "loved it", "wonderful acting", "brilliant and fun", "superb"]
    neg = ["terrible film", "hated it", "boring plot", "awful and slow", "worst ever"]
    rows = ["text,label"]
    for _ in range(80):
        rows.append(f"\"{rng.choice(pos)}! {rng.choice(pos)}\",pos")
        rows.append(f"\"{rng.choice(neg)}. {rng.choice(neg)}\",neg")
    path = tmp / "reviews.csv"
    path.write_text("\n".join(rows))
    return path


def write_text_folder(tmp: Path) -> Path:
    root = tmp / "texts"
    for cls, words in (("sports", "goal match team score"), ("tech", "code chip data cloud")):
        (root / cls).mkdir(parents=True)
        for i in range(30):
            (root / cls / f"{i}.txt").write_text(" ".join(np.random.default_rng(i).permutation(
                words.split())))
    return root


def write_audio(tmp: Path) -> Path:
    root = tmp / "audio"
    sr = 8000
    for cls, freq in (("low", 220.0), ("high", 1760.0)):
        (root / cls).mkdir(parents=True)
        for i in range(24):
            t = np.arange(int(sr * 0.25)) / sr
            sig = 0.6 * np.sin(2 * np.pi * (freq + 5 * i) * t)
            pcm = (sig * 32767).astype(np.int16)
            with wave.open(str(root / cls / f"{i}.wav"), "wb") as fh:
                fh.setnchannels(1)
                fh.setsampwidth(2)
                fh.setframerate(sr)
                fh.writeframes(pcm.tobytes())
    return root


def write_series(tmp: Path) -> Path:
    t = np.arange(400)
    value = np.sin(t / 10) + 0.1 * np.random.default_rng(2).normal(size=400)
    other = np.cos(t / 15)
    path = tmp / "series.csv"
    path.write_text("t,value,other\n" + "\n".join(
        f"{i},{v:.5f},{o:.5f}" for i, v, o in zip(t, value, other, strict=True)))
    return path


def write_images(tmp: Path) -> Path:
    from PIL import Image

    root = tmp / "images"
    rng = np.random.default_rng(3)
    for cls, channel in (("red", 0), ("blue", 2)):
        (root / cls).mkdir(parents=True)
        for i in range(24):
            arr = (rng.random((20, 20, 3)) * 60).astype(np.uint8)
            arr[..., channel] = 200
            Image.fromarray(arr).save(root / cls / f"{i}.png")
    return root


def write_npz(tmp: Path) -> Path:
    rng = np.random.default_rng(4)
    y = rng.integers(0, 3, 120)
    x = rng.normal(size=(120, 1, 8, 8)).astype(np.float32) + y[:, None, None, None]
    path = tmp / "arrays.npz"
    np.savez(path, x=x, y=y)
    return path


def write_json(tmp: Path) -> Path:
    rng = np.random.default_rng(5)
    path = tmp / "records.jsonl"
    path.write_text("\n".join(json.dumps({"x": list(map(float, rng.normal(size=6) + k)),
                                          "y": int(k)})
                              for k in rng.integers(0, 2, 100)))
    return path


# ----------------------------------------------------------------- scenarios

def _multiclass(tmp):
    return chain("synthetic_multiclass", "20", MLP(4), [
        ("data.synthetic", {"n_features": 20, "n_classes": 4, "n_samples": 300}),
        ("train.loss_cross_entropy", {"label_smoothing": 0.1}),
        ("train.adamw", {"lr": 0.01}), ("train.warmup_cosine_lr", {"warmup_epochs": 1}),
        ("eval.accuracy", {}), ("eval.f1", {"average": "macro"}), ("eval.roc_auc", {}),
        ("eval.top_k_accuracy", {"k": 2}), ("eval.mcc", {}), ("eval.confusion_matrix", {}),
        ("eval.log_loss", {}), ("eval.predict", {"n_samples": 3}), TRAINER])


def _regression(tmp):
    return chain("synthetic_regression", "8", [("core.dense", {"units": 16}), ("core.gelu", {}),
                                               ("core.dense", {"units": 1})], [
        ("data.synthetic", {"kind": "regression", "n_features": 8, "n_samples": 300}),
        ("train.loss_huber", {"delta": 1.0}), ("train.sgd", {"lr": 0.01, "momentum": 0.9}),
        ("train.step_lr", {"step_size": 1, "gamma": 0.5}), ("prep.normalize", {"mode": "fit"}),
        ("eval.mae", {}), ("eval.rmse", {}), ("eval.r2", {}), ("eval.mape", {}), TRAINER])


def _binary(tmp):
    return chain("synthetic_binary", "10", MLP(1), [
        ("data.synthetic", {"n_features": 10, "n_classes": 2, "n_samples": 300}),
        ("train.loss_bce_logits", {}), ("train.adam", {"lr": 0.01}),
        ("prep.class_balance", {"strategy": "class weights"}),
        ("eval.accuracy", {}), ("eval.roc_auc", {}), ("eval.precision", {}),
        ("eval.average_precision", {}), TRAINER])


def _multilabel(tmp):
    return chain("synthetic_multilabel", "12", MLP(5), [
        ("data.synthetic", {"n_features": 12, "n_classes": 5, "n_samples": 300}),
        ("train.loss_bce_logits", {}), ("train.rmsprop", {"lr": 0.005}),
        ("eval.accuracy", {}), ("eval.roc_auc", {}), TRAINER])


def _sklearn_kfold(tmp):
    return chain("iris_kfold", "4", MLP(3), [
        ("data.sklearn", {"dataset": "iris"}), ("prep.normalize", {"mode": "fit"}),
        ("train.kfold", {"k": 3}), ("train.nadam", {"lr": 0.01}), ("eval.accuracy", {}),
        TRAINER])


def _table(tmp):
    path = write_table(tmp)
    return chain("table_file", "5", MLP(2), [
        ("data.csv", {"path": str(path), "target_column": "label"}),
        ("prep.drop_columns", {"columns": "id"}), ("prep.impute", {"strategy": "median"}),
        ("prep.one_hot", {}), ("prep.log_transform", {"columns": ""}),
        ("prep.clip_outliers", {"method": "iqr"}), ("prep.robust_scale", {}),
        ("prep.split", {"val_fraction": 0.2, "test_fraction": 0.2}),
        ("train.loss_cross_entropy", {}), ("train.adam", {"lr": 0.01}),
        ("eval.accuracy", {}), ("eval.balanced_accuracy", {}), TRAINER])


def _text(tmp):
    path = write_texts(tmp)
    return chain("text_sentiment", "16", [
        ("core.embedding", {"num_embeddings": 200, "embedding_dim": 16}),
        ("core.mean_over_time", {}), ("core.dense", {"units": 2})], [
        ("data.text_csv", {"path": str(path)}), ("prep.text_clean", {}),
        ("prep.tokenize", {"max_length": 16, "vocab_size": 200}),
        ("train.loss_cross_entropy", {}), ("train.adam", {"lr": 0.01}),
        ("eval.accuracy", {}), TRAINER], dtype="int64")


def _text_folder(tmp):
    root = write_text_folder(tmp)
    return chain("text_folder_char_lstm", "40", [
        ("core.embedding", {"num_embeddings": 60, "embedding_dim": 8}),
        ("core.lstm", {"hidden_size": 16, "return_sequences": False}),
        ("core.dense", {"units": 2})], [
        ("data.text_folder", {"root": str(root)}),
        ("prep.tokenize", {"method": "char", "max_length": 40, "vocab_size": 60}),
        ("eval.accuracy", {}), TRAINER], dtype="int64")


def _audio(tmp):
    root = write_audio(tmp)
    return chain("audio_mel_cnn", "1, 32, 13", [
        ("core.conv2d", {"out_channels": 4, "kernel_size": 3, "padding": 1}),
        ("core.relu", {}), ("core.global_avgpool2d", {}), ("core.dense", {"units": 2})], [
        ("data.audio_folder", {"root": str(root), "sample_rate": 8000, "duration": 0.25}),
        ("prep.audio_features", {"kind": "mel_spectrogram", "n_fft": 256, "hop_length": 160,
                                 "n_mels": 32}),
        ("prep.normalize", {"mode": "fit"}), ("eval.accuracy", {}), TRAINER])


def _audio_specaug(tmp):
    root = write_audio(tmp)
    return chain("audio_mfcc_specaug", "1, 13, 13", [
        ("core.flatten", {}), ("core.dense", {"units": 2})], [
        ("data.audio_folder", {"root": str(root), "sample_rate": 8000, "duration": 0.25}),
        ("prep.audio_features", {"kind": "mfcc", "n_fft": 256, "hop_length": 160,
                                 "n_mels": 32, "n_mfcc": 13}),
        ("prep.spec_augment", {"freq_mask": 2, "time_mask": 2}),
        ("eval.accuracy", {}), TRAINER])


def _series_lstm(tmp):
    path = write_series(tmp)
    return chain("series_lstm", "16, 2", [
        ("core.lstm", {"hidden_size": 16, "return_sequences": False}),
        ("core.dense", {"units": 2})], [
        ("data.timeseries_csv", {"path": str(path), "target_columns": "value",
                                 "feature_columns": "value, other", "window": 16, "horizon": 2}),
        ("prep.normalize", {"mode": "fit"}), ("train.loss_mse", {}),
        ("eval.mae", {}), ("eval.rmse", {}), TRAINER])


def _series_conv(tmp):
    path = write_series(tmp)
    return chain("series_conv1d", "2, 24", [
        ("core.conv1d", {"out_channels": 8, "kernel_size": 3, "padding": 1}),
        ("core.relu", {}), ("core.global_avgpool1d", {}), ("core.dense", {"units": 1})], [
        ("data.timeseries_csv", {"path": str(path), "target_columns": "value",
                                 "feature_columns": "value, other", "window": 24,
                                 "layout": "channels [C, L]"}),
        ("prep.minmax", {}), ("train.loss_l1", {}), ("eval.mae", {}), TRAINER])


def _images(tmp):
    root = write_images(tmp)
    return chain("image_folder_aug", "3, 16, 16", [
        ("core.conv2d", {"out_channels": 8, "kernel_size": 3, "padding": 1}),
        ("core.batch_norm2d", {}), ("core.relu", {}), ("core.maxpool2d", {}),
        ("core.global_avgpool2d", {}), ("core.dense", {"units": 2})], [
        ("data.image_folder", {"root": str(root)}),
        ("prep.resize", {"height": 16, "width": 16}),
        ("prep.random_flip", {"mode": "both"}), ("prep.random_rotation", {"degrees": 10}),
        ("prep.normalize", {"mode": "fit"}),
        ("train.loss_cross_entropy", {}), ("train.adam", {"lr": 0.01}),
        ("eval.accuracy", {}), TRAINER])


def _images_torch_heavy(tmp):
    root = write_images(tmp)
    return chain("image_folder_heavy_aug", "3, 16, 16", [
        ("core.conv2d", {"out_channels": 8, "kernel_size": 3, "padding": 1}),
        ("core.relu", {}), ("core.global_avgpool2d", {}), ("core.dense", {"units": 2})], [
        ("data.image_folder", {"root": str(root)}),
        ("prep.random_resized_crop", {"size": 16, "scale_min": 0.5}),
        ("prep.color_jitter", {}), ("prep.gaussian_blur", {"kernel_size": 3}),
        ("prep.trivial_augment", {}), ("prep.random_erasing", {}),
        ("prep.mixup_cutmix", {"mode": "both"}),
        ("prep.normalize", {"mode": "fixed", "mean": "0.5, 0.5, 0.5", "std": "0.25, 0.25, 0.25"}),
        ("prep.class_balance", {"strategy": "oversample"}),
        ("train.loss_cross_entropy", {}), ("train.radam", {"lr": 0.01}),
        ("train.one_cycle_lr", {"max_lr": 0.01}), ("eval.accuracy", {}),
        ("train.trainer", {"epochs": 2, "batch_size": 8, "device": "cpu",
                           "grad_clip_norm": 1.0, "accumulation_steps": 2})])


def _npz(tmp):
    path = write_npz(tmp)
    return chain("npz_cnn", "1, 8, 8", [
        ("core.conv2d", {"out_channels": 4}), ("core.relu", {}), ("core.flatten", {}),
        ("core.dense", {"units": 3})], [
        ("data.numpy", {"path": str(path)}), ("prep.normalize", {"mode": "fit"}),
        ("train.loss_focal", {}), ("eval.accuracy", {}), TRAINER])


def _json(tmp):
    path = write_json(tmp)
    return chain("json_records", "6", MLP(2), [
        ("data.json", {"path": str(path)}), ("prep.minmax", {}), ("train.adagrad", {"lr": 0.05}),
        ("train.plateau_lr", {"patience": 1}), ("eval.accuracy", {}), TRAINER])


SCENARIOS = [
    Scenario("multiclass", _multiclass, expect=("accuracy", "f1", "roc_auc", "mcc")),
    Scenario("regression", _regression, expect=("mae", "rmse", "r2")),
    Scenario("binary", _binary, expect=("accuracy", "roc_auc", "precision")),
    Scenario("multilabel", _multilabel, expect=("accuracy",)),
    Scenario("sklearn_kfold", _sklearn_kfold, frameworks=("pytorch",)),
    Scenario("table", _table, expect=("accuracy", "balanced_accuracy")),
    Scenario("text", _text, expect=("accuracy",)),
    Scenario("text_folder", _text_folder, expect=("accuracy",)),
    Scenario("audio", _audio, expect=("accuracy",)),
    Scenario("audio_specaug", _audio_specaug, frameworks=("pytorch",), expect=("accuracy",)),
    Scenario("series_lstm", _series_lstm, expect=("mae",)),
    Scenario("series_conv", _series_conv, expect=("mae",)),
    Scenario("images", _images, expect=("accuracy",)),
    Scenario("images_heavy", _images_torch_heavy, frameworks=("pytorch",), expect=("accuracy",)),
    Scenario("npz", _npz, expect=("accuracy",)),
    Scenario("json", _json, expect=("accuracy",)),
]


_INFER_CHECK = r"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location("trained", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.load_predictor(".")
mode = sys.argv[2]
if mode == "equivalence":
    # raw test samples through prepare_inputs == the training pipeline's test arrays
    expected = m.make_arrays()["test"][0]
    m.load_inference_state(".")
    x_raw, y, _ = m.load_raw()
    _, _, test_i = m.split_indices(len(y), y)
    if hasattr(x_raw, "iloc"):
        raw = x_raw.iloc[test_i].to_dict("records")
    elif isinstance(x_raw, list):
        raw = [x_raw[i] for i in test_i]
    else:
        raw = [x_raw[i].tolist() for i in test_i]
    got = m.prepare_inputs(raw)
    assert got.shape == expected.shape, (got.shape, expected.shape)
    assert np.allclose(got, expected, atol=1e-5), float(np.abs(got - expected).max())
    raw = raw[:3]
elif mode == "images":
    raw = [str(p) for p in sorted(Path("images").rglob("*.png"))[:3]]
    import base64
    raw.append(base64.b64encode(Path(raw[0]).read_bytes()).decode())
elif mode == "series":
    import pandas as pd
    frame = pd.read_csv("series.csv")
    window = m.INPUT_SHAPE[0] if m.INPUT_SHAPE[1] == 2 else m.INPUT_SHAPE[1]
    raw = [frame[["value", "other"]].to_numpy()[i:i + window].tolist() for i in (0, 50)]
out = m.infer(raw)
assert len(out) == len(raw), out
print("INFER-OK " + json.dumps(out[0]))
"""


def check_inference(sc: Scenario, framework: str, workdir: Path, script: Path) -> str:
    mode = ("images" if sc.name.startswith("images") else
            "series" if sc.name.startswith("series") else "equivalence")
    env = {**os.environ, "KERAS_BACKEND": "torch", "PYTHONWARNINGS": "ignore"}
    proc = subprocess.run([sys.executable, "-c", _INFER_CHECK, script.name, mode], cwd=workdir,
                          capture_output=True, text=True, timeout=600, env=env)
    if "INFER-OK" not in proc.stdout:
        raise AssertionError(f"{sc.name}/{framework} inference failed:\n"
                             f"{(proc.stdout + proc.stderr)[-3000:]}")
    return proc.stdout.split("INFER-OK ", 1)[1].strip()


def run_scenario(sc: Scenario, framework: str, workdir: Path, verbose: bool = False) -> str:
    graph = sc.build(workdir)
    issues = [i for i in graph.validate() if i.severity == "error"]
    if issues:
        raise AssertionError(f"{sc.name}: graph invalid: {issues}")
    code = generate_training(graph, framework)
    script = workdir / f"{sc.name}_{framework}.py"
    script.write_text(code)
    env = {**os.environ, "KERAS_BACKEND": "torch", "PYTHONWARNINGS": "ignore"}
    proc = subprocess.run([sys.executable, script.name], cwd=workdir, capture_output=True,
                          text=True, timeout=600, env=env)
    out = proc.stdout + proc.stderr
    if verbose:
        print(out)
    if proc.returncode != 0:
        raise AssertionError(f"{sc.name}/{framework} failed:\n{out[-3000:]}")
    if "kfold" not in sc.name:
        assert "epoch 2/2" in out or "early stopping" in out, out[-2000:]
        test_line = next((ln for ln in out.splitlines() if ln.startswith("test:")), "")
        for key in sc.expect:
            assert f"{key}=" in test_line, f"{key} missing from: {test_line}"
        assert (workdir / "inference_state.pkl").exists(), "inference state not saved"
        check_inference(sc, framework, workdir, script)
    else:
        assert "cv accuracy" in out, out[-2000:]
    return out


def main() -> int:
    verbose = "-v" in sys.argv
    sel = sys.argv[sys.argv.index("-k") + 1] if "-k" in sys.argv else ""
    frameworks = ("pytorch", "keras") if "--keras" in sys.argv else ("pytorch",)
    failures = 0
    for sc in SCENARIOS:
        if sel and sel not in sc.name:
            continue
        for fw in frameworks:
            if fw not in sc.frameworks:
                continue
            with tempfile.TemporaryDirectory() as tmp:
                try:
                    out = run_scenario(sc, fw, Path(tmp), verbose)
                    test_line = next((ln for ln in out.splitlines() if ln.startswith(("test:", "cv "))), "")
                    print(f"ok    {sc.name:16} {fw:8} {test_line[:110]}")
                except Exception as exc:  # noqa: BLE001
                    failures += 1
                    print(f"FAIL  {sc.name:16} {fw:8} {str(exc)[-1500:]}")
    print(f"{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
