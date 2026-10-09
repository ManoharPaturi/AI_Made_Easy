"""Speech: keyword spotting, CTC speech recognition, audio tagging — data, metrics, rules,
profiles, training and serving."""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ai_made_easy.core import api
from ai_made_easy.core.fixes import fix_for_issue
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.tasks import task_of

torch = pytest.importorskip("torch")

KWS = [("audio.mel_spectrogram", {"n_mels": 32}), ("core.conv2d", {"out_channels": 8}),
       ("core.relu", {}), ("core.maxpool2d", {}), ("core.conv2d", {"out_channels": 16}),
       ("core.relu", {}), ("core.global_avgpool2d", {}), ("core.dense", {"units": 4})]
ASR = [("audio.mel_spectrogram", {"n_mels": 32, "layout": "sequence [frames, bins]"}),
       ("core.lstm", {"hidden_size": 32, "bidirectional": True}), ("core.dense", {"units": 29})]


@pytest.fixture(scope="module")
def rt():
    from ai_made_easy.core.speech.runtime import namespace

    return namespace()


def design(layers: list, data: dict, *, samples: int = 16000, extra: tuple = (),
           epochs: int = 1, name: str = "speech_case") -> dict:
    nodes = [{"id": "in", "type": "core.input", "params": {"shape": f"1, {samples}"},
              "position": [0, 0]}]
    edges, prev = [], "in"
    for i, (type_id, params) in enumerate(layers):
        nodes.append({"id": f"l{i}", "type": type_id, "params": params, "position": [0, 0]})
        edges.append({"from": f"{prev}/out", "to": f"l{i}/in"})
        prev = f"l{i}"
    nodes += [{"id": "out", "type": "core.output", "params": {}, "position": [0, 0]},
              {"id": "data", "type": "data.synthetic_speech",
               "params": {"n_clips": 60, "n_keywords": 4, **data}, "position": [0, 0]},
              {"id": "tr", "type": "train.trainer",
               "params": {"epochs": epochs, "batch_size": 16, "device": "cpu"},
               "position": [0, 0]}, *extra]
    edges.append({"from": f"{prev}/out", "to": "out/in"})
    return {"name": name, "nodes": nodes, "edges": edges}


def issues(data: dict) -> list:
    return [i for i in Graph.from_dict(data).validate()
            if "optimizer" not in i.message and "loss function" not in i.message]


# ================================================================ tasks / samples

@pytest.mark.parametrize("task, expected", [("keywords", "keyword_spotting"),
                                            ("transcripts", "speech_recognition"),
                                            ("tags", "audio_tagging")])
def test_dataset_task_decides_the_speech_task(task, expected):
    found = task_of(Graph.from_dict(design(KWS, {"task": task})))
    assert found.id == expected and found.trainer_kind == "speech"


@pytest.mark.parametrize("sample, task", [("keyword_spotting.json", "keyword_spotting"),
                                          ("speech_recognition_ctc.json",
                                           "speech_recognition")])
def test_samples_validate_and_resolve(sample, task):
    graph = Graph.from_dict(api.read_sample(sample))
    assert graph.validate() == []
    assert task_of(graph).id == task


# ================================================================ data / metrics

def test_synthetic_speech(rt):
    records, names = rt["synthetic_speech"]("keywords", 8, 4, 1.0, 8000, 0.0, 0)
    assert names == ["yes", "no", "up", "down"] and [r["label"] for r in records[:5]] == \
        [0, 1, 2, 3, 0]
    assert records[0]["wave"].shape == (8000,) and np.abs(records[0]["wave"]).max() > 0.1
    records, names = rt["synthetic_speech"]("transcripts", 20, 3, 1.5, 8000, 0.0, 1)
    assert names == list(rt["ALPHABET"])
    assert all(set(r["label"]) <= set("yesnoup ") and r["label"] for r in records)
    records, names = rt["synthetic_speech"]("tags", 20, 3, 1.0, 8000, 0.0, 2)
    assert names == rt["SOUND_TAGS"] and all(r["label"] for r in records)
    again, _ = rt["synthetic_speech"]("tags", 20, 3, 1.0, 8000, 0.0, 2)
    np.testing.assert_array_equal(records[3]["wave"], again[3]["wave"])


def test_wav_round_trip_and_resampling(rt, tmp_path):
    t = np.arange(8000) / 8000
    wave = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    rt["write_wav"](tmp_path / "a.wav", wave, 8000)
    back, rate = rt["decode_wav_bytes"]((tmp_path / "a.wav").read_bytes())
    assert rate == 8000 and np.abs(back - wave).max() < 1e-3
    up = rt["read_wav"](tmp_path / "a.wav", 16000, 20000)
    assert up.shape == (20000,) and np.abs(up[16000:]).max() == 0
    spectrum = np.abs(np.fft.rfft(up[:16000]))
    assert abs(int(spectrum.argmax()) - 440) <= 1    # pitch kept after resampling
    with pytest.raises(ValueError, match="not a readable WAV"):
        rt["decode_wav_bytes"](b"not audio")


def test_metrics_hand_cases(rt):
    assert rt["edit_distance"]("kitten", "sitting") == 3
    assert rt["error_rate"](["go up"], ["go up"], False) == 0
    assert rt["error_rate"](["go up", "no"], ["go", "yes"], True) == pytest.approx(2 / 3)
    assert rt["ctc_greedy"]([0, 7, 7, 0, 15, 0, 28, 0, 21, 16, 16], rt["ALPHABET"]) == "go up"
    assert rt["ctc_greedy"]([7, 0, 7], rt["ALPHABET"]) == "gg"   # blank separates repeats
    y = np.array([[1, 0], [0, 1], [1, 0]])
    perfect = rt["tagging_scores"](y, np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]]))
    assert perfect["map"] == 1.0 and perfect["f1"] == 1.0
    assert rt["average_precision"](np.array([0, 1]), np.array([0.9, 0.1])) == 0.5
    scores = rt["classification_scores"](np.array([0, 1, 1, 2]), np.array([0, 1, 2, 2]), 3)
    assert scores["accuracy"] == 0.75 and scores["confusion"][1] == [0, 1, 1]


def test_manifest_reader(rt, tmp_path):
    for i, name in enumerate(("a", "b", "c")):
        rt["write_wav"](tmp_path / f"{name}.wav", np.full(4000, 0.1 * i, np.float32), 8000)
    (tmp_path / "m.csv").write_text("path,label\na.wav,Go Up!\nb.wav,no\nc.wav,yes\n")
    records, names = rt["read_manifest"](tmp_path / "m.csv", "transcripts", "path", "label",
                                         ";", rt["ALPHABET"], 16000, 0.5)
    assert records[0]["label"] == "go up" and records[0]["wave"].shape == (8000,)
    records, names = rt["read_manifest"](tmp_path / "m.csv", "keywords", "path", "label",
                                         ";", rt["ALPHABET"], 8000, 0.5)
    assert names == ["Go Up!", "no", "yes"] and records[2]["label"] == 2
    (tmp_path / "t.csv").write_text("file,tags\na.wav,dog; car\nb.wav,car\n")
    records, names = rt["read_manifest"](tmp_path / "t.csv", "tags", "file", "tags", ";",
                                         "", 8000, 0.5)
    assert names == ["car", "dog"] and records[0]["label"] == [0, 1]
    with pytest.raises(ValueError, match="not found"):
        rt["read_manifest"](tmp_path / "t.csv", "tags", "path", "tags", ";", "", 8000, 0.5)


# ================================================================ rules

def test_input_shape_rule_and_fix():
    data = design(KWS, {"task": "keywords"}, samples=8000)
    issue = next(i for i in issues(data) if "Set the Input shape" in i.message)
    fixed = fix_for_issue(Graph.from_dict(data), issue)
    assert fixed[2].nodes["in"].params["shape"] == "1, 16000"


def test_class_count_rule_and_fix():
    data = design(KWS, {"task": "keywords", "n_keywords": 6})
    issue = next(i for i in issues(data) if "set units to 6" in i.message)
    assert fix_for_issue(Graph.from_dict(data), issue)[2].nodes["l7"].params["units"] == 6
    assert any("one score per keyword" in i.message
               for i in issues(design(KWS[:-2], {"task": "keywords"})))


def test_ctc_rules():
    good = design(ASR, {"task": "transcripts"})
    assert not [i for i in issues(good) if i.severity in ("error", "warning")]
    wrong_vocab = design([*ASR[:-1], ("core.dense", {"units": 27})], {"task": "transcripts"})
    assert any("set units to 29" in i.message for i in issues(wrong_vocab))
    pooled = design(KWS, {"task": "transcripts"})
    assert any("per-frame character scores" in i.message for i in issues(pooled))
    too_few = design([("audio.mel_spectrogram", {"n_mels": 32, "hop_length": 4000,
                                                 "n_fft": 4000,
                                                 "layout": "sequence [frames, bins]"}),
                      ("core.dense", {"units": 29})], {"task": "transcripts"})
    assert any("one frame per character" in i.message for i in issues(too_few))
    ctc = ({"id": "ctc", "type": "speech.loss_ctc", "params": {}, "position": [0, 0]},)
    assert any("trains speech recognition" in i.message
               for i in issues(design(KWS, {"task": "keywords"}, extra=ctc)))


def test_sample_rate_rules():
    mel = design(KWS, {"task": "keywords", "sample_rate": 8000}, samples=8000)
    issue = next(i for i in issues(mel) if "assumes 16000 Hz" in i.message)
    assert fix_for_issue(Graph.from_dict(mel), issue)[2].nodes["l0"].params[
        "sample_rate"] == 8000


def test_tagging_softmax_warning():
    data = design([*KWS[:-1], ("core.dense", {"units": 5}), ("core.softmax", {})],
                  {"task": "tags"})
    assert any("softmax makes tags compete" in i.message for i in issues(data))


# ================================================================ profile

def test_profile_speech(rt, tmp_path):
    from ai_made_easy.core.data.profile import profile_dataset
    from ai_made_easy.core.registry import get_registry

    defaults = {p.name: p.default for p in get_registry().get("data.synthetic_speech").params}
    p = profile_dataset("data.synthetic_speech", {**defaults, "n_clips": 40})
    assert p.task == "keyword_spotting" and len(p.classes) == 6
    rt["write_wav"](tmp_path / "a.wav", np.zeros(24000, np.float32), 8000)
    (tmp_path / "m.csv").write_text("path,label\na.wav,Hello World 2\nmissing.wav,hi\n")
    manifest = {p.name: p.default for p in get_registry().get("data.speech_csv").params}
    manifest.update(path="m.csv", task="transcripts")
    p = profile_dataset("data.speech_csv", manifest, tmp_path)
    found = " ".join(f.message for f in p.findings)
    assert "missing" in found and "longer than duration" in found
    assert "outside the alphabet" in found and "resampled to 16000" in found
    assert "does not exist" in profile_dataset("data.speech_csv", {**manifest, "path": "x"},
                                               tmp_path).error


# ================================================================ training / serving

def _train(data: dict, tmp_path: Path) -> Path:
    from ai_made_easy.core.training.generate import generate_training

    code = generate_training(Graph.from_dict(data), "pytorch")
    assert "Needs:  torch, numpy, pillow" in code
    (tmp_path / "train.py").write_text(code)
    proc = subprocess.run([sys.executable, "train.py"], cwd=tmp_path, capture_output=True,
                          text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-3000:]
    assert "resources: peak_memory_mb" in proc.stdout
    return tmp_path


def _infer(folder: Path, items: list) -> list:
    code = ("import sys, json; sys.path.insert(0, '.'); import train; "
            "train.load_predictor('.'); "
            f"print('OUT ' + json.dumps(train.infer({items!r})))")
    proc = subprocess.run([sys.executable, "-c", code], cwd=folder, capture_output=True,
                          text=True, timeout=300, env={**os.environ, "PYTHONWARNINGS": "ignore"})
    assert "OUT " in proc.stdout, proc.stderr[-3000:]
    return json.loads(proc.stdout.split("OUT ", 1)[1])


@pytest.mark.parametrize("task, layers, samples", [
    ("keywords", KWS, 16000), ("tags", [*KWS[:-1], ("core.dense", {"units": 5})], 16000),
    ("transcripts", ASR, 24000)])
def test_speech_tasks_train_and_serve(task, layers, samples, rt, tmp_path):
    data = design(layers, {"task": task, "duration": samples / 16000}, samples=samples)
    run = _train(data, tmp_path)
    metrics = json.loads((run / "metrics.json").read_text())
    expected = {"keywords": {"accuracy", "macro_f1", "confusion"}, "tags": {"map", "f1"},
                "transcripts": {"cer", "wer"}}[task]
    assert expected <= set(metrics)
    assert list((run / "eval_samples").glob("*.png"))
    rt["write_wav"](tmp_path / "clip.wav", np.zeros(8000, np.float32), 8000)
    clip = base64.b64encode((tmp_path / "clip.wav").read_bytes()).decode()
    out = _infer(run, [clip, [0.0] * 100, {"waveform": [0.0] * 4000, "sample_rate": 8000}])
    key = {"keywords": "label", "tags": "tags", "transcripts": "text"}[task]
    assert len(out) == 3 and all(key in o for o in out)


def test_speech_run_deploys(tmp_path, isolated_home):
    pytest.importorskip("fastapi")
    from ai_made_easy.core.deploy.package import build_package
    from ai_made_easy.core.runner.manager import RunManager
    from ai_made_easy.core.runs.history import RunHistory

    mgr = RunManager(RunHistory(tmp_path / "runs"))
    api.set_manager(mgr)
    try:
        run_id = mgr.start(Graph.from_dict(design(KWS, {"task": "keywords"}, name="kws")))
        status = mgr.wait(run_id, 600)
        assert status["state"] == "finished", mgr.history.get(run_id).error[-2000:]
        assert "accuracy" in mgr.history.get(run_id).final_metrics
        result = build_package(mgr.history.path(run_id), tmp_path / "pkg")
        assert result.verified, result.log[-3000:]
        meta = json.loads((tmp_path / "pkg/metadata.json").read_text())
        assert meta["task"] == "keyword_spotting" and meta["input_kind"] == "speech"
        reqs = (tmp_path / "pkg/requirements.txt").read_text()
        assert "torch" in reqs and "numpy" in reqs
        assert api.run_samples(run_id)["samples"]
    finally:
        api.set_manager(None)
