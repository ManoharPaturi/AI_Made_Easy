"""V3 Phase 1: resource budgets — estimates, device profiles, lints, fixes, calibration."""
from __future__ import annotations

import copy
import importlib.util
import json

import pytest
from conftest import tiny_classifier_dict

from ai_made_easy.core import api, budget
from ai_made_easy.core.fixes import fix_for_issue
from ai_made_easy.core.graph import Graph

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _sample(name: str, **trainer) -> Graph:
    data = api.read_sample(name)
    for node in data["nodes"]:
        if "weights" in node.get("params", {}):
            node["params"]["weights"] = "none"
        if node["type"] == "train.trainer":
            node["params"].update(trainer)
    return Graph.from_dict(data)


def _with_budget(graph: Graph, **values) -> Graph:
    budget.set_budget(graph, **values)
    return graph


# ================================================================ devices

def test_builtin_devices():
    table = budget.devices()
    assert {"cpu_laptop", "apple_m_series", "rtx_3060", "rtx_4090", "t4", "a100_40", "a100_80",
            "h100", "iphone_ane", "jetson_orin_nano", "raspberry_pi_5"} <= set(table)
    for device in table.values():
        assert device.memory_gb > 0 and device.tflops_fp32 > 0 and device.bandwidth_gbs > 0
        assert device.tflops_fp16 >= device.tflops_fp32
    with pytest.raises(KeyError, match="unknown device"):
        budget.get_device("toaster")


def test_user_devices(isolated_home):
    mine = budget.Device("lab_box", "Lab box", "cuda", 10, 5, 10, 200)
    budget.save_user_device(mine)
    assert budget.get_device("lab_box") == mine
    assert "rtx_3060" in budget.devices()  # built-ins stay
    budget.user_devices_path().write_text("not json")
    assert "lab_box" not in budget.devices() and "t4" in budget.devices()


# ================================================================ estimates

@pytest.mark.skipif(not HAS_TORCH, reason="needs torch")
@pytest.mark.parametrize("sample", ["mnist_cnn.json", "mlp_mnist.json",
                                    "cifar10_cnn_augmented.json", "skip_connection_mlp.json",
                                    "cifar10_transfer_resnet18.json"])
def test_flops_match_torch_counter(sample):
    """Analytic FLOPs are within 10% of PyTorch's own FLOP counter."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    from ai_made_easy.core.codegen import generate

    if sample.startswith("cifar10_transfer"):
        pytest.importorskip("torchvision")
    graph = _sample(sample)
    estimated = budget.estimate(graph).flops
    namespace: dict = {"__name__": "budget_test"}
    exec(compile(generate(graph, "pytorch"), "<gen>", "exec"), namespace)  # noqa: S102
    model = namespace["build_model"]().eval()
    entry = next(n for n in graph.nodes.values() if n.type_id == "core.input")
    x = torch.zeros(1, *graph.infer_shapes()[entry.instance_id])
    with FlopCounterMode(display=False) as counter, torch.no_grad():
        model(x)
    measured = counter.get_total_flops()
    assert measured > 0
    assert abs(estimated / measured - 1) < 0.10, (estimated, measured)


def test_lstm_flops_formula():
    """torch's FLOP counter misses the fused CPU LSTM kernel on Linux, so check the
    recurrent layer against the textbook count: 2 x weights x time steps."""
    graph = _sample("lstm_classifier.json")
    est = budget.estimate(graph)
    shapes = graph.infer_shapes()
    lstm = next(layer for layer in est.layers if layer.type_id == "core.lstm")
    edge = graph.input_edge_for(lstm.node_id, "in")
    steps = shapes[edge.source_id][0]
    assert lstm.params > 0 and lstm.flops == 2 * lstm.params * steps


def test_estimate_parts():
    graph = _sample("mlp_mnist.json", batch_size=64)
    est = budget.estimate(graph, "rtx_3060")
    assert est.params == sum(layer.params for layer in est.layers) == est.trainable_params
    assert est.batch_size == 64 and est.optimizer_states == 2
    flatten = next(layer for layer in est.layers if layer.type_id == "core.flatten")
    assert flatten.activations == 0  # a view keeps no memory
    # memory grows with the batch, mixed precision shrinks activations
    assert est.train_memory_bytes(128) > est.train_memory_bytes(64)
    assert est.train_memory_bytes(64, True) < est.train_memory_bytes(64, False) + est.params * 2
    assert est.latency_ms() > 0 and est.step_time_ms() > 0
    slow = budget.estimate(graph, "raspberry_pi_5")
    assert slow.latency_ms() > est.latency_ms()
    report = api.estimate_budget(graph.to_dict(), "t4")
    assert report["estimate"]["device"]["id"] == "t4"
    assert report["checks"][0]["kind"] == "train_memory"
    json.dumps(report)


def test_frozen_backbone_trains_only_the_head():
    pytest.importorskip("torchvision")
    graph = _sample("cifar10_transfer_resnet18.json")
    est = budget.estimate(graph)
    assert est.params > 11_000_000 and est.trainable_params < 100_000
    backbone = next(layer for layer in est.layers if layer.type_id == "core.pretrained_backbone")
    assert not backbone.trainable and backbone.flops > 1e8


def test_optimizer_state_counts():
    data = tiny_classifier_dict()
    opt = next(n for n in data["nodes"] if n["type"].startswith("train.") and n["id"] == "opt")
    opt["type"], opt["params"] = "train.adam", {}
    assert budget.estimate(Graph.from_dict(data)).optimizer_states == 2
    opt["type"], opt["params"] = "train.sgd", {"momentum": 0.0}
    assert budget.estimate(Graph.from_dict(data)).optimizer_states == 0
    opt["params"] = {"momentum": 0.9}
    assert budget.estimate(Graph.from_dict(data)).optimizer_states == 1


def test_summary_carries_flops():
    out = api.summarize(api.read_sample("mnist_cnn.json"))
    assert out["total_flops"] == sum(layer["flops"] for layer in out["layers"]) > 1_000_000
    conv = next(layer for layer in out["layers"] if layer["type"] == "core.conv2d")
    assert conv["flops"] > 0


# ================================================================ budget settings

def test_budget_settings_roundtrip():
    graph = Graph.from_dict(tiny_classifier_dict())
    assert budget.budget_of(graph)["device"] == "" and budget.check(graph) == []
    budget.set_budget(graph, device="t4", max_latency_ms=5)
    again = Graph.from_dict(graph.to_dict())
    assert budget.budget_of(again) == {"device": "t4", "max_train_memory_gb": 0.0,
                                       "max_latency_ms": 5.0, "max_params_m": 0.0,
                                       "max_model_mb": 0.0}
    with pytest.raises(ValueError, match="unknown budget"):
        budget.set_budget(graph, gpus=2)
    with pytest.raises(KeyError):
        budget.set_budget(graph, device="toaster")
    graph.meta["budget"] = {"device": "t4", "max_latency_ms": "lots"}
    assert budget.budget_of(graph)["max_latency_ms"] == 0.0


# ================================================================ lints + fixes

def _messages(graph: Graph) -> list[str]:
    return [i.message for i in graph.validate()]


def test_memory_lint_and_mixed_precision_fix():
    graph = _with_budget(_sample("cifar10_cnn_augmented.json", batch_size=256,
                                 mixed_precision=False), max_train_memory_gb=0.1)
    issue = next(i for i in graph.validate() if i.message.startswith("Training needs about"))
    assert "batch 256" in issue.message and issue.node_id
    label, _desc, fixed = fix_for_issue(graph, issue)
    est = budget.estimate(graph)
    if est.train_memory_bytes(mixed_precision=True) <= 0.1 * budget.GB:
        assert label == "Use mixed precision"
        trainer = next(n for n in fixed.nodes.values() if n.type_id == "train.trainer")
        assert trainer.params["mixed_precision"] is True
    else:
        assert label == "Fit batch size"


def test_memory_fix_keeps_the_effective_batch():
    graph = _with_budget(_sample("cifar10_cnn_augmented.json", batch_size=512,
                                 mixed_precision=True, accumulation_steps=1),
                         max_train_memory_gb=0.06)
    issue = next(i for i in graph.validate() if i.message.startswith("Training needs about"))
    label, _desc, fixed = fix_for_issue(graph, issue)
    assert label == "Fit batch size"
    trainer = next(n for n in fixed.nodes.values() if n.type_id == "train.trainer")
    batch, steps = trainer.params["batch_size"], trainer.params["accumulation_steps"]
    assert batch < 512 and batch * steps == 512
    assert budget.budget_of(fixed)["max_train_memory_gb"] == 0.06  # budget survives the fix
    assert not any(m.startswith("Training needs about") for m in _messages(fixed))


def test_latency_params_and_size_lints():
    graph = _with_budget(_sample("mnist_cnn.json"), device="raspberry_pi_5",
                         max_latency_ms=0.0001, max_params_m=0.001, max_model_mb=0.001)
    messages = _messages(graph)
    assert any("over the 0.0001 ms budget" in m for m in messages)
    assert any("parameters, over the 0.001M budget" in m for m in messages)
    assert any("MB, over the 0.001 MB budget" in m for m in messages)
    relaxed = _with_budget(copy.deepcopy(graph), max_latency_ms=1e4, max_params_m=100,
                           max_model_mb=100)
    assert not any("budget" in m for m in _messages(relaxed))


def test_no_budget_no_lint():
    graph = _sample("cifar10_cnn_augmented.json", batch_size=4096)
    assert not any("Training needs" in m for m in _messages(graph))


# ================================================================ sweeps

def test_sweep_skips_over_budget_trials(isolated_home):
    from ai_made_easy.core.sweeps import SweepRunner

    graph = _with_budget(_sample("mlp_mnist.json"), max_params_m=0.15)
    dense = next(n for n in graph.nodes.values() if n.type_id == "core.dense")
    big = copy.deepcopy(graph)
    big.nodes[dense.instance_id].params["units"] = 4096
    assert SweepRunner._over_budget(graph) == ""
    assert "params" in SweepRunner._over_budget(big)


# ================================================================ calibration

def test_worker_parses_resources_line():
    import io

    from ai_made_easy.worker.train_worker import LineTap

    sink = io.StringIO()
    LineTap(sink).write("resources: peak_memory_mb=812.5 step_ms=31.25 batch_size=64 "
                        "device=mps\n")
    event = json.loads(sink.getvalue())
    assert event == {"type": "resources", "peak_memory_mb": 812.5, "step_ms": 31.25,
                     "batch_size": 64.0, "device": "mps"}


def test_resources_event_lands_in_the_run_record(isolated_home):
    from ai_made_easy.core.runner.manager import TrainingRun
    from ai_made_easy.core.runs.history import RunHistory

    history = RunHistory()
    record = history.create(tiny_classifier_dict())
    run = TrainingRun(record.run_id, isolated_home / "s.py", isolated_home, history)
    run.record({"type": "resources", "peak_memory_mb": 100.0, "step_ms": 5.0})
    assert history.get(record.run_id).resources == {"peak_memory_mb": 100.0, "step_ms": 5.0}


def test_training_script_reports_resources():
    from ai_made_easy.core.training.generate import generate_training

    code = generate_training(Graph.from_dict(tiny_classifier_dict()), "pytorch")
    assert "def report_resources" in code and "resources: peak_memory_mb" in code


# ================================================================ surfaces

def test_server_budget_endpoints(isolated_home, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from ai_made_easy.server.app import create_app

    with TestClient(create_app(token="", projects_dir=tmp_path)) as client:
        devices = client.get("/api/devices").json()["devices"]
        assert any(d["id"] == "h100" for d in devices)
        body = {"graph": api.read_sample("mnist_cnn.json"), "device": "rtx_3060",
                "limits": {"max_latency_ms": 1e-6}}
        report = client.post("/api/budget", json=body).json()
        assert report["estimate"]["latency_ms"] > 0
        assert any(c["kind"] == "latency" and c["over"] for c in report["checks"])
        bad = client.post("/api/budget", json={**body, "device": "toaster"})
        assert bad.status_code == 400 and "unknown device" in bad.text
        summary = client.post("/api/summary", json={"graph": api.read_sample("mnist_cnn.json")})
        assert summary.json()["total_flops"] > 0


def test_cli_budget(tmp_path, capsys):
    from ai_made_easy.cli import main

    path = tmp_path / "p.json"
    data = api.read_sample("mnist_cnn.json")
    path.write_text(json.dumps(data))
    assert main(["budget", "--list"]) == 0
    assert "raspberry_pi_5" in capsys.readouterr().out
    assert main(["budget", str(path), "--device", "t4"]) == 0
    out = capsys.readouterr().out
    assert "forward FLOPs" in out and "NVIDIA T4" in out
    data["meta"] = {"budget": {"device": "t4", "max_params_m": 0.0001}}
    path.write_text(json.dumps(data))
    assert main(["budget", str(path), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["checks"]


def test_canvas_adapter_keeps_meta():
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets

    from ai_made_easy.ui.canvas.adapter import CanvasController

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    adapter = CanvasController()
    graph = Graph.from_dict(api.read_sample("iris_mlp.json"))
    budget.set_budget(graph, device="t4")
    adapter.load_ir(graph)
    assert budget.budget_of(adapter.to_ir())["device"] == "t4"
    adapter.clear()
    assert adapter.to_ir().meta == {}
