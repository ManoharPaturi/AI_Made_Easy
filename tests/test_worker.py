"""Worker + protocol tests: JSON event streaming from real training runs."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ai_made_easy.core.codegen import export_training
from ai_made_easy.core.graph import Edge, Graph, NodeInstance
from ai_made_easy.core.runner.protocol import parse_event, worker_script_path


def build(*specs: tuple[str, str, dict]) -> Graph:
    g = Graph(name="tiny")
    for nid, type_id, params in specs:
        g.add_node(NodeInstance(nid, type_id, params))
    g.add_edge(Edge("in", "out", "d1", "in"))
    g.add_edge(Edge("d1", "out", "out", "in"))
    return g


def run_worker(script: Path, cwd: Path) -> list[dict]:
    proc = subprocess.run(
        [sys.executable, str(worker_script_path()), str(script)],
        capture_output=True, text=True, timeout=300, cwd=cwd,
    )
    events = [e for e in (parse_event(l) for l in proc.stdout.splitlines()) if e]
    assert proc.returncode == 0, proc.stderr
    return events


def test_parse_event_is_tolerant():
    assert parse_event("") is None
    assert parse_event("   ") is None
    assert parse_event("some garbage stdout") == {"type": "log", "line": "some garbage stdout"}
    ev = parse_event('{"type": "epoch", "epoch": 1, "total": 3, "metrics": {}}')
    assert ev["type"] == "epoch" and ev["epoch"] == 1
    assert parse_event('{"no_type": 1}')["type"] == "log"
    assert parse_event("[1, 2, 3]")["type"] == "log"


def test_worker_streams_epochs_from_real_training(tmp_path: Path):
    g = build(
        ("in", "core.input", {"shape": "16"}),
        ("d1", "core.dense", {"units": 8}),
        ("out", "core.output", {}),
        ("data", "data.synthetic", {"kind": "classification", "n_samples": 240,
                                     "n_features": 16, "n_classes": 3,
                                     "noise": 0.3, "seed": 7}),
        ("opt", "train.sgd", {"lr": 0.05, "momentum": 0.9, "nesterov": False}),
        ("loss", "train.loss_cross_entropy", {"label_smoothing": 0.0}),
        ("trainer", "train.trainer", {"epochs": 2, "batch_size": 32,
                                       "device": "cpu", "seed": 7,
                                       "early_stopping_patience": 0}),
        ("f1", "eval.f1", {"average": "macro"}),
    )
    assert g.validate() == []
    script = export_training(g, "pytorch", tmp_path)
    events = run_worker(script, tmp_path)

    types = [e["type"] for e in events]
    assert types[-1] == "done" and events[-1]["returncode"] == 0
    epochs = [e for e in events if e["type"] == "epoch"]
    assert len(epochs) == 2
    assert [e["epoch"] for e in epochs] == [1, 2]
    assert all(e["total"] == 2 for e in epochs)
    assert {"train_loss", "val_loss", "accuracy", "f1"} <= set(epochs[0]["metrics"])
    assert all(isinstance(v, float) for v in epochs[0]["metrics"].values())
    logs = [e for e in events if e["type"] == "log"]
    assert any("device" in e["line"] for e in logs)
    assert any("parameters" in e["line"] for e in logs)


def test_worker_reports_crashes_as_error_events(tmp_path: Path):
    bad = tmp_path / "bad_script.py"
    bad.write_text("raise RuntimeError('boom')\n")
    proc = subprocess.run(
        [sys.executable, str(worker_script_path()), str(bad)],
        capture_output=True, text=True, timeout=60, cwd=tmp_path,
    )
    events = [e for e in (parse_event(l) for l in proc.stdout.splitlines()) if e]
    errors = [e for e in events if e["type"] == "error"]
    assert errors and "RuntimeError: boom" in errors[0]["traceback"]
    assert events[-1]["type"] == "done" and events[-1]["returncode"] == 1
