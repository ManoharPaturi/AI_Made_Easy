"""Test-session environment and shared fixtures.

Headless Qt, Keras on the PyTorch backend, and an isolated ``$AIME_HOME`` so
tests never touch the user's run history, registry or settings.
"""
from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("AIME_HOME", tempfile.mkdtemp(prefix="aime_test_home_"))

import pytest  # noqa: E402


def tiny_classifier_dict(epochs: int = 2, lr: float = 0.05, units: int = 8,
                         name: str = "tiny") -> dict:
    """A small trainable MLP on synthetic data (a few seconds on CPU)."""
    nodes = [
        ("in", "core.input", {"shape": "16"}),
        ("d1", "core.dense", {"units": units}),
        ("act", "core.relu", {}),
        ("d2", "core.dense", {"units": 3}),
        ("out", "core.output", {}),
        ("data", "data.synthetic", {"kind": "classification", "n_samples": 240,
                                    "n_features": 16, "n_classes": 3,
                                    "noise": 0.3, "seed": 7}),
        ("opt", "train.sgd", {"lr": lr, "momentum": 0.9, "nesterov": False}),
        ("loss", "train.loss_cross_entropy", {"label_smoothing": 0.0}),
        ("trainer", "train.trainer", {"epochs": epochs, "batch_size": 32,
                                      "device": "cpu", "seed": 7,
                                      "early_stopping_patience": 0}),
        ("acc", "eval.accuracy", {}),
    ]
    chain = ["in", "d1", "act", "d2", "out"]
    return {
        "name": name,
        "nodes": [{"id": i, "type": t, "params": p, "position": [0, 0]} for i, t, p in nodes],
        "edges": [{"from": f"{a}/out", "to": f"{b}/in"} for a, b in zip(chain, chain[1:], strict=False)],
        "meta": {},
    }


@pytest.fixture
def tiny_classifier() -> dict:
    return tiny_classifier_dict()


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """A fresh ``$AIME_HOME`` for one test."""
    monkeypatch.setenv("AIME_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


@pytest.fixture(autouse=True)
def _forget_api_manager():
    """The process-wide RunManager keeps the run history of the test that made it (the
    server makes one at startup): later tests must get a fresh one for their own home."""
    from ai_made_easy.core import api

    yield
    api.set_manager(None)


@pytest.fixture(autouse=True)
def _forget_custom_blocks():
    """Custom blocks registered by a test must not leak into later tests."""
    from ai_made_easy.core.registry import get_registry

    registry = get_registry()
    before = {b.type_id for b in registry.all()}
    yield
    for block in registry.all():
        if block.type_id.startswith("custom.") and block.type_id not in before:
            registry.unregister(block.type_id)


# pytest's "internal error" code until a session actually finishes: a crash before or
# during collection must never turn into a passing CI job
_exit_status = {"code": 3}


def pytest_sessionfinish(session, exitstatus):  # noqa: ANN001
    _exit_status["code"] = int(exitstatus)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):  # noqa: ANN001
    """On CI, skip native teardown: Qt, torch and Keras destructors can segfault at
    interpreter exit (exit code 139) after every test has passed."""
    if os.environ.get("CI"):
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(_exit_status["code"])


@pytest.fixture(autouse=True)
def _no_recovery_prompt(monkeypatch):
    """Desktop contexts from earlier tests keep their two-minute autosave timers, and one
    can write a recovery file while the next window boots; the modal "Recover unsaved
    work" prompt would then block a headless run. Tests call the autosave parts directly."""
    try:
        from ai_made_easy.ui.context import AppContext
    except ImportError:  # no Qt in this environment
        return
    monkeypatch.setattr(AppContext, "_offer_recovery", lambda self: None)


def _flush_qt_deletes() -> None:
    """Run the deletions queued by ``deleteLater()``. Without an event loop Qt never
    gets to them, so every test's window would stay alive until Python exits."""
    if "PySide6.QtWidgets" not in sys.modules:
        return
    from PySide6 import QtCore, QtWidgets

    if QtWidgets.QApplication.instance() is not None:
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


@pytest.fixture(autouse=True)
def _delete_closed_windows():
    yield
    _flush_qt_deletes()


@pytest.fixture(scope="session", autouse=True)
def _close_windows_at_exit():
    """Close any app window still open while Python is fully running, so its services
    and timers stop before interpreter exit tears Qt down in no particular order.
    Closing only: by now Python may have collected parts of old widgets (pyqtgraph's
    layout items), and forcing a C++ delete through them crashes."""
    yield
    if "PySide6.QtWidgets" not in sys.modules:
        return
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance()
    if app is None:
        return
    for window in app.topLevelWidgets():
        ctx = getattr(window, "ctx", None)
        if ctx is not None:
            ctx.project_store.mark_clean()   # no "save changes?" prompt
            window.close()
