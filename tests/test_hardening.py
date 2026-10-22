"""3.3 Phase 0: the bugs and hardening fixes, one test each."""
from __future__ import annotations

import io
import logging
import os
import re
import tempfile
import threading
import time
import zipfile
from pathlib import Path

import pytest
from conftest import tiny_classifier_dict

from ai_made_easy.core.runner import manager as manager_mod
from ai_made_easy.core.runner.manager import FINAL_STATES, RunManager, TrainingRun
from ai_made_easy.core.runs import history as history_mod
from ai_made_easy.core.runs.history import RunHistory

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ run liveness

def test_liveness_never_signals_and_spots_reused_pids(monkeypatch):
    sent: list = []
    monkeypatch.setattr(os, "kill", lambda *a: sent.append(a))
    psutil = pytest.importorskip("psutil")
    me = psutil.Process(os.getpid()).create_time()
    assert history_mod._alive(os.getpid(), me + 1)
    # a process that started long after the run was created is a reused PID
    assert not history_mod._alive(os.getpid(), me - 10 * history_mod._REUSE_SLACK)
    assert not history_mod._alive(999_999_999, time.time())
    assert sent == []  # psutil checks without os.kill (which terminates on Windows)


def test_run_ids_cannot_leave_the_runs_folder(tmp_path):
    h = RunHistory(tmp_path)
    for bad in ("C:x", "../x", "a/b", "a\\b", ".hidden", "a\0b"):
        with pytest.raises(KeyError):
            h.path(bad)


# ------------------------------------------------------------------ the run manager

def _fake_run(tmp_path: Path, mgr: RunManager, state: str = "running") -> TrainingRun:
    record = mgr.history.create(tiny_classifier_dict())
    run = TrainingRun(record.run_id, tmp_path / "s.py", mgr.history.path(record.run_id),
                      mgr.history)
    run.state = state
    mgr._runs[record.run_id] = run
    return run


def test_stop_does_not_overwrite_a_finished_run(tmp_path):
    mgr = RunManager(RunHistory(tmp_path))
    run = _fake_run(tmp_path, mgr)
    stops: list = []

    class Backend(manager_mod.backends.LocalBackend):
        def stop(self, r):  # noqa: ANN001
            stops.append(r.run_id)

    run.backend = Backend()
    run.record({"type": "done", "returncode": 0})
    assert mgr.stop(run.run_id)["state"] == "finished"
    assert stops == []
    other = _fake_run(tmp_path, mgr)
    other.backend = Backend()
    assert mgr.stop(other.run_id)["state"] == "stopped"
    assert stops == [other.run_id]


def test_attaching_twice_starts_one_poller(tmp_path, monkeypatch):
    mgr = RunManager(RunHistory(tmp_path))
    record = mgr.history.create(tiny_classifier_dict())
    mgr.history.update(record.run_id, status="running", backend="ssh:box",
                       remote={"dir": "/r"})
    started: list = []

    class Remote(manager_mod.backends.LocalBackend):
        kind, spec, name, live = "ssh", "ssh:box", "ssh", False

        def reattach(self, run):  # noqa: ANN001
            time.sleep(0.05)
            started.append(run.run_id)

    monkeypatch.setattr(manager_mod.backends, "resolve", lambda spec: Remote())
    threads = [threading.Thread(target=mgr.attach, args=(record.run_id,)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert started == [record.run_id]


def test_run_memory_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(manager_mod, "MAX_EVENTS", 50)
    monkeypatch.setattr(manager_mod, "MAX_LOG_LINES", 20)
    mgr = RunManager(RunHistory(tmp_path))
    run = _fake_run(tmp_path, mgr)
    for i in range(200):
        run.record({"type": "log", "line": str(i)})
    assert len(run.events) == 50 and run.event_count == 200
    assert run.logs[-1] == "199" and len(run.logs) == 20
    events, seen = run.events_since(190)
    assert [e["line"] for e in events] == [str(i) for i in range(190, 200)] and seen == 200
    events, _ = run.events_since(0)  # older ones are gone from memory, not repeated
    assert len(events) == 50


def test_finished_runs_are_pruned_from_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(manager_mod, "KEEP_FINISHED", 3)
    mgr = RunManager(RunHistory(tmp_path))
    runs = [_fake_run(tmp_path, mgr, state="finished") for _ in range(6)]
    live = _fake_run(tmp_path, mgr)
    mgr._prune()
    assert set(mgr._runs) == {r.run_id for r in runs[-3:]} | {live.run_id}
    assert all(r.state in FINAL_STATES for r in runs)


def test_results_zip_size_is_checked_before_unpacking(tmp_path, monkeypatch):
    monkeypatch.setenv("AIME_MAX_UPLOAD_MB", "0.001")  # 1 KB upload, 4 KB unpacked
    mgr = RunManager(RunHistory(tmp_path))
    bomb = tmp_path / "bomb.zip"
    with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("bundle.json", "{}")
        z.writestr("big.bin", b"\0" * 100_000)
    with pytest.raises(ValueError, match="unpacks to"):
        mgr.import_results(bomb)


# ------------------------------------------------------------------ interpreters

def test_subprocesses_use_python_executable():
    """``sys.executable`` is the app binary in the frozen bundle: only
    ``python_executable()`` may fall back to it. Workers and generated scripts run in
    the training interpreter, where it is right."""
    allowed = {"core/runner/manager.py", "worker/", "core/deploy/templates.py"}
    pattern = re.compile(r"subprocess\.\w+\(\s*\[[^\]]*sys\.executable|or sys\.executable")
    offenders = []
    for path in (ROOT / "ai_made_easy").rglob("*.py"):
        rel = path.relative_to(ROOT / "ai_made_easy").as_posix()
        if rel.startswith("server/static") or any(rel.startswith(a) for a in allowed):
            continue
        if pattern.search(path.read_text(encoding="utf-8")):
            offenders.append(rel)
    assert offenders == []


# ------------------------------------------------------------------ the web server

@pytest.fixture()
def client(isolated_home, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from ai_made_easy.core import api
    from ai_made_easy.server.app import create_app

    api.set_manager(RunManager(RunHistory()))
    with TestClient(create_app(token="", projects_dir=tmp_path / "projects")) as c:
        yield c
    api.set_manager(None)


def test_run_export_leaves_no_temp_folder(client):
    before = set(Path(tempfile.gettempdir()).glob("aime_runs_*"))
    got = client.get("/api/export/runs?format=csv")
    assert got.status_code == 200
    assert set(Path(tempfile.gettempdir()).glob("aime_runs_*")) == before


def test_oversized_uploads_are_refused(client, monkeypatch):
    monkeypatch.setenv("AIME_MAX_UPLOAD_MB", "0.01")  # ~10 KB
    big = b"x" * 50_000
    assert client.post("/api/runs/import", content=big).status_code == 413
    got = client.post("/api/import", data={"kind": "onnx"},
                      files={"file": ("m.onnx", io.BytesIO(big), "application/octet-stream")})
    assert got.status_code == 413
    assert client.post("/api/runs/import", content=b"").status_code == 400


def test_concurrent_saves_of_one_project_do_not_collide(client):
    graph = tiny_classifier_dict()
    codes: list = []

    def save() -> None:
        codes.append(client.put("/api/projects/same", json=graph).status_code)

    threads = [threading.Thread(target=save) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert codes == [200] * 8
    assert client.get("/api/projects/same").status_code == 200


def test_server_start_fails_runs_a_crash_left_running(isolated_home, tmp_path):
    pytest.importorskip("fastapi")
    from ai_made_easy.core import api
    from ai_made_easy.server.app import create_app

    h = RunHistory()
    rec = h.create(tiny_classifier_dict())
    h.update(rec.run_id, status="running", pid=999_999_999)
    api.set_manager(RunManager(h))
    try:
        create_app(token="", projects_dir=tmp_path / "p")
        assert h.get(rec.run_id).status == "failed"
    finally:
        api.set_manager(None)


def test_websocket_ends_for_a_run_nothing_is_running(client, monkeypatch):
    from ai_made_easy.core import api
    from ai_made_easy.server import app as app_mod

    monkeypatch.setattr(app_mod, "_STALE_POLLS", 1)
    h = api.manager().history
    rec = h.create(tiny_classifier_dict())
    h.update(rec.run_id, status="running", backend="ssh:gone")  # remote, not followed
    with client.websocket_connect(f"/api/runs/{rec.run_id}/events") as ws:
        while (msg := ws.receive_json())["type"] != "status":
            pass
    assert msg["status"] == "running"


def test_access_log_hides_the_token():
    pytest.importorskip("fastapi")
    from ai_made_easy.server.app import RedactToken

    record = logging.LogRecord("uvicorn.access", logging.INFO, "", 0,
                               '%s - "%s %s HTTP/%s" %d', ("127.0.0.1", "GET",
                               "/api/export/runs?format=csv&token=s3cret", "1.1", 200), None)
    RedactToken().filter(record)
    assert "s3cret" not in record.getMessage() and "token=***" in record.getMessage()
