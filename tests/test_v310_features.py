"""3.10 features. Phase 0: Keras runs and runs on other machines resume too."""
from __future__ import annotations

import json
import time

import pytest
from test_backends import fake_ssh  # noqa: F401 — fixture
from test_v39_features import _resumable_design, _run

from ai_made_easy.core import api, resume
from ai_made_easy.core.codegen import export_training
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.runner.manager import RunManager
from ai_made_easy.core.runs.history import RESUME_FILES, RunHistory

_CUT = ("import keras\n"
        "class Cut(keras.callbacks.Callback):\n"
        "    def on_epoch_begin(self, epoch, logs=None):\n"
        "        if epoch == 2:\n"
        "            raise SystemExit('power cut')\n"
        "m.CALLBACKS = [Cut()]")


@pytest.mark.slow
def test_a_resumed_keras_run_ends_exactly_where_an_uninterrupted_one_does(tmp_path):
    keras = pytest.importorskip("keras")
    if keras.backend.backend() != "torch":
        pytest.skip("compared on the torch backend")
    design = _resumable_design()
    trainer = next(n for n in design["nodes"] if n["type"] == "train.trainer")
    trainer["params"]["early_stopping_patience"] = 3      # its counters are restored too
    graph = Graph.from_dict(design)
    straight, broken = tmp_path / "straight", tmp_path / "broken"
    for folder in (straight, broken):
        folder.mkdir()
    done = _run(straight, export_training(graph, "keras", straight))
    assert done.returncode == 0, done.stderr[-2000:]
    script = export_training(graph, "keras", broken)
    cut = _run(broken, script, _CUT)
    assert cut.returncode != 0 and "epoch 2/4" in cut.stdout and "epoch 3/4" not in cut.stdout
    assert json.loads((broken / "resume.json").read_text(encoding="utf-8"))["epoch"] == 2
    resumed = _run(broken, script, "m.RESUME = True")
    assert resumed.returncode == 0, resumed.stderr[-2000:]
    assert "resuming after epoch 2" in resumed.stdout and "epoch 1/4" not in resumed.stdout

    def epochs(text: str) -> list[str]:
        return [line.rsplit(" (", 1)[0] for line in text.splitlines()
                if line.startswith(("epoch 3/4", "epoch 4/4"))]

    assert epochs(resumed.stdout) == epochs(done.stdout)
    assert json.loads((straight / "metrics.json").read_text(encoding="utf-8")) == json.loads(
        (broken / "metrics.json").read_text(encoding="utf-8"))
    assert not any((broken / name).exists() for name in RESUME_FILES)


@pytest.mark.slow
def test_a_stopped_keras_run_resumes_as_the_same_run(isolated_home, tmp_path):
    pytest.importorskip("keras")
    mgr = RunManager(RunHistory(tmp_path / "runs"))
    api.set_manager(mgr)
    try:
        run_id = mgr.start(Graph.from_dict(_resumable_design(epochs=5)), framework="keras")
        assert mgr.wait(run_id, 600)["state"] == "finished"
        folder = mgr.history.path(run_id)
        _run(folder, next(folder.glob("*_train_keras.py")), _CUT)   # left as a stop leaves it
        mgr.history.update(run_id, status="stopped")
        assert api.resume_info(run_id) == {"run_id": run_id, "resumable": True,
                                           "after_epoch": 2}
        assert api.resume_run(run_id) == {"run_id": run_id, "after_epoch": 2}
        assert mgr.wait(run_id, 600)["state"] == "finished"
        assert [e["epoch"] for e in mgr.history.epochs(run_id)] == [1, 2, 3, 4, 5]
        assert not any((folder / name).exists() for name in RESUME_FILES)
    finally:
        api.set_manager(None)


def _wait(condition, timeout: float = 120) -> None:
    deadline = time.time() + timeout
    while not condition() and time.time() < deadline:
        time.sleep(0.2)


@pytest.mark.slow
def test_a_stopped_ssh_run_resumes_on_its_host(isolated_home, fake_ssh):  # noqa: F811
    pytest.importorskip("torch")
    mgr = RunManager()
    api.set_manager(mgr)
    try:
        design = _resumable_design(epochs=400)
        next(n for n in design["nodes"] if n["id"] == "data")["params"]["n_samples"] = 6000
        run_id = mgr.start(Graph.from_dict(design), on="ssh:gpu-box")
        _wait(lambda: mgr.history.get(run_id).epochs_done >= 2)
        mgr.stop(run_id)
        assert mgr.wait(run_id, 120)["state"] == "stopped"
        folder = mgr.history.path(run_id)
        assert (folder / "resume.pt").is_file(), "the saved state did not come back"
        after = api.resume_info(run_id)["after_epoch"]
        assert after >= 2
        assert api.resume_run(run_id)["after_epoch"] == after
        _wait(lambda: mgr.history.get(run_id).epochs_done > after + 1)
        logs = mgr.get(run_id).logs
        assert any(f"resuming after epoch {after}" in line for line in logs), logs[-20:]
        assert any("connected" in line for line in logs)
        mgr.stop(run_id)
        mgr.wait(run_id, 120)
        record = mgr.history.get(run_id)
        assert record.backend == "ssh:gpu-box"
        numbers = [e["epoch"] for e in mgr.history.epochs(run_id)]
        assert numbers == list(range(1, len(numbers) + 1)) and len(numbers) > after + 1
    finally:
        api.set_manager(None)


def test_a_resumed_remote_run_carries_its_state(isolated_home, tmp_path):
    import io
    import tarfile

    mgr = RunManager(RunHistory(tmp_path / "runs"))
    graph = Graph.from_dict(_resumable_design())
    record = mgr.history.create(graph.to_dict(), framework="pytorch")
    folder = mgr.history.path(record.run_id)
    training = export_training(graph, "pytorch", folder)
    (folder / "resume.pt").write_bytes(b"state")
    (folder / "resume.json").write_text('{"epoch": 2, "epochs": 4}', encoding="utf-8")
    bundle = resume._bundle(record.run_id, folder, training, graph)
    (folder / resume.SCRIPT).write_text("# wrapper", encoding="utf-8")
    with tarfile.open(fileobj=io.BytesIO(bundle.tar_bytes()), mode="r:gz") as tar:
        names = set(tar.getnames())
    assert {resume.SCRIPT, training.name, "resume.pt", "resume.json", "bundle.json"} <= names
    assert bundle.metadata()["script"] == resume.SCRIPT


def test_a_finished_run_drops_its_resume_files(tmp_path):
    history = RunHistory(tmp_path / "runs")
    record = history.create(_resumable_design(), framework="keras")
    folder = history.path(record.run_id)
    for name in RESUME_FILES:
        (folder / name).write_text("x", encoding="utf-8")
    history.finalize(record.run_id, "stopped", 1)
    assert all((folder / name).exists() for name in RESUME_FILES)
    history.finalize(record.run_id, "finished", 0)
    assert not any((folder / name).exists() for name in RESUME_FILES)


def test_the_keras_template_saves_and_clears_its_resume_state():
    from ai_made_easy.core.training.generate import generate_training

    source = generate_training(Graph.from_dict(_resumable_design()), "keras")
    assert "RESUME = False" in source and "class SaveResume" in source
    assert "initial_epoch=first_epoch" in source
    assert 'for name in (RESUME_FILE, "resume.extra.pkl", "resume.json")' in source
