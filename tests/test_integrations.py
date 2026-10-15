"""3.2 Phase 3 — work with your tools: MLflow / W&B tracking, registry mirrors, the
Hugging Face Hub, Kaggle and Hub datasets, and run export.

MLflow runs for real on a local SQLite store and W&B in offline mode; the Hub and
Kaggle clients are faked (real calls only run in the nightly job).
"""
from __future__ import annotations

import glob
import json
import shutil
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from conftest import tiny_classifier_dict

from ai_made_easy.core import tracking
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.integrations import datasets, hub

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def home(isolated_home, monkeypatch):
    monkeypatch.delenv("AIME_TRACKING", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    import ai_made_easy.core.api as api

    monkeypatch.setattr(api, "_manager", None, raising=False)
    return isolated_home


def _train(meta: dict | None = None, epochs: int = 3):
    pytest.importorskip("torch")
    from ai_made_easy.core.runner.manager import RunManager

    graph = tiny_classifier_dict(epochs=epochs)
    graph["meta"] = meta or {}
    mgr = RunManager()
    run_id = mgr.start(Graph.from_dict(graph), "pytorch", project="trk")
    status = mgr.wait(run_id, timeout=600)
    assert status["state"] == "finished", status
    return mgr, run_id


# ===================================================================== tracking

def test_tracker_selection(monkeypatch):
    monkeypatch.delenv("AIME_TRACKING", raising=False)
    assert tracking.parse("MLflow, W&B") == ["mlflow", "wandb"]
    assert tracking.parse(["wandb", "wandb", "off"]) == ["wandb"]
    with pytest.raises(tracking.TrackingError, match="unknown tracker"):
        tracking.parse("neptune")
    assert tracking.selected({"tracking": "wandb"}) == ["wandb"]
    monkeypatch.setenv("AIME_TRACKING", "mlflow")
    assert tracking.selected({"tracking": "wandb"}) == ["wandb", "mlflow"]
    assert tracking.selected(None) == ["mlflow"]


def _wandb_history(root: Path) -> list[dict]:
    """Epoch rows from a W&B offline run file (a LevelDB-style log of protobufs)."""
    from wandb.proto import wandb_internal_pb2 as pb

    raw = Path(glob.glob(str(root / "wandb" / "offline-run-*" / "*.wandb"))[0]).read_bytes()
    pos, block, buf, rows = 7, 32768, b"", []
    while pos + 7 <= len(raw):
        left = block - pos % block
        if left < 7:
            pos += left
            continue
        _crc, length, kind = struct.unpack("<IHB", raw[pos:pos + 7])
        pos += 7
        if length == 0 and kind == 0:
            break
        buf += raw[pos:pos + length]
        pos += length
        if kind in (1, 4):
            rec = pb.Record()
            rec.ParseFromString(buf)
            buf = b""
            if rec.WhichOneof("record_type") == "history":
                items = {(i.key or "/".join(i.nested_key)): json.loads(i.value_json)
                         for i in rec.history.item}
                rows.append({k: v for k, v in items.items() if not k.startswith("_")})
    return rows


def test_run_mirrored_to_mlflow_and_wandb(home):
    pytest.importorskip("mlflow")
    pytest.importorskip("wandb")
    mgr, run_id = _train({"tracking": "mlflow,wandb"})
    record = mgr.history.get(run_id)
    local = mgr.history.epochs(run_id)
    assert len(local) == 3
    assert {t["tracker"] for t in record.tracking} == {"mlflow", "wandb"}

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking.mlflow_uri())
    remote = next(t["id"] for t in record.tracking if t["tracker"] == "mlflow")
    run = client.get_run(remote)
    assert run.info.status == "FINISHED"
    assert run.data.tags["aime.run_id"] == run_id
    assert run.data.params["trainer.epochs"] == "3"
    for key in local[0]["metrics"]:
        if not isinstance(local[0]["metrics"][key], (int, float)):
            continue
        history = {m.step: m.value for m in client.get_metric_history(remote, key)}
        assert history == pytest.approx({e["epoch"]: e["metrics"][key] for e in local})
    for key, value in record.final_metrics.items():
        assert run.data.metrics[f"final_{key}"] == pytest.approx(value)
    files = {f.path for f in client.list_artifacts(remote)}
    assert "design.json" in files and "run" in files
    assert "metrics.json" in {Path(f.path).name for f in client.list_artifacts(remote, "run")}

    rows = _wandb_history(home / "wandb")
    assert [r["epoch"] for r in rows] == [e["epoch"] for e in local]
    for row, event in zip(rows, local, strict=True):
        for key, value in event["metrics"].items():
            if isinstance(value, (int, float)):
                assert row[key] == pytest.approx(value)


def test_a_failing_tracker_never_stops_the_run(home, monkeypatch):
    pytest.importorskip("mlflow")
    monkeypatch.setenv("AIME_TRACKING", "mlflow")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "bogus-scheme://nowhere")
    mgr, run_id = _train(epochs=1)
    assert any("MLflow tracking is off" in line for line in mgr.get(run_id).logs)
    assert mgr.history.get(run_id).tracking == []


def test_registry_mirrors_and_mlflow_models_open_as_designs(home, tmp_path):
    pytest.importorskip("mlflow")
    import torch

    from ai_made_easy.core import api

    mgr, run_id = _train(epochs=1)
    monkey_manager(api, mgr)
    mv = api.register_model(run_id, "tiny")
    mv = api.set_model_stage("tiny", mv["version"], "production", also=["mlflow"])
    where = mv["mirrors"][0]
    assert where["target"] == "mlflow" and where["uri"] == "models:/tiny/1"
    import mlflow

    mlflow.set_tracking_uri(tracking.mlflow_uri())
    folder = Path(mlflow.artifacts.download_artifacts("models:/tiny@production",
                                                      dst_path=str(tmp_path / "dl")))
    assert list(folder.glob("*_best.pt")) and (folder / "run.json").exists()

    # an MLflow PyTorch model (MLmodel + a saved module) opens through the importer
    model = torch.nn.Sequential(torch.nn.Linear(6, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2))
    art = tmp_path / "art" / "model"
    (art / "data").mkdir(parents=True)
    torch.save(model, art / "data" / "model.pth")
    (art / "MLmodel").write_text("flavors:\n  pytorch:\n    model_data: data\n")
    client = mlflow.MlflowClient(tracking_uri=tracking.mlflow_uri())
    exp = client.create_experiment("imports", artifact_location=(tmp_path / "mlart").as_uri())
    run = client.create_run(exp)
    client.log_artifacts(run.info.run_id, str(art), "model")
    kind, file = tracking.open_mlflow_model(f"runs:/{run.info.run_id}/model",
                                            tmp_path / "open")
    assert kind == "torch" and file.name == "model.pth"
    result = api.open_mlflow_model(f"runs:/{run.info.run_id}/model", input_shape=[6])
    types = [n["type"] for n in result["graph"]["nodes"]]
    assert types.count("core.dense") == 2, types
    assert result["ok"], result


def monkey_manager(api, mgr) -> None:  # noqa: ANN001
    api._manager = mgr


# ============================================================================ hub

class FakeHub:
    """The HfApi calls the integration makes, backed by folders."""

    def __init__(self, root: Path, signed_in: bool = True) -> None:
        self.root, self.signed_in, self.repos = root, signed_in, {}

    def whoami(self) -> dict:
        if not self.signed_in:
            raise OSError("no token")
        return {"name": "me"}

    def create_repo(self, repo_id, repo_type="model", private=True, exist_ok=False,  # noqa: ANN001
                    space_sdk=None):  # noqa: ANN001
        self.repos[repo_id] = {"type": repo_type, "private": private, "sdk": space_sdk}
        (self.root / repo_type / repo_id).mkdir(parents=True, exist_ok=True)
        return f"https://huggingface.co/{repo_id}"

    def upload_folder(self, folder_path, repo_id, repo_type="model",  # noqa: ANN001
                      commit_message=""):  # noqa: ANN001
        shutil.copytree(folder_path, self.root / repo_type / repo_id, dirs_exist_ok=True)

    def snapshot_download(self, repo_id, repo_type="model", revision=None,  # noqa: ANN001
                          local_dir=None):  # noqa: ANN001
        shutil.copytree(self.root / repo_type / repo_id, local_dir, dirs_exist_ok=True)
        return local_dir


def test_hub_push_pull_round_trip_predicts(home, tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from ai_made_easy.core import api

    fake = FakeHub(tmp_path / "hubstore")
    monkeypatch.setattr(hub, "_api", lambda: fake)
    mgr, run_id = _train(epochs=2)
    monkey_manager(api, mgr)

    pushed = api.push_to_hub("me/tiny-mlp", run_id=run_id)
    assert pushed["private"] and fake.repos["me/tiny-mlp"]["type"] == "model"
    repo = tmp_path / "hubstore" / "model" / "me" / "tiny-mlp"
    card = (repo / "README.md").read_text()
    assert card.startswith("---\nlibrary_name: pytorch\ntags:\n- ai-made-easy\n")
    assert "# Model card" in card and "aime hub pull me/tiny-mlp" in card
    assert {"aime.json", "run.json", "design.json"} <= {p.name for p in repo.iterdir()}
    assert list(repo.glob("*_best.pt"))

    mv = api.pull_from_hub("me/tiny-mlp")
    assert mv["name"] == "tiny-mlp" and mv["run_id"] == run_id
    package = api.deploy_model("tiny-mlp", "latest", str(tmp_path / "server"), ["onnx"])
    assert package["verified"] and package["formats"]["onnx"]["ok"]   # loads and predicts

    space = api.deploy_to_space("me/tiny-api", run_id=run_id)
    assert fake.repos["me/tiny-api"] == {"type": "space", "private": True, "sdk": "docker"}
    readme = (tmp_path / "hubstore" / "space" / "me" / "tiny-api" / "README.md").read_text()
    assert readme.startswith("---\ntitle: tiny-api\nsdk: docker\napp_port: 8000\n")
    assert space["url"] == "https://huggingface.co/spaces/me/tiny-api"


def test_hub_errors_name_the_fix(home, tmp_path, monkeypatch):
    monkeypatch.setattr(hub, "_api", lambda: FakeHub(tmp_path, signed_in=False))
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(hub.HubError, match="hf auth login"):
        hub.push_model(run, "me/x")
    with pytest.raises(hub.HubError, match="owner/name"):
        hub.push_model(run, "no-owner")
    monkeypatch.setattr(hub, "_api", lambda: FakeHub(tmp_path))
    with pytest.raises(hub.HubError, match="not a run folder"):
        hub.push_model(run, "me/x")
    (tmp_path / "model" / "me" / "other").mkdir(parents=True)
    (tmp_path / "model" / "me" / "other" / "model.safetensors").write_text("x")
    with pytest.raises(hub.HubError, match="not pushed by AI Made Easy"):
        hub.pull_model("me/other")


# ======================================================================= datasets

class FakeKaggle:
    calls: list = []

    def __init__(self) -> None:
        pass

    def download(self, ref, dest, *, competition=False, file=""):  # noqa: ANN001
        FakeKaggle.calls.append((ref, competition, file))
        dest = Path(dest)
        rows = "a,b,label\n" + "".join(f"{i},{i % 3},{i % 2}\n" for i in range(60))
        if competition:   # competitions arrive as one zip
            with zipfile.ZipFile(dest / f"{ref}.zip", "w") as zf:
                zf.writestr("train.csv", rows)
                zf.writestr("test.csv", rows.replace(",label", "").replace(",0\n", "\n"))
                zf.writestr("sample_submission.csv", "id,label\n")
        else:
            (dest / "rows.csv").write_text(rows)

    def submit(self, competition, file, message):  # noqa: ANN001
        return type("R", (), {"message": f"{competition}: {Path(file).name} ({message})"})()


def test_sources_parse():
    assert datasets.parse("kaggle:owner/name") == ("kaggle", "owner/name", "")
    assert datasets.parse("kaggle:owner/name/file.csv") == ("kaggle", "owner/name",
                                                            "file.csv")
    assert datasets.parse("kaggle-competition:titanic") == ("kaggle-competition", "titanic",
                                                            "")
    assert datasets.parse("hf:user/repo:test") == ("hf", "user/repo", "test")
    assert datasets.parse("hf:mnist") == ("hf", "mnist", "train")
    assert not datasets.is_source("data/train.csv")
    for bad in ("kaggle:only", "hf:a b", "kaggle-competition:../x", "s3:bucket"):
        with pytest.raises(datasets.SourceError):
            datasets.parse(bad)


def test_kaggle_datasets_download_once_and_detect(home, monkeypatch):
    from ai_made_easy.core import api
    from ai_made_easy.core.runner.backends import kaggle

    FakeKaggle.calls = []
    monkeypatch.setattr(kaggle, "KaggleClient", FakeKaggle)
    found = api.fetch_dataset("kaggle-competition:titanic")
    assert found["path"].endswith("train.csv")
    assert found["facts"]["kind"] == "table" and found["facts"]["rows"] == 60
    assert found["facts"]["target"] == "label"
    assert api.detect_data("kaggle-competition:titanic")["rows"] == 60   # cached
    assert FakeKaggle.calls == [("titanic", True, "")]
    assert api.fetch_dataset("kaggle:me/rows")["path"].endswith("rows.csv")
    ranked = api.recommend_recipes("multiclass", path="kaggle:me/rows", target="label")
    assert ranked["suggestions"] and ranked["facts"]["rows"] == 60

    with pytest.raises(api.ApiError, match="confirmation"):
        api.submit_to_kaggle("titanic", found["path"], "first")
    result = api.submit_to_kaggle("titanic", found["path"], "first", confirm=True)
    assert result["result"] == "titanic: train.csv (first)"


def test_unsafe_zip_is_rejected(tmp_path):
    with zipfile.ZipFile(tmp_path / "bad.zip", "w") as zf:
        zf.writestr("../escape.txt", "x")
    with pytest.raises(datasets.SourceError, match="unsafe"):
        datasets._unzip_all(tmp_path)


def test_hub_datasets_become_tables_or_class_folders(home, monkeypatch):
    hf = pytest.importorskip("datasets")
    from PIL import Image

    table = hf.Dataset.from_dict({"x": [float(i) for i in range(40)],
                                  "label": [i % 2 for i in range(40)]})
    monkeypatch.setattr(datasets, "_load_hf", lambda repo, split: table)
    path = datasets.fetch("hf:me/table")
    assert path.name == "train.parquet"
    from ai_made_easy.core import api

    assert api.detect_data(str(path))["rows"] == 40

    images = hf.Dataset.from_dict(
        {"image": [Image.new("RGB", (8, 8), (i * 20, 0, 0)) for i in range(6)],
         "label": [0, 1, 0, 1, 0, 1]},
        features=hf.Features({"image": hf.Image(),
                              "label": hf.ClassLabel(names=["cat", "dog"])}))
    monkeypatch.setattr(datasets, "_load_hf", lambda repo, split: images)
    folder = datasets.fetch("hf:me/pets")
    assert sorted(p.name for p in (folder / "images").iterdir()) == ["cat", "dog"]
    assert len(list((folder / "images" / "dog").glob("*.png"))) == 3


# ===================================================================== run export

def test_runs_export_csv_parquet_and_epochs(home, tmp_path):
    import pandas as pd

    from ai_made_easy.core import api

    mgr, run_id = _train(epochs=2)
    monkey_manager(api, mgr)
    out = api.export_runs(str(tmp_path / "runs.csv"))
    frame = pd.read_csv(out["path"])
    assert frame["run_id"].tolist() == [run_id] and frame["status"][0] == "finished"
    assert "param.trainer.epochs" in frame and any(c.startswith("final.") for c in frame)
    pytest.importorskip("pyarrow")
    epochs = api.export_runs(str(tmp_path / "epochs.parquet"), epochs=True)
    frame = pd.read_parquet(epochs["path"])
    assert frame["epoch"].tolist() == [1, 2] and "val_loss" in frame
    with pytest.raises(api.ApiError, match="unknown format"):
        api.export_runs(str(tmp_path / "runs.xlsx"))


def test_cli_tools(home, tmp_path, monkeypatch):
    env = {"AIME_HOME": str(home), "PYTHONPATH": str(ROOT), "KERAS_BACKEND": "torch",
           "PATH": "/usr/bin:/bin"}

    def aime(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-m", "ai_made_easy.cli", *args],
                              capture_output=True, text=True, env=env, timeout=300)

    status = aime("tracking")
    assert status.returncode == 0 and "tracking: off" in status.stdout
    project = tmp_path / "p.json"
    project.write_text(json.dumps({**tiny_classifier_dict(), "meta": {"tracking": "wandb"}}))
    assert "tracking: wandb" in aime("tracking", str(project)).stdout
    exported = aime("runs", "export", "-o", str(tmp_path / "none.csv"))
    assert exported.returncode == 0 and "wrote 0 row(s)" in exported.stdout
    bad = aime("data", "fetch", "kaggle:only")
    assert bad.returncode == 1 and "owner/name" in bad.stderr
    usage = aime("hub", "push", "only-one")
    assert usage.returncode == 2


# ======================================================================== desktop

def test_desktop_tracking_fetch_and_share(qtbot, home, monkeypatch):
    pytest.importorskip("PySide6")
    from ai_made_easy.core.runner.backends import kaggle
    from ai_made_easy.ui.features.experiments import ExperimentsPage
    from ai_made_easy.ui.features.tools import FetchDatasetDialog, TrackingDialog

    dialog = TrackingDialog(None, ["wandb"])
    qtbot.addWidget(dialog)
    assert dialog.checks["wandb"].isChecked() and not dialog.checks["mlflow"].isChecked()
    dialog.checks["mlflow"].setChecked(True)
    assert dialog.selected() == ["mlflow", "wandb"]

    monkeypatch.setattr(kaggle, "KaggleClient", FakeKaggle)
    fetch = FetchDatasetDialog(None)
    qtbot.addWidget(fetch)
    fetch.source.setText("kaggle-competition:titanic")
    fetch.fetch_btn.click()
    qtbot.waitUntil(lambda: fetch.use_btn.isEnabled(), timeout=30000)
    assert fetch.path.endswith("train.csv") and "60 rows" in fetch.result.text()

    page = ExperimentsPage()
    qtbot.addWidget(page)
    assert not page.model_share_btn.isEnabled()
    shared = []
    page.model_share_requested.connect(lambda n, v, t: shared.append((n, v, t)))
    page._selected_model = lambda: ("tiny", 2)
    page.model_share_btn.menu().findChild(type(page.model_share_btn.menu().actions()[0]),
                                          "share_space").trigger()
    assert shared == [("tiny", 2, "space")]


def test_routes(home, tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from ai_made_easy.core.runner.backends import kaggle
    from ai_made_easy.server.app import create_app

    monkeypatch.setattr(kaggle, "KaggleClient", FakeKaggle)
    monkeypatch.setattr(hub, "_api", lambda: FakeHub(tmp_path / "hubstore"))
    with TestClient(create_app(token="", projects_dir=tmp_path)) as client:
        status = client.get("/api/tracking").json()
        assert status["selected"] == [] and {t["id"] for t in status["trackers"]} == \
            {"mlflow", "wandb"}
        chosen = client.post("/api/tracking", json={"graph": {"meta": {"tracking": "wandb"}}})
        assert chosen.json()["selected"] == ["wandb"]
        csv = client.get("/api/export/runs?format=csv")
        assert csv.status_code == 200
        assert 'filename="runs.csv"' in csv.headers["content-disposition"]
        assert client.get("/api/export/runs?format=xlsx").status_code == 400
        fetched = client.post("/api/data/fetch", json={"source": "kaggle-competition:titanic"})
        assert fetched.json()["facts"]["rows"] == 60
        bad = client.post("/api/data/fetch", json={"source": "kaggle:only"})
        assert bad.status_code == 400 and "owner/name" in bad.json()["detail"]
        refused = client.post("/api/kaggle/submit", json={
            "competition": "titanic", "file": fetched.json()["path"], "message": "m"})
        assert refused.status_code == 400 and "confirmation" in refused.json()["detail"]
        ok = client.post("/api/kaggle/submit", json={
            "competition": "titanic", "file": fetched.json()["path"], "message": "m",
            "confirm": True})
        assert ok.status_code == 200
        missing = client.post("/api/hub/push", json={"repo_id": "me/x", "run_id": "nope"})
        assert missing.status_code in (400, 404)


# ========================================================== online (nightly job only)

_ONLINE = bool(__import__("os").environ.get("AIME_ONLINE"))


@pytest.mark.skipif(not _ONLINE, reason="needs AIME_ONLINE=1 (the nightly job)")
def test_real_hub_dataset(home):
    pytest.importorskip("datasets")
    from ai_made_easy.core import api

    found = api.fetch_dataset("hf:scikit-learn/iris")
    assert found["facts"]["rows"] == 150


@pytest.mark.skipif(not (_ONLINE and __import__("os").environ.get("KAGGLE_KEY")),
                    reason="needs AIME_ONLINE=1 and a Kaggle token (the nightly job's secret)")
def test_real_kaggle_dataset(home):
    pytest.importorskip("kaggle")
    from ai_made_easy.core import api

    found = api.fetch_dataset("kaggle:uciml/iris")
    assert found["facts"]["rows"] == 150


@pytest.mark.skipif(not (_ONLINE and __import__("os").environ.get("HF_TOKEN")
                         and __import__("os").environ.get("AIME_HF_TEST_REPO")),
                    reason="needs AIME_ONLINE=1, HF_TOKEN and AIME_HF_TEST_REPO (a private "
                           "repo the nightly job may overwrite)")
def test_real_hub_push_round_trip(home, tmp_path):
    import os

    from ai_made_easy.core import api

    mgr, run_id = _train(epochs=1)
    monkey_manager(api, mgr)
    repo = os.environ["AIME_HF_TEST_REPO"]
    api.push_to_hub(repo, run_id=run_id)
    mv = api.pull_from_hub(repo, name="hub-round-trip")
    package = api.deploy_model(mv["name"], mv["version"], str(tmp_path / "server"), ["onnx"])
    assert package["verified"] and package["formats"]["onnx"]["ok"]
