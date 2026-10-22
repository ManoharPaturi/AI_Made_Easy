"""3.6 Phase 1: try a whole file of inputs, and see an image model's Grad-CAM in Try it."""
from __future__ import annotations

import base64
import csv
import io
import sys
import zipfile
from pathlib import Path

import pytest
from conftest import tiny_classifier_dict

pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from verify_training import chain, write_images  # noqa: E402

from ai_made_easy.core import api, tryit  # noqa: E402
from ai_made_easy.core.graph import Graph  # noqa: E402
from ai_made_easy.core.runner.manager import RunManager  # noqa: E402
from ai_made_easy.core.runs.history import RunHistory  # noqa: E402


def _train(mgr: RunManager, graph: Graph) -> str:
    run_id = mgr.start(graph, "auto")
    assert mgr.wait(run_id, 900)["state"] == "finished", mgr.history.get(run_id).error
    return run_id


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("trybatch")
    mgr = RunManager(RunHistory(root / "runs"))
    images = write_images(root)
    cnn = chain("cams", "3, 16, 16", [("core.conv2d", {"out_channels": 6}), ("core.relu", {}),
                                      ("core.global_avgpool2d", {}),
                                      ("core.dense", {"units": 2})],
                [("data.image_folder", {"root": str(images)}),
                 ("prep.resize", {"height": 16, "width": 16}),
                 ("train.trainer", {"epochs": 1, "batch_size": 8, "device": "cpu"})])
    table = root / "table.csv"
    table.write_text("size,weight,label\n" + "\n".join(
        f"{i % 7},{(i * 3) % 5},{'big' if i % 7 > 3 else 'small'}" for i in range(120)), encoding="utf-8")
    tab = chain("tab", "2", [("core.dense", {"units": 8}), ("core.relu", {}),
                             ("core.dense", {"units": 2})],
                [("data.csv", {"path": str(table), "target_column": "label"}),
                 ("train.trainer", {"epochs": 2, "batch_size": 16, "device": "cpu"})])
    out = {"arrays": _train(mgr, Graph.from_dict(tiny_classifier_dict(epochs=1))),
           "images": _train(mgr, cnn), "images_root": images, "records": _train(mgr, tab)}
    return mgr, out


@pytest.fixture()
def runs(trained):
    mgr, out = trained
    api.set_manager(mgr)
    yield out
    api.try_service().close()


def _rows(text: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(text)))


def test_a_csv_of_arrays_comes_back_with_a_prediction_per_row(runs):
    data = "\n".join(",".join(str((i + j) % 3 - 1.0) for j in range(16)) for i in range(70))
    got = api.try_file(runs["arrays"], "rows.csv", data.encode())
    rows = _rows(got["csv"])
    assert got["rows"] == 70 and len(rows) == 70                 # more than one chunk of 64
    assert rows[0]["row"] == "1" and rows[0]["label"]
    probs = [k for k in rows[0] if k.startswith("p(")]
    assert len(probs) == 3 and abs(sum(float(rows[0][k]) for k in probs) - 1) < 1e-3
    same = api.try_run(runs["arrays"], [[float((0 + j) % 3 - 1.0) for j in range(16)]])
    assert same["predictions"][0]["label"] == rows[0]["label"]
    assert got["filename"] == "rows_predictions.csv"


def test_bad_files_say_what_is_wrong(runs):
    with pytest.raises(api.ApiError, match="has 3 values; the model takes 16"):
        api.try_file(runs["arrays"], "rows.csv", b"1,2,3\n")
    with pytest.raises(api.ApiError, match="CSV"):
        api.try_file(runs["arrays"], "rows.txt", b"1,2,3\n")


def test_a_table_model_predicts_a_csv(runs):
    assert api.try_describe(runs["records"])["batch_formats"] == ".csv"
    got = api.try_file(runs["records"], "people.csv", b"size,weight\n6,2\n1,4\n")
    rows = _rows(got["csv"])
    assert [r["row"] for r in rows] == ["1", "2"] and all(r["label"] for r in rows)


def _zip_of_images(root: Path) -> bytes:
    buf = io.BytesIO()
    files = sorted(p for p in root.rglob("*.png"))[:5]
    with zipfile.ZipFile(buf, "w") as archive:
        for p in files:
            archive.write(p, f"pics/{p.name}")
        archive.writestr("pics/readme.txt", "not an image")
        archive.writestr("__MACOSX/._x.png", b"junk")
    return buf.getvalue()


def test_a_zip_of_images_comes_back_with_a_row_per_image(runs):
    got = api.try_file(runs["images"], "pics.zip", _zip_of_images(runs["images_root"]))
    rows = _rows(got["csv"])
    assert len(rows) == 5 and all(r["file"].startswith("pics/") for r in rows)
    assert all(r["label"] for r in rows)


def test_image_predictions_can_carry_a_grad_cam(runs):
    form = api.try_describe(runs["images"])
    assert form["can_explain"] and form["batch_formats"] == ".zip"
    image = sorted(runs["images_root"].rglob("*.png"))[0]
    payload = "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode()
    got = api.try_run(runs["images"], [payload], explain=True)
    pred = got["predictions"][0]
    assert pred["label"] and pred["gradcam"]
    from PIL import Image

    cam = Image.open(io.BytesIO(base64.b64decode(pred["gradcam"])))
    assert cam.size == Image.open(image).size and cam.mode == "RGB"
    plain = api.try_run(runs["images"], [payload])["predictions"][0]
    assert "gradcam" not in plain and plain["label"] == pred["label"]


def test_models_without_a_convolution_do_not_offer_grad_cam(runs):
    assert api.try_describe(runs["arrays"])["can_explain"] is False
    with pytest.raises(api.ApiError, match="Grad-CAM"):
        api.try_run(runs["arrays"], [[0.0] * 16], explain=True)


def test_the_server_returns_a_csv_download(runs):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from ai_made_easy.server.app import create_app

    client = TestClient(create_app())
    data = "\n".join(",".join("0.5" for _ in range(16)) for _ in range(3))
    got = client.post(f"/api/runs/{runs['arrays']}/try/file",
                      files={"file": ("x.csv", data.encode(), "text/csv")})
    assert got.status_code == 200 and got.headers["content-type"].startswith("text/csv")
    assert got.headers["x-rows"] == "3" and "x_predictions.csv" in got.headers[
        "content-disposition"]
    bad = client.post(f"/api/runs/{runs['arrays']}/try/file",
                      files={"file": ("x.csv", b"1,2\n", "text/csv")})
    assert bad.status_code == 400 and "the model takes 16" in bad.json()["detail"]
    explained = client.post(f"/api/runs/{runs['arrays']}/try",
                            json={"inputs": [[0.0] * 16], "explain": True})
    assert explained.status_code == 400


def test_batch_items_reads_text_files_and_csvs(tmp_path):
    texts, names = tryit.batch_items("texts", "t.txt", b"good film\n\nbad film\n", tmp_path)
    assert texts == ["good film", "bad film"] and names == ["1", "2"]
    texts, _ = tryit.batch_items("texts", "t.csv", b"id,text\n1,hello\n2,bye\n", tmp_path)
    assert texts == ["hello", "bye"]
    rows, _ = tryit.batch_items("records", "t.csv", b"a,b\n1.5,x\n,y\n", tmp_path)
    assert rows == [{"a": 1.5, "b": "x"}, {"a": None, "b": "y"}]


def test_desktop_try_dialog_predicts_a_file_and_shows_grad_cam(runs, tmp_path):
    pytest.importorskip("PySide6")
    import time

    from PySide6 import QtCore, QtWidgets

    from ai_made_easy.ui.app import _ensure_qt_plugin_path
    from ai_made_easy.ui.features.try_dialog import TryDialog

    _ensure_qt_plugin_path()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def wait(check, seconds=60.0):
        deadline = time.time() + seconds
        while not check() and time.time() < deadline:
            app.processEvents()
            QtCore.QThread.msleep(20)
        assert check()

    dialog = TryDialog(None, api.manager().history, runs["arrays"])
    wait(lambda: dialog.predict_btn.isEnabled())
    assert dialog.file_btn.isEnabled() and dialog.explain_box.isHidden()
    source, target = tmp_path / "in.csv", tmp_path / "out.csv"
    source.write_text("\n".join(",".join("0.1" for _ in range(16)) for _ in range(4)), encoding="utf-8")
    dialog.predict_file(source, target)
    wait(lambda: "predictions saved" in dialog.result.toPlainText())
    assert len(_rows(target.read_text(encoding="utf-8"))) == 4
    dialog.deleteLater()

    dialog = TryDialog(None, api.manager().history, runs["images"])
    wait(lambda: dialog.predict_btn.isEnabled())
    assert not dialog.explain_box.isHidden()
    dialog.explain_box.setChecked(True)
    dialog.set_file(sorted(runs["images_root"].rglob("*.png"))[0])
    dialog.predict()
    wait(lambda: "% sure" in dialog.result.toPlainText())
    assert not dialog.cam.isHidden() and not dialog.cam.pixmap().isNull()
    assert "gradcam" not in dialog.result.toPlainText()
    dialog.deleteLater()
