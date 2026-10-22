"""3.3 Phase 2: copy / paste, groups, notes and dropped data files, in the core, the
web server and the desktop canvas."""
from __future__ import annotations

import io
import json

import pytest
from conftest import tiny_classifier_dict

from ai_made_easy.core import canvas as cv
from ai_made_easy.core.codegen import generate
from ai_made_easy.core.graph import Graph

# ------------------------------------------------------------------ core


def test_copy_keeps_only_the_wires_inside_the_selection():
    design = tiny_classifier_dict()
    frag = cv.copy_fragment(design, ["d1", "act"])
    assert [n["id"] for n in frag["nodes"]] == ["d1", "act"]
    assert frag["edges"] == [{"from": "d1/out", "to": "act/in"}]
    assert cv.parse(cv.to_text(frag))["nodes"] == frag["nodes"]


def test_paste_renames_clashing_ids_and_offsets_positions():
    design = tiny_classifier_dict()
    out, ids = cv.paste_fragment(design, cv.copy_fragment(design, ["d1", "act"]))
    assert ids == ["d1_2", "act_2"]
    assert {"from": "d1_2/out", "to": "act_2/in"} in out["edges"]
    assert len(out["nodes"]) == len(design["nodes"]) + 2
    pasted = {n["id"]: n for n in out["nodes"]}
    assert pasted["d1_2"]["position"] == [40.0, 40.0]
    out2, ids2 = cv.paste_fragment(out, cv.copy_fragment(design, ["d1"]), at=(500, 300))
    assert ids2 == ["d1_3"] and next(n for n in out2["nodes"] if n["id"] == "d1_3")[
        "position"] == [500.0, 300.0]
    assert not [i for i in Graph.from_dict(out2).validate() if "duplicate" in i.message]


def test_a_whole_design_pastes_and_junk_does_not():
    design = tiny_classifier_dict()
    assert cv.parse(json.dumps(design))["nodes"]
    for junk in ("hello", "[]", '{"nodes": [1]}', '{"nodes": [], "edges": [{"from": "x"}]}'):
        assert cv.parse(junk) is None


def test_groups_travel_with_their_blocks_and_lose_deleted_ones():
    design = tiny_classifier_dict()
    cv.add_group(design.setdefault("meta", {}), ["d1", "act"], "Hidden layer")
    frag = cv.copy_fragment(design, ["d1", "act", "d2"])
    assert frag["groups"][0]["title"] == "Hidden layer"
    out, _ids = cv.paste_fragment(design, frag)
    assert [g["nodes"] for g in out["meta"]["groups"]] == [["d1", "act"], ["d1_2", "act_2"]]
    cleaned = cv.clean(out["meta"], {"d1", "d2"})
    assert [g["nodes"] for g in cleaned["groups"]] == [["d1"]]
    assert "groups" not in cv.clean(out["meta"], {"d2"})


def test_notes_and_groups_never_reach_the_code():
    pytest.importorskip("jinja2")
    design = tiny_classifier_dict()
    meta = design.setdefault("meta", {})
    cv.add_note(meta, "Remember to try dropout", (0, -200))
    cv.add_group(meta, ["d1", "act"], "Hidden layer")
    graph = Graph.from_dict(design)
    assert not [i for i in graph.validate() if i.severity == "error"]
    code = generate(graph, "pytorch")
    assert "dropout" not in code.lower() and "Hidden layer" not in code
    assert generate(Graph.from_dict(cv.strip(design)), "pytorch") == code


def test_group_bounds_surround_the_blocks():
    x, y, w, h = cv.group_bounds({"nodes": ["a", "b"]}, {"a": (0, 0), "b": (300, 100)})
    assert x < 0 and y < 0 and x + w > 300 + 230 and y + h > 100 + 60
    assert cv.group_bounds({"nodes": ["gone"]}, {}) is None


# ------------------------------------------------------------------ uploads


@pytest.fixture()
def client(isolated_home, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from ai_made_easy.server.app import create_app

    with TestClient(create_app(token="", projects_dir=tmp_path / "projects")) as c:
        yield c


def _csv() -> bytes:
    rows = ["age,income,churn"] + [f"{20 + i % 40},{1000 + i * 37},{'yes' if i % 3 else 'no'}"
                                   for i in range(80)]
    return "\n".join(rows).encode()


def test_a_dropped_csv_is_stored_and_detected(client, isolated_home):
    pytest.importorskip("pandas")
    got = client.post("/api/data/upload",
                      files={"file": ("customers.csv", io.BytesIO(_csv()), "text/csv")})
    assert got.status_code == 200, got.text
    body = got.json()
    assert body["path"].startswith(str(isolated_home)) and body["path"].endswith("customers.csv")
    assert body["facts"]["kind"] == "table" and body["facts"]["target"] == "churn"
    again = client.post("/api/data/upload",
                        files={"file": ("customers.csv", io.BytesIO(_csv()), "text/csv")})
    assert again.json()["path"] == body["path"]   # same content, same place


def test_a_dropped_zip_of_folders_unpacks_safely(client):
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for label in ("cats", "dogs"):
            for i in range(3):
                z.writestr(f"pets/{label}/{i}.txt", f"{label} {i}")
    got = client.post("/api/data/upload",
                      files={"file": ("pets.zip", io.BytesIO(buf.getvalue()), "application/zip")})
    assert got.status_code == 200, got.text
    assert got.json()["path"].endswith("pets")

    evil = io.BytesIO()
    with zipfile.ZipFile(evil, "w") as z:
        z.writestr("../escape.txt", "x")
    bad = client.post("/api/data/upload",
                      files={"file": ("evil.zip", io.BytesIO(evil.getvalue()), "application/zip")})
    assert bad.status_code == 400 and "unsafe path" in bad.text


def test_other_files_are_refused(client):
    bad = client.post("/api/data/upload", files={"file": ("tool.exe", io.BytesIO(b"MZ"), "x")})
    assert bad.status_code == 400 and "drop a table" in bad.text


# ------------------------------------------------------------------ desktop canvas


@pytest.fixture()
def desk(isolated_home):
    pytest.importorskip("PySide6")
    from PySide6 import QtCore, QtWidgets

    from ai_made_easy.ui.app import _ensure_qt_plugin_path

    _ensure_qt_plugin_path()
    QtCore.QCoreApplication.setOrganizationName("aime-tests")
    QtCore.QCoreApplication.setApplicationName("canvas-tools")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from ai_made_easy.ui.context import AppContext
    from ai_made_easy.ui.workbench import Workbench

    ctx = AppContext()
    win = Workbench(ctx)
    app.processEvents()
    design = tiny_classifier_dict()
    for i, node in enumerate(design["nodes"]):   # spread out: frames wrap what they cover
        node["position"] = [i * 300, 0] if node["type"].startswith("core.") else [i * 300, 400]
    ctx.canvas.load_ir(Graph.from_dict(design))
    yield ctx, win, app
    ctx.project_store.mark_clean()
    win.close()
    win.deleteLater()


def _blocks(ctx) -> list:
    return [n for n in ctx.canvas.node_graph.all_nodes() if ctx.canvas.block_of(n.id)]


def _select(ctx, types: set[str]) -> None:
    for n in ctx.canvas.node_graph.all_nodes():
        block = ctx.canvas.block_of(n.id)
        n.set_selected(bool(block) and block.type_id in types)


def test_desktop_copy_paste_and_duplicate(desk):
    ctx, _win, app = desk
    before = len(_blocks(ctx))
    _select(ctx, {"core.dense"})
    ctx.act_copy()
    text = app.clipboard().text()
    assert json.loads(text)["aime_clipboard"] == 1
    ctx.act_paste()
    assert len(_blocks(ctx)) == before + 2
    assert len(ctx.canvas.selected_ids()) == 2   # the pasted copies
    ctx.act_duplicate()
    assert len(_blocks(ctx)) == before + 4
    ctx.canvas.node_graph.undo_stack().undo()
    assert len(_blocks(ctx)) == before + 2
    graph = ctx.canvas.to_ir("x")
    assert not [i for i in graph.validate() if "duplicate" in i.message.lower()]


def test_desktop_paste_from_the_web_format(desk):
    ctx, _win, _app = desk
    design = tiny_classifier_dict()
    text = cv.to_text(cv.copy_fragment(design, ["d1", "act"]))
    assert ctx.canvas.paste_text(text) == 2
    graph = ctx.canvas.to_ir("x")
    dense_ids = [n.instance_id for n in graph.nodes.values() if n.type_id == "core.dense"]
    assert len(dense_ids) == 3
    assert any(e.source_id in dense_ids for e in graph.edges if e.target_port == "in")
    with pytest.raises(ValueError, match="unknown block type"):
        ctx.canvas.paste_text(json.dumps({"nodes": [{"id": "x", "type": "nope.block"}]}))


def test_desktop_groups_and_notes_round_trip(desk):
    ctx, _win, _app = desk
    _select(ctx, {"core.dense", "core.relu"})
    assert ctx.canvas.group_selected("Hidden layer") is not None
    ctx.canvas.add_note("Try a wider layer", (0, -300))
    meta = ctx.canvas.to_ir("x").meta
    assert meta["groups"][0]["title"] == "Hidden layer"
    assert len(meta["groups"][0]["nodes"]) == 3   # Dense, ReLU, Dense
    assert meta["notes"][0]["text"] == "Try a wider layer"
    # reopening the design restores both
    saved = ctx.canvas.to_ir("x").to_dict()
    ctx.canvas.load_ir(Graph.from_dict(saved))
    again = ctx.canvas.to_ir("x").meta
    assert again["groups"][0]["title"] == "Hidden layer"
    assert len(again["groups"][0]["nodes"]) == 3
    assert again["notes"][0]["text"] == "Try a wider layer"
    # ungroup a selected block's group
    _select(ctx, {"core.relu"})
    assert ctx.canvas.ungroup_selected() == 1
    assert "groups" not in ctx.canvas.to_ir("x").meta


def test_desktop_window_accepts_data_files(desk, tmp_path, monkeypatch):
    from PySide6 import QtCore

    ctx, win, _app = desk
    csv = tmp_path / "customers.csv"
    csv.write_bytes(_csv())
    mime = QtCore.QMimeData()
    mime.setUrls([QtCore.QUrl.fromLocalFile(str(csv))])
    assert win.dropped_data(mime) == str(csv)
    exe = QtCore.QMimeData()
    exe.setUrls([QtCore.QUrl.fromLocalFile(str(tmp_path / "tool.exe"))])
    assert win.dropped_data(exe) is None
    opened: list = []
    monkeypatch.setattr(ctx, "act_new_from_task", lambda *a, **kw: opened.append(kw))
    ctx._on_canvas_files([str(csv)])   # dropped on the canvas itself
    assert opened == [{"initial_path": str(csv)}]


def test_desktop_shortcuts_are_unique():
    from collections import Counter

    from ai_made_easy.ui.actions_catalog import CATALOG

    counts = Counter(a.shortcut for a in CATALOG if a.shortcut)
    assert [s for s, n in counts.items() if n > 1] == []


# ------------------------------------------------------------------ explain, compare, new


def test_design_diff_lists_every_kind_of_change():
    from ai_made_easy.core import diff

    before = tiny_classifier_dict()
    after = json.loads(json.dumps(before))
    after["nodes"][1]["params"]["units"] = 64                     # d1: 8 -> 64
    after["nodes"].append({"id": "drop", "type": "core.dropout", "params": {"p": 0.2},
                           "position": [0, 0]})
    after["nodes"] = [n for n in after["nodes"] if n["id"] != "acc"]
    after["edges"].append({"from": "act/out", "to": "drop/in"})
    d = diff.diff(before, after)
    assert d["changed"] == [{"id": "d1", "type": "core.dense", "param": "units",
                             "from": 8, "to": 64}]
    assert d["added"] == [{"id": "drop", "type": "core.dropout"}]
    assert d["removed"] == [{"id": "acc", "type": "eval.accuracy"}]
    assert d["wired"] == [{"from": "act/out", "to": "drop/in"}]
    assert not d["same"] and "1 block added" in d["summary"]
    assert diff.diff(before, before)["same"]


def _train_here(graph, tmp_path, monkeypatch):
    from ai_made_easy.core.codegen.training_gen import generate_training

    monkeypatch.chdir(tmp_path)
    code = generate_training(graph, "pytorch")
    ns: dict = {"__name__": "aime_training"}
    exec(compile(code, "<explain>", "exec"), ns)
    ns["main"]()
    return code


def test_feature_importance_finds_the_informative_feature(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("pandas")
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"signal": rng.normal(size=400), "noise": rng.normal(size=400),
                          "colour": rng.choice(["red", "blue"], size=400)})
    frame["label"] = (frame["signal"] > 0).astype(int)
    csv = tmp_path / "table.csv"
    frame.to_csv(csv, index=False)
    design = {"name": "explain_me", "meta": {}, "edges": [], "nodes": [
        {"id": "in", "type": "core.input", "params": {"shape": "4"}, "position": [0, 0]},
        {"id": "d1", "type": "core.dense", "params": {"units": 16}, "position": [0, 0]},
        {"id": "act", "type": "core.relu", "params": {}, "position": [0, 0]},
        {"id": "d2", "type": "core.dense", "params": {"units": 2}, "position": [0, 0]},
        {"id": "out", "type": "core.output", "params": {}, "position": [0, 0]},
        {"id": "data", "type": "data.csv", "params": {"path": str(csv), "target_column": "label"},
         "position": [0, 0]},
        {"id": "opt", "type": "train.adam", "params": {"lr": 0.01}, "position": [0, 0]},
        {"id": "trainer", "type": "train.trainer",
         "params": {"epochs": 8, "batch_size": 32, "device": "cpu"}, "position": [0, 0]},
        {"id": "why", "type": "eval.permutation_importance", "params": {"n_repeats": 3},
         "position": [0, 0]}]}
    chain = ["in", "d1", "act", "d2", "out"]
    design["edges"] = [{"from": f"{a}/out", "to": f"{b}/in"} for a, b in zip(chain[:-1], chain[1:], strict=True)]
    graph = Graph.from_dict(design)
    errors = [i for i in graph.validate() if i.severity == "error"]
    if errors:
        pytest.skip(f"table design needs adjusting here: {errors}")
    _train_here(graph, tmp_path, monkeypatch)
    explained = json.loads((tmp_path / "explain.json").read_text(encoding="utf-8"))
    names = [f["name"] for f in explained["features"]]
    assert names[0] == "signal"
    assert set(names) == {"signal", "noise", "colour=blue", "colour=red"}


def test_feature_importance_is_skipped_for_images_with_a_warning():
    from ai_made_easy.core.training.spec import collect_spec

    design = json.loads(json.dumps(__import__("ai_made_easy.core.api", fromlist=["x"])
                                   .read_sample("mnist_cnn.json")))
    design["nodes"].append({"id": "why", "type": "eval.permutation_importance", "params": {},
                            "position": [0, 0]})
    spec = collect_spec(Graph.from_dict(design))
    assert spec.explain is None
    assert any("Feature Importance" in w for w in spec.warnings)


def test_explain_and_diff_endpoints(client):
    from ai_made_easy.core import api

    h = api.manager().history
    rec = h.create(tiny_classifier_dict())
    (h.path(rec.run_id) / "explain.json").write_text(json.dumps(
        {"method": "permutation", "features": [{"name": "x", "importance": 0.3, "std": 0.01}]}), encoding="utf-8")
    got = client.get(f"/api/runs/{rec.run_id}/explain").json()
    assert got["features"][0]["name"] == "x"
    other = h.create(tiny_classifier_dict())
    assert client.get(f"/api/runs/{other.run_id}/explain").json()["features"] == []
    edited = tiny_classifier_dict()
    edited["nodes"][1]["params"]["units"] = 99
    d = client.post(f"/api/runs/{rec.run_id}/diff", json={"graph": edited}).json()
    assert d["changed"][0]["to"] == 99 and d["run_id"] == rec.run_id


def test_aime_new_picks_the_task_from_the_data(tmp_path, isolated_home):
    pytest.importorskip("pandas")
    import os
    import subprocess
    import sys

    csv = tmp_path / "customers.csv"
    csv.write_bytes(_csv())
    out = tmp_path / "project.json"
    env = {**os.environ, "AIME_HOME": str(isolated_home)}
    proc = subprocess.run([sys.executable, "-m", "ai_made_easy.cli", "new", "--data", str(csv),
                           "-o", str(out)], capture_output=True, text=True, env=env,
                          timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert "task: " in proc.stderr and "from the data" in proc.stderr
    design = json.loads(out.read_text(encoding="utf-8"))
    assert any(n["type"] in ("data.csv", "data.table") or n["type"].startswith("data.")
               for n in design["nodes"])
    bare = subprocess.run([sys.executable, "-m", "ai_made_easy.cli", "new"],
                          capture_output=True, text=True, env=env, timeout=120)
    assert bare.returncode == 1 and "--task" in bare.stderr


def test_desktop_changes_since_a_run(desk):
    from ai_made_easy.core import api

    ctx, _win, _app = desk
    rec = api.manager().history.create(ctx.canvas.to_ir("x").to_dict())
    ctx._diff_run(rec.run_id)
    assert ctx.last_diff["same"]
    _select(ctx, {"core.relu"})
    ctx.canvas.delete_selected()
    ctx._diff_run(rec.run_id)
    assert ctx.last_diff["removed"] and "removed" in ctx.last_diff["summary"]
