"""3.4 — the items 3.3 deferred: SHAP values, collapsing groups and SimCLR (NT-Xent on
two augmented views of each image). Tiny synthetic data."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from verify_training import chain, write_images  # noqa: E402

from ai_made_easy.core import canvas as cv  # noqa: E402
from ai_made_easy.core.codegen import CodegenError  # noqa: E402
from ai_made_easy.core.codegen.training_gen import generate_training  # noqa: E402
from ai_made_easy.core.registry import get_registry  # noqa: E402

TRAINER = ("train.trainer", {"epochs": 6, "batch_size": 32, "device": "cpu", "seed": 0})


def _run(graph, tmp_path, monkeypatch) -> str:
    monkeypatch.chdir(tmp_path)
    code = generate_training(graph, "pytorch")
    ns: dict = {"__name__": "aime_training"}
    exec(compile(code, f"<{graph.name}>", "exec"), ns)
    ns["main"]()
    return code


# ------------------------------------------------------------------ SHAP

def test_shap_is_an_optional_extra():
    defn = get_registry().get("eval.shap_values")
    assert defn.extra == "explain" and defn.requires == ("shap",)


def test_shap_values_rank_the_informative_feature(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("shap")
    pd = pytest.importorskip("pandas")
    import numpy as np

    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"signal": rng.normal(size=400), "noise": rng.normal(size=400),
                          "other": rng.normal(size=400)})
    frame["label"] = (frame["signal"] > 0).astype(int)
    csv = tmp_path / "table.csv"
    frame.to_csv(csv, index=False)
    graph = chain("shap", "3", [("core.dense", {"units": 16}), ("core.relu", {}),
                                ("core.dense", {"units": 2})],
                  [("data.csv", {"path": str(csv), "target_column": "label"}),
                   ("train.adam", {"lr": 0.02}), ("train.trainer", {**TRAINER[1], "epochs": 10}),
                   ("eval.shap_values", {"background": 50, "max_samples": 80})])
    code = _run(graph, tmp_path, monkeypatch)
    assert "shap.GradientExplainer" in code and "import shap" in code
    explained = json.loads((tmp_path / "explain.json").read_text())
    assert explained["method"] == "shap" and explained["background"] == 50
    assert explained["features"][0]["name"] == "signal"


# ------------------------------------------------------------------ collapsing groups

def test_collapse_flags_round_trip_in_the_design():
    meta: dict = {}
    gid = cv.add_group(meta, ["a", "b"])
    assert cv.set_collapsed(meta, gid, True) and meta["groups"][0]["collapsed"] is True
    assert cv.hidden_by(meta) == {"a": gid, "b": gid}
    assert cv.set_collapsed(meta, gid, False) and "collapsed" not in meta["groups"][0]
    assert cv.hidden_by(meta) == {} and not cv.set_collapsed(meta, "nope", True)
    # code generation never sees groups, folded or not
    design = {"nodes": [], "edges": [], "meta": {"groups": [{"id": "g", "nodes": ["a"],
                                                             "collapsed": True}]}}
    assert "groups" not in cv.strip(design)["meta"]


@pytest.fixture()
def desk(isolated_home):
    pytest.importorskip("PySide6")
    from PySide6 import QtCore, QtWidgets

    from ai_made_easy.core.graph import Graph
    from ai_made_easy.ui.app import _ensure_qt_plugin_path
    from tests.conftest import tiny_classifier_dict

    _ensure_qt_plugin_path()
    QtCore.QCoreApplication.setOrganizationName("aime-tests")
    QtCore.QCoreApplication.setApplicationName("v34")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from ai_made_easy.ui.context import AppContext
    from ai_made_easy.ui.workbench import Workbench

    ctx = AppContext()
    win = Workbench(ctx)
    app.processEvents()
    design = tiny_classifier_dict()
    for i, node in enumerate(design["nodes"]):
        node["position"] = [i * 300, 0] if node["type"].startswith("core.") else [i * 300, 400]
    ctx.canvas.load_ir(Graph.from_dict(design))
    yield ctx, win, app
    ctx.project_store.mark_clean()
    win.close()
    win.deleteLater()


def _core_nodes(ctx) -> list:
    return [n for n in ctx.canvas.node_graph.all_nodes()
            if (b := ctx.canvas.block_of(n.id)) and b.type_id.startswith("core.")]


def test_desktop_groups_fold_into_a_card_and_open_where_they_moved(desk):
    ctx, win, app = desk
    members = _core_nodes(ctx)[:3]
    for n in ctx.canvas.node_graph.all_nodes():
        n.set_selected(n in members)
    group = ctx.canvas.group_selected("Encoder")
    before = {n.id: tuple(n.pos()) for n in members}
    for n in members:                                    # a block inside the group
        n.set_selected(n is members[0])
    win.actions["edit.collapse_group"].trigger()
    app.processEvents()
    assert group.folded and tuple(group.size()) == group.FOLDED_SIZE
    assert all(not n.view.isVisible() for n in members)
    pipes = [p for n in members for port in n.output_ports() for p in port.view.connected_pipes]
    assert pipes and all(not p.isVisible() for p in pipes)
    meta = ctx.canvas.to_ir().meta
    assert meta["groups"][0]["collapsed"] is True
    assert sorted(meta["groups"][0]["nodes"]) == sorted(before)    # members kept while folded
    group.set_pos(group.x_pos() + 100, group.y_pos() + 50)
    assert ctx.canvas.set_group_collapsed(group, False)
    app.processEvents()
    assert all(n.view.isVisible() for n in members)
    for n in members:
        assert tuple(n.pos()) == pytest.approx((before[n.id][0] + 100, before[n.id][1] + 50))
    assert "collapsed" not in ctx.canvas.to_ir().meta["groups"][0]


def test_desktop_folded_groups_load_and_give_their_blocks_back(desk):
    ctx, _win, app = desk
    graph = ctx.canvas.to_ir()
    ids = [n.id for n in _core_nodes(ctx)[:2]]
    graph.meta["groups"] = [{"id": "group1", "title": "Stem", "nodes": ids,
                             "color": "#91A7FF", "collapsed": True}]
    ctx.canvas.load_ir(graph)
    app.processEvents()
    group = ctx.canvas._annotations("group")[0]
    hidden = ctx.canvas.group_members(group)   # the loaded blocks (new canvas ids)
    assert len(hidden) == len(ids)
    assert group.folded and all(not n.view.isVisible() for n in hidden)
    group.set_selected(True)
    assert ctx.canvas.delete_selected() == 1                    # deleting the card ...
    app.processEvents()
    assert all(n.view.isVisible() for n in hidden)              # ... keeps its blocks
    assert len(_core_nodes(ctx)) >= 2


# ------------------------------------------------------------------ SimCLR

def _simclr_graph(root: Path):
    return chain("simclr", "3, 16, 16", [
        ("core.conv2d", {"out_channels": 8, "kernel_size": 3, "padding": 1}),
        ("core.relu", {}), ("core.global_avgpool2d", {}), ("core.dense", {"units": 8})], [
        ("data.image_folder", {"root": str(root)}),
        ("prep.resize", {"height": 16, "width": 16}),
        ("prep.random_flip", {"mode": "both"}), ("prep.random_rotation", {"degrees": 15}),
        ("train.loss_nt_xent", {"classes": 2, "temperature": 0.5}),
        ("train.adam", {"lr": 0.01}), ("eval.accuracy", {}),
        ("train.trainer", {**TRAINER[1], "epochs": 3, "batch_size": 16})])


def test_simclr_trains_on_two_views_of_each_image(tmp_path, monkeypatch, capsys):
    pytest.importorskip("torchvision")
    pytest.importorskip("PIL")
    graph = _simclr_graph(write_images(tmp_path))
    assert not [i for i in graph.validate() if i.severity == "error"], graph.validate()
    code = _run(graph, tmp_path, monkeypatch)
    assert "TwoViews(build_transforms(train=True" in code
    assert "xb, yb, loss_fn.paired = views(xb, yb)" in code
    out = capsys.readouterr().out
    assert "test:" in out and "accuracy=" in out.split("test:")[-1]
    assert Path(tmp_path, "calibration.npz").exists()          # single images, not pairs
    import numpy as np

    assert np.load(tmp_path / "calibration.npz")["x"].shape[1:] == (3, 16, 16)


def test_simclr_needs_images():
    graph = chain("simclr_table", "6", [("core.dense", {"units": 8})],
                  [("data.synthetic", {"n_features": 6, "n_classes": 2}),
                   ("train.loss_nt_xent", {"classes": 2}), ("train.adam", {}), TRAINER])
    with pytest.raises(CodegenError, match="two augmented views"):
        generate_training(graph, "pytorch")


def test_other_scripts_have_no_views():
    code = generate_training(chain("plain_images", "3, 16, 16", [
        ("core.conv2d", {"out_channels": 4, "kernel_size": 3, "padding": 1}),
        ("core.global_avgpool2d", {}), ("core.dense", {"units": 2})],
        [("data.image_folder", {"root": "imgs"}), ("prep.resize", {"height": 16, "width": 16}),
         ("train.adam", {}), TRAINER]), "pytorch")
    assert "TwoViews" not in code and "views(" not in code


def test_simclr_recipe_is_valid():
    from ai_made_easy.core.recipes import build

    graph = build("image_simclr", task="multiclass")
    assert "train.loss_nt_xent" in {n.type_id for n in graph.nodes.values()}
    assert not [i for i in graph.validate() if i.severity == "error"]
    code = generate_training(graph, "pytorch")
    assert "NTXentLoss(EMBED_DIM, NUM_CLASSES" in code and "TwoViews" in code
