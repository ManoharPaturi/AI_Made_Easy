"""Custom blocks: versioning and .aimeblocks packs."""
from __future__ import annotations

import json
import zipfile

import pytest

from ai_made_easy.core.block_packs import (
    PackError,
    export_pack,
    import_pack,
    list_blocks,
    save_block,
    templates_dir,
)


def _block(units: int = 8) -> dict:
    return {"nodes": [{"id": "d", "type": "core.dense", "params": {"units": units}},
                      {"id": "r", "type": "core.relu", "params": {}}],
            "edges": [{"from": "d/out", "to": "r/in"}], "entry": "d", "exit": "r"}


def test_versions_increase_only_on_change(isolated_home):
    save_block(_block(), "Dense ReLU")
    save_block(_block(), "Dense ReLU")
    assert list_blocks()[0]["version"] == 1
    save_block(_block(16), "Dense ReLU")
    (info,) = list_blocks()
    assert info["version"] == 2 and info["type_id"] == "custom.dense_relu"
    with pytest.raises(PackError):
        save_block({"nodes": []}, "bad")


def test_pack_round_trip_and_conflicts(isolated_home, tmp_path):
    save_block(_block(), "Dense ReLU")
    save_block(_block(4), "Small")
    pack = export_pack(tmp_path / "mine.aimeblocks")
    with zipfile.ZipFile(pack) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    assert {b["slug"] for b in manifest["blocks"]} == {"dense_relu", "small"}
    # identical blocks are skipped
    assert import_pack(pack)["skipped"] == ["dense_relu", "small"]
    # a different block with a taken name is renamed, not overwritten
    save_block(_block(32), "Small")
    result = import_pack(pack)
    assert result["renamed"] == {"small": "small_2"}
    assert json.loads((templates_dir() / "small.json").read_text())["nodes"][0][
        "params"]["units"] == 32
    assert import_pack(pack, overwrite=True)["imported"] == ["dense_relu", "small"]
    with pytest.raises(PackError):
        import_pack(tmp_path / "missing.aimeblocks")
    with pytest.raises(PackError):
        export_pack(tmp_path / "x.aimeblocks", ["nope"])


def test_archives_carry_custom_blocks_and_restore_runs(isolated_home, tmp_path):
    pytest.importorskip("torch")
    from conftest import tiny_classifier_dict

    from ai_made_easy.core.block_packs import install_for_project
    from ai_made_easy.core.bundle import read_bundle, restore_run, write_bundle
    from ai_made_easy.core.graph import Graph
    from ai_made_easy.core.runner.manager import RunManager
    from ai_made_easy.core.runs.history import RunHistory

    save_block(_block(), "Dense ReLU")
    mgr = RunManager(RunHistory(tmp_path / "runs"))
    run_id = mgr.start(Graph.from_dict(tiny_classifier_dict(epochs=1)))
    assert mgr.wait(run_id, 300)["state"] == "finished"
    graph = tiny_classifier_dict()
    graph["nodes"].append({"id": "c", "type": "custom.dense_relu", "params": {}})
    archive = write_bundle(tmp_path / "p.aime", graph, name="p",
                           workdir=mgr.history.path(run_id))
    # another machine: same block name, different content
    (templates_dir() / "dense_relu.json").write_text(json.dumps({**_block(64), "name": "x"}))
    bundle = read_bundle(archive)
    assert set(bundle["custom_blocks"]) == {"dense_relu"}
    result = install_for_project(bundle["graph"], bundle["custom_blocks"])
    assert result["renamed"] == {"dense_relu": "dense_relu_2"}
    assert bundle["graph"]["nodes"][-1]["type"] == "custom.dense_relu_2"

    from ai_made_easy.core.block_packs import register_custom_blocks

    assert {b.type_id for b in register_custom_blocks()} >= {"custom.dense_relu_2"}
    history = RunHistory(tmp_path / "other_runs")
    restored = restore_run(bundle, history)
    rec = history.get(restored)
    assert rec.status == "finished" and "archive" in rec.tags
    files = {p.name for p in history.path(restored).iterdir()}
    assert {"tiny_best.pt", "tiny_train_pytorch.py", "inference_state.pkl"} <= files

    from ai_made_easy.core.deploy import build_package

    assert build_package(history.path(restored), tmp_path / "pkg").verified
