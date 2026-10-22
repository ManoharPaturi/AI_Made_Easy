"""3.7: a project with accented names, end to end. Folder, columns, classes and the design
name are all non-ASCII: save and reopen the project, train, read the run back, and
predict with Try it (one input and a whole file). On Windows this exercises the UTF-8
file handling (3.6) and the UTF-8 pipes between the app and its child processes (3.7)."""
from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from verify_training import chain  # noqa: E402

from ai_made_easy.core import api  # noqa: E402
from ai_made_easy.core.bundle import read_bundle, write_bundle  # noqa: E402
from ai_made_easy.core.graph import Graph  # noqa: E402
from ai_made_easy.core.runner.manager import RunManager  # noqa: E402
from ai_made_easy.core.runs.history import RunHistory  # noqa: E402

CLASSES = ("grand ☀", "petit é")


def test_accented_project_trains_reopens_and_predicts(tmp_path):
    root = tmp_path / "projét données"
    root.mkdir()
    table = root / "mesures é.csv"
    rows = [f"{i % 7},{(i * 3) % 5},{CLASSES[0] if i % 7 > 3 else CLASSES[1]}"
            for i in range(120)]
    table.write_text("größe,gewicht,étiquette\n" + "\n".join(rows) + "\n", encoding="utf-8")
    graph = chain("Modèle à étiquettes", "2",
                  [("core.dense", {"units": 8}), ("core.relu", {}), ("core.dense", {"units": 2})],
                  [("data.csv", {"path": str(table), "target_column": "étiquette"}),
                   ("train.trainer", {"epochs": 2, "batch_size": 16, "device": "cpu"})])

    bundle = write_bundle(root / "Modèle.aime", graph.to_dict(), name="Modèle à étiquettes",
                          card_md="# Modèle à étiquettes\nClasses: grand ☀, petit é\n")
    reopened = Graph.from_dict(read_bundle(bundle)["graph"])
    assert reopened.name == "Modèle à étiquettes"
    assert not [i for i in reopened.validate() if i.severity == "error"]

    mgr = RunManager(RunHistory(root / "runs"))
    run_id = mgr.start(reopened, "auto")
    record = mgr.wait(run_id, 900)
    assert record["state"] == "finished", mgr.history.get(run_id).error
    assert mgr.history.get(run_id).name == "Modèle à étiquettes"
    classes = json.loads((mgr.history.path(run_id) / "classes.json").read_text(encoding="utf-8"))
    assert sorted(classes) == sorted(CLASSES)

    api.set_manager(mgr)
    try:
        one = api.try_run(run_id, [{"größe": 6, "gewicht": 1}])
        assert one["predictions"][0]["label"] in CLASSES
        got = api.try_file(run_id, "entrées é.csv",
                           "größe,gewicht\n6,2\n1,4\n".encode("utf-8"))
        out = list(csv.DictReader(io.StringIO(got["csv"])))
        assert len(out) == 2 and {r["label"] for r in out} <= set(CLASSES)
        assert any(k == f"p({CLASSES[0]})" for k in out[0])
    finally:
        api.try_service().close()
