"""Desktop: Train runs pipeline designs; Experiments ▸ Pipelines shows their stages."""
from __future__ import annotations

import importlib.util
import time

import pytest

pytest.importorskip("PySide6")

from test_pipelines import pipeline, stage  # noqa: E402
from test_ui import app, ctx  # noqa: E402, F401 — shared fixtures

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None,
                                reason="needs torch")


def test_train_runs_a_pipeline(isolated_home, ctx, app):  # noqa: F811
    data = pipeline([stage("t", "pipeline.train", design="embedded:student", epochs=2),
                     stage("e", "pipeline.evaluate")], [("t", "e")])
    assert ctx.open_imported(data)
    app.processEvents()
    deadline = time.time() + 30
    while not ctx.validation_store.valid and time.time() < deadline:
        app.processEvents()
        time.sleep(0.05)
    ctx.act_train()
    service = ctx.experiment_service
    assert service.pipeline is not None
    pipeline_id = service.pipeline.pipeline_id
    deadline = time.time() + 600
    while service.pipeline is not None:
        assert time.time() < deadline
        app.processEvents()
        time.sleep(0.05)
    record = service.pipelines.get(pipeline_id)
    assert record.state == "finished", {k: v["message"] for k, v in record.stages.items()}
    page = ctx.experiments_page
    ctx._refresh_experiments()
    page.show_pipeline(pipeline_id)
    assert page.tabs.tabText(page.tabs.currentIndex()) == "Pipelines"
    assert page.stages_table.rowCount() == 2
    assert page.stages_table.item(1, 1).text() == "finished"
    assert "ece" in page.stages_table.item(1, 2).text()
