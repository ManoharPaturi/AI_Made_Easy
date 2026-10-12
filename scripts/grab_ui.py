"""Render the workbench to PNG files (design review / documentation).

Usage: python scripts/grab_ui.py [out_prefix] [dark|light] [design|data|automl]

``data`` opens a customer-churn style table design with the Data dock raised.
``automl`` first runs a small multi-objective AutoML search (accuracy against latency on
a Raspberry Pi 5) on two interleaved spirals in a throwaway ``$AIME_HOME``, then shows its
Pareto front.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from ai_made_easy.ui.app import _ensure_qt_plugin_path

_ensure_qt_plugin_path()

from PySide6 import QtCore, QtWidgets  # noqa: E402


def main() -> int:
    prefix = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/aime_ui")
    theme = sys.argv[2] if len(sys.argv) > 2 else "dark"
    scene = sys.argv[3] if len(sys.argv) > 3 else "design"
    if scene == "automl":                      # never mix the capture with real history
        os.environ["AIME_HOME"] = tempfile.mkdtemp(prefix="aime-capture-")
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setOrganizationName("AI Made Easy")
    app.setApplicationName("AI Made Easy (capture)")
    from ai_made_easy.ui.context import AppContext
    from ai_made_easy.ui.workbench import Workbench

    best = _automl_search() if scene == "automl" else None
    ctx = AppContext()
    win = Workbench(ctx)
    win.resize(1600, 960)
    win.show()

    def select_first() -> None:
        ctx._apply_theme(theme, announce=False)
        if scene == "data":
            ctx.graph_service.load(_churn_design())
            ctx.project_store.set_name("churn_mlp")
            ctx.act_zoom_fit()
            ctx.act_data()
            win.resizeDocks([win.docks["data"]], [430], QtCore.Qt.Orientation.Vertical)
            return
        if scene == "automl":
            from ai_made_easy.core.graph import Graph

            ctx.graph_service.load(Graph.from_dict(best))
            ctx.project_store.set_name("spirals_automl")
            ctx.act_zoom_fit()
            page = ctx.experiments_page
            page.scope.setCurrentIndex(1)
            page.tabs.setCurrentIndex(1)
            ctx.act_experiments()
            page.sweeps_table.selectRow(0)
            win.resizeDocks([win.docks["experiments"]], [600], QtCore.Qt.Orientation.Vertical)
            return
        ids = [n.id for n in ctx.canvas.node_graph.all_nodes()]
        conv = [i for i in ids if ctx.canvas.block_of(i) and
                ctx.canvas.block_of(i).type_id == "core.conv2d"]
        if conv:                                  # full-size blocks around the first Conv2D
            graph = ctx.canvas.node_graph
            for _ in range(20):
                if graph.viewer().transform().m11() >= 0.8:
                    break
                graph.set_zoom(graph.viewer().get_zoom() + 0.25)
            ctx._locate(conv[0])
            win.docks["training"].raise_()

    def shoot() -> None:
        ctx.data_service.wait()
        QtWidgets.QApplication.processEvents()
        ctx.project_store.mark_clean()
        win.grab().save(f"{prefix}_full.png")
        print(f"saved {prefix}_full.png", flush=True)
        app.quit()

    QtCore.QTimer.singleShot(1500, select_first)
    QtCore.QTimer.singleShot(3500, shoot)
    return app.exec()


def _automl_search() -> dict:
    """Accuracy against latency on two spirals (where size pays); returns the best design."""
    import time

    from ai_made_easy.core import api

    started = api.start_automl({"task": "binary", "path": str(_spirals_csv()),
                                "target": "arm", "max_trials": 10, "epochs": 15,
                                "objectives": ["latency_ms"], "device": "raspberry_pi_5",
                                "seed": 2})
    while api.get_sweep(started["sweep_id"])["state"] in ("created", "running"):
        time.sleep(0.5)
    return api.sweep_best_graph(started["sweep_id"])


def _spirals_csv() -> Path:
    """Two interleaved spirals plus noise columns: small models stay near chance."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 3000
    t = rng.uniform(0.5, 3.5 * np.pi, n)
    arm = rng.integers(0, 2, n)
    r = t * (1 + 0.04 * rng.normal(size=n))
    frame = pd.DataFrame({"x": r * np.cos(t + np.pi * arm), "y": r * np.sin(t + np.pi * arm),
                          **{f"noise_{i}": rng.normal(size=n) for i in range(4)},
                          "arm": np.where(arm == 1, "b", "a")})
    folder = Path(tempfile.mkdtemp(prefix="aime-spirals-"))
    frame.to_csv(folder / "spirals.csv", index=False)
    return folder / "spirals.csv"


def _churn_design():
    """A small churn table with realistic problems, wired to an MLP."""
    import json

    import numpy as np
    import pandas as pd

    from ai_made_easy.core.graph import Graph

    rng = np.random.default_rng(3)
    n = 1200
    tenure = rng.integers(1, 72, n)
    charges = np.round(rng.normal(65, 30, n).clip(18, 120), 2)
    churn = np.where(rng.random(n) < 0.12 + 0.25 * (tenure < 12), "yes", "no")
    frame = pd.DataFrame({
        "customer_id": [f"C{100000 + i}" for i in range(n)], "tenure_months": tenure,
        "monthly_charges": charges, "total_charges": np.round(tenure * charges, 2),
        "contract": rng.choice(["monthly", "one year", "two year"], n, p=[.55, .25, .2]),
        "support_calls": rng.poisson(1.5, n), "country": "US", "churn": churn,
        "churn_flag": np.where(churn == "yes", 1, 0)})
    frame.loc[rng.choice(n, 40, replace=False), "total_charges"] = np.nan
    folder = Path("/tmp/aime-demo") if Path("/tmp").is_dir() else Path(tempfile.mkdtemp())
    folder.mkdir(exist_ok=True)
    path = folder / "churn.csv"
    frame.to_csv(path, index=False)
    data = json.loads((Path(__file__).parent.parent / "ai_made_easy" / "samples" /
                       "iris_mlp.json").read_text())
    for node in data["nodes"]:
        if node["type"] == "data.sklearn":
            node.update(type="data.csv", params={"path": str(path), "target_column": "churn"})
        if node["type"] == "core.input":
            node["params"]["shape"] = "7"
        if node["params"].get("units") == 3:  # two classes
            node["params"]["units"] = 2
    data["name"] = "churn_mlp"
    return Graph.from_dict(data)


if __name__ == "__main__":
    sys.exit(main())
