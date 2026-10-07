"""End-to-end check for classic ML (scikit-learn / XGBoost / LightGBM / CatBoost)
pipelines: generate the script, run it, assert it reports metrics.

Usage: python scripts/verify_classic.py [-k substring] [--sweep] [-v]
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from verify_training import write_images, write_series, write_table, write_texts  # noqa: E402

from ai_made_easy.core.classic import catalog as ml  # noqa: E402
from ai_made_easy.core.codegen.training_gen import generate_training  # noqa: E402
from ai_made_easy.core.graph import Graph, NodeInstance  # noqa: E402


def pipeline(name: str, blocks: list[tuple[str, dict]]) -> Graph:
    g = Graph(name=name)
    for i, (tid, params) in enumerate(blocks):
        g.add_node(NodeInstance(f"b{i}", tid, params))
    return g


def _table(tmp):
    path = write_table(tmp)
    return pipeline("classic_table", [
        ("data.csv", {"path": str(path), "target_column": "label"}),
        ("prep.drop_columns", {"columns": "id"}), ("prep.impute", {"strategy": "median"}),
        ("prep.one_hot", {}), ("prep.normalize", {}),
        ("prep.class_balance", {"strategy": "class weights"}),
        ("ml.random_forest_classifier", {"n_estimators": 50}), ("train.kfold", {"k": 3}),
        ("eval.accuracy", {}), ("eval.f1", {}), ("eval.roc_auc", {}), ("eval.log_loss", {}),
        ("eval.mcc", {}), ("eval.confusion_matrix", {})])


def _wine_search(tmp):
    return pipeline("classic_wine_search", [
        ("data.sklearn", {"dataset": "wine"}), ("prep.normalize", {}), ("ml.pca", {}),
        ("ml.logistic_regression", {}),
        ("ml.hyperparameter_search", {"param_grid": "C: 0.1, 1.0, 10.0", "cv": 3}),
        ("eval.accuracy", {}), ("eval.balanced_accuracy", {})])


def _diabetes_random(tmp):
    return pipeline("classic_diabetes", [
        ("data.sklearn", {"dataset": "diabetes"}), ("ml.polynomial", {"degree": 2}),
        ("prep.normalize", {}), ("ml.ridge", {}),
        ("ml.hyperparameter_search", {"method": "random", "n_iter": 4,
                                      "param_grid": "alpha: 0.01, 0.1, 1.0, 10.0, 100.0",
                                      "cv": 3}),
        ("eval.mae", {}), ("eval.r2", {}), ("eval.rmse", {})])


def _text_nb(tmp):
    path = write_texts(tmp)
    return pipeline("classic_text_nb", [
        ("data.text_csv", {"path": str(path)}), ("prep.text_clean", {}), ("ml.tfidf", {}),
        ("ml.multinomial_nb", {}), ("eval.accuracy", {}), ("eval.f1", {})])


def _text_svc(tmp):
    path = write_texts(tmp)
    return pipeline("classic_text_svc", [
        ("data.text_csv", {"path": str(path)}), ("ml.count_vectorizer", {"ngram_max": 2}),
        ("ml.linear_svc", {}), ("eval.accuracy", {}), ("eval.roc_auc", {})])


def _kmeans(tmp):
    return pipeline("classic_kmeans", [
        ("data.synthetic", {"n_features": 4, "n_classes": 3, "n_samples": 300}),
        ("prep.normalize", {}), ("ml.kmeans", {"n_clusters": 3}),
        ("eval.silhouette", {}), ("eval.adjusted_rand", {}), ("eval.davies_bouldin", {})])


def _dbscan(tmp):
    return pipeline("classic_dbscan", [
        ("data.synthetic", {"kind": "moons", "n_features": 2, "n_samples": 300}),
        ("ml.dbscan", {"eps": 0.2}), ("eval.silhouette", {}), ("eval.nmi", {})])


def _isolation(tmp):
    return pipeline("classic_isolation", [
        ("data.synthetic", {"n_features": 5, "n_samples": 300}), ("ml.isolation_forest", {})])


def _series(tmp):
    path = write_series(tmp)
    return pipeline("classic_series", [
        ("data.timeseries_csv", {"path": str(path), "target_columns": "value", "window": 12}),
        ("ml.hist_gradient_boosting_regressor", {}), ("eval.mae", {}), ("eval.r2", {})])


def _images(tmp):
    root = write_images(tmp)
    return pipeline("classic_images", [
        ("data.image_folder", {"root": str(root)}), ("ml.pca", {"n_components": 10}),
        ("ml.svc", {}), ("eval.accuracy", {})])


def _boosters(tmp):
    return pipeline("classic_xgb", [
        ("data.sklearn", {"dataset": "breast_cancer"}), ("ml.xgb_classifier", {"n_estimators": 50}),
        ("eval.accuracy", {}), ("eval.roc_auc", {}), ("eval.average_precision", {})])


SCENARIOS = {
    "table": (_table, ("accuracy", "f1", "roc_auc", "mcc")),
    "wine_search": (_wine_search, ("accuracy", "balanced_accuracy")),
    "diabetes_random": (_diabetes_random, ("mae", "r2")),
    "text_nb": (_text_nb, ("accuracy", "f1")),
    "text_svc": (_text_svc, ("accuracy", "roc_auc")),
    "kmeans": (_kmeans, ("silhouette", "adjusted_rand")),
    "dbscan": (_dbscan, ()),
    "isolation": (_isolation, ("anomaly_rate",)),
    "series": (_series, ("mae", "r2")),
    "images": (_images, ("accuracy",)),
    "xgboost": (_boosters, ("accuracy", "roc_auc")),
}


def sweep_graph(est: ml.Estimator) -> Graph:
    """Smallest sensible pipeline for an estimator (used by the full sweep)."""
    blocks: list[tuple[str, dict]]
    if est.task == ml.CLASSIFICATION:
        blocks = [("data.sklearn", {"dataset": "iris"}),
                  ("prep.minmax" if est.nonnegative else "prep.normalize", {}),
                  (est.type_id, {}), ("eval.accuracy", {})]
    elif est.task == ml.REGRESSION:
        blocks = [("data.sklearn", {"dataset": "diabetes"}), ("prep.normalize", {}),
                  (est.type_id, {}), ("eval.r2", {})]
    elif est.task == ml.CLUSTERING:
        blocks = [("data.synthetic", {"n_features": 3, "n_classes": 3, "n_samples": 200}),
                  (est.type_id, {}), ("eval.silhouette", {})]
    else:
        blocks = [("data.synthetic", {"n_features": 3, "n_samples": 200}), (est.type_id, {})]
    return pipeline(f"sweep_{est.type_id.split('.')[-1]}", blocks)


_INFER_CHECK = r"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location("trained", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
model = m.load_predictor(".")
X, y = m.load_data()
if Path("images").exists():
    raw = [str(p) for p in sorted(Path("images").rglob("*.png"))[:3]]
    direct = model.predict(m.prepare_inputs(raw))
elif hasattr(X, "iloc"):
    raw = X.iloc[:3].to_dict("records")
    direct = model.predict(X.iloc[:3]) if hasattr(model, "predict") else None
elif isinstance(X, list):
    raw = X[:3]
    direct = model.predict(X[:3])
else:
    raw = [row.tolist() for row in X[:3]]
    direct = model.predict(X[:3]) if hasattr(model, "predict") else None
if not hasattr(model, "predict"):
    print("INFER-OK skipped (no predict)")
    raise SystemExit(0)
out = m.infer(raw)
assert len(out) == len(raw), out
first = out[0]
if "label" in first:
    assert [o["label"] for o in out] == [str(v) for v in np.ravel(direct)], (out, direct)
elif "prediction" in first:
    assert np.allclose([o["prediction"] for o in out], np.asarray(direct, dtype=float)), out
print("INFER-OK " + json.dumps(first))
"""


def check_inference(script: Path, workdir: Path) -> str:
    proc = subprocess.run([sys.executable, "-c", _INFER_CHECK, script.name], cwd=workdir,
                          capture_output=True, text=True, timeout=600,
                          env={**os.environ, "PYTHONWARNINGS": "ignore"})
    if "INFER-OK" not in proc.stdout:
        raise AssertionError(f"{script.name} inference failed:\n"
                             f"{(proc.stdout + proc.stderr)[-3000:]}")
    return proc.stdout


def run_graph(graph: Graph, workdir: Path, expect=(), verbose=False) -> str:
    errors = [i for i in graph.validate() if i.severity == "error"]
    if errors:
        raise AssertionError(f"{graph.name}: invalid graph {errors}")
    code = generate_training(graph, "sklearn")
    script = workdir / f"{graph.name}.py"
    script.write_text(code)
    proc = subprocess.run([sys.executable, script.name], cwd=workdir, capture_output=True,
                          text=True, timeout=600,
                          env={**os.environ, "PYTHONWARNINGS": "ignore"})
    out = proc.stdout + proc.stderr
    if verbose:
        print(out)
    if proc.returncode != 0:
        raise AssertionError(f"{graph.name} failed:\n{out[-3000:]}")
    test_line = next((ln for ln in out.splitlines() if ln.startswith("test:")), "")
    for key in expect:
        assert f"{key}=" in test_line, f"{key} missing from {test_line!r}"
    assert "saved model" in out, out[-1500:]
    check_inference(script, workdir)
    return test_line


def main() -> int:
    verbose = "-v" in sys.argv
    sel = sys.argv[sys.argv.index("-k") + 1] if "-k" in sys.argv else ""
    failures = 0
    jobs = [(name, build, expect) for name, (build, expect) in SCENARIOS.items()
            if sel in name]
    if "--sweep" in sys.argv:
        jobs += [(f"sweep:{e.type_id}", (lambda tmp, e=e: sweep_graph(e)), ())
                 for e in ml.ESTIMATORS if sel in e.type_id]
    for name, build, expect in jobs:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                line = run_graph(build(Path(tmp)), Path(tmp), expect, verbose)
                print(f"ok    {name:40} {line[:90]}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL  {name:40} {str(exc)[-1200:]}")
    print(f"{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
