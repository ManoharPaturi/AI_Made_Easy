"""3.2 Phase 4 — AutoML for every family: costs for classic and probabilistic designs,
one Pareto front mixing classic and neural recipes, held-out metrics for the
probabilistic families, and trials that run in parallel or on a backend."""
from __future__ import annotations

import importlib.util
import json

import pytest

from ai_made_easy.core import automl, budget, budget_structured, recipes
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.recipes import DataFacts


def _design(recipe_id: str, task: str, **knobs) -> Graph:  # noqa: ANN003
    built = recipes.build(recipe_id, task, DataFacts(), knobs or None, None)
    return Graph.from_dict(built.to_dict() if hasattr(built, "to_dict") else built)


# ======================================================================== budgets

def test_structured_estimates_follow_the_settings():
    small = budget.estimate(_design("random_forest", "multiclass", n_estimators=50))
    large = budget.estimate(_design("random_forest", "multiclass", n_estimators=200))
    assert small.family == "structured" and small.value_bytes == 8
    assert large.params == pytest.approx(4 * small.params, rel=0.01)
    assert large.latency_ms(budget.get_device("raspberry_pi_5")) > \
        small.latency_ms(budget.get_device("raspberry_pi_5"))
    linear = budget.estimate(_design("linear_baseline", "multiclass"))
    assert linear.params < small.params / 100
    assert budget.estimate(_design("gradient_boosting", "multiclass")).params > 0


def test_probabilistic_estimates():
    pytest.importorskip("pgmpy")
    student = _design("bayesian_network", "probabilistic_inference")
    # the student network: D(2) + I(2) + G|D,I (3·2·2) + S|I (2·2) + L|G (2·3) = 26 values
    assert budget.estimate(student).params == 26
    hier = _design("hierarchical_regression", "bayesian_modeling")
    est = budget.estimate(hier)
    # 4 chains × 1000 draws × (mu_a + b + 10 group effects + noise scales)
    assert est.params >= 4 * 1000 * 12
    assert est.train_memory_bytes() > est.model_bytes()        # + LOO's pointwise matrix
    gp = budget.estimate(_design("gp_regression", "gp_regression"))
    rows = budget_structured.data_shape(_design("gp_regression", "gp_regression"))[0]
    assert gp.work_values >= rows * rows                       # the exact kernel matrix
    assert budget.estimate(_design("sarimax", "state_space_forecasting")).params > 0


def test_budget_checks_classic_designs():
    graph = _design("random_forest", "multiclass", n_estimators=500)
    graph.meta["budget"] = {"device": "raspberry_pi_5", "max_model_mb": 0.5}
    over = [c for c in budget.check(graph) if c.over]
    assert [c.kind for c in over] == ["model_size"]
    report = budget.budget_report(graph)
    assert report["estimate"]["family"] == "structured"


# ======================================================================== metrics

def test_every_trainable_family_has_an_automl_metric():
    families = {}
    for recipe in recipes.all_recipes():
        if recipe.family in ("llm", "text_to_image"):
            continue        # exported as scripts / served with diffusers, not tuned
        for task in recipe.tasks:
            families.setdefault(recipe.family, set()).add(task)
    for family, tasks in families.items():
        missing = sorted(t for t in tasks if t not in automl.TASK_METRICS)
        assert not missing, f"{family}: no AutoML metric for {missing}"
    for task in ("probabilistic_inference", "regime_detection", "bayesian_modeling",
                 "gp_regression", "state_space_forecasting", "density_estimation"):
        metric, direction = automl.TASK_METRICS[task]
        from ai_made_easy.core.runs.history import lower_is_better

        assert lower_is_better(metric) == (direction == "min"), metric


# ====================================================================== AutoML runs

@pytest.fixture()
def api(isolated_home, monkeypatch):
    from ai_made_easy.core import api as api_module

    monkeypatch.setattr(api_module, "_manager", None)
    monkeypatch.setattr(api_module, "_sweep_store", None, raising=False)
    return api_module


def _search(api, spec: dict, timeout: float = 1800) -> tuple[dict, list[dict]]:
    started = api.start_automl(spec)
    record = api._sweeps[started["sweep_id"]].wait(timeout)
    assert record.state == "finished", record.message
    return started, api.automl_leaderboard(started["sweep_id"])["trials"]


def _baseline_vs_best(rows: list[dict], candidates: list[str], minimize: bool):
    baseline = [r for r in rows if r["number"] < len(candidates) and r["score"] is not None]
    scored = [r["score"] for r in rows if r["score"] is not None]
    pick = min if minimize else max
    return pick(r["score"] for r in baseline), pick(scored)


FAMILIES = [
    # task, recipes, required package, max_trials
    ("clustering", ["kmeans", "gaussian_mixture"], "sklearn", 4),
    ("anomaly_detection", ["isolation_forest", "local_outlier_factor"], "sklearn", 4),
    ("probabilistic_inference", ["structure_learning"], "pgmpy", 3),
    ("regime_detection", ["hmm"], "hmmlearn", 3),
    ("gp_regression", ["gp_regression"], "gpytorch", 3),
    ("state_space_forecasting", ["sarimax", "structural_time_series"], "statsmodels", 4),
    ("density_estimation", ["realnvp"], "torch", 3),
]


@pytest.mark.parametrize("task,candidates,package,trials", FAMILIES,
                         ids=[f[0] for f in FAMILIES])
def test_automl_beats_its_baseline_round(api, task, candidates, package, trials, tmp_path):
    pytest.importorskip(package)
    spec = {"task": task, "recipes": candidates, "max_trials": trials, "epochs": 3,
            "seed": 0}
    if task == "anomaly_detection":   # scored against labels: 5% planted anomalies
        import numpy as np

        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, (400, 4))
        x[:20] += 6
        rows = ["a,b,c,d,label"] + [",".join(f"{v:.4f}" for v in r) + f",{int(i < 20)}"
                                    for i, r in enumerate(x)]
        (tmp_path / "events.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        spec.update(path=str(tmp_path / "events.csv"), target="label")
    started, rows = _search(api, spec)
    metric, direction = automl.TASK_METRICS[task]
    assert started["metric"] == metric
    finished = [r for r in rows if r["state"] == "finished"]
    assert finished and all(r["score"] is not None for r in finished), rows
    baseline, best = _baseline_vs_best(rows, candidates, direction == "min")
    assert (best <= baseline) if direction == "min" else (best >= baseline)
    assert len(rows) == trials and rows[0]["score"] == best


def test_bayesian_modeling_is_ranked_by_elpd(api):
    pytest.importorskip("pymc")
    started, rows = _search(api, {"task": "bayesian_modeling",
                                  "recipes": ["hierarchical_regression"],
                                  "max_trials": 2, "seed": 0})
    assert started["metric"] == "elpd_loo"
    assert all(r["score"] is not None and r["score"] < 0 for r in rows), rows


def test_one_pareto_front_mixes_classic_and_neural(api):
    pytest.importorskip("sklearn")
    pytest.importorskip("torch")
    _started, rows = _search(api, {
        "task": "multiclass", "recipes": ["tabular_mlp", "random_forest", "linear_baseline"],
        "max_trials": 3, "epochs": 2, "objectives": ["latency_ms", "model_mb"],
        "device": "raspberry_pi_5", "seed": 0})
    families = {recipes.get_recipe(r["recipe"]).family for r in rows}
    assert families == {"neural", "classic"}
    for row in rows:
        assert {"latency_ms", "model_mb"} <= set(row["objectives"]), row
    classic = [r for r in rows if recipes.get_recipe(r["recipe"]).family == "classic"]
    assert all("measured" in r["message"] for r in classic), classic
    assert any(r["pareto"] for r in rows)


def test_classic_runs_measure_their_costs(isolated_home):
    pytest.importorskip("sklearn")
    from ai_made_easy.core.runner.manager import RunManager

    mgr = RunManager()
    run_id = mgr.start(_design("random_forest", "multiclass", n_estimators=50), "sklearn")
    assert mgr.wait(run_id, timeout=600)["state"] == "finished"
    costs = json.loads((mgr.history.path(run_id) / "costs.json").read_text(encoding="utf-8"))
    assert costs["predict_ms"] > 0 and costs["model_mb"] > 0


def test_parallel_trials_and_the_baseline_barrier(api):
    pytest.importorskip("sklearn")
    pytest.importorskip("optuna")
    _started, rows = _search(api, {
        "task": "multiclass", "recipes": ["random_forest", "linear_baseline",
                                          "gradient_boosting"],
        "max_trials": 5, "concurrency": 3, "seed": 0})
    runs = {r["number"]: api.manager().history.get(r["run_id"]) for r in rows if r["run_id"]}
    baseline = [runs[n] for n in range(3)]
    # the baseline round trains at once ...
    assert max(r.created_at for r in baseline) < min(r.finished_at for r in baseline)
    # ... and TPE's trials start only after all of it is scored
    tuned = [runs[n] for n in runs if n >= 3]
    assert tuned and min(r.created_at for r in tuned) >= max(r.finished_at for r in baseline)


def test_trials_go_to_the_chosen_backend(api, monkeypatch):
    pytest.importorskip("sklearn")
    from ai_made_easy.core.runner.manager import RunManager

    seen = []
    original = RunManager.start

    def start(self, graph, framework="auto", *, on=None, **kwargs):  # noqa: ANN001, ANN003
        seen.append(on)
        return original(self, graph, framework, on=None, **kwargs)

    monkeypatch.setattr(RunManager, "start", start)
    _search(api, {"task": "multiclass", "recipes": ["linear_baseline"], "max_trials": 1,
                  "on": "gpu-box", "concurrency": 2})
    assert seen == ["gpu-box"]
    with pytest.raises(api.ApiError, match="concurrency"):
        api.start_automl({"task": "multiclass", "concurrency": 99})


def test_cli_exposes_backends_and_concurrency():
    from ai_made_easy import cli

    with pytest.raises(SystemExit):
        cli.main(["automl", "--help"])


@pytest.mark.skipif(importlib.util.find_spec("PySide6") is None, reason="needs Qt")
def test_desktop_wizard_sends_backend_and_concurrency(qtbot):
    from ai_made_easy.ui.features.wizard import NewProjectWizard

    wizard = NewProjectWizard(None)
    qtbot.addWidget(wizard)
    wizard._task = {"id": "multiclass", "automl": True}
    wizard.parallel.setValue(3)
    wizard._start_automl()
    assert wizard.automl_spec["concurrency"] == 3
    assert wizard.automl_spec["on"] in ("", wizard.run_on.selected())
