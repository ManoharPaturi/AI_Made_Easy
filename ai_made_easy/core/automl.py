"""AutoML: search over recipes and their settings for a task, within the budget.

An AutoML search is a sweep whose first dimension is the recipe itself. The first
trials train every candidate recipe with its defaults (a fair baseline for each);
later trials let Optuna's TPE sampler pick a recipe and its settings (learning rate,
width, depth, ...) conditioned on the scores so far (random search without Optuna).
Trials whose design breaks the project budget are skipped before they run, and running
trials are pruned by the median rule: after the warm-up epochs a trial whose metric is
worse than the median of the finished trials at the same epoch is stopped early.

Records are sweep records (``spec["automl"]`` holds the search), so the Experiments
leaderboard, the CLI and the web list them with the sweeps.
"""
from __future__ import annotations

import random
import statistics
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from ai_made_easy.core.graph import Graph
from ai_made_easy.core.sweeps import SweepError, SweepRecord, SweepRunner, SweepSpec, TrialRow

# task -> (metric the runs report, direction)
TASK_METRICS = {
    "multiclass": ("accuracy", "max"), "binary": ("accuracy", "max"),
    "multilabel": ("f1", "max"), "regression": ("rmse", "min"),
    "distribution": ("val_loss", "min"), "keyword_spotting": ("accuracy", "max"),
    "audio_tagging": ("map", "max"), "speech_recognition": ("cer", "min"),
    "detection": ("map", "max"), "keypoints": ("map", "max"),
    "instance_segmentation": ("mask_map", "max"), "semantic_segmentation": ("miou", "max"),
    "forecasting": ("mase", "min"), "node_classification": ("accuracy", "max"),
    "graph_classification": ("accuracy", "max"), "link_prediction": ("roc_auc", "max"),
    "recommendation": ("ndcg_at_10", "max"), "reinforcement_learning": ("mean_reward", "max"),
    "density_estimation": ("nll", "min"), "vae_generation": ("val_loss", "min"),
    "gan_generation": ("val_loss", "min"), "diffusion_generation": ("val_loss", "min"),
    "language_modeling": ("val_loss", "min"), "sequence_to_sequence": ("val_loss", "min"),
}
# knobs that set how long a trial trains: fixed by the search, never tuned
LENGTH_KNOBS = ("epochs", "total_timesteps", "iterations", "draws", "max_iter",
                "n_estimators")
WARMUP_EPOCHS = 2
POLL_SECONDS = 0.5


class AutoMLError(ValueError):
    pass


@dataclass
class AutoMLSpec:
    task: str
    facts: dict = field(default_factory=dict)     # DataFacts.to_dict(); {} = demo data
    recipes: list[str] = field(default_factory=list)  # candidates ([] = every ready one)
    modality: str = ""
    max_trials: int = 12
    metric: str = ""                              # "" = the task's metric
    direction: str = ""
    epochs: int = 0                               # epochs per trial (0 = recipe default)
    tune: bool = True                             # False: every recipe once, with defaults
    prune: bool = True
    budget: dict = field(default_factory=dict)
    seed: int = 0
    framework: str = "auto"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> AutoMLSpec:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in dict(data).items() if k in known})

    def resolved_metric(self) -> tuple[str, bool]:
        """(metric, minimize)."""
        metric, direction = TASK_METRICS.get(self.task, ("val_loss", "min"))
        metric = self.metric or metric
        if self.direction:
            return metric, self.direction == "min"
        if self.metric:
            from ai_made_easy.core.runs.history import lower_is_better

            return metric, lower_is_better(metric)
        return metric, direction == "min"


def _facts(spec_dict: dict):  # noqa: ANN202 — DataFacts
    from ai_made_easy.core.recipes import DataFacts

    return DataFacts.from_dict(spec_dict.get("facts") or {})


def candidates(spec: AutoMLSpec) -> list:
    """The recipes the search may use: installed, buildable, valid and within budget."""
    from ai_made_easy.core import recipes

    if spec.task not in TASK_METRICS:
        raise AutoMLError(f"AutoML does not support {spec.task!r} yet; supported tasks: "
                          f"{', '.join(sorted(TASK_METRICS))}")
    facts = _facts(spec.to_dict())
    wanted = set(spec.recipes)
    out = []
    for sug in recipes.recommend(spec.task, facts, spec.budget or None,
                                 modality=spec.modality or None):
        if wanted and sug.recipe.id not in wanted:
            continue
        if sug.ready:
            out.append(sug.recipe)
    if not out:
        raise AutoMLError("no recipe can train this task on this data within the budget"
                          + (f" (asked for {sorted(wanted)})" if wanted else ""))
    return out


def _knob_values(recipe, values: dict, epochs: int) -> dict:  # noqa: ANN001
    knobs = {k: v for k, v in values.items() if k != "recipe"}
    if epochs and any(k.name == "epochs" for k in recipe.knobs):
        knobs["epochs"] = epochs
    return knobs


def trial_design(spec_dict: dict, values: dict) -> Graph:
    """The design a trial trains: its recipe built with its settings."""
    from ai_made_easy.core import recipes

    spec = AutoMLSpec.from_dict(spec_dict)
    recipe = recipes.get_recipe(values["recipe"])
    graph = recipes.build(recipe, spec.task, _facts(spec_dict),
                          _knob_values(recipe, values, spec.epochs), spec.budget or None)
    return graph


class AutoMLRunner(SweepRunner):
    """Runs the trials of one AutoML search on a background thread."""

    def __init__(self, manager, store, spec: AutoMLSpec, project: str = "",  # noqa: ANN001
                 listener=None):  # noqa: ANN001
        import threading

        self.automl = spec
        self.recipes = candidates(spec)
        metric, minimize = spec.resolved_metric()
        self.spec = SweepSpec(dimensions=[], metric=metric, direction="min" if minimize
                              else "max", strategy="tpe" if spec.tune else "grid",
                              max_trials=max(int(spec.max_trials), 1), seed=spec.seed,
                              framework=spec.framework, skip_over_budget=True)
        self.manager, self.store, self.listener = manager, store, listener
        self._stop = threading.Event()
        self._current_run: str | None = None
        self._curves: dict[int, list[float]] = {}
        first = trial_design(spec.to_dict(), {"recipe": self.recipes[0].id})
        self.graph_dict = first.to_dict()
        self.record = SweepRecord(
            sweep_id=time.strftime("automl-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4],
            name=f"automl_{spec.task}", graph=self.graph_dict, project=project,
            spec={**self.spec.to_dict(), "automl": spec.to_dict(),
                  "candidates": [r.id for r in self.recipes]})
        store.save(self.record)
        self.thread = None

    # ------------------------------------------------------------- proposals
    def _defaults(self, recipe) -> dict:  # noqa: ANN001
        return {"recipe": recipe.id, **{k.name: k.default for k in recipe.knobs
                                        if k.name not in LENGTH_KNOBS}}

    def _tunable(self, recipe) -> list:  # noqa: ANN001
        return [k for k in recipe.knobs if k.name not in LENGTH_KNOBS]

    def _random(self, rng: random.Random) -> dict:
        recipe = rng.choice(self.recipes)
        values = {"recipe": recipe.id}
        for knob in self._tunable(recipe):
            values[knob.name] = knob.dimension().sample(rng)
        return values

    def _proposals(self):  # noqa: ANN202
        budget = self.spec.max_trials
        baseline = [self._defaults(r) for r in self.recipes][:budget]
        for values in baseline:
            yield values, None
        remaining = budget - len(baseline)
        if remaining <= 0 or not self.automl.tune:
            return
        study = None
        try:
            import optuna

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(
                direction="minimize" if self.spec.minimize else "maximize",
                sampler=optuna.samplers.TPESampler(seed=self.spec.seed))
            for row in self.record.trials:            # teach TPE the baseline
                if row["score"] is not None:
                    study.add_trial(optuna.trial.create_trial(
                        params=self._optuna_params(row["values"]),
                        distributions=self._distributions(row["values"]["recipe"]),
                        value=row["score"]))
        except ImportError:
            self.record.message = "Optuna is not installed; using random search"
        rng = random.Random(self.spec.seed)
        for _ in range(remaining):
            if study is None:
                yield self._random(rng), None
                continue
            trial = study.ask()
            recipe_id = trial.suggest_categorical("recipe", [r.id for r in self.recipes])
            recipe = next(r for r in self.recipes if r.id == recipe_id)
            values = {"recipe": recipe_id}
            for knob in self._tunable(recipe):
                dim = knob.dimension()
                dim.node = recipe_id                       # conditional parameter names
                values[knob.name] = dim.suggest(trial)

            def tell(score, _trial=trial, _study=study):  # noqa: ANN001, ANN202
                import optuna

                if score == "pruned":
                    _study.tell(_trial, state=optuna.trial.TrialState.PRUNED)
                elif score is None:
                    _study.tell(_trial, state=optuna.trial.TrialState.FAIL)
                else:
                    _study.tell(_trial, score)

            yield values, tell

    def _optuna_params(self, values: dict) -> dict:
        recipe_id = values["recipe"]
        return {"recipe": recipe_id, **{f"{recipe_id}.{k}": v for k, v in values.items()
                                        if k != "recipe"}}

    def _distributions(self, recipe_id: str) -> dict:
        import optuna.distributions as od

        recipe = next(r for r in self.recipes if r.id == recipe_id)
        out: dict[str, Any] = {"recipe": od.CategoricalDistribution(
            [r.id for r in self.recipes])}
        for knob in self._tunable(recipe):
            key = f"{recipe_id}.{knob.name}"
            if knob.kind == "choice":
                out[key] = od.CategoricalDistribution(list(knob.values))
            elif knob.kind == "int":
                out[key] = od.IntDistribution(int(knob.low), int(knob.high), log=knob.log)
            else:
                out[key] = od.FloatDistribution(float(knob.low), float(knob.high),
                                                log=knob.log)
        return out

    # ------------------------------------------------------------- trials
    def _trial_graph(self, values: dict) -> Graph:
        graph = trial_design(self.automl.to_dict(), values)
        return graph

    def _epoch_value(self, event: dict) -> tuple[str, float] | None:
        metrics = event.get("metrics") or {}
        for key in (self.spec.metric, "val_loss"):
            value = metrics.get(key)
            if isinstance(value, (int, float)) and value == value:
                return key, float(value)
        return None

    def _should_prune(self, number: int, epoch: int, key: str, value: float) -> bool:
        if not self.automl.prune or epoch < WARMUP_EPOCHS:
            return False
        others = [c[epoch - 1] for n, c in self._curves.items()
                  if n != number and len(c) >= epoch and c[epoch - 1] is not None]
        if len(others) < 2:
            return False
        median = statistics.median(others)
        minimize = self.spec.minimize if key == self.spec.metric else True
        return value > median if minimize else value < median

    def _await(self, row: TrialRow) -> dict:
        curve: list[float | None] = []
        seen = 0
        while True:
            status = self.manager.status(row.run_id)
            events = self.manager.metrics(row.run_id)
            for event in events[seen:]:
                found = self._epoch_value(event)
                curve.append(found[1] if found else None)
                if found and row.state != "pruned" and self._should_prune(
                        row.number, len(curve), *found):
                    row.state = "pruned"
                    row.message = (f"pruned at epoch {len(curve)}: {found[0]} "
                                   f"{found[1]:.4g} is worse than the median")
                    self.manager.stop(row.run_id)
            seen = len(events)
            if status["state"] in ("finished", "failed", "stopped"):
                break
            time.sleep(POLL_SECONDS)
        status = self.manager.wait(row.run_id, timeout=60)
        self._curves[row.number] = curve
        return status


def start(manager, store, spec: AutoMLSpec | dict, project: str = "",  # noqa: ANN001
          listener=None) -> AutoMLRunner:  # noqa: ANN001
    spec = spec if isinstance(spec, AutoMLSpec) else AutoMLSpec.from_dict(spec)
    try:
        runner = AutoMLRunner(manager, store, spec, project=project, listener=listener)
    except (SweepError, KeyError) as exc:
        raise AutoMLError(str(exc)) from exc
    runner.start()
    return runner


def leaderboard(record: SweepRecord) -> list[dict]:
    """Trials best first, with each trial's recipe title."""
    from ai_made_easy.core import recipes

    spec = record.spec
    minimize = spec.get("direction") == "min"
    rows = []
    for t in record.trials:
        try:
            title = recipes.get_recipe(t["values"]["recipe"]).title
        except (recipes.RecipeError, KeyError):
            title = t["values"].get("recipe", "?")
        rows.append({**t, "recipe": t["values"].get("recipe"), "title": title})
    scored = sorted((r for r in rows if r["score"] is not None),
                    key=lambda r: r["score"], reverse=not minimize)
    return scored + [r for r in rows if r["score"] is None]
