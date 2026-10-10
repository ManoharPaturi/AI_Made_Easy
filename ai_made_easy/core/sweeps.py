"""Hyperparameter sweeps over any block parameter of a graph.

A sweep samples values for chosen ``node.param`` dimensions, applies them to
a copy of the graph, validates the result (invalid trials are skipped, never
run), and trains each trial through the :class:`RunManager` as a child run
(``parent = sweep_id``). Strategies: ``grid``, ``random`` and ``tpe``
(Optuna; falls back to random when Optuna is not installed).

Sweep records live in ``$AIME_HOME/sweeps/<sweep_id>.json``. Pure Python.
"""
from __future__ import annotations

import copy
import itertools
import json
import math
import random
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ai_made_easy.core.graph import Graph
from ai_made_easy.core.paths import subdir
from ai_made_easy.core.runs.history import FINAL_STATES, lower_is_better

STRATEGIES = ("grid", "random", "tpe")
KINDS = ("float", "int", "choice")
MAX_GRID = 1000


class SweepError(ValueError):
    pass


@dataclass
class Dimension:
    node: str
    param: str
    kind: str = "float"  # float | int | choice
    low: float | None = None
    high: float | None = None
    log: bool = False
    step: int = 1
    values: list[Any] = field(default_factory=list)  # choice values / explicit grid
    points: int = 3  # grid resolution for float ranges

    @property
    def key(self) -> str:
        return f"{self.node}.{self.param}"

    def check(self) -> None:
        if self.kind not in KINDS:
            raise SweepError(f"{self.key}: kind must be one of {KINDS}")
        if self.kind == "choice":
            if not self.values:
                raise SweepError(f"{self.key}: give at least one value")
            return
        if self.values:
            return
        if self.low is None or self.high is None:
            raise SweepError(f"{self.key}: give a low and high bound")
        if self.low > self.high:
            raise SweepError(f"{self.key}: low must not exceed high")
        if self.log and self.low <= 0:
            raise SweepError(f"{self.key}: log scale needs a positive lower bound")

    def grid(self) -> list[Any]:
        if self.values:
            return list(self.values)
        if self.kind == "int":
            return list(range(int(self.low), int(self.high) + 1, max(int(self.step), 1)))
        n = max(int(self.points), 1)
        if n == 1:
            return [self.low]
        if self.log:
            lo, hi = math.log10(self.low), math.log10(self.high)
            return [float(f"{10 ** (lo + (hi - lo) * i / (n - 1)):.6g}") for i in range(n)]
        return [float(f"{self.low + (self.high - self.low) * i / (n - 1):.6g}")
                for i in range(n)]

    def sample(self, rng: random.Random) -> Any:
        if self.kind == "choice" or self.values:
            return rng.choice(list(self.values))
        if self.kind == "int":
            steps = (int(self.high) - int(self.low)) // max(int(self.step), 1)
            return int(self.low) + rng.randint(0, steps) * max(int(self.step), 1)
        if self.log:
            return float(f"{10 ** rng.uniform(math.log10(self.low), math.log10(self.high)):.6g}")
        return float(f"{rng.uniform(self.low, self.high):.6g}")

    def suggest(self, trial) -> Any:  # noqa: ANN001 — optuna.trial.Trial
        if self.kind == "choice" or self.values:
            return trial.suggest_categorical(self.key, list(self.values))
        if self.kind == "int":
            return trial.suggest_int(self.key, int(self.low), int(self.high),
                                     step=max(int(self.step), 1), log=self.log and self.step == 1)
        return trial.suggest_float(self.key, float(self.low), float(self.high), log=self.log)


@dataclass
class SweepSpec:
    dimensions: list[Dimension]
    metric: str = "val_loss"
    direction: str = ""  # "min" | "max"; inferred from the metric name when empty
    strategy: str = "random"
    max_trials: int = 10
    seed: int = 0
    framework: str = "auto"
    skip_over_budget: bool = True  # trials that break the project's resource budget

    @property
    def minimize(self) -> bool:
        if self.direction:
            return self.direction == "min"
        return lower_is_better(self.metric)

    def check(self, graph: Graph) -> None:
        if not self.dimensions:
            raise SweepError("add at least one parameter to sweep")
        if self.strategy not in STRATEGIES:
            raise SweepError(f"strategy must be one of {STRATEGIES}")
        if self.direction not in ("", "min", "max"):
            raise SweepError("direction must be 'min' or 'max'")
        if self.max_trials < 1:
            raise SweepError("max_trials must be at least 1")
        seen = set()
        for dim in self.dimensions:
            dim.check()
            if dim.key in seen:
                raise SweepError(f"{dim.key} is listed twice")
            seen.add(dim.key)
            node = graph.nodes.get(dim.node)
            if node is None:
                raise SweepError(f"no block {dim.node!r} in the graph")
            if dim.param not in {p.name for p in node.definition().params}:
                raise SweepError(f"{node.definition().display_name} has no parameter "
                                 f"{dim.param!r}")
        if self.strategy == "grid" and self.grid_size() > MAX_GRID:
            raise SweepError(f"the grid has {self.grid_size()} points (max {MAX_GRID}); "
                             "use random or tpe")

    def grid_size(self) -> int:
        return math.prod(len(d.grid()) for d in self.dimensions)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> SweepSpec:
        data = dict(data)
        dims = [Dimension(**d) for d in data.pop("dimensions", [])]
        known = set(cls.__dataclass_fields__) - {"dimensions"}
        return cls(dimensions=dims, **{k: v for k, v in data.items() if k in known})


def sweepable_params(graph: Graph) -> list[dict]:
    """Numeric and choice parameters a sweep can vary, with suggested ranges."""
    out = []
    for node in graph.nodes.values():
        definition = node.definition()
        values = node.resolved_params()
        for spec in definition.params:
            current = values.get(spec.name)
            row = {"node": node.instance_id, "param": spec.name, "key":
                   f"{node.instance_id}.{spec.name}", "block": definition.display_name,
                   "current": current}
            if spec.type == "float" and isinstance(current, (int, float)):
                lr_like = spec.name in ("lr", "learning_rate", "weight_decay", "eps") or (
                    current and 0 < current < 0.01)
                low = spec.minimum if spec.minimum is not None else (
                    current / 10 if lr_like and current else 0.0)
                high = spec.maximum if spec.maximum is not None else (
                    current * 10 if lr_like and current else max(current * 2, 1.0))
                if lr_like and current:
                    low, high = max(low, current / 10), min(high, current * 10)
                row.update(kind="float", low=low, high=high, log=bool(lr_like and low > 0))
            elif spec.type == "int" and isinstance(current, int) and not isinstance(current, bool):
                low = spec.minimum if spec.minimum is not None else max(1, current // 2)
                high = spec.maximum if spec.maximum is not None else max(current * 2, low)
                row.update(kind="int", low=low, high=high, log=False)
            elif spec.type == "enum" and spec.options:
                row.update(kind="choice", values=list(spec.options))
            elif spec.type == "bool":
                row.update(kind="choice", values=[False, True])
            else:
                continue
            out.append(row)
    return out


def apply_values(graph_dict: dict, values: dict[str, Any]) -> dict:
    """Copy of the graph with ``node.param`` values substituted."""
    data = copy.deepcopy(graph_dict)
    nodes = {n["id"]: n for n in data.get("nodes", [])}
    for key, value in values.items():
        node_id, param = key.rsplit(".", 1)
        nodes[node_id].setdefault("params", {})[param] = value
    return data


# ------------------------------------------------------------------ store

@dataclass
class TrialRow:
    number: int
    values: dict[str, Any]
    run_id: str = ""
    # pending | running | finished | failed | invalid | over_budget | stopped | pruned
    state: str = "pending"
    score: float | None = None
    message: str = ""


@dataclass
class SweepRecord:
    sweep_id: str
    name: str
    spec: dict
    graph: dict
    project: str = ""
    state: str = "created"  # created | running | finished | stopped | failed
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    trials: list[dict] = field(default_factory=list)
    best: dict | None = None  # {"number", "run_id", "score", "values"}
    message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class SweepStore:
    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root else subdir("sweeps")
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _file(self, sweep_id: str) -> Path:
        if not sweep_id or "/" in sweep_id or "\\" in sweep_id or sweep_id.startswith("."):
            raise KeyError(f"invalid sweep id {sweep_id!r}")
        return self.root / f"{sweep_id}.json"

    def save(self, record: SweepRecord) -> None:
        with self._lock:
            file = self._file(record.sweep_id)
            tmp = file.with_suffix(".tmp")
            tmp.write_text(json.dumps(record.to_dict(), indent=1, default=str))
            tmp.replace(file)

    def get(self, sweep_id: str) -> SweepRecord:
        file = self._file(sweep_id)
        if not file.exists():
            raise KeyError(f"unknown sweep {sweep_id!r}")
        with self._lock:
            data = json.loads(file.read_text())
        known = set(SweepRecord.__dataclass_fields__)
        return SweepRecord(**{k: v for k, v in data.items() if k in known})

    def list(self, project: str | None = None) -> list[SweepRecord]:
        out = []
        for file in sorted(self.root.glob("*.json"), reverse=True):
            try:
                rec = self.get(file.stem)
            except (KeyError, ValueError, TypeError):
                continue
            if project is None or rec.project == project:
                out.append(rec)
        return out

    def delete(self, sweep_id: str) -> None:
        self._file(sweep_id).unlink(missing_ok=False)


# ----------------------------------------------------------------- runner

SweepListener = Callable[[str, dict], None]


class SweepRunner:
    """Runs one sweep's trials sequentially on a background thread."""

    def __init__(self, manager, store: SweepStore, graph: Graph, spec: SweepSpec,  # noqa: ANN001
                 project: str = "", listener: SweepListener | None = None):
        spec.check(graph)
        self.manager = manager
        self.store = store
        self.spec = spec
        self.graph_dict = graph.to_dict()
        self.listener = listener
        self._stop = threading.Event()
        self._current_run: str | None = None
        self.record = SweepRecord(
            sweep_id=time.strftime("sw-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4],
            name=graph.name, spec=spec.to_dict(), graph=self.graph_dict, project=project)
        store.save(self.record)
        self.thread: threading.Thread | None = None

    @property
    def sweep_id(self) -> str:
        return self.record.sweep_id

    # -------------------------------------------------------------- control
    def start(self) -> str:
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self.sweep_id

    def stop(self) -> None:
        self._stop.set()
        if self._current_run and self.manager.is_live(self._current_run):
            self.manager.stop(self._current_run)

    def wait(self, timeout: float = 3600.0) -> SweepRecord:
        if self.thread is not None:
            self.thread.join(timeout)
        return self.store.get(self.sweep_id)

    # ------------------------------------------------------------- internal
    def _emit(self, event: dict) -> None:
        if self.listener is not None:
            try:
                self.listener(self.sweep_id, event)
            except Exception:  # noqa: BLE001
                pass

    def _proposals(self):
        """Yield (values, tell) pairs; ``tell(score_or_None)`` feeds TPE."""
        spec = self.spec
        if spec.strategy == "grid":
            axes = [d.grid() for d in spec.dimensions]
            for combo in itertools.islice(itertools.product(*axes), spec.max_trials):
                yield {d.key: v for d, v in zip(spec.dimensions, combo, strict=True)}, None
            return
        study = None
        if spec.strategy == "tpe":
            try:
                import optuna

                optuna.logging.set_verbosity(optuna.logging.WARNING)
                study = optuna.create_study(
                    direction="minimize" if spec.minimize else "maximize",
                    sampler=optuna.samplers.TPESampler(seed=spec.seed))
            except ImportError:
                self.record.message = "Optuna is not installed; using random search"
        rng = random.Random(spec.seed)
        seen: set[str] = set()
        for _ in range(spec.max_trials):
            if study is not None:
                trial = study.ask()
                values = {d.key: d.suggest(trial) for d in spec.dimensions}

                def tell(score, _trial=trial, _study=study):
                    import optuna

                    if score == "pruned":
                        _study.tell(_trial, state=optuna.trial.TrialState.PRUNED)
                    elif score is None:
                        _study.tell(_trial, state=optuna.trial.TrialState.FAIL)
                    else:
                        _study.tell(_trial, score)

                yield values, tell
            else:
                for _attempt in range(50):  # avoid exact repeats when the space allows
                    values = {d.key: d.sample(rng) for d in spec.dimensions}
                    sig = json.dumps(values, sort_keys=True, default=str)
                    if sig not in seen:
                        break
                seen.add(sig)
                yield values, None

    def _trial_graph(self, values: dict) -> Graph:
        """The design a trial trains (the base graph with the trial's values)."""
        return Graph.from_dict(apply_values(self.graph_dict, values))

    def _await(self, row: TrialRow) -> dict:
        """Wait for the trial's run (AutoML watches the epochs and may prune it)."""
        return self.manager.wait(row.run_id, timeout=24 * 3600)

    def _score(self, run_id: str) -> float | None:
        rec = self.manager.history.get(run_id)
        for source in (rec.final_metrics, rec.best_metrics):
            value = source.get(self.spec.metric)
            if isinstance(value, (int, float)) and math.isfinite(value):
                return float(value)
        return None

    def _save_trial(self, row: TrialRow) -> None:
        trials = [t for t in self.record.trials if t["number"] != row.number]
        trials.append(asdict(row))
        self.record.trials = sorted(trials, key=lambda t: t["number"])
        scored = [t for t in self.record.trials if t["score"] is not None]
        if scored:
            pick = min if self.spec.minimize else max
            best = pick(scored, key=lambda t: t["score"])
            self.record.best = {k: best[k] for k in ("number", "run_id", "score", "values")}
        self.store.save(self.record)

    @staticmethod
    def _over_budget(graph: Graph) -> str:
        from ai_made_easy.core import budget

        try:
            over = [c for c in budget.check(graph) if c.over]
        except Exception:  # noqa: BLE001 — no estimate, no skip
            return ""
        return "; ".join(f"{c.kind.replace('_', ' ')} {c.used:.2f} {c.unit} > {c.limit:g} {c.unit}"
                         for c in over)

    def _run(self) -> None:
        self.record.state = "running"
        self.store.save(self.record)
        self._emit({"type": "sweep_started", "total": self.spec.max_trials})
        try:
            for number, (values, tell) in enumerate(self._proposals()):
                if self._stop.is_set():
                    break
                row = TrialRow(number=number, values=values)
                try:
                    trial_graph = self._trial_graph(values)
                except Exception as exc:  # noqa: BLE001 — recorded on the trial
                    row.state, row.message = "invalid", str(exc)
                    self._save_trial(row)
                    if tell:
                        tell(None)
                    self._emit({"type": "trial", **asdict(row)})
                    continue
                errors = [i for i in trial_graph.validate() if i.severity == "error"]
                if errors:
                    row.state, row.message = "invalid", errors[0].message
                    self._save_trial(row)
                    if tell:
                        tell(None)
                    self._emit({"type": "trial", **asdict(row)})
                    continue
                over = self._over_budget(trial_graph) if self.spec.skip_over_budget else ""
                if over:
                    row.state, row.message = "over_budget", over
                    self._save_trial(row)
                    if tell:
                        tell(None)
                    self._emit({"type": "trial", **asdict(row)})
                    continue
                trial_graph.name = f"{self.record.name}_t{number}"
                try:
                    row.run_id = self.manager.start(
                        trial_graph, self.spec.framework, project=self.record.project,
                        parent=self.sweep_id, trial=values)
                except Exception as exc:  # noqa: BLE001 — recorded on the trial
                    row.state, row.message = "failed", str(exc)
                    self._save_trial(row)
                    if tell:
                        tell(None)
                    continue
                self._current_run = row.run_id
                row.state = "running"
                self._save_trial(row)
                self._emit({"type": "trial", **asdict(row)})
                status = self._await(row)
                self._current_run = None
                if row.state != "pruned":
                    row.state = status["state"] if status["state"] in FINAL_STATES \
                        else "failed"
                row.score = self._score(row.run_id) if row.state == "finished" else None
                if row.state == "finished" and row.score is None:
                    row.message = f"the run did not report {self.spec.metric!r}"
                self._save_trial(row)
                if tell:
                    tell("pruned" if row.state == "pruned" else row.score)
                self._emit({"type": "trial", **asdict(row)})
            self.record.state = "stopped" if self._stop.is_set() else "finished"
        except Exception as exc:  # noqa: BLE001
            self.record.state, self.record.message = "failed", str(exc)
        self.record.finished_at = time.time()
        self.store.save(self.record)
        self._emit({"type": "sweep_done", "state": self.record.state,
                    "best": self.record.best})


def best_graph(record: SweepRecord) -> dict:
    """The sweep's base graph with the best trial's values applied (AutoML: the best
    trial's recipe built with its settings)."""
    if not record.best:
        raise SweepError("the sweep has no successful trial yet")
    if "automl" in record.spec:
        from ai_made_easy.core.automl import trial_design

        return trial_design(record.spec["automl"], record.best["values"]).to_dict()
    return apply_values(record.graph, record.best["values"])
