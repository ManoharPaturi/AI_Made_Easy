"""Experiment tracking: mirror runs to MLflow and Weights & Biases as they go.

One :class:`Tracker` interface, two implementations. The run manager mirrors from the
events every backend already reports (local, SSH, Kaggle, Colab, Modal), so remote runs
are tracked the same way as local ones:

* start: the block parameters, where it trains, and the design JSON
* every epoch: its metrics, with the epoch as the step
* done: the final metrics (``metrics.json``), the run's files and the status

Choose trackers per project (``meta.tracking`` in the design: ``"mlflow"``,
``"wandb"`` or both) or for every run with ``AIME_TRACKING=mlflow,wandb``. Sign-in is
each tool's own: ``MLFLOW_TRACKING_URI`` (default: a local SQLite store in
``$AIME_HOME/mlflow``) and ``wandb login`` / ``WANDB_MODE=offline``. A tracker that
fails is reported in the run's log and switched off; it never stops a run.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ai_made_easy.core.paths import subdir

TRACKERS = ("mlflow", "wandb")
LABELS = {"mlflow": "MLflow", "wandb": "Weights & Biases"}
EXTRA = "pip install 'ai-made-easy[tracking]'"
# run files mirrored as artifacts when a run finishes (models are mirrored by
# registering a version, see ``register_version``)
ARTIFACTS = ("metrics.json", "classes.json", "predictions.json", "inference.json",
             "epochs.jsonl", "export_report.json")
MODEL_GLOBS = ("*_best.pt", "*_best.keras", "*_model.joblib")


class TrackingError(RuntimeError):
    pass


def parse(value: Any) -> list[str]:
    """``"mlflow, wandb"`` / ``["mlflow"]`` -> known tracker ids, in order."""
    items = value if isinstance(value, (list, tuple)) else str(value or "").split(",")
    out = []
    for item in items:
        name = str(item).strip().lower().replace("w&b", "wandb")
        if name in ("", "none", "off"):
            continue
        if name not in TRACKERS:
            raise TrackingError(f"unknown tracker {name!r}; one of {', '.join(TRACKERS)}")
        if name not in out:
            out.append(name)
    return out


def selected(meta: dict | None = None) -> list[str]:
    """Trackers for a run: the design's ``meta.tracking`` plus ``$AIME_TRACKING``."""
    names = parse((meta or {}).get("tracking"))
    for name in parse(os.environ.get("AIME_TRACKING", "")):
        if name not in names:
            names.append(name)
    return names


def mlflow_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or \
        f"sqlite:///{(subdir('mlflow') / 'mlflow.db').as_posix()}"


def _numbers(metrics: dict | None) -> dict[str, float]:
    return {k: float(v) for k, v in (metrics or {}).items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)}


def _flat_params(params: dict) -> dict[str, str]:
    """Block parameters as strings (MLflow caps values at 6000 characters)."""
    return {k[:250]: json.dumps(v, default=str)[:6000] if not isinstance(v, str) else v[:6000]
            for k, v in params.items()}


class Tracker:
    """Mirror one run. Subclasses implement the ``_``-prefixed hooks."""

    name = ""

    def __init__(self, record: dict) -> None:
        self.record = record
        self.url = ""
        self.remote_id = ""

    # ------------------------------------------------------------ interface
    def start(self) -> None: ...

    def epoch(self, event: dict) -> None: ...

    def finish(self, status: str, run_dir: Path, final: dict) -> None: ...

    # -------------------------------------------------------------- helpers
    def tags(self) -> dict[str, str]:
        r = self.record
        return {"aime.run_id": r.get("run_id", ""), "aime.framework": r.get("framework", ""),
                "aime.kind": r.get("kind", "train"), "aime.project": r.get("project", ""),
                "aime.backend": r.get("backend") or "local",
                "aime.parent": r.get("parent", "")}

    def info(self) -> dict:
        return {"tracker": self.name, "id": self.remote_id, "url": self.url}


class MLflowTracker(Tracker):
    name = "mlflow"

    def __init__(self, record: dict) -> None:
        super().__init__(record)
        try:
            from mlflow.tracking import MlflowClient
        except ImportError as exc:
            raise TrackingError(f"MLflow tracking needs mlflow: {EXTRA}") from exc
        self.client = MlflowClient(tracking_uri=mlflow_uri())

    def _experiment(self) -> str:
        name = self.record.get("project") or "AI Made Easy"
        found = self.client.get_experiment_by_name(name)
        if found is not None:
            return found.experiment_id
        location = None
        if mlflow_uri().startswith("sqlite:") and "MLFLOW_TRACKING_URI" not in os.environ:
            location = (subdir("mlflow") / "artifacts").as_uri()
        return self.client.create_experiment(name, artifact_location=location)

    def start(self) -> None:
        run = self.client.create_run(self._experiment(),
                                     run_name=f"{self.record.get('name', 'run')} "
                                              f"{self.record.get('run_id', '')}",
                                     tags=self.tags())
        self.remote_id = run.info.run_id
        params = _flat_params(self.record.get("params") or {})
        items = list(params.items())
        from mlflow.entities import Param

        for i in range(0, len(items), 100):
            self.client.log_batch(self.remote_id,
                                  params=[Param(k, v) for k, v in items[i:i + 100]])
        self.client.log_dict(self.remote_id, self.record.get("graph") or {}, "design.json")
        self.url = f"{mlflow_uri()} (run {self.remote_id})"

    def epoch(self, event: dict) -> None:
        import time

        from mlflow.entities import Metric

        step = int(event.get("epoch") or 0)
        now = int(time.time() * 1000)
        metrics = [Metric(k, v, now, step) for k, v in _numbers(event.get("metrics")).items()]
        if metrics:
            self.client.log_batch(self.remote_id, metrics=metrics)

    def finish(self, status: str, run_dir: Path, final: dict) -> None:
        import time

        from mlflow.entities import Metric

        now = int(time.time() * 1000)
        metrics = [Metric(f"final_{k}", v, now, 0) for k, v in _numbers(final).items()]
        if metrics:
            self.client.log_batch(self.remote_id, metrics=metrics)
        for name in ARTIFACTS:
            if (run_dir / name).is_file():
                self.client.log_artifact(self.remote_id, str(run_dir / name), "run")
        self.client.set_terminated(self.remote_id,
                                   {"finished": "FINISHED", "stopped": "KILLED"}.get(status,
                                                                                    "FAILED"))


class WandbTracker(Tracker):
    name = "wandb"

    def __init__(self, record: dict) -> None:
        super().__init__(record)
        try:
            import wandb
        except ImportError as exc:
            raise TrackingError(f"W&B tracking needs wandb: {EXTRA}") from exc
        self.wandb = wandb
        self.run = None

    def start(self) -> None:
        r = self.record
        self.run = self.wandb.init(
            project=os.environ.get("WANDB_PROJECT") or r.get("project") or "ai-made-easy",
            name=f"{r.get('name', 'run')} {r.get('run_id', '')}",
            config={**(r.get("params") or {}), "framework": r.get("framework", "")},
            tags=[t for t in (r.get("framework"), r.get("backend") or "local") if t],
            group=r.get("parent") or None, job_type=r.get("kind", "train"),
            dir=str(subdir("wandb")), reinit="create_new")
        self.remote_id = self.run.id
        self.url = getattr(self.run, "url", "") or f"offline run {self.run.id}"
        with tempfile.TemporaryDirectory() as tmp:
            design = Path(tmp) / "design.json"
            design.write_text(json.dumps(r.get("graph") or {}, indent=1), encoding="utf-8")
            artifact = self.wandb.Artifact(f"design-{r.get('run_id', 'run')}", type="design")
            artifact.add_file(str(design))
            self.run.log_artifact(artifact)

    def epoch(self, event: dict) -> None:
        metrics = _numbers(event.get("metrics"))
        if metrics:
            step = int(event.get("epoch") or 0)
            self.run.log({**metrics, "epoch": step}, step=step)

    def finish(self, status: str, run_dir: Path, final: dict) -> None:
        for key, value in _numbers(final).items():
            self.run.summary[key] = value
        self.run.summary["status"] = status
        files = [run_dir / n for n in ARTIFACTS if (run_dir / n).is_file()]
        if files:
            artifact = self.wandb.Artifact(f"run-{self.record.get('run_id', 'run')}",
                                           type="run-files")
            for file in files:
                artifact.add_file(str(file))
            self.run.log_artifact(artifact)
        self.run.finish(exit_code=0 if status == "finished" else 1)


_CLASSES = {"mlflow": MLflowTracker, "wandb": WandbTracker}


class RunMirror:
    """The trackers of one run, fed from its events on a thread of their own (tracking
    never slows the run's event pump). A tracker that fails is switched off."""

    def __init__(self, record: dict, names: list[str], log=None,  # noqa: ANN001
                 on_start=None) -> None:  # noqa: ANN001
        import queue
        import threading

        self.record = record
        self.names = list(names)
        self.log = log or (lambda line: None)
        self.on_start = on_start
        self.trackers: list[Tracker] = []
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._work, name="aime-tracking", daemon=True)
        self._thread.start()

    def info(self) -> list[dict]:
        return [t.info() for t in self.trackers]

    def _start(self) -> None:
        for name in self.names:
            try:
                tracker = _CLASSES[name](self.record)
                tracker.start()
            except Exception as exc:  # noqa: BLE001 — tracking never stops a run
                self.log(f"{LABELS[name]} tracking is off for this run: {exc}")
                continue
            self.trackers.append(tracker)
            self.log(f"tracking with {LABELS[name]}: {tracker.url}")
        if self.on_start is not None:
            self.on_start(self.info())

    def _each(self, method: str, *args: Any) -> None:
        for tracker in list(self.trackers):
            try:
                getattr(tracker, method)(*args)
            except Exception as exc:  # noqa: BLE001
                self.log(f"{LABELS[tracker.name]} tracking stopped: {exc}")
                self.trackers.remove(tracker)

    def _work(self) -> None:
        self._start()
        while True:
            item = self._queue.get()
            if item is None:
                return
            method, args = item
            self._each(method, *args)

    def epoch(self, event: dict) -> None:
        self._queue.put(("epoch", (event,)))

    def finish(self, status: str, run_dir: Path) -> None:
        final: dict = {}
        metrics = run_dir / "metrics.json"
        if metrics.is_file():
            try:
                final = json.loads(metrics.read_text(encoding="utf-8"))
            except ValueError:
                final = {}
        self._queue.put(("finish", (status, run_dir, final)))
        self._queue.put(None)

    def join(self, timeout: float = 120.0) -> None:
        self._thread.join(timeout)


# ===================================================================== registries

def register_version(target: str, version_dir: Path | str, name: str, version: int,
                     *, stage: str = "", metrics: dict | None = None) -> dict:
    """Register a model-registry version in MLflow's model registry or as a W&B model
    artifact. Returns where it went."""
    version_dir = Path(version_dir)
    files = [p for p in sorted(version_dir.iterdir()) if p.is_file()]
    if target == "mlflow":
        try:
            from mlflow.tracking import MlflowClient
        except ImportError as exc:
            raise TrackingError(f"MLflow needs mlflow: {EXTRA}") from exc
        client = MlflowClient(tracking_uri=mlflow_uri())
        tracker = MLflowTracker({"name": name, "project": "AI Made Easy registry",
                                 "run_id": f"{name}-v{version}", "params": {},
                                 "kind": "register"})
        run = client.create_run(tracker._experiment(), run_name=f"register {name} v{version}",
                                tags={"aime.model": name, "aime.version": str(version)})
        for file in files:
            client.log_artifact(run.info.run_id, str(file), "model")
        for key, value in _numbers(metrics).items():
            client.log_metric(run.info.run_id, key, value)
        client.set_terminated(run.info.run_id)
        from mlflow.exceptions import MlflowException

        try:
            client.create_registered_model(name)
        except MlflowException:
            pass  # already registered
        mv = client.create_model_version(name, f"{run.info.artifact_uri}/model",
                                         run_id=run.info.run_id,
                                         tags={"aime.version": str(version)})
        if stage and stage != "none":
            client.set_registered_model_alias(name, stage, mv.version)
        return {"target": "mlflow", "name": name, "version": str(mv.version),
                "uri": f"models:/{name}/{mv.version}"}
    if target == "wandb":
        try:
            import wandb
        except ImportError as exc:
            raise TrackingError(f"W&B needs wandb: {EXTRA}") from exc
        run = wandb.init(project=os.environ.get("WANDB_PROJECT") or "ai-made-easy",
                         job_type="register", name=f"register {name} v{version}",
                         dir=str(subdir("wandb")), reinit="create_new")
        artifact = wandb.Artifact(name, type="model", metadata={
            "aime_version": version, "metrics": _numbers(metrics)})
        for file in files:
            artifact.add_file(str(file))
        aliases = [f"v{version}-aime"] + ([stage] if stage and stage != "none" else [])
        run.log_artifact(artifact, aliases=aliases)
        run.finish()
        return {"target": "wandb", "name": name, "version": f"v{version}-aime",
                "uri": f"{name}:{aliases[-1]}"}
    raise TrackingError(f"unknown registry {target!r}; one of {', '.join(TRACKERS)}")


def open_mlflow_model(uri: str, dest: Path | str | None = None) -> tuple[str, Path]:
    """Download an MLflow model (``models:/name/1``, ``runs:/id/model``, a path) and
    return ``(kind, file)`` for the importers: ``onnx`` or ``torch``."""
    try:
        import mlflow
    except ImportError as exc:
        raise TrackingError(f"opening MLflow models needs mlflow: {EXTRA}") from exc
    mlflow.set_tracking_uri(mlflow_uri())
    folder = Path(mlflow.artifacts.download_artifacts(
        artifact_uri=uri, dst_path=str(dest) if dest else tempfile.mkdtemp(prefix="aime_mlf_")))
    flavors: dict = {}
    mlmodel = next(iter(sorted(folder.rglob("MLmodel"))), None)
    if mlmodel is not None:
        import yaml

        flavors = (yaml.safe_load(mlmodel.read_text(encoding="utf-8")) or {}).get("flavors", {})
    onnx = sorted(folder.rglob("*.onnx"))
    if "onnx" in flavors or (onnx and "pytorch" not in flavors):
        if not onnx:
            raise TrackingError(f"{uri} has no .onnx file")
        return "onnx", onnx[0]
    torch_files = sorted(p for p in folder.rglob("*")
                         if p.suffix in (".pth", ".pt") and p.is_file())
    if "pytorch" in flavors or torch_files:
        if not torch_files:
            raise TrackingError(f"{uri} has no PyTorch model file")
        return "torch", torch_files[0]
    raise TrackingError(f"{uri} is not a PyTorch or ONNX model "
                        f"(flavors: {', '.join(flavors) or 'none'})")
