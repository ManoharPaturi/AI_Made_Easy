"""Running pipelines: every compute stage is a child run (``parent`` = the pipeline id) in
the run history; Export and Register act on the model their input produced.

Each stage gets a fingerprint from its kind, settings, the design it trains (and that
design's data fingerprint) and its inputs' fingerprints. A stage whose fingerprint
matches a finished stage of any earlier attempt is reused instead of rerun, so running a
pipeline again resumes it after the last finished stage and skips unchanged work.

Records live in ``$AIME_HOME/pipelines/<pipeline_id>.json``. Pure Python.
"""
from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from ai_made_easy.core.paths import subdir
from ai_made_easy.core.pipelines import stages as S
from ai_made_easy.core.pipelines.designs import canonical
from ai_made_easy.core.pipelines.plan import PASS_THROUGH, Plan, Stage, build_plan

FINAL = ("finished", "failed", "stopped")
POLL_SECONDS = 0.5


class PipelineError(ValueError):
    pass


@dataclass
class PipelineRecord:
    pipeline_id: str
    name: str
    graph: dict
    project: str = ""
    state: str = "created"          # created | running | finished | failed | stopped
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    order: list[str] = field(default_factory=list)
    stages: dict[str, dict] = field(default_factory=dict)
    resumed_from: str = ""
    message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class PipelineStore:
    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root else subdir("pipelines")
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _file(self, pipeline_id: str) -> Path:
        if not pipeline_id or "/" in pipeline_id or "\\" in pipeline_id \
                or pipeline_id.startswith("."):
            raise KeyError(f"invalid pipeline id {pipeline_id!r}")
        return self.root / f"{pipeline_id}.json"

    def save(self, record: PipelineRecord) -> None:
        with self._lock:
            file = self._file(record.pipeline_id)
            tmp = file.with_suffix(".tmp")
            tmp.write_text(json.dumps(record.to_dict(), indent=1, default=str))
            tmp.replace(file)

    def get(self, pipeline_id: str) -> PipelineRecord:
        file = self._file(pipeline_id)
        if not file.exists():
            raise KeyError(f"unknown pipeline {pipeline_id!r}")
        with self._lock:
            data = json.loads(file.read_text())
        known = set(PipelineRecord.__dataclass_fields__)
        return PipelineRecord(**{k: v for k, v in data.items() if k in known})

    def list(self, project: str | None = None) -> list[PipelineRecord]:
        out = []
        for file in sorted(self.root.glob("pl-*.json"), reverse=True):
            try:
                rec = self.get(file.stem)
            except (KeyError, ValueError, TypeError):
                continue
            if project is None or rec.project == project:
                out.append(rec)
        return out

    def workspace(self, pipeline_id: str, stage_id: str) -> Path:
        return self.root / pipeline_id / stage_id


def _fingerprint(stage: Stage, inputs: list[str]) -> str:
    from ai_made_easy import __version__
    from ai_made_easy.core.data.fingerprint import graph_fingerprint

    design = None
    data = ""
    if stage.design is not None:
        design = canonical(stage.design)
        try:
            data = graph_fingerprint(stage.design.to_dict())
        except Exception:  # noqa: BLE001
            data = ""
    payload = {"kind": stage.kind, "params": stage.params, "design": design, "data": data,
               "inputs": inputs, "version": __version__}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


Listener = Callable[[str, dict], None]


class PipelineRunner:
    """Runs one pipeline's stages in order on a background thread."""

    def __init__(self, manager, store: PipelineStore, graph, project: str = "",  # noqa: ANN001
                 base=None, listener: Listener | None = None, resumed_from: str = ""):  # noqa: ANN001
        from ai_made_easy.core.pipelines.rules import pipeline_issues

        errors = [i for i in pipeline_issues(graph) if i.severity == "error"]
        if errors:
            raise PipelineError("the pipeline has errors: "
                                + "; ".join(i.message for i in errors[:4]))
        self.manager, self.store, self.listener = manager, store, listener
        self.graph = graph
        self.plan: Plan = build_plan(graph, base)
        self._stop = threading.Event()
        self._current: str | None = None
        self.record = PipelineRecord(
            pipeline_id=time.strftime("pl-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4],
            name=graph.name, graph=graph.to_dict(), project=project, order=self.plan.order,
            resumed_from=resumed_from,
            stages={s.id: {"id": s.id, "kind": s.kind, "label": s.label, "state": "pending",
                           "run_id": "", "fingerprint": "", "metrics": {}, "message": "",
                           "output": {}, "started_at": None, "finished_at": None,
                           "inputs": list(s.inputs)} for s in self.plan})
        store.save(self.record)
        self.thread: threading.Thread | None = None

    @property
    def pipeline_id(self) -> str:
        return self.record.pipeline_id

    # ---------------------------------------------------------------- control
    def start(self) -> str:
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self.pipeline_id

    def stop(self) -> None:
        self._stop.set()
        if self._current and self.manager.is_live(self._current):
            self.manager.stop(self._current)

    def wait(self, timeout: float = 3600.0) -> PipelineRecord:
        if self.thread is not None:
            self.thread.join(timeout)
        return self.store.get(self.pipeline_id)

    # ---------------------------------------------------------------- internal
    def _emit(self, event: dict) -> None:
        if self.listener is not None:
            try:
                self.listener(self.pipeline_id, event)
            except Exception:  # noqa: BLE001
                pass

    def _update(self, stage_id: str, **values) -> None:
        self.record.stages[stage_id].update(values)
        self.store.save(self.record)
        self._emit({"type": "stage", **self.record.stages[stage_id]})

    def _cached(self, fingerprint: str) -> dict | None:
        """A finished stage with this fingerprint from any pipeline run."""
        for rec in self.store.list():
            if rec.pipeline_id == self.pipeline_id:
                continue
            for row in rec.stages.values():
                if row.get("fingerprint") != fingerprint or row.get("state") not in (
                        "finished", "cached"):
                    continue
                run_id = row.get("run_id")
                if run_id:
                    try:
                        if self.manager.history.get(run_id).status != "finished":
                            continue
                    except KeyError:
                        continue
                path = (row.get("output") or {}).get("path")
                if path and not Path(path).exists():
                    continue
                return row
        return None

    def _run(self) -> None:
        self.record.state = "running"
        self.store.save(self.record)
        self._emit({"type": "pipeline_started", "stages": self.plan.order})
        outputs: dict[str, dict] = {}
        fingerprints: dict[str, str] = {}
        failed = False
        try:
            for stage in self.plan:
                if self._stop.is_set() or failed:
                    self._update(stage.id, state="stopped" if self._stop.is_set()
                                 else "skipped",
                                 message="" if self._stop.is_set() else "an earlier stage failed")
                    continue
                inputs = [outputs[i] for i in stage.inputs]
                fp = _fingerprint(stage, [fingerprints[i] for i in stage.inputs])
                fingerprints[stage.id] = fp
                cached = self._cached(fp)
                if cached is not None:
                    outputs[stage.id] = cached["output"]
                    self._update(stage.id, state="cached", fingerprint=fp,
                                 run_id=cached.get("run_id", ""),
                                 metrics=cached.get("metrics", {}),
                                 output=cached["output"], message="unchanged: reused")
                    continue
                self._update(stage.id, state="running", fingerprint=fp,
                             started_at=time.time())
                try:
                    result = self._execute(stage, inputs)
                except Exception as exc:  # noqa: BLE001 — recorded on the stage
                    failed = True
                    self._update(stage.id, state="stopped" if self._stop.is_set()
                                 else "failed", message=str(exc)[-2000:],
                                 finished_at=time.time())
                    continue
                outputs[stage.id] = result["output"]
                self._update(stage.id, state="finished", finished_at=time.time(), **result)
            self.record.state = "stopped" if self._stop.is_set() else \
                "failed" if failed else "finished"
        except Exception as exc:  # noqa: BLE001
            self.record.state, self.record.message = "failed", str(exc)
        self.record.finished_at = time.time()
        self.store.save(self.record)
        self._emit({"type": "pipeline_done", "state": self.record.state})

    def _await(self, run_id: str) -> dict:
        self._current = run_id
        while True:
            status = self.manager.status(run_id)
            if status["state"] in FINAL:
                break
            if self._stop.is_set():
                self.manager.stop(run_id)
            time.sleep(POLL_SECONDS)
        self.manager.wait(run_id, timeout=60)
        self._current = None
        rec = self.manager.history.get(run_id)
        if rec.status != "finished":
            raise PipelineError(rec.error.strip().splitlines()[-1] if rec.error.strip()
                                else f"the run {rec.status}")
        return {"run_id": run_id, "metrics": {k: v for k, v in rec.final_metrics.items()
                                              if isinstance(v, (int, float))}}

    def _trial(self, stage: Stage) -> dict:
        return {"pipeline_stage": stage.id, "pipeline_kind": stage.kind}

    def _start(self, design, stage: Stage) -> dict:  # noqa: ANN001
        run_id = self.manager.start(design, project=self.record.project,
                                    parent=self.pipeline_id, trial=self._trial(stage),
                                    tags=["pipeline"])
        result = self._await(run_id)
        result["output"] = {"model_dir": str(self.manager.history.path(run_id))}
        return result

    def _start_stage(self, design, stage: Stage, write) -> dict:  # noqa: ANN001
        run_id = self.manager.start_stage(design, write, kind=stage.kind.split(".", 1)[1],
                                          project=self.record.project,
                                          parent=self.pipeline_id, trial=self._trial(stage),
                                          tags=["pipeline"])
        result = self._await(run_id)
        result["output"] = {"model_dir": str(self.manager.history.path(run_id))}
        return result

    def _execute(self, stage: Stage, inputs: list[dict]) -> dict:
        p = stage.params
        source = Path(inputs[0]["model_dir"]) if inputs else None
        task = self._task_of(stage)
        if stage.kind == "pipeline.train":
            return self._start(_with_epochs(stage.design, p["epochs"]), stage)
        if stage.kind == "pipeline.cross_validate":
            design = _with_epochs(stage.design, p["epochs"])
            design = _with_kfold(design, int(p["k"]))
            result = self._start(design, stage)
            result["output"] = {}                            # scores only, no model
            return result
        if stage.kind == "pipeline.finetune":
            return self._start_stage(stage.model, stage, lambda wd: S.finetune(
                wd, source, p, stage.design))
        if stage.kind == "pipeline.distill":
            return self._start_stage(stage.design, stage, lambda wd: S.distill(
                wd, source, stage.design, p, task))
        if stage.kind == "pipeline.prune":
            return self._start_stage(stage.model, stage, lambda wd: S.prune(wd, source, p))
        if stage.kind == "pipeline.quantize":
            return self._start_stage(stage.model, stage, lambda wd: S.quantize(wd, source, p))
        if stage.kind == "pipeline.evaluate":
            classification = task not in ("regression", "distribution")
            result = self._start_stage(stage.model, stage, lambda wd: S.evaluate(
                wd, source, p, classification))
            result["output"] = dict(inputs[0])               # passes the model on
            return result
        if stage.kind == "pipeline.ensemble":
            members = [Path(i["model_dir"]) for i in inputs]
            first = self.plan.stages[stage.inputs[0]].model
            result = self._start_stage(first, stage, lambda wd: S.ensemble(
                wd, members, p, task))
            result["output"] = {}
            return result
        if stage.kind == "pipeline.export":
            return self._export(stage, source, inputs[0])
        if stage.kind == "pipeline.deploy":
            return self._register(stage, source, inputs[0])
        raise PipelineError(f"unknown stage {stage.kind}")

    def _task_of(self, stage: Stage) -> str:
        from ai_made_easy.core.tasks import task_of

        graph = stage.design if stage.kind == "pipeline.distill" else None
        if graph is None and stage.inputs:
            graph = self.plan.stages[stage.inputs[0]].model
        task = task_of(graph) if graph is not None else None
        return task.id if task else ""

    def _export(self, stage: Stage, source: Path, upstream: dict) -> dict:
        from ai_made_easy.core.deploy.package import build_package

        out = Path(stage.params["out_dir"]).expanduser() if stage.params["out_dir"] \
            else self.store.workspace(self.pipeline_id, stage.id)
        formats = tuple(f.strip() for f in str(stage.params["formats"]).split(",")
                        if f.strip())
        result = build_package(source, out, formats=formats)
        ok = {k: v.get("file") for k, v in result.formats.items() if v.get("ok")}
        failed = [k for k, v in result.formats.items() if not v.get("ok")]
        return {"output": {**upstream, "path": str(out)}, "metrics": {},
                "message": f"package in {out}; exported {', '.join(ok) or 'nothing'}"
                           + (f"; failed: {', '.join(failed)}" if failed else "")}

    def _register(self, stage: Stage, source: Path, upstream: dict) -> dict:
        from ai_made_easy.core.deploy import ModelRegistry
        from ai_made_easy.core.deploy.package import build_package

        registry = ModelRegistry()
        record = json.loads((source / "run.json").read_text())
        name = stage.params["name"] or record.get("name") or self.record.name
        version = registry.register(source, name, description=f"pipeline {self.pipeline_id}"
                                                               f" · stage {stage.id}")
        if stage.params["stage"] != "none":
            registry.set_stage(name, version.version, stage.params["stage"])
        output = {**upstream, "model": f"{name}:{version.version}"}
        message = f"registered {name} v{version.version} ({stage.params['stage']})"
        if stage.params["package"]:
            out = self.store.workspace(self.pipeline_id, stage.id)
            build_package(source, out, name=name, version=str(version.version))
            output["path"] = str(out)
            message += f"; package in {out}"
        return {"output": output, "metrics": {}, "message": message}


def _with_epochs(design, epochs: int):  # noqa: ANN001, ANN202
    design = copy.deepcopy(design)
    if epochs:
        for node in design.nodes.values():
            if node.type_id == "train.trainer":
                node.params["epochs"] = int(epochs)
    return design


def _with_kfold(design, k: int):  # noqa: ANN001, ANN202
    from ai_made_easy.core.graph import NodeInstance

    design = copy.deepcopy(design)
    existing = [n for n in design.nodes.values() if n.type_id == "train.kfold"]
    if existing:
        existing[0].params["k"] = k
    else:
        design.add_node(NodeInstance("pipeline_kfold", "train.kfold", {"k": k}, (0, 0)))
    return design


def resume(manager, store: PipelineStore, pipeline_id: str, base=None,  # noqa: ANN001
           listener: Listener | None = None) -> PipelineRunner:
    """Run a pipeline again: its finished stages are reused, the rest run."""
    from ai_made_easy.core.graph import Graph

    old = store.get(pipeline_id)
    runner = PipelineRunner(manager, store, Graph.from_dict(old.graph), old.project, base,
                            listener, resumed_from=pipeline_id)
    runner.start()
    return runner


__all__ = ["PASS_THROUGH", "PipelineError", "PipelineRecord", "PipelineRunner",
           "PipelineStore", "resume"]
