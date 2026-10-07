"""Headless application API: the one surface the CLI, the MCP server and the
web server call. Graphs travel as JSON dicts (the project-file format);
every function returns JSON-serialisable data. Pure Python, no Qt.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ai_made_easy.core.graph import Graph

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"

_manager = None


class ApiError(ValueError):
    """A request the API cannot honour (unknown target, bad input …)."""


def manager():
    """Process-wide RunManager (lazy, so importing the API stays cheap)."""
    global _manager
    if _manager is None:
        from ai_made_easy.core.runner.manager import RunManager

        _manager = RunManager()
    return _manager


def set_manager(mgr) -> None:  # noqa: ANN001 — tests and servers inject their own
    global _manager
    _manager = mgr


def _graph(graph: dict | Graph) -> Graph:
    if isinstance(graph, Graph):
        return graph
    if not isinstance(graph, dict):
        raise ApiError("graph must be a project JSON object")
    return Graph.from_dict(graph)


# ---------------------------------------------------------------- blocks

def list_blocks(category: str | None = None) -> dict:
    from ai_made_easy.core.registry import get_registry

    blocks = get_registry().list_blocks()
    if category:
        blocks = [b for b in blocks if b["category"].lower() == category.lower()]
    return {"count": len(blocks), "blocks": blocks}


def list_samples() -> dict:
    names = sorted(p.name for p in SAMPLES_DIR.glob("*.json"))
    return {"count": len(names), "samples": names}


def read_sample(name: str) -> dict:
    import json

    path = SAMPLES_DIR / Path(name).name
    if not path.exists():
        raise ApiError(f"no sample named {name!r}")
    return json.loads(path.read_text())


# ----------------------------------------------------------------- graph

def validate(graph: dict | Graph) -> dict:
    """Issues (with Quick Fix labels where one exists) and inferred shapes."""
    from ai_made_easy.core.fixes import fix_for_issue

    g = _graph(graph)
    issues = g.validate()
    shapes, _ = g.infer_shapes_detailed()
    rows = []
    for i in issues:
        try:
            fix = fix_for_issue(g, i)
        except Exception:  # noqa: BLE001 — a fix is optional
            fix = None
        rows.append({"severity": i.severity, "node": i.node_id, "message": i.message,
                     "fix": {"label": fix[0], "description": fix[1]} if fix else None})
    return {"valid": not any(i.severity == "error" for i in issues),
            "issues": rows,
            "shapes": {k: list(v) for k, v in shapes.items()}}


def apply_fix(graph: dict | Graph, issue_index: int) -> dict:
    """Apply the Quick Fix of ``validate(graph)["issues"][issue_index]``."""
    from ai_made_easy.core.fixes import fix_for_issue

    g = _graph(graph)
    issues = g.validate()
    if not 0 <= issue_index < len(issues):
        raise ApiError("no issue at that index (the graph changed?)")
    fix = fix_for_issue(g, issues[issue_index])
    if fix is None:
        raise ApiError("this issue has no automatic fix")
    return {"label": fix[0], "graph": fix[2].to_dict()}


def targets() -> dict:
    from ai_made_easy.core.targets import RENDERERS, target_label

    return {"targets": [{"id": t, "label": target_label(t)} for t in RENDERERS]}


def generate(graph: dict | Graph, target: str = "pytorch_model") -> str:
    from ai_made_easy.core.targets import RENDERERS

    if target not in RENDERERS:
        raise ApiError(f"unknown target {target!r}; one of {sorted(RENDERERS)}")
    return RENDERERS[target](_graph(graph))


def summarize(graph: dict | Graph) -> dict:
    from ai_made_easy.core.summary import summarize as _summarize

    s = _summarize(_graph(graph))
    return {"total_params": s.total_params,
            "total_params_display": s.total_params_display,
            "layers": [{"name": L.name, "type": L.type_id,
                        "output_shape": L.output_shape, "params": L.params}
                       for L in s.layers]}


def expand(graph: dict | Graph, node_id: str) -> dict:
    from ai_made_easy.core.composites import expand_in_graph

    expanded = expand_in_graph(_graph(graph), node_id)
    return {"graph": expanded.to_dict(), "node_count": len(expanded.nodes)}


# ------------------------------------------------------------------ runs

def start_training(graph: dict | Graph, framework: str = "auto", project: str = "",
                   tags: list[str] | None = None) -> dict:
    g = _graph(graph)
    errors = [i for i in g.validate() if i.severity == "error"]
    if errors:
        raise ApiError("the graph has errors: " + "; ".join(i.message for i in errors[:5]))
    run_id = manager().start(g, framework, project=project, tags=tags)
    return {"run_id": run_id}


def run_status(run_id: str) -> dict:
    return manager().status(run_id)


def run_metrics(run_id: str) -> dict:
    return {"run_id": run_id, "epochs": manager().metrics(run_id)}


def stop_run(run_id: str) -> dict:
    return manager().stop(run_id)


def list_runs(project: str | None = None, parent: str | None = None) -> dict:
    records = manager().history.list(project=project, parent=parent)
    rows = []
    for r in records:
        row = r.to_dict()
        row.pop("graph", None)
        row.pop("params", None)
        rows.append(row)
    return {"count": len(rows), "runs": rows}


def get_run(run_id: str) -> dict:
    data = manager().history.get(run_id).to_dict()
    data["epochs"] = manager().history.epochs(run_id)
    data["artifacts"] = sorted(p.name for p in manager().history.path(run_id).iterdir())
    return data


def update_run(run_id: str, *, tags: list[str] | None = None, note: str | None = None) -> dict:
    fields: dict[str, Any] = {}
    if tags is not None:
        fields["tags"] = [str(t) for t in tags]
    if note is not None:
        fields["note"] = str(note)
    return manager().history.update(run_id, **fields).to_dict()


def delete_run(run_id: str) -> dict:
    mgr = manager()
    if mgr.is_live(run_id) and mgr.status(run_id)["state"] == "running":
        raise ApiError("stop the run before deleting it")
    mgr.history.delete(run_id)
    return {"deleted": run_id}


def compare_runs(run_ids: list[str]) -> dict:
    from ai_made_easy.core.runs.history import compare

    if len(run_ids) < 2:
        raise ApiError("select at least two runs to compare")
    history = manager().history
    result = compare([history.get(r) for r in run_ids])
    result["epochs"] = {r: history.epochs(r) for r in run_ids}
    return result


# ---------------------------------------------------------------- sweeps

_sweeps: dict = {}
_sweep_store = None


def sweep_store():
    global _sweep_store
    from ai_made_easy.core.sweeps import SweepStore

    if _sweep_store is None or _sweep_store.root != SweepStore().root:
        _sweep_store = SweepStore()
    return _sweep_store


def sweepable_params(graph: dict | Graph) -> dict:
    from ai_made_easy.core.sweeps import sweepable_params as _params

    return {"params": _params(_graph(graph))}


def start_sweep(graph: dict | Graph, spec: dict, project: str = "",
                listener=None) -> dict:  # noqa: ANN001
    from ai_made_easy.core.sweeps import SweepError, SweepRunner, SweepSpec

    g = _graph(graph)
    errors = [i for i in g.validate() if i.severity == "error"]
    if errors:
        raise ApiError("the graph has errors: " + "; ".join(i.message for i in errors[:5]))
    try:
        runner = SweepRunner(manager(), sweep_store(), g, SweepSpec.from_dict(spec),
                             project=project, listener=listener)
    except (SweepError, TypeError) as exc:
        raise ApiError(str(exc)) from exc
    _sweeps[runner.sweep_id] = runner
    runner.start()
    return {"sweep_id": runner.sweep_id}


def get_sweep(sweep_id: str) -> dict:
    return sweep_store().get(sweep_id).to_dict()


def list_sweeps(project: str | None = None) -> dict:
    rows = []
    for rec in sweep_store().list(project):
        row = rec.to_dict()
        row.pop("graph", None)
        rows.append(row)
    return {"count": len(rows), "sweeps": rows}


def stop_sweep(sweep_id: str) -> dict:
    runner = _sweeps.get(sweep_id)
    if runner is None:
        raise ApiError("that sweep is not running in this process")
    runner.stop()
    return {"sweep_id": sweep_id, "stopping": True}


def wait_sweep(sweep_id: str, timeout: float = 3600.0) -> dict:
    runner = _sweeps.get(sweep_id)
    if runner is not None:
        runner.wait(timeout)
    return get_sweep(sweep_id)


def sweep_best_graph(sweep_id: str) -> dict:
    from ai_made_easy.core.sweeps import SweepError, best_graph

    try:
        return best_graph(sweep_store().get(sweep_id))
    except SweepError as exc:
        raise ApiError(str(exc)) from exc
