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


# ---------------------------------------------------------------- deploy

def model_registry():
    from ai_made_easy.core.deploy import ModelRegistry

    return ModelRegistry()


def deploy_formats() -> dict:
    from ai_made_easy.core.deploy import FORMAT_LABELS, FORMATS

    return {fw: [{"id": f, "label": FORMAT_LABELS[f]} for f in fmts]
            for fw, fmts in FORMATS.items()}


def _package(source: Path, out_dir: str, formats, name=None, version="1") -> dict:  # noqa: ANN001
    from ai_made_easy.core.deploy import DeployError, build_package

    try:
        return build_package(source, out_dir, formats=tuple(formats or ()), name=name,
                             version=str(version), python=manager().python).to_dict()
    except DeployError as exc:
        raise ApiError(str(exc)) from exc


def deploy_run(run_id: str, out_dir: str, formats: list[str] | None = None,
               name: str | None = None) -> dict:
    """Build a model-server package (FastAPI + Dockerfile) from a finished run."""
    return _package(manager().history.path(run_id), out_dir, formats, name)


def register_model(run_id: str, name: str, description: str = "") -> dict:
    from ai_made_easy.core.deploy import RegistryError

    try:
        return model_registry().register(manager().history.path(run_id), name,
                                         description).to_dict()
    except RegistryError as exc:
        raise ApiError(str(exc)) from exc


def list_models() -> dict:
    rows = [v.to_dict() for v in model_registry().list()]
    return {"count": len(rows), "models": rows}


def set_model_stage(name: str, version: int, stage: str) -> dict:
    from ai_made_easy.core.deploy import RegistryError

    try:
        return model_registry().set_stage(name, int(version), stage).to_dict()
    except RegistryError as exc:
        raise ApiError(str(exc)) from exc


def delete_model(name: str, version: int) -> dict:
    model_registry().delete(name, int(version))
    return {"deleted": f"{name}/{version}"}


def deploy_model(name: str, version: str | int, out_dir: str,
                 formats: list[str] | None = None) -> dict:
    """Package a registered model version ('latest', 'production' or a number)."""
    registry = model_registry()
    mv = registry.get(name, version)
    return _package(registry.path(name, mv.version), out_dir, formats, name, mv.version)


# ---------------------------------------------------------------- import

def import_model(kind: str, source: str, attr: str = "", input_shape: list[int] | None = None,
                 dtype: str = "float32", kwargs: dict | None = None, name: str = "") -> dict:
    """Import a PyTorch module, ONNX file or Keras model as an editable graph."""
    from ai_made_easy.core.importers import ModelImportError
    from ai_made_easy.core.importers import import_model as _import

    try:
        return _import(kind, source=source, attr=attr, input_shape=input_shape, dtype=dtype,
                       kwargs=kwargs, name=name, python=manager().python).to_dict()
    except ModelImportError as exc:
        raise ApiError(str(exc)) from exc


# ---------------------------------------------------------------- data

_profiles = None


def profile_cache():
    global _profiles
    if _profiles is None:
        from ai_made_easy.core.data.lints import ProfileCache

        _profiles = ProfileCache()
    return _profiles


def _dataset_node(graph: Graph, node_id: str | None):
    from ai_made_easy.core.data.lints import dataset_nodes

    nodes = dataset_nodes(graph)
    if node_id:
        nodes = [n for n in nodes if n[0] == node_id]
    if not nodes:
        raise ApiError(f"no dataset block {node_id!r}" if node_id else
                       "the design has no dataset block")
    return nodes[0]


def profile_data(graph: dict | Graph | None = None, node_id: str | None = None, *,
                 path: str | None = None, target: str | None = None,
                 base: str | None = None) -> dict:
    """Profile the dataset behind a dataset block, or any table / folder ``path``."""
    from ai_made_easy.core.data.profile import profile_path

    if path:
        if not Path(path).expanduser().exists():
            raise ApiError(f"{path} does not exist")
        return profile_path(path, target=target).to_dict()
    if graph is None:
        raise ApiError("pass a graph or a path")
    _nid, type_id, params = _dataset_node(_graph(graph), node_id)
    return profile_cache().profile(type_id, params, base).to_dict()


def data_issues(graph: dict | Graph, base: str | None = None) -> dict:
    """Data warnings (missing values, leakage, imbalance, ...) for the design's datasets."""
    from ai_made_easy.core.data.lints import data_issues as _issues

    issues = _issues(_graph(graph), profile_cache(), base)
    return {"issues": [{"severity": i.severity, "message": i.message, "node_id": i.node_id}
                       for i in issues]}


def split_preview(graph: dict | Graph, base: str | None = None) -> dict:
    """Samples per class in train / validation / test, as the training script splits."""
    from ai_made_easy.core.data.lints import task_of
    from ai_made_easy.core.data.splits import split_preview as _preview
    from ai_made_easy.core.training import data_catalog as dcat

    g = _graph(graph)
    _nid, type_id, params = _dataset_node(g, None)
    split = next((dict(n.resolved_params()) for n in g.nodes.values()
                  if n.type_id == "prep.split"), None)
    if split is None:
        split = {p.name: p.default for p in dcat.BLOCKS["prep.split"].params}
    result = _preview(type_id, params, split, base=base, task=task_of(g) or "multiclass",
                      modality=dcat.BLOCKS[type_id].modality or "")
    if isinstance(result, str):
        raise ApiError(result)
    return result.to_dict()


def data_fingerprint(graph: dict | Graph, base: str | None = None) -> dict:
    from ai_made_easy.core.data.fingerprint import fingerprint

    _nid, type_id, params = _dataset_node(_graph(graph), None)
    return {"fingerprint": fingerprint(type_id, params, base)}


def augmentation_preview(graph: dict | Graph, out_dir: str, images: int = 4,
                         variants: int = 6, base: str | None = None) -> dict:
    """Render evaluation and random training views of dataset images (PNG files)."""
    from ai_made_easy.core.data.augment import PreviewError
    from ai_made_easy.core.data.augment import augmentation_preview as _preview

    try:
        return _preview(_graph(graph), out_dir, base=base, python=manager().python,
                        images=images, variants=variants)
    except PreviewError as exc:
        raise ApiError(str(exc)) from exc
