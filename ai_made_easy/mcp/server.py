"""MCP server: expose the AI Made Easy engine as Model Context Protocol tools.

Any MCP client (Claude Desktop, ZCode, ...) can design, inspect, generate
code for, and train models by driving the exact same pure-Python core the
desktop UI uses. Graphs travel as JSON — the same files the app saves.

Run:    aime-mcp            (stdio transport, for MCP client configs)
"""
from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from ai_made_easy.core import api

mcp = FastMCP("ai-made-easy")

SAMPLES_DIR = api.SAMPLES_DIR


# ------------------------------------------------------------ pure impls
# Thin JSON adapters over core.api (kept as functions so tests and other
# hosts can call them without the MCP transport).

def _err(exc: Exception) -> str:
    return json.dumps({"error": str(exc)})


def _impl_list_blocks(category: str | None = None) -> str:
    return json.dumps(api.list_blocks(category))


def _impl_list_samples() -> str:
    return json.dumps(api.list_samples())


def _impl_read_sample(name: str) -> str:
    try:
        return json.dumps(api.read_sample(name))
    except api.ApiError as exc:
        return _err(exc)


def _impl_validate(graph: dict) -> dict:
    return api.validate(graph)


def _impl_generate(graph: dict, target: str = "pytorch_model") -> str:
    try:
        return api.generate(graph, target)
    except api.ApiError as exc:
        return _err(exc)


def _impl_summarize(graph: dict) -> dict:
    return api.summarize(graph)


def _impl_expand(graph: dict, node_id: str) -> dict:
    return api.expand(graph, node_id)


def _impl_start_training(graph: dict, framework: str = "auto") -> dict:
    result = api.start_training(graph, framework)
    result["hint"] = "poll get_run_status(run_id); metrics arrive per epoch"
    return result


def _impl_run_status(run_id: str) -> dict:
    return api.run_status(run_id)


def _impl_run_metrics(run_id: str) -> dict:
    return api.run_metrics(run_id)


def _impl_stop_run(run_id: str) -> dict:
    return api.stop_run(run_id)


# ------------------------------------------------------------- MCP tools

@mcp.tool()
def list_blocks(category: str | None = None) -> str:
    """Every registered block with its full JSON schema (params, ports,
    category, composite flag). Filter by category optionally. Start here to
    learn the vocabulary before building graphs."""
    return _impl_list_blocks(category)


@mcp.tool()
def list_samples() -> str:
    """Sample project graphs shipped with the app — good starting points."""
    return _impl_list_samples()


@mcp.tool()
def read_sample(name: str) -> str:
    """Load one sample graph as JSON (pass a name from list_samples)."""
    return _impl_read_sample(name)


@mcp.tool()
def validate_graph(graph: dict[str, Any]) -> str:
    """Validate a block graph: structure, wiring, and shape inference.
    Returns issues with severities; 'valid' is true when no errors exist."""
    return json.dumps(_impl_validate(graph))


@mcp.tool()
def generate_code(graph: dict[str, Any], target: str = "pytorch_model") -> str:
    """Generate code for a graph. Targets: pytorch_model, keras_model,
    pytorch_train, keras_train, sklearn_train (runnable training scripts), llm (LLM workflow
    script from llm.* blocks)."""
    return _impl_generate(graph, target)


@mcp.tool()
def summarize_model(graph: dict[str, Any]) -> str:
    """Analytic model summary: per-layer output shapes and parameter counts
    (no torch needed)."""
    return json.dumps(_impl_summarize(graph))


@mcp.tool()
def expand_architecture(graph: dict[str, Any], node_id: str) -> str:
    """Expand a composite architecture block (arch.*) into primitive blocks.
    Returns the updated graph JSON — codegen needs expanded graphs."""
    return json.dumps(_impl_expand(graph, node_id))


@mcp.tool()
def start_training(graph: dict[str, Any], framework: str = "auto") -> str:
    """Start training the graph in a managed subprocess (framework: auto,
    pytorch, keras or sklearn). Returns a run_id; poll get_run_status /
    get_run_metrics; stop with stop_run. Every run is kept in the run history."""
    return json.dumps(_impl_start_training(graph, framework))


@mcp.tool()
def get_run_status(run_id: str) -> str:
    """Current state of a training run (state, epochs done, latest metrics,
    log tail)."""
    return json.dumps(_impl_run_status(run_id))


@mcp.tool()
def get_run_metrics(run_id: str) -> str:
    """All epoch events of a training run (losses + scores per epoch)."""
    return json.dumps(_impl_run_metrics(run_id))


@mcp.tool()
def stop_run(run_id: str) -> str:
    """Stop a running training run."""
    return json.dumps(_impl_stop_run(run_id))


@mcp.tool()
def list_runs(project: str | None = None) -> str:
    """Run history (newest first): status, final and best metrics, duration."""
    return json.dumps(api.list_runs(project))


@mcp.tool()
def compare_runs(run_ids: list[str]) -> str:
    """Compare two or more runs: metrics side by side and the parameters that
    differ between them."""
    try:
        return json.dumps(api.compare_runs(run_ids), default=str)
    except (api.ApiError, KeyError) as exc:
        return _err(exc)


@mcp.tool()
def sweepable_parameters(graph: dict[str, Any]) -> str:
    """Parameters of a graph a hyperparameter sweep can vary, with suggested
    ranges (keys look like 'opt.lr')."""
    return json.dumps(api.sweepable_params(graph), default=str)


@mcp.tool()
def start_sweep(graph: dict[str, Any], spec: dict[str, Any]) -> str:
    """Start a hyperparameter sweep. spec = {"dimensions": [{"node", "param",
    "kind": float|int|choice, "low", "high", "log", "values"}], "metric",
    "direction": min|max, "strategy": tpe|random|grid, "max_trials"}.
    Returns a sweep_id; poll get_sweep."""
    try:
        return json.dumps(api.start_sweep(graph, spec))
    except api.ApiError as exc:
        return _err(exc)


@mcp.tool()
def get_sweep(sweep_id: str) -> str:
    """Sweep state, every trial (values, score, run_id) and the best trial."""
    try:
        record = api.get_sweep(sweep_id)
        record.pop("graph", None)
        return json.dumps(record, default=str)
    except KeyError as exc:
        return _err(exc)


@mcp.tool()
def deploy_run(run_id: str, out_dir: str, formats: list[str] | None = None) -> str:
    """Build a model-server package (FastAPI app, Dockerfile, pinned requirements,
    optional ONNX / TorchScript / Core ML exports) from a finished training run."""
    try:
        return json.dumps(api.deploy_run(run_id, out_dir, formats), default=str)
    except (api.ApiError, KeyError) as exc:
        return _err(exc)


@mcp.tool()
def register_model(run_id: str, name: str) -> str:
    """Copy a finished run into the model registry as a new version."""
    try:
        return json.dumps(api.register_model(run_id, name), default=str)
    except (api.ApiError, KeyError) as exc:
        return _err(exc)


@mcp.tool()
def list_models() -> str:
    """Registered model versions with stage and metrics."""
    return json.dumps(api.list_models(), default=str)


@mcp.tool()
def import_model(kind: str, source: str, attr: str = "", input_shape: list[int] | None = None,
                 dtype: str = "float32") -> str:
    """Import an existing model as an editable graph: kind = pytorch (source = .py
    file or module, attr = class / factory, input_shape required), onnx (.onnx)
    or keras (.keras / .h5). Returns the graph and a fidelity report."""
    try:
        return json.dumps(api.import_model(kind, source, attr, input_shape, dtype), default=str)
    except api.ApiError as exc:
        return _err(exc)


@mcp.tool()
def list_tasks(family: str = "") -> str:
    """Tasks a design can be trained for (losses, metrics, output role, serving schema),
    and the model families (neural, classic, llm, ...)."""
    return json.dumps({**api.list_tasks(family or None), **api.list_families()})


@mcp.tool()
def estimate_budget(graph: dict, device: str = "", max_train_memory_gb: float = 0,
                    max_latency_ms: float = 0, max_params_m: float = 0) -> str:
    """Estimate FLOPs, training memory, model size and inference latency of a neural
    design on a target device (see ``devices``: rtx_3060, t4, a100_40, apple_m_series,
    iphone_ane, raspberry_pi_5, ...) and check it against the given limits (0 = none)."""
    limits = {k: v for k, v in {"max_train_memory_gb": max_train_memory_gb,
                                "max_latency_ms": max_latency_ms,
                                "max_params_m": max_params_m}.items() if v}
    try:
        report = api.estimate_budget(graph, device or None, limits)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": str(exc)})
    report["devices"] = [d["id"] for d in api.list_devices()["devices"]]
    return json.dumps(report)


@mcp.tool()
def profile_data(graph: dict | None = None, path: str = "", target: str = "") -> str:
    """Profile a dataset: the dataset block of ``graph``, or any table / class-folder
    ``path``. Returns rows, column statistics, class counts and findings (missing
    values, constant / ID-like columns, target leakage, duplicates, imbalance)."""
    try:
        return json.dumps(api.profile_data(graph, path=path or None, target=target or None),
                          default=str)
    except api.ApiError as exc:
        return _err(exc)


@mcp.tool()
def data_issues(graph: dict) -> str:
    """Data warnings for the design's dataset in the context of its pipeline."""
    try:
        return json.dumps(api.data_issues(graph))
    except api.ApiError as exc:
        return _err(exc)


@mcp.tool()
def split_preview(graph: dict) -> str:
    """Samples per class in train / validation / test, exactly as training splits them."""
    try:
        return json.dumps(api.split_preview(graph))
    except api.ApiError as exc:
        return _err(exc)


def main() -> None:
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
