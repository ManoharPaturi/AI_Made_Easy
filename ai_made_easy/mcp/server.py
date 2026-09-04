"""MCP server: expose the AI Made Easy engine as Model Context Protocol tools.

Any MCP client (Claude Desktop, ZCode, ...) can design, inspect, generate
code for, and train models by driving the exact same pure-Python core the
desktop UI uses. Graphs travel as JSON — the same files the app saves.

Run:    aime-mcp            (stdio transport, for MCP client configs)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from ai_made_easy.core.codegen import generate
from ai_made_easy.core.codegen.llm_gen import generate_llm_script
from ai_made_easy.core.codegen.training_gen import generate_training
from ai_made_easy.core.composites import expand_in_graph
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.runner.manager import RunManager
from ai_made_easy.core.summary import summarize

mcp = FastMCP("ai-made-easy")

SAMPLES_DIR = Path(__file__).parent.parent.parent / "samples"
_runs = RunManager()

_GENERATE_TARGETS = {
    "pytorch_model": lambda g: generate(g, "pytorch"),
    "keras_model": lambda g: generate(g, "keras"),
    "pytorch_train": lambda g: generate_training(g, "pytorch"),
    "keras_train": lambda g: generate_training(g, "keras"),
    "llm": lambda g: generate_llm_script(g),
}


# ------------------------------------------------------------ pure impls

def _impl_list_blocks(category: str | None = None) -> str:
    blocks = get_registry().list_blocks()
    if category:
        blocks = [b for b in blocks if b["category"].lower() == category.lower()]
    return json.dumps({"count": len(blocks), "blocks": blocks})


def _impl_list_samples() -> str:
    names = sorted(p.name for p in SAMPLES_DIR.glob("*.json"))
    return json.dumps({"count": len(names), "samples": names})


def _impl_read_sample(name: str) -> str:
    path = SAMPLES_DIR / Path(name).name
    if not path.exists():
        return json.dumps({"error": f"no sample named {name!r}"})
    return path.read_text()


def _impl_validate(graph: dict) -> dict:
    g = Graph.from_dict(graph)
    issues = g.validate()
    return {"valid": not any(i.severity == "error" for i in issues),
            "issues": [{"severity": i.severity, "node": i.node_id,
                        "message": i.message} for i in issues]}


def _impl_generate(graph: dict, target: str = "pytorch_model") -> str:
    if target not in _GENERATE_TARGETS:
        return json.dumps({"error": f"unknown target {target!r}; one of "
                                    f"{sorted(_GENERATE_TARGETS)}"})
    return _GENERATE_TARGETS[target](Graph.from_dict(graph))


def _impl_summarize(graph: dict) -> dict:
    s = summarize(Graph.from_dict(graph))
    return {"total_params": s.total_params,
            "total_params_display": s.total_params_display,
            "layers": [{"name": L.name, "type": L.type_id,
                        "output_shape": L.output_shape, "params": L.params}
                       for L in s.layers]}


def _impl_expand(graph: dict, node_id: str) -> dict:
    expanded = expand_in_graph(Graph.from_dict(graph), node_id)
    return {"graph": expanded.to_dict(),
            "node_count": len(expanded.nodes)}


def _impl_start_training(graph: dict, framework: str = "pytorch") -> dict:
    run_id = _runs.start(Graph.from_dict(graph), framework)
    return {"run_id": run_id,
            "hint": "poll get_run_status(run_id); metrics arrive per epoch"}


def _impl_run_status(run_id: str) -> dict:
    return _runs.status(run_id)


def _impl_run_metrics(run_id: str) -> dict:
    return {"run_id": run_id, "epochs": _runs.metrics(run_id)}


def _impl_stop_run(run_id: str) -> dict:
    return _runs.stop(run_id)


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
    pytorch_train, keras_train (runnable training scripts), llm (LLM workflow
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
def start_training(graph: dict[str, Any], framework: str = "pytorch") -> str:
    """Start training the graph in a managed subprocess. Returns a run_id;
    poll get_run_status / get_run_metrics; stop with stop_run."""
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


def main() -> None:
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
