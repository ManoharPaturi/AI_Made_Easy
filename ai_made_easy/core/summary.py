"""Analytic model summary: per-layer output shapes + parameter counts,
computed from the IR alone — no torch import, instant, works in the UI,
the CLI, and (later) MCP tool responses.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ai_made_easy.core.graph import Graph
from ai_made_easy.core.spec import shape_volume


@dataclass
class LayerSummary:
    name: str
    type_id: str
    output_shape: list[int]
    params: int


@dataclass
class ModelSummary:
    total_params: int = 0
    layers: list[LayerSummary] = field(default_factory=list)

    @property
    def total_params_display(self) -> str:
        n = self.total_params
        if n >= 1_000_000:
            return f"{n / 1_000_000:.2f}M"
        if n >= 1_000:
            return f"{n / 1_000:.1f}K"
        return str(n)


def summarize(graph: Graph) -> ModelSummary:
    shapes = graph.infer_shapes()
    summary = ModelSummary()
    for node in graph.model_nodes():
        defn = node.definition()
        params = 0
        if defn.param_fn is not None:
            in_shapes = [
                shapes[e.source_id]
                for e in (
                    graph.input_edge_for(node.instance_id, p.name)
                    for p in defn.inputs
                )
                if e is not None
            ]
            params = int(defn.param_fn(in_shapes, node.resolved_params()))
        summary.layers.append(
            LayerSummary(
                name=defn.display_name,
                type_id=defn.type_id,
                output_shape=shapes[node.instance_id],
                params=params,
            )
        )
        summary.total_params += params
    return summary


# --------------------------------------------------------------- formulas

def linear_params(in_shapes, params) -> int:
    (s,) = in_shapes
    return s[0] * int(params["units"]) + (int(params["units"]) if params["bias"] else 0)


def conv_params(rank: int):
    def fn(in_shapes, params) -> int:
        (s,) = in_shapes
        k = int(params["kernel_size"])
        cout = int(params["out_channels"])
        return s[0] * cout * k**rank + cout

    return fn


def embedding_params(in_shapes, params) -> int:
    return int(params["num_embeddings"]) * int(params["embedding_dim"])


def lstm_params(in_shapes, params) -> int:
    (s,) = in_shapes
    h, layers = int(params["hidden_size"]), int(params["num_layers"])
    dirs = 2 if params["bidirectional"] else 1
    total, cin = 0, s[-1]
    for layer in range(layers):
        inp = cin if layer == 0 else h * dirs
        total += dirs * (4 * h * inp + 4 * h * h + 8 * h)
    return total


def gru_params(in_shapes, params) -> int:
    (s,) = in_shapes
    h, layers = int(params["hidden_size"]), int(params["num_layers"])
    dirs = 2 if params["bidirectional"] else 1
    total, cin = 0, s[-1]
    for layer in range(layers):
        inp = cin if layer == 0 else h * dirs
        total += dirs * (3 * h * inp + 3 * h * h + 6 * h)
    return total


def mha_params(in_shapes, params) -> int:
    embed = in_shapes[0][-1]
    return 4 * embed * embed + 4 * embed


def transformer_encoder_params(in_shapes, params) -> int:
    d = in_shapes[0][-1]
    f = int(params["dim_feedforward"])
    attn = 4 * d * d + 4 * d
    ffn = d * f + f + f * d + d
    norms = 4 * d
    return attn + ffn + norms


def _num_features(in_shapes) -> int:
    s = in_shapes[0]
    return s[0] if len(s) >= 3 else s[-1]


def batch_norm_params(in_shapes, params) -> int:
    return 2 * _num_features(in_shapes)


def layer_norm_params(in_shapes, params) -> int:
    return 2 * shape_volume(in_shapes[0])


def group_norm_params(in_shapes, params) -> int:
    return 2 * in_shapes[0][0]


def prelu_params(in_shapes, params) -> int:
    return 1
