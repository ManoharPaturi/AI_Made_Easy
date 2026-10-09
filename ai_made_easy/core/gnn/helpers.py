"""PyTorch Geometric helpers emitted into generated model code.

A model with graph layers takes ``forward(x, edge_index=None, batch=None)``; its first line
builds a small mutable graph context ``g`` (edges and the graph id of every node) that
message-passing layers read and TopK / SAG pooling update. Without edges the model sees
isolated nodes, so ``model(x)`` still runs (export checks, shape tests).
"""
from __future__ import annotations

CONTEXT = '''\
import torch_geometric.nn as pyg


def graph_context(x: torch.Tensor, edge_index: torch.Tensor | None,
                  batch: torch.Tensor | None) -> dict:
    """Edges [2, E] and the graph id of every node (defaults: no edges, one graph)."""
    if edge_index is None:
        edge_index = torch.empty(2, 0, dtype=torch.long, device=x.device)
    if batch is None:
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    return {"edge_index": edge_index, "batch": batch}
'''

POOL = '''\
class GraphPool(nn.Module):
    """TopK / SAG pooling: keeps the highest-scoring nodes and the edges between them."""

    def __init__(self, kind: str, channels: int, ratio: float) -> None:
        super().__init__()
        cls = pyg.TopKPooling if kind == "topk" else pyg.SAGPooling
        self.pool = cls(channels, ratio=ratio)

    def forward(self, x: torch.Tensor, g: dict) -> torch.Tensor:
        x, edge_index, _attr, batch, _perm, _score = self.pool(
            x, g["edge_index"], batch=g["batch"])
        g["edge_index"], g["batch"] = edge_index, batch
        return x
'''

GNN_HELPERS = {"GraphContext": CONTEXT, "GraphPool": POOL}


def register() -> None:
    from ai_made_easy.core.codegen.helpers import PYTORCH_HELPERS

    PYTORCH_HELPERS.update(GNN_HELPERS)
