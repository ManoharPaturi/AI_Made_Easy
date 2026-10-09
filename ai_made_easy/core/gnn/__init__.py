"""Graph neural networks (PyTorch Geometric): message-passing layers, pooling, link
prediction, graph datasets and the ``graph`` trainer (node classification, graph
classification, link prediction)."""
from ai_made_easy.core.gnn import blocks as _blocks
from ai_made_easy.core.gnn import helpers as _helpers

_helpers.register()
_blocks.register_all()

from ai_made_easy.core.gnn import tasks  # noqa: E402, F401


def _register_trainer() -> None:
    from ai_made_easy.core.gnn import template
    from ai_made_easy.core.training.generate import register_trainer

    register_trainer("graph", template.render, needs_spec=False)


_register_trainer()
from ai_made_easy.core.gnn import rules as _rules  # noqa: E402, F401
