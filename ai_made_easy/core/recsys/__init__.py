"""Recommenders: matrix factorisation, NCF, two-tower retrieval and DLRM-lite with the
``recommendation`` trainer (BPR / BCE / MSE objectives, Recall@K, NDCG@K, MRR)."""
from ai_made_easy.core.recsys import blocks as _blocks
from ai_made_easy.core.recsys import helpers as _helpers

_helpers.register()
_blocks.register_all()

from ai_made_easy.core.recsys import tasks  # noqa: E402, F401


def _register_trainer() -> None:
    from ai_made_easy.core.recsys import template
    from ai_made_easy.core.training.generate import register_trainer

    register_trainer("recommendation", template.render, needs_spec=False)


_register_trainer()
from ai_made_easy.core.recsys import rules as _rules  # noqa: E402, F401
