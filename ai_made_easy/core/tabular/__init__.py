"""Deep learning for tables: ResNet-MLP, FT-Transformer, TabTransformer, TabNet and
categorical entity embeddings, trained by the supervised PyTorch pipeline."""
from ai_made_easy.core.tabular import blocks as _blocks
from ai_made_easy.core.tabular import helpers as _helpers

_helpers.register()
_blocks.register_all()

from ai_made_easy.core.tabular import rules as _rules  # noqa: E402, F401


def _register_profiler() -> None:
    from ai_made_easy.core.data.profile import register_profiler
    from ai_made_easy.core.tabular.synthetic import profile

    register_profiler("data.synthetic_table", profile)


_register_profiler()
