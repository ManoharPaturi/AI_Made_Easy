"""Time-series forecasting: models with distribution heads, multi-series data, backtests.

Importing the package registers the blocks, the task and its resolver, the rules,
the codegen helpers, the training loop and the data profiler.
"""
from ai_made_easy.core.forecast import blocks as _blocks
from ai_made_easy.core.forecast import helpers as _helpers

_helpers.register()
_blocks.register_all()

from ai_made_easy.core.forecast import tasks  # noqa: E402, F401
from ai_made_easy.core.forecast import template as _template  # noqa: E402

_template.register()
from ai_made_easy.core.forecast import rules as _rules  # noqa: E402, F401
from ai_made_easy.core.forecast import profile as _profile  # noqa: E402

_profile.register()
