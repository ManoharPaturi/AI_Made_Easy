"""Normalizing flows: invertible layers (RealNVP affine couplings, masked autoregressive
layers, neural spline couplings, ActNorm, permutations), the density-estimation task and
its maximum-likelihood training loop."""
from ai_made_easy.core.flows import blocks as _blocks
from ai_made_easy.core.flows import helpers as _helpers

_helpers.register()
_blocks.register_all()

from ai_made_easy.core.flows import tasks  # noqa: E402, F401


def _register_trainer() -> None:
    from ai_made_easy.core.flows import template
    from ai_made_easy.core.training.generate import register_trainer

    register_trainer("flow", template.render, needs_spec=False)


_register_trainer()
from ai_made_easy.core.flows import rules as _rules  # noqa: E402, F401
