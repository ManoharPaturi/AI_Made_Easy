"""Bayesian deep learning: MC dropout, Bayes-by-Backprop layers, mixture density heads,
the last-layer Laplace approximation, calibration and conformal prediction for
supervised PyTorch designs."""
from ai_made_easy.core.bayes import blocks as _blocks
from ai_made_easy.core.bayes import helpers as _helpers

_helpers.register()
_blocks.register_all()

from ai_made_easy.core.bayes import rules as _rules  # noqa: E402, F401
