"""Probabilistic programming (PyMC): distributions wired into hierarchical models.

Importing the package registers the blocks, the ``ppl`` family with its ``pymc``
framework and script generator, and the task.
"""
from ai_made_easy.core.ppl import blocks as _blocks

_blocks.register_all()

from ai_made_easy.core.ppl import tasks  # noqa: E402, F401


def _register_family() -> None:
    from ai_made_easy.core.families import Family, register_family, register_framework
    from ai_made_easy.core.ppl.rules import ppl_issues
    from ai_made_easy.core.ppl.template import render
    from ai_made_easy.core.targets import register_target

    register_framework("pymc", "pymc", "pymc")
    register_family(Family(
        "ppl", "Probabilistic program",
        "Bayesian models from distributions, expressions and data (PyMC).",
        detect=lambda types: any(t.startswith("ppl.") for t in types), priority=30,
        frameworks=("pymc",), targets=("pymc_train",), extras=("probabilistic",),
        validate=ppl_issues, generators={"pymc": render}))
    register_target("pymc_train", render, "PyMC script")


_register_family()
