"""Probabilistic graphical models (pgmpy) and hidden Markov models (hmmlearn).

Importing the package registers the blocks, the ``pgm`` family with its ``pgmpy``
framework and script generator, and the tasks.
"""
from ai_made_easy.core.pgm import blocks as _blocks

_blocks.register_all()

from ai_made_easy.core.pgm import tasks  # noqa: E402, F401


def _register_family() -> None:
    from ai_made_easy.core.families import Family, register_family, register_framework
    from ai_made_easy.core.pgm.rules import pgm_issues
    from ai_made_easy.core.pgm.template import render
    from ai_made_easy.core.targets import register_target

    register_framework("pgmpy", "pgmpy", "pgmpy")
    register_family(Family(
        "pgm", "Graphical model",
        "Bayesian / Markov networks, naive Bayes, dynamic networks and HMMs (pgmpy).",
        detect=lambda types: any(t.startswith("pgm.") for t in types), priority=30,
        frameworks=("pgmpy",), targets=("pgmpy_train",), extras=("probabilistic",),
        validate=pgm_issues, acyclic=False, generators={"pgmpy": render}))
    register_target("pgmpy_train", render, "pgmpy script")


_register_family()
