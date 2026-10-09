"""Gaussian processes: kernel algebra on the canvas, GPyTorch or scikit-learn scripts.

Importing the package registers the blocks, the ``gp`` family (frameworks ``gpytorch``
and ``sklearn``, chosen by the model's library) and the task.
"""
from ai_made_easy.core.gp import blocks as _blocks

_blocks.register_all()

from ai_made_easy.core.gp import tasks  # noqa: E402, F401


def _library(graph) -> str:  # noqa: ANN001
    from ai_made_easy.core.gp.kernels import model_node

    model = model_node(graph)
    return model.resolved_params()["library"] if model is not None else "gpytorch"


def _register_family() -> None:
    from ai_made_easy.core.families import Family, register_family, register_framework
    from ai_made_easy.core.gp.rules import gp_issues
    from ai_made_easy.core.gp.template import render
    from ai_made_easy.core.targets import register_target

    register_framework("gpytorch", "gpytorch", "gpytorch")
    register_family(Family(
        "gp", "Gaussian process",
        "Kernel-based regression and classification with uncertainty (GPyTorch or "
        "scikit-learn).",
        detect=lambda types: any(t.startswith("gp.") for t in types), priority=30,
        frameworks=("gpytorch", "sklearn"), targets=("gp_gpytorch", "gp_sklearn"),
        extras=("probabilistic",), validate=gp_issues, choose=_library,
        generators={"gpytorch": lambda g: render(g, "gpytorch"),
                    "sklearn": lambda g: render(g, "sklearn")}))
    register_target("gp_gpytorch", lambda g: render(g, "gpytorch"), "GPyTorch script")
    register_target("gp_sklearn", lambda g: render(g, "sklearn"), "scikit-learn GP script")


_register_family()
