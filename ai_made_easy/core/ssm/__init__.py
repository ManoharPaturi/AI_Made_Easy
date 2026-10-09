"""State-space models: structural time series (level, trend, seasonality, cycles,
autoregression, regression) and SARIMAX, fitted with the Kalman filter (statsmodels).

Importing the package registers the blocks, the ``ssm`` family (framework
``statsmodels``) and the task.
"""
from ai_made_easy.core.ssm import blocks as _blocks

_blocks.register_all()

from ai_made_easy.core.ssm import tasks  # noqa: E402, F401


def _register_family() -> None:
    from ai_made_easy.core.families import Family, register_family, register_framework
    from ai_made_easy.core.ssm.rules import ssm_issues
    from ai_made_easy.core.ssm.template import render
    from ai_made_easy.core.targets import register_target

    register_framework("statsmodels", "statsmodels", "statsmodels")
    register_family(Family(
        "ssm", "State-space model",
        "Structural time series and SARIMAX with the Kalman filter (statsmodels).",
        detect=lambda types: any(t.startswith("ssm.") for t in types), priority=30,
        frameworks=("statsmodels",), targets=("ssm_statsmodels",),
        extras=("probabilistic",), validate=ssm_issues,
        generators={"statsmodels": render}))
    register_target("ssm_statsmodels", render, "statsmodels script")


_register_family()
