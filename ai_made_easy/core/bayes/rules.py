"""Design rules for Bayesian layers, the Laplace approximation and calibration blocks."""
from __future__ import annotations

from ai_made_easy.core.bayes.blocks import CALIBRATION, STOCHASTIC, quantile_levels
from ai_made_easy.core.lints import LintContext, _issue, _name, register_rule

REGRESSION_LOSSES = ("train.loss_mse", "train.loss_l1", "train.loss_huber",
                     "train.loss_smooth_l1")


def bayes_rules(ctx: LintContext) -> list:
    types = {n.type_id for n in ctx.graph.nodes.values()}
    if not types & {*STOCHASTIC, "bayes.mdn", "bayes.quantile_head", "bayes.laplace",
                    *CALIBRATION}:
        return []
    out = []
    last = ctx.last_compute()
    loss = ctx.loss_type()
    classification = loss is not None and loss not in REGRESSION_LOSSES
    for node in ctx.nodes_of("bayes.mdn"):
        if last is None or node.instance_id != last.instance_id:
            out.append(_issue("error", f"{_name(node)} must be the last layer: it outputs the "
                                       "mixture's parameters", node.instance_id))
        if classification:
            out.append(_issue("error", "a mixture density head predicts real-valued targets: "
                                       "use an MSE loss block (training uses the mixture's "
                                       "likelihood instead)", node.instance_id))
    for node in ctx.nodes_of("bayes.quantile_head"):
        if last is None or node.instance_id != last.instance_id:
            out.append(_issue("error", f"{_name(node)} must be the last layer: it outputs the "
                                       "quantiles", node.instance_id))
        if classification:
            out.append(_issue("error", "a quantile head predicts real-valued targets: use an "
                                       "MSE loss block (training uses the pinball loss "
                                       "instead)", node.instance_id))
        if ctx.nodes_of("bayes.mdn", "bayes.laplace"):
            out.append(_issue("error", "a quantile head does not combine with a mixture "
                                       "density head or the Laplace approximation: keep one",
                              node.instance_id))
        conformal = ctx.nodes_of("eval.conformal")
        levels = quantile_levels(node.resolved_params())
        if conformal and len(levels) >= 2:
            alpha = float(conformal[0].resolved_params()["alpha"])
            if abs((levels[-1] - levels[0]) - (1 - alpha)) > 0.02:
                out.append(_issue("info", f"the head's outer quantiles ({levels[0]:g}, "
                                          f"{levels[-1]:g}) form a {levels[-1] - levels[0]:.0%} "
                                          f"interval but Conformal Prediction targets "
                                          f"{1 - alpha:.0%}: CQR still calibrates it, but "
                                          f"quantiles {alpha / 2:g} and {1 - alpha / 2:g} give "
                                          "tighter intervals", node.instance_id))
        elif conformal:
            out.append(_issue("error", "conformalized quantile regression needs at least two "
                                       "quantiles (a lower and an upper one)",
                              node.instance_id))
    for node in ctx.nodes_of("bayes.laplace"):
        if last is None or last.type_id != "core.dense":
            out.append(_issue("error", "the Laplace approximation needs the model to end with "
                                       "a Dense layer (no activation after it)",
                              node.instance_id))
        if ctx.nodes_of("bayes.mdn"):
            out.append(_issue("error", "the Laplace approximation and a mixture density head "
                                       "do not combine: keep one", node.instance_id))
    if not classification:
        for node in ctx.nodes_of("eval.ece", "eval.temperature_scaling"):
            out.append(_issue("warning", f"{_name(node)} measures class probabilities: it "
                                         "has no effect on regression", node.instance_id))
    data = next((n for n in ctx.graph.nodes.values() if n.type_id == "data.timeseries_csv"),
                None)
    if data is not None:
        out.append(_issue("warning", "uncertainty and conformal intervals are reported in the "
                                     "scaled target units for time-series windows",
                          data.instance_id))
    stochastic = ctx.nodes_of(*STOCHASTIC)
    if stochastic and all(int(n.resolved_params()["samples"]) <= 1 for n in stochastic):
        out.append(_issue("info", "samples = 1: predictions use a single stochastic pass "
                                  "(raise samples to average several)",
                          stochastic[0].instance_id))
    return out


register_rule(bayes_rules)
