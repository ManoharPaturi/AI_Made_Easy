"""Design rules for forecasting (registered as lints; Quick Fixes in core.fixes).

"set <param> to <value>" and "Set the Input shape to '…'" phrasing becomes a
one-click fix on the flagged block.
"""
from __future__ import annotations

from ai_made_easy.core.forecast.blocks import FORECAST_MODELS, USES_FUTURE, _columns_count
from ai_made_easy.core.forecast.tasks import dataset_of, forecast_task, model_of
from ai_made_easy.core.lints import LintContext, _issue, _name, register_rule
from ai_made_easy.core.sequence.helpers import tcn_receptive_field

POINT_LOSSES = {"train.loss_mse", "train.loss_l1", "train.loss_huber", "train.loss_smooth_l1"}
DEFAULT_DATA = {"window": 48, "horizon": 12, "channels": 1, "future": 0, "counts": False}


def data_layout(node) -> dict:  # noqa: ANN001
    """Window, horizon, channels (target + past + future) and future count of a dataset."""
    if node is None:
        return dict(DEFAULT_DATA)
    p = dict(node.resolved_params())
    if node.type_id == "data.synthetic_series":
        future, past, counts = int(bool(p["promotions"])), 0, bool(p["counts"])
    else:
        future, past = _columns_count(p["future_covariates"]), _columns_count(p["past_covariates"])
        counts = None   # unknown until the data profile reads the file
    return {"window": int(p["window"]), "horizon": int(p["horizon"]),
            "channels": 1 + past + future, "future": future, "counts": counts}


def forecast_rules(ctx: LintContext) -> list:
    if forecast_task(ctx.graph) is None:
        return []
    graph = ctx.graph
    data_node = dataset_of(graph)
    data = data_layout(data_node)
    model = model_of(graph)
    out: list = []
    models = [n for n in graph.nodes.values() if n.type_id in FORECAST_MODELS]
    if len(models) > 1:
        out.append(_issue("error", "a design trains one forecaster: keep one Forecasting "
                                   "model block", models[1].instance_id))
    inputs = [n for n in graph.nodes.values() if n.type_id == "core.input"]
    want = [data["window"], data["channels"]]
    have = ctx.shapes.get(inputs[0].instance_id) if inputs else None
    if inputs and have is not None and list(have) != want:
        source = _name(data_node) if data_node is not None else "the generated series"
        out.append(_issue(
            "error", f"{source} gives windows of {want[0]} steps × {want[1]} channel(s) "
                     "(target, then past and future covariates) but the Input is "
                     f"{list(have)}. Set the Input shape to '{want[0]}, {want[1]}'",
            inputs[0].instance_id))
    if model is not None:
        out += _model_rules(ctx, model, data, data_node)
    else:
        out += _generic_rules(ctx, data)
    out += _tcn_rules(ctx, data)
    return out


def _model_rules(ctx: LintContext, model, data: dict, data_node) -> list:  # noqa: ANN001
    p = dict(model.resolved_params())
    out = []
    if int(p["horizon"]) != data["horizon"]:
        out.append(_issue("error", f"{_name(model)} predicts {p['horizon']} steps but the "
                                   f"dataset holds out {data['horizon']}: set horizon to "
                                   f"{data['horizon']}", model.instance_id))
    prods, cons = ctx.producers(model.instance_id), ctx.consumers(model.instance_id)
    if (prods and prods[0].type_id != "core.input") or \
            (cons and cons[0].type_id != "core.output"):
        out.append(_issue("error", f"{_name(model)} reads the history window from the Input "
                                   "and feeds the Output: remove the blocks in between",
                          model.instance_id))
    if model.type_id in USES_FUTURE:
        f = int(p["future_covariates"])
        if f != data["future"]:
            out.append(_issue("error", f"the dataset has {data['future']} known-future "
                                       f"covariate(s) but {_name(model)} expects {f}: set "
                                       f"future_covariates to {data['future']}",
                              model.instance_id))
    elif data["future"]:
        out.append(_issue("info", f"{_name(model)} only sees the covariates' past values; "
                                     "TiDE, TCN and RNN forecasters also read their known "
                                     "future (promotions, holidays) over the horizon",
                          model.instance_id))
    head = p["head"]
    if head == "negbin" and data["counts"] is False and data_node is not None:
        out.append(_issue("warning", "the negbin head models non-negative counts but this "
                                     "series is continuous: use student_t or quantile",
                          model.instance_id))
    if head == "point" and data["counts"]:
        out.append(_issue("info", "these are counts: the negbin head gives calibrated "
                                  "intervals and never predicts negative values",
                          model.instance_id))
    losses = ctx.nodes_of(*POINT_LOSSES)
    if losses and head != "point":
        out.append(_issue("warning", f"{_name(losses[0])} is ignored: the {head} head trains "
                                     "with its own likelihood / quantile loss",
                          losses[0].instance_id))
    return out


def _generic_rules(ctx: LintContext, data: dict) -> list:
    """Designs built from layers (e.g. TCN + Linear) must output one value per future step."""
    last = ctx.last_compute()
    out_node = ctx.output_node()
    if last is None or out_node is None:
        return []
    shape = ctx.shapes.get(last.instance_id)
    if shape is not None and list(shape) != [data["horizon"]]:
        return [_issue("error", f"a forecasting design outputs one value per future step "
                                f"([{data['horizon']}]) but this one outputs {list(shape)}: "
                                "end with Flatten + Linear to the horizon, or use a "
                                "Forecasting model block", last.instance_id)]
    return []


def _tcn_rules(ctx: LintContext, data: dict) -> list:
    out = []
    for node in ctx.nodes_of("seq.tcn", "forecast.tcn"):
        p = dict(node.resolved_params())
        levels, kernel = int(p["levels"]), int(p["kernel_size"])
        field = tcn_receptive_field(levels, kernel)
        if field < data["window"]:
            need = levels
            while tcn_receptive_field(need, kernel) < data["window"] and need < 12:
                need += 1
            out.append(_issue("warning", f"{_name(node)} sees only the last {field} steps of "
                                         f"the {data['window']}-step window (receptive field): "
                                         f"set levels to {need}", node.instance_id))
    return out


register_rule(forecast_rules)
