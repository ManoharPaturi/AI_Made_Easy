"""Forecasting blocks: models with distribution heads, datasets and metrics."""
from __future__ import annotations

from dataclasses import replace

from ai_made_easy.core.blocks._dsl import P, ShapeError, nn_block
from ai_made_easy.core.forecast import helpers as h
from ai_made_easy.core.palette import family_color
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import BlockDefinition, PortSpec

CATEGORY = "Forecasting"
HEADS = ("point", "quantile", "student_t", "negbin")
# models that read known-future covariates
USES_FUTURE = {"forecast.tcn", "forecast.tide", "forecast.rnn"}
FORECAST_MODELS: dict[str, str] = {}   # type id -> helper class
FORECAST_DATA: dict[str, dict] = {}
FORECAST_METRICS: dict[str, dict] = {}


def quantile_levels(p: dict) -> tuple[float, ...]:
    try:
        levels = tuple(sorted(float(v) for v in str(p.get("quantiles", "")).split(",")
                              if v.strip()))
    except ValueError:
        return ()
    return levels


def head_size(p: dict) -> int:
    return h.head_k(p["head"], len(quantile_levels(p)) or 1)


def _out_shape(p: dict) -> list[int]:
    horizon = int(p["horizon"])
    return [horizon] if p["head"] == "point" else [horizon, head_size(p)]


def _shape(name: str, check=None):
    def fn(in_shapes, p):
        s = in_shapes[0]
        if len(s) != 2:
            raise ShapeError(f"{name} reads a history window [L, C] (time steps × channels; "
                             f"channel 0 is the target), got {s}")
        if p["head"] == "quantile":
            levels = quantile_levels(p)
            if not levels or any(not 0 < q < 1 for q in levels):
                raise ShapeError("quantiles must be comma-separated levels between 0 and 1")
        if check:
            check(s, p)
        return _out_shape(p)
    return fn


def _patch_check(s, p):
    if int(p["patch_len"]) > s[0]:
        raise ShapeError(f"patch_len {p['patch_len']} is longer than the {s[0]}-step window")


def _heads_args(c: dict) -> str:
    return f"\"{c['head']}\", {quantile_levels(c) or (0.5,)!r}"


def _cost(fn):
    def wrapped(in_shapes, p):
        params, macs = fn(in_shapes[0], p, int(p["horizon"]) * head_size(p))
        return params, 2 * macs
    return wrapped


COMMON = (P("horizon", "int", 12, lo=1, help="Future steps to predict"),
          P("head", "enum", "point", options=HEADS,
            help="point: one value per step; quantile: prediction intervals; student_t: "
                 "heavy-tailed distribution; negbin: counts (sales, visits)"),
          P("quantiles", "str", "0.1, 0.5, 0.9", help="Levels for the quantile head"))
FUTURE = P("future_covariates", "int", 0, lo=0,
           help="Known-future covariates (promotions, holidays) given for the horizon")


def _model(type_id: str, name: str, helper: str, params: tuple, torch, cost, desc: str,
           check=None) -> BlockDefinition:
    defn = nn_block(type_id, name, CATEGORY, family="model", params=(*COMMON, *params),
                    shape=_shape(name, check), layout="ir", torch=torch,
                    torch_helpers=("ForecastHead", *(("TCN",) if helper == "TCNForecaster"
                                                      else ()), helper),
                    param_fn=lambda s, p: _cost(cost)(s, p)[0], desc=desc)
    FORECAST_MODELS[type_id] = helper
    return replace(defn, meta={"task": "forecasting", "cost": _cost(cost)},
                   outputs=(PortSpec("out", role="distribution"),), library="PyTorch")


def _models() -> list[BlockDefinition]:
    return [
        _model("forecast.dlinear", "DLinear", "DLinear",
               (P("kernel", "int", 25, lo=1, help="Moving-average window for the trend"),),
               lambda c: (f"DLinear({c['input_shape'][0]}, {int(c['horizon'])}, "
                          f"{_heads_args(c)}, kernel={int(c['kernel'])})"),
               lambda s, p, out: h.dlinear_cost(s[0], out),
               "Trend + seasonal decomposition with one linear map each: a strong, tiny "
               "baseline."),
        _model("forecast.nbeats", "N-BEATS", "NBEATS",
               (P("stacks", "int", 2, lo=1), P("blocks", "int", 3, lo=1),
                P("layers", "int", 4, lo=1), P("width", "int", 256, lo=1)),
               lambda c: (f"NBEATS({c['input_shape'][0]}, {int(c['horizon'])}, "
                          f"{_heads_args(c)}, stacks={int(c['stacks'])}, "
                          f"blocks={int(c['blocks'])}, layers={int(c['layers'])}, "
                          f"width={int(c['width'])})"),
               lambda s, p, out: h.nbeats_cost(s[0], out, int(p["stacks"]), int(p["blocks"]),
                                               int(p["layers"]), int(p["width"])),
               "Doubly residual MLP stacks that subtract what they explain (N-BEATS)."),
        _model("forecast.nhits", "N-HiTS", "NHITS",
               (P("pools", "str", "8, 4, 1", help="Pooling kernel per stack (coarse → fine)"),
                P("blocks", "int", 1, lo=1), P("layers", "int", 2, lo=1),
                P("width", "int", 256, lo=1)),
               lambda c: (f"NHITS({c['input_shape'][0]}, {int(c['horizon'])}, "
                          f"{_heads_args(c)}, pools={_pools(c)!r}, blocks={int(c['blocks'])}, "
                          f"layers={int(c['layers'])}, width={int(c['width'])})"),
               lambda s, p, out: h.nhits_cost(s[0], int(p["horizon"]), head_size(p), _pools(p),
                                              int(p["blocks"]), int(p["layers"]),
                                              int(p["width"])),
               "N-BEATS with multi-rate pooling and interpolation: fast and accurate for long "
               "horizons."),
        _model("forecast.tcn", "TCN Forecaster", "TCNForecaster",
               (P("channels", "int", 64, lo=1), P("levels", "int", 4, lo=1, hi=12),
                P("kernel_size", "int", 3, lo=2), P("dropout", "float", 0.1, lo=0.0, hi=1.0),
                FUTURE),
               lambda c: (f"TCNForecaster({c['input_shape'][1]}, {int(c['horizon'])}, "
                          f"{_heads_args(c)}, channels={int(c['channels'])}, "
                          f"levels={int(c['levels'])}, kernel={int(c['kernel_size'])}, "
                          f"dropout={float(c['dropout'])}, "
                          f"future_covariates={int(c['future_covariates'])})"),
               lambda s, p, out: h.tcn_forecaster_cost(
                   s[1], s[0], out, int(p["channels"]), int(p["levels"]),
                   int(p["kernel_size"]), int(p["horizon"]), int(p["future_covariates"])),
               "Dilated causal convolutions over all channels; uses known-future covariates."),
        _model("forecast.patchtst", "PatchTST", "PatchTST",
               (P("patch_len", "int", 16, lo=1), P("stride", "int", 8, lo=1),
                P("d_model", "int", 64, lo=8), P("heads", "int", 4, lo=1),
                P("layers", "int", 2, lo=1), P("dropout", "float", 0.1, lo=0.0, hi=1.0)),
               lambda c: (f"PatchTST({c['input_shape'][0]}, {int(c['horizon'])}, "
                          f"{_heads_args(c)}, patch_len={int(c['patch_len'])}, "
                          f"stride={int(c['stride'])}, d_model={int(c['d_model'])}, "
                          f"heads={int(c['heads'])}, layers={int(c['layers'])}, "
                          f"dropout={float(c['dropout'])})"),
               lambda s, p, out: h.patchtst_cost(s[0], out, int(p["patch_len"]),
                                                 int(p["stride"]), int(p["d_model"]),
                                                 int(p["layers"])),
               "Transformer over patches of the series (PatchTST): long look-back windows.",
               check=_patch_check),
        _model("forecast.tide", "TiDE", "TiDE",
               (P("hidden", "int", 128, lo=8), P("decoder_dim", "int", 16, lo=1),
                P("dropout", "float", 0.1, lo=0.0, hi=1.0), FUTURE),
               lambda c: (f"TiDE({c['input_shape'][0]}, {c['input_shape'][1]}, "
                          f"{int(c['horizon'])}, {_heads_args(c)}, hidden={int(c['hidden'])}, "
                          f"decoder_dim={int(c['decoder_dim'])}, dropout={float(c['dropout'])}, "
                          f"future_covariates={int(c['future_covariates'])})"),
               lambda s, p, out: h.tide_cost(s[0], s[1], int(p["horizon"]), head_size(p),
                                             int(p["hidden"]), int(p["decoder_dim"]),
                                             int(p["future_covariates"])),
               "Dense encoder-decoder (TiDE) that reads past and known-future covariates."),
        _model("forecast.rnn", "RNN Forecaster (DeepAR-style)", "RNNForecaster",
               (P("hidden", "int", 64, lo=1), P("layers", "int", 2, lo=1),
                P("dropout", "float", 0.1, lo=0.0, hi=1.0), FUTURE),
               lambda c: (f"RNNForecaster({c['input_shape'][1]}, {int(c['horizon'])}, "
                          f"{_heads_args(c)}, hidden={int(c['hidden'])}, "
                          f"layers={int(c['layers'])}, dropout={float(c['dropout'])}, "
                          f"future_covariates={int(c['future_covariates'])})"),
               lambda s, p, out: h.rnn_forecaster_cost(
                   s[1], s[0], out, int(p["hidden"]), int(p["layers"]), int(p["horizon"]),
                   int(p["future_covariates"])),
               "LSTM encoder with a probabilistic multi-horizon head (DeepAR-style); pair with "
               "student_t or negbin."),
        _model("forecast.transformer", "Transformer Forecaster (Informer-style)",
               "TransformerForecaster",
               (P("d_model", "int", 64, lo=8), P("heads", "int", 4, lo=1),
                P("layers", "int", 2, lo=1), P("dropout", "float", 0.1, lo=0.0, hi=1.0)),
               lambda c: (f"TransformerForecaster({c['input_shape'][0]}, {c['input_shape'][1]}, "
                          f"{int(c['horizon'])}, {_heads_args(c)}, d_model={int(c['d_model'])}, "
                          f"heads={int(c['heads'])}, layers={int(c['layers'])}, "
                          f"dropout={float(c['dropout'])})"),
               lambda s, p, out: h.transformer_forecaster_cost(s[0], s[1], out,
                                                               int(p["d_model"]),
                                                               int(p["layers"])),
               "Attention encoder with convolutional distilling between layers (Informer)."),
    ]


def _pools(p: dict) -> tuple[int, ...]:
    try:
        return tuple(max(1, int(v)) for v in str(p.get("pools", "")).split(",")
                     if v.strip()) or (1,)
    except ValueError:
        return (1,)


# ------------------------------------------------------------------ data

def _columns_count(spec) -> int:
    return len([c for c in str(spec or "").split(",") if c.strip()])


def _window_checks(p: dict) -> list:
    out = []
    if int(p["window"]) < int(p["horizon"]):
        out.append(("warning", f"the look-back window ({p['window']}) is shorter than the "
                               f"horizon ({p['horizon']}): models rarely forecast further "
                               "ahead than they look back"))
    if int(p["val_windows"]) + int(p["test_windows"]) == 0:
        out.append(("warning", "no validation or test windows: nothing measures the forecasts"))
    return out


def _data(type_id: str, name: str, params: tuple, desc: str, season: int = 1) -> BlockDefinition:
    FORECAST_DATA[type_id] = {}
    windows = (P("window", "int", 48, lo=2, help="Look-back steps the model reads"),
               P("horizon", "int", 12, lo=1, help="Future steps to predict"),
               P("stride", "int", 1, lo=1, help="Step between training windows"),
               P("season_length", "int", season, lo=1,
                 help="Seasonal period (7 daily data with weekly cycle, 24 hourly, 12 monthly): "
                      "the MASE baseline"),
               P("val_windows", "int", 2, lo=0, help="Horizons held out for validation"),
               P("test_windows", "int", 2, lo=0, help="Horizons held out for the final test"))
    return BlockDefinition(
        type_id=type_id, display_name=name, category="Data", color=family_color("data"),
        params=(*params, *windows), description=desc, library="PyTorch",
        checks_fn=_window_checks, meta={"modality": "timeseries", "forecast": True})


def _datasets() -> list[BlockDefinition]:
    return [
        _data("data.forecast_csv", "Forecasting Table",
              (P("path", "str", "series.csv"),
               P("format", "enum", "csv", options=("csv", "tsv", "parquet")),
               P("time_column", "str", "date"), P("target_column", "str", "value"),
               P("id_column", "str", "", help="Column naming each series (empty = one series)"),
               P("past_covariates", "str", "",
                 help="Comma-separated columns only known up to now (e.g. temperature)"),
               P("future_covariates", "str", "",
                 help="Comma-separated columns known ahead (promotions, holidays, prices)")),
              "One or many time series in long format (date, id, value, covariates); "
              "rolling-origin backtest split."),
        _data("data.synthetic_series", "Synthetic Series",
              (P("n_series", "int", 30, lo=1), P("length", "int", 400, lo=20),
               P("trend", "float", 0.2, lo=-0.9, hi=10.0, help="Relative growth over the series"),
               P("noise", "float", 0.1, lo=0.0), P("counts", "bool", False,
                                                    help="Poisson counts (try the negbin head)"),
               P("promotions", "bool", True,
                 help="Add a known-future promotion flag that lifts demand"),
               P("seed", "int", 0, lo=0)),
              "Generated seasonal demand series with trend, noise and promotions: try "
              "forecasting without data files.", season=24),
    ]


# ------------------------------------------------------------------ metrics

def _metric(type_id: str, name: str, key: str, desc: str) -> BlockDefinition:
    FORECAST_METRICS[type_id] = {"key": key}
    return BlockDefinition(
        type_id=type_id, display_name=name, category="Metrics",
        color=family_color("evaluation"), description=desc, library="PyTorch",
        meta={"kind": "metrics", "key": key, "tasks": ["forecasting"], "forecast": True})


def _metrics() -> list[BlockDefinition]:
    return [
        _metric("eval.mase", "MASE", "mase",
                "Mean absolute error relative to a seasonal-naive forecast (< 1 beats it)."),
        _metric("eval.smape", "sMAPE", "smape", "Symmetric mean absolute percentage error."),
        _metric("eval.wape", "WAPE", "wape", "Total absolute error over total actual volume."),
        _metric("eval.crps", "CRPS", "crps",
                "Continuous ranked probability score: accuracy of the whole forecast "
                "distribution."),
        _metric("eval.pinball", "Pinball Loss", "pinball", "Mean quantile loss."),
    ]


def register_all() -> None:
    reg = get_registry()
    for defn in (*_models(), *_datasets(), *_metrics()):
        reg.register(defn)
