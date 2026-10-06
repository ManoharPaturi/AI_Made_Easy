"""Recurrent layers over batch-first sequences [L, C]."""
from __future__ import annotations

from ai_made_easy.core.blocks._dsl import P, nn_block
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import ShapeError

reg = get_registry()

_GATES = {"lstm": 4, "gru": 3, "rnn": 1}


def _rnn_params(kind: str) -> tuple:
    params = (
        P("hidden_size", "int", 64, lo=1),
        P("num_layers", "int", 1, lo=1, help="Stacked recurrent layers"),
        P("bias", "bool", True),
        P("bidirectional", "bool", False),
        P("dropout", "float", 0.0, lo=0.0, hi=1.0,
          help="Dropout between stacked layers (needs num_layers > 1)"),
        P("return_sequences", "bool", True,
          help="True: every time step [L, H]; False: final state [H]"),
    )
    if kind == "rnn":
        params = params[:1] + (P("nonlinearity", "enum", "tanh",
                                 options=("tanh", "relu")),) + params[1:]
    return params


def _rnn_shape(name: str):
    def fn(in_shapes, params):
        s = in_shapes[0]
        if len(s) != 2:
            raise ShapeError(f"{name} expects a batch-first sequence [L, C], got {s}")
        width = int(params["hidden_size"]) * (2 if params["bidirectional"] else 1)
        return [s[0], width] if params["return_sequences"] else [width]

    return fn


def _rnn_checks(p):
    if float(p.get("dropout", 0.0)) > 0 and int(p.get("num_layers", 1)) == 1:
        return [("warning", "dropout only applies between stacked layers; it has no "
                            "effect with num_layers = 1")]
    return []


def _rnn_count(kind: str):
    gates = _GATES[kind]

    def fn(in_shapes, params):
        h, layers = int(params["hidden_size"]), int(params["num_layers"])
        dirs = 2 if params["bidirectional"] else 1
        bias = 2 * gates * h if params["bias"] else 0
        total, cin = 0, in_shapes[0][-1]
        for layer in range(layers):
            inp = cin if layer == 0 else h * dirs
            total += dirs * (gates * h * inp + gates * h * h + bias)
        return total

    return fn


def _torch_rnn(kind: str):
    cls = {"lstm": "nn.LSTM", "gru": "nn.GRU", "rnn": "nn.RNN"}[kind]

    def fn(c):
        args = [f"input_size={c['input_size']}", f"hidden_size={c['hidden_size']}",
                f"num_layers={c['num_layers']}"]
        if kind == "rnn":
            args.append(f"nonlinearity=\"{c['nonlinearity']}\"")
        if not c["bias"]:
            args.append("bias=False")
        args.append("batch_first=True")
        if float(c["dropout"]) > 0 and int(c["num_layers"]) > 1:
            args.append(f"dropout={c['dropout']}")
        if c["bidirectional"]:
            args.append("bidirectional=True")
        return f"{cls}({', '.join(args)})"

    return fn


def _torch_rnn_expr(kind: str):
    def fn(c):
        call = f"self.{c['self_var']}({c['i0']})"
        if c["return_sequences"]:
            return f"{call}[0]"
        h_n = f"{call}[1][0]" if kind == "lstm" else f"{call}[1]"
        if c["bidirectional"]:
            return f"torch.cat([{h_n}[-2], {h_n}[-1]], dim=-1)"
        return f"{h_n}[-1]"

    return fn


def _keras_rnn(kind: str):
    cls = {"lstm": "LSTM", "gru": "GRU", "rnn": "SimpleRNN"}[kind]

    def fn(c):
        layers_n = int(c["num_layers"])
        expr = c["i0"]
        for i in range(layers_n):
            last = i == layers_n - 1
            seq = bool(c["return_sequences"]) or not last
            args = [f"units={c['hidden_size']}", f"return_sequences={seq}"]
            if kind == "rnn":
                args.append(f"activation=\"{c['nonlinearity']}\"")
            if kind == "gru":
                args.append("reset_after=True")
            if not c["bias"]:
                args.append("use_bias=False")
            layer = f"layers.{cls}({', '.join(args)})"
            if c["bidirectional"]:
                layer = f"layers.Bidirectional({layer})"
            expr = f"{layer}({expr})"
            if not last and float(c["dropout"]) > 0:
                expr = f"layers.Dropout({c['dropout']})({expr})"
        return expr

    return fn


for _kind, _name, _desc in (
        ("lstm", "LSTM", "Long short-term memory recurrent layer."),
        ("gru", "GRU", "Gated recurrent unit layer."),
        ("rnn", "SimpleRNN", "Elman recurrent layer (tanh or ReLU).")):
    reg.register(nn_block(
        f"core.{_kind}", _name, "Recurrent", family="recurrent",
        params=_rnn_params(_kind), shape=_rnn_shape(_name), layout="ir",
        param_fn=_rnn_count(_kind), checks=_rnn_checks,
        torch=_torch_rnn(_kind), torch_expr=_torch_rnn_expr(_kind),
        keras_expr=_keras_rnn(_kind), desc=_desc,
    ))


def _select_shape(in_shapes, params):
    s = in_shapes[0]
    if len(s) != 2:
        raise ShapeError(f"Select Timestep expects a sequence [L, C], got {s}")
    idx = int(params["index"])
    if not -s[0] <= idx < s[0]:
        raise ShapeError(f"time index {idx} is out of range for length {s[0]}")
    return [s[1]]


reg.register(nn_block(
    "core.select_timestep", "Select Timestep", "Recurrent", family="recurrent",
    params=(P("index", "int", -1, lo=-100000, help="-1 = last time step"),),
    shape=_select_shape, layout="ir",
    torch_expr="{i0}[:, {index}]", keras_expr="{i0}[:, {index}]",
    desc="Pick one time step of a [L, C] sequence → [C].",
))
