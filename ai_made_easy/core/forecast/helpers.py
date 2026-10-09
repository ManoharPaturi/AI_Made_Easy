"""PyTorch helpers for forecasting models, emitted into generated code.

Every forecaster maps a history window [B, L, C] (channel 0 = the target, the
rest covariates over the history) to H future steps through a shared
distribution head: point [B, H], quantiles [B, H, Q], Student-t parameters
[B, H, 3] (loc, scale, df) or negative-binomial parameters [B, H, 2] (mean,
dispersion). ``forecast(x, future)`` also takes known future covariates
[B, H, F] for models that use them; ``forward(x)`` is the designed graph view.
``*_cost`` twins count parameters / multiply-accumulates without torch.
"""
from __future__ import annotations

import math

HEAD = '''\
class ForecastHead(nn.Module):
    """Raw outputs [B, H * k] -> point / quantile / Student-t / negative-binomial forecasts."""

    def __init__(self, horizon: int, head: str = "point", quantiles: tuple = (0.1, 0.5, 0.9)) -> None:
        super().__init__()
        self.horizon, self.head = horizon, head
        self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32),
                             persistent=False)
        self.k = {"point": 1, "quantile": len(quantiles), "student_t": 3, "negbin": 2}[head]

    @property
    def size(self) -> int:
        return self.horizon * self.k

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        out = raw.reshape(raw.shape[0], self.horizon, self.k)
        softplus = nn.functional.softplus
        if self.head == "point":
            return out[..., 0]
        if self.head == "quantile":  # non-crossing: lowest quantile + positive increments
            return torch.cat([out[..., :1], softplus(out[..., 1:])], dim=-1).cumsum(-1)
        if self.head == "student_t":
            return torch.stack([out[..., 0], softplus(out[..., 1]) + 1e-3,
                                2.0 + softplus(out[..., 2])], dim=-1)
        return torch.stack([softplus(out[..., 0]) + 1e-3, softplus(out[..., 1]) + 1e-3], dim=-1)


class Forecaster(nn.Module):
    """Base: forward(x) = forecast(x) without future covariates."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forecast(x, None)

    def _future(self, x: torch.Tensor, future, count: int) -> torch.Tensor:
        if count == 0:
            return x.new_zeros(x.shape[0], 0)
        if future is None:
            future = x.new_zeros(x.shape[0], self.out.horizon, count)
        return future.reshape(x.shape[0], -1)
'''

DLINEAR = '''\
class DLinear(Forecaster):
    """Trend / seasonal decomposition with one linear map each (Zeng et al., 2023)."""

    def __init__(self, window: int, horizon: int, head: str = "point",
                 quantiles: tuple = (0.1, 0.5, 0.9), kernel: int = 25) -> None:
        super().__init__()
        self.kernel = max(1, min(kernel, window) // 2 * 2 + 1)
        self.out = ForecastHead(horizon, head, quantiles)
        self.linear = nn.Linear(2 * window, self.out.size)

    def forecast(self, x: torch.Tensor, future=None) -> torch.Tensor:
        y = x[..., 0]
        pad = self.kernel // 2
        padded = torch.cat([y[:, :1].repeat(1, pad), y, y[:, -1:].repeat(1, pad)], dim=1)
        trend = nn.functional.avg_pool1d(padded.unsqueeze(1), self.kernel, 1).squeeze(1)
        return self.out(self.linear(torch.cat([y - trend, trend], dim=1)))
'''

NBEATS = '''\
class NBEATS(Forecaster):
    """N-BEATS generic architecture (Oreshkin et al., 2020): doubly residual MLP blocks."""

    def __init__(self, window: int, horizon: int, head: str = "point",
                 quantiles: tuple = (0.1, 0.5, 0.9), stacks: int = 2, blocks: int = 3,
                 layers: int = 4, width: int = 256) -> None:
        super().__init__()
        self.out = ForecastHead(horizon, head, quantiles)
        self.blocks = nn.ModuleList()
        for _ in range(stacks * blocks):
            mlp = [nn.Linear(window, width), nn.ReLU()]
            for _ in range(layers - 1):
                mlp += [nn.Linear(width, width), nn.ReLU()]
            self.blocks.append(nn.ModuleDict({"mlp": nn.Sequential(*mlp),
                                              "back": nn.Linear(width, window),
                                              "fore": nn.Linear(width, self.out.size)}))

    def forecast(self, x: torch.Tensor, future=None) -> torch.Tensor:
        residual, total = x[..., 0], 0
        for block in self.blocks:
            h = block["mlp"](residual)
            residual = residual - block["back"](h)
            total = total + block["fore"](h)
        return self.out(total)
'''

NHITS = '''\
class NHITS(Forecaster):
    """N-HiTS (Challu et al., 2023): multi-rate pooling and hierarchical interpolation."""

    def __init__(self, window: int, horizon: int, head: str = "point",
                 quantiles: tuple = (0.1, 0.5, 0.9), pools: tuple = (8, 4, 1), blocks: int = 1,
                 layers: int = 2, width: int = 256) -> None:
        super().__init__()
        self.out = ForecastHead(horizon, head, quantiles)
        self.horizon = horizon
        self.blocks = nn.ModuleList()
        for pool in pools:
            pool = max(1, min(pool, window))
            inputs = math.ceil(window / pool)
            coeffs = math.ceil(horizon / pool)
            for _ in range(blocks):
                mlp = [nn.Linear(inputs, width), nn.ReLU()]
                for _ in range(layers - 1):
                    mlp += [nn.Linear(width, width), nn.ReLU()]
                self.blocks.append(nn.ModuleDict({
                    "pool": nn.MaxPool1d(pool, pool, ceil_mode=True),
                    "mlp": nn.Sequential(*mlp), "back": nn.Linear(width, window),
                    "fore": nn.Linear(width, coeffs * self.out.k)}))

    def forecast(self, x: torch.Tensor, future=None) -> torch.Tensor:
        residual, total, k = x[..., 0], 0, self.out.k
        for block in self.blocks:
            h = block["mlp"](block["pool"](residual.unsqueeze(1)).squeeze(1))
            residual = residual - block["back"](h)
            coeffs = block["fore"](h).reshape(x.shape[0], -1, k).transpose(1, 2)
            total = total + nn.functional.interpolate(coeffs, size=self.horizon, mode="linear",
                                                      align_corners=True)
        return self.out(total.transpose(1, 2).reshape(x.shape[0], -1))
'''

TCN_FORECASTER = '''\
class TCNForecaster(Forecaster):
    """Dilated causal convolutions over the history; the last step predicts all H steps."""

    def __init__(self, channels_in: int, horizon: int, head: str = "point",
                 quantiles: tuple = (0.1, 0.5, 0.9), channels: int = 64, levels: int = 4,
                 kernel: int = 3, dropout: float = 0.1, future_covariates: int = 0) -> None:
        super().__init__()
        self.out = ForecastHead(horizon, head, quantiles)
        self.future_covariates = future_covariates
        self.tcn = TCN(channels_in, channels, levels, kernel, dropout)
        self.mlp = nn.Sequential(nn.Linear(channels + horizon * future_covariates, channels),
                                 nn.ReLU(), nn.Linear(channels, self.out.size))

    def forecast(self, x: torch.Tensor, future=None) -> torch.Tensor:
        h = self.tcn(x)[:, -1]
        return self.out(self.mlp(torch.cat([h, self._future(x, future,
                                                            self.future_covariates)], 1)))
'''

PATCHTST = '''\
class PatchTST(Forecaster):
    """Patch transformer (Nie et al., 2023): the target series is split into patches that
    a transformer encoder reads as tokens."""

    def __init__(self, window: int, horizon: int, head: str = "point",
                 quantiles: tuple = (0.1, 0.5, 0.9), patch_len: int = 16, stride: int = 8,
                 d_model: int = 64, heads: int = 4, layers: int = 2,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        patches = (window - patch_len) // stride + 1
        self.out = ForecastHead(horizon, head, quantiles)
        self.embed = nn.Linear(patch_len, d_model)
        self.position = nn.Parameter(torch.zeros(patches, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, 2 * d_model, dropout,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.project = nn.Linear(patches * d_model, self.out.size)

    def forecast(self, x: torch.Tensor, future=None) -> torch.Tensor:
        patches = x[..., 0].unfold(1, self.patch_len, self.stride)    # [B, P, patch]
        z = self.encoder(self.embed(patches) + self.position)
        return self.out(self.project(z.flatten(1)))
'''

TIDE = '''\
class TiDEResidual(nn.Module):
    def __init__(self, din: int, hidden: int, dout: int, dropout: float) -> None:
        super().__init__()
        self.dense = nn.Sequential(nn.Linear(din, hidden), nn.ReLU(), nn.Linear(hidden, dout),
                                   nn.Dropout(dropout))
        self.skip = nn.Linear(din, dout)
        self.norm = nn.LayerNorm(dout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.dense(x) + self.skip(x))


class TiDE(Forecaster):
    """TiDE (Das et al., 2023): dense encoder-decoder using past and known future covariates."""

    def __init__(self, window: int, channels_in: int, horizon: int, head: str = "point",
                 quantiles: tuple = (0.1, 0.5, 0.9), hidden: int = 128, decoder_dim: int = 16,
                 dropout: float = 0.1, future_covariates: int = 0) -> None:
        super().__init__()
        self.out = ForecastHead(horizon, head, quantiles)
        self.horizon, self.future_covariates, self.decoder_dim = (horizon, future_covariates,
                                                                  decoder_dim)
        self.encoder = TiDEResidual(window * channels_in + horizon * future_covariates, hidden,
                                    hidden, dropout)
        self.decoder = TiDEResidual(hidden, hidden, horizon * decoder_dim, dropout)
        self.temporal = TiDEResidual(decoder_dim + future_covariates, max(hidden // 4, 8),
                                     self.out.k, dropout)
        self.lookback = nn.Linear(window, self.out.size)

    def forecast(self, x: torch.Tensor, future=None) -> torch.Tensor:
        batch = x.shape[0]
        fut = self._future(x, future, self.future_covariates)
        h = self.decoder(self.encoder(torch.cat([x.reshape(batch, -1), fut], 1)))
        steps = h.reshape(batch, self.horizon, self.decoder_dim)
        if self.future_covariates:
            steps = torch.cat([steps, fut.reshape(batch, self.horizon, -1)], dim=-1)
        raw = self.temporal(steps).reshape(batch, -1) + self.lookback(x[..., 0])
        return self.out(raw)
'''

RNN_FORECASTER = '''\
class RNNForecaster(Forecaster):
    """DeepAR-style forecaster: an LSTM reads the history, an MLP emits every horizon's
    distribution at once (direct multi-horizon, no sampling loop)."""

    def __init__(self, channels_in: int, horizon: int, head: str = "student_t",
                 quantiles: tuple = (0.1, 0.5, 0.9), hidden: int = 64, layers: int = 2,
                 dropout: float = 0.1, future_covariates: int = 0) -> None:
        super().__init__()
        self.out = ForecastHead(horizon, head, quantiles)
        self.future_covariates = future_covariates
        self.rnn = nn.LSTM(channels_in, hidden, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.mlp = nn.Sequential(nn.Linear(hidden + horizon * future_covariates, hidden),
                                 nn.ReLU(), nn.Linear(hidden, self.out.size))

    def forecast(self, x: torch.Tensor, future=None) -> torch.Tensor:
        h = self.rnn(x)[0][:, -1]
        return self.out(self.mlp(torch.cat([h, self._future(x, future,
                                                            self.future_covariates)], 1)))
'''

TRANSFORMER_FORECASTER = '''\
class TransformerForecaster(Forecaster):
    """Informer-style encoder: attention layers with convolutional distilling (halving the
    sequence) between them, pooled into a multi-horizon head."""

    def __init__(self, window: int, channels_in: int, horizon: int, head: str = "point",
                 quantiles: tuple = (0.1, 0.5, 0.9), d_model: int = 64, heads: int = 4,
                 layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.out = ForecastHead(horizon, head, quantiles)
        self.embed = nn.Linear(channels_in, d_model)
        self.position = nn.Parameter(torch.zeros(window, d_model))
        self.layers = nn.ModuleList(nn.TransformerEncoderLayer(
            d_model, heads, 2 * d_model, dropout, batch_first=True) for _ in range(layers))
        self.distil = nn.ModuleList(nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1), nn.BatchNorm1d(d_model), nn.ELU(),
            nn.MaxPool1d(3, 2, padding=1)) for _ in range(layers - 1))
        self.project = nn.Linear(d_model, self.out.size)

    def forecast(self, x: torch.Tensor, future=None) -> torch.Tensor:
        z = self.embed(x) + self.position
        for i, layer in enumerate(self.layers):
            z = layer(z)
            if i < len(self.distil):
                z = self.distil[i](z.transpose(1, 2)).transpose(1, 2)
        return self.out(self.project(z.mean(1)))
'''

_MATH = "import math\n\n\n"
FORECAST_HELPERS: dict[str, str] = {
    "ForecastHead": HEAD, "DLinear": DLINEAR, "NBEATS": NBEATS, "NHITS": _MATH + NHITS,
    "TCNForecaster": TCN_FORECASTER, "PatchTST": PATCHTST, "TiDE": TIDE,
    "RNNForecaster": RNN_FORECASTER, "TransformerForecaster": TRANSFORMER_FORECASTER,
}


def register() -> None:
    from ai_made_easy.core.codegen.helpers import PYTORCH_HELPERS

    PYTORCH_HELPERS.update(FORECAST_HELPERS)


# ------------------------------------------------------------- analytic costs

HEAD_SIZES = {"point": 1, "student_t": 3, "negbin": 2}


def head_k(head: str, quantiles: int) -> int:
    return quantiles if head == "quantile" else HEAD_SIZES[head]


def _lin(i: int, o: int) -> int:
    return i * o + o


def _mlp(din: int, width: int, layers: int) -> int:
    return _lin(din, width) + (layers - 1) * _lin(width, width)


def encoder_layer_params(d: int) -> int:
    """nn.TransformerEncoderLayer(d, heads, 2d): attention, feed-forward, two norms."""
    return (4 * d * d + 4 * d) + (_lin(d, 2 * d) + _lin(2 * d, d)) + 4 * d


def dlinear_cost(window: int, out: int) -> tuple[int, int]:
    params = _lin(2 * window, out)
    return params, params + 2 * window  # + moving average


def nbeats_cost(window: int, out: int, stacks: int, blocks: int, layers: int,
                width: int) -> tuple[int, int]:
    per = _mlp(window, width, layers) + _lin(width, window) + _lin(width, out)
    return stacks * blocks * per, stacks * blocks * per


def nhits_cost(window: int, horizon: int, k: int, pools: tuple, blocks: int, layers: int,
               width: int) -> tuple[int, int]:
    params = 0
    for pool in pools:
        pool = max(1, min(pool, window))
        inputs, coeffs = math.ceil(window / pool), math.ceil(horizon / pool)
        params += blocks * (_mlp(inputs, width, layers) + _lin(width, window)
                            + _lin(width, coeffs * k))
    return params, params


def tcn_forecaster_cost(cin: int, window: int, out: int, channels: int, levels: int,
                        kernel: int, horizon: int, future: int) -> tuple[int, int]:
    from ai_made_easy.core.sequence.helpers import tcn_cost

    tp, tm = tcn_cost(cin, channels, levels, kernel, window)
    head = _lin(channels + horizon * future, channels) + _lin(channels, out)
    return tp + head, tm + head


def patchtst_cost(window: int, out: int, patch: int, stride: int, d: int,
                  layers: int) -> tuple[int, int]:
    patches = (window - patch) // stride + 1
    params = (_lin(patch, d) + patches * d + layers * encoder_layer_params(d)
              + _lin(patches * d, out))
    macs = (patches * (patch * d + layers * (8 * d * d)) + layers * 2 * patches * patches * d
            + patches * d * out)
    return params, macs


def tide_cost(window: int, cin: int, horizon: int, k: int, hidden: int, decoder: int,
              future: int) -> tuple[int, int]:
    def rb(i: int, h: int, o: int) -> int:
        return _lin(i, h) + _lin(h, o) + _lin(i, o) + 2 * o

    enc = rb(window * cin + horizon * future, hidden, hidden)
    dec = rb(hidden, hidden, horizon * decoder)
    temporal = rb(decoder + future, max(hidden // 4, 8), k)
    params = enc + dec + temporal + _lin(window, horizon * k)
    return params, enc + dec + horizon * temporal + window * horizon * k


def rnn_forecaster_cost(cin: int, window: int, out: int, hidden: int, layers: int,
                        horizon: int, future: int) -> tuple[int, int]:
    lstm = sum(4 * hidden * ((cin if i == 0 else hidden) + hidden) + 8 * hidden
               for i in range(layers))
    head = _lin(hidden + horizon * future, hidden) + _lin(hidden, out)
    return lstm + head, lstm * window + head


def transformer_forecaster_cost(window: int, cin: int, out: int, d: int,
                                layers: int) -> tuple[int, int]:
    distil = (layers - 1) * (3 * d * d + d + 2 * d)
    params = (_lin(cin, d) + window * d + layers * encoder_layer_params(d) + distil
              + _lin(d, out))
    macs, length = window * cin * d, window
    for i in range(layers):
        macs += length * 8 * d * d + 2 * length * length * d
        if i < layers - 1:
            macs += length * 3 * d * d
            length = (length - 1) // 2 + 1
    return params, macs + d * out
