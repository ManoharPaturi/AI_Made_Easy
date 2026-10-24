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

TFT = '''\
class GRN(nn.Module):
    """Gated residual network (TFT): ELU MLP, GLU gate, skip connection and LayerNorm."""

    def __init__(self, din: int, hidden: int, dout: int, dropout: float) -> None:
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(din, hidden), nn.Linear(hidden, hidden)
        self.gate, self.drop = nn.Linear(hidden, 2 * dout), nn.Dropout(dropout)
        self.skip = nn.Linear(din, dout) if din != dout else nn.Identity()
        self.norm = nn.LayerNorm(dout)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.fc2(nn.functional.elu(self.fc1(a))))
        return self.norm(self.skip(a) + nn.functional.glu(self.gate(h), -1))


class VariableSelection(nn.Module):
    """Embeds every input variable, weighs them with a softmax per time step and mixes them:
    ``weights`` keeps the last importances [B, T, variables] for inspection."""

    def __init__(self, variables: int, d: int, dropout: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.randn(variables, d) * 0.1)
        self.shift = nn.Parameter(torch.zeros(variables, d))
        self.select = GRN(variables * d, d, variables, dropout)
        self.grns = nn.ModuleList(GRN(d, d, d, dropout) for _ in range(variables))
        self.weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = x.unsqueeze(-1) * self.scale + self.shift                      # [B, T, V, d]
        w = torch.softmax(self.select(e.flatten(-2)), -1)
        self.weights = w.detach()
        v = torch.stack([g(e[..., i, :]) for i, g in enumerate(self.grns)], -2)
        return (w.unsqueeze(-1) * v).sum(-2)


class TFTLite(Forecaster):
    """Temporal Fusion Transformer, lite (Lim et al., 2021) without static covariates:
    variable selection on the past channels and the known-future covariates, an LSTM
    encoder-decoder, gated skips, causal multi-head attention over past and future steps
    and a position-wise GRN before the distribution head."""

    def __init__(self, window: int, channels_in: int, horizon: int, head: str = "quantile",
                 quantiles: tuple = (0.1, 0.5, 0.9), d_model: int = 32, heads: int = 4,
                 dropout: float = 0.1, future_covariates: int = 0) -> None:
        super().__init__()
        d = d_model
        self.out = ForecastHead(horizon, head, quantiles)
        self.horizon, self.future_covariates = horizon, future_covariates
        self.past = VariableSelection(channels_in, d, dropout)
        if future_covariates:
            self.future = VariableSelection(future_covariates, d, dropout)
        else:   # no known future: one learned input per horizon step
            self.queries = nn.Parameter(torch.zeros(horizon, d))
        self.encoder = nn.LSTM(d, d, batch_first=True)
        self.decoder = nn.LSTM(d, d, batch_first=True)
        self.lstm_gate, self.lstm_norm = nn.Linear(d, 2 * d), nn.LayerNorm(d)
        self.enrich = GRN(d, d, d, dropout)
        self.attention = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.attn_gate, self.attn_norm = nn.Linear(d, 2 * d), nn.LayerNorm(d)
        self.positionwise = GRN(d, d, d, dropout)
        self.final_gate, self.final_norm = nn.Linear(d, 2 * d), nn.LayerNorm(d)
        self.project = nn.Linear(d, self.out.k)
        steps = window + horizon
        self.register_buffer("mask", torch.ones(steps, steps).triu(1).bool()[-horizon:],
                             persistent=False)

    def forecast(self, x: torch.Tensor, future=None) -> torch.Tensor:
        batch, glu = x.shape[0], nn.functional.glu
        past = self.past(x)
        if self.future_covariates:
            fut = (future.reshape(batch, self.horizon, -1) if future is not None
                   else x.new_zeros(batch, self.horizon, self.future_covariates))
            ahead = self.future(fut)
        else:
            ahead = self.queries.expand(batch, -1, -1)
        enc, state = self.encoder(past)
        dec, _ = self.decoder(ahead, state)
        z = self.lstm_norm(torch.cat([past, ahead], 1) + glu(self.lstm_gate(
            torch.cat([enc, dec], 1)), -1))
        e = self.enrich(z)
        q = e[:, -self.horizon:]
        a = self.attention(q, e, e, attn_mask=self.mask, need_weights=False)[0]
        y = self.positionwise(self.attn_norm(q + glu(self.attn_gate(a), -1)))
        y = self.final_norm(z[:, -self.horizon:] + glu(self.final_gate(y), -1))
        return self.out(self.project(y).reshape(batch, -1))
'''

NLINEAR = '''\
class NLinear(Forecaster):
    """NLinear (Zeng et al., 2023): subtract the last value, one linear map over the
    window, add the last value back; robust when the level of a series shifts."""

    def __init__(self, window: int, horizon: int, head: str = "point",
                 quantiles: tuple = (0.1, 0.5, 0.9)) -> None:
        super().__init__()
        self.out = ForecastHead(horizon, head, quantiles)
        self.linear = nn.Linear(window, self.out.size)

    def forecast(self, x: torch.Tensor, future=None) -> torch.Tensor:
        y = x[..., 0]
        last = y[:, -1:]
        raw = self.linear(y - last).reshape(y.shape[0], self.out.horizon, self.out.k)
        level = torch.zeros_like(raw)
        level[..., 0] = last        # back to the series' level: point, lowest quantile, loc
        return self.out((raw + level).reshape(y.shape[0], -1))
'''

ITRANSFORMER = '''\
class ITransformer(Forecaster):
    """iTransformer (Liu et al., 2024): each variable's whole window is one token, so
    attention runs across variables (how the target relates to its covariates) and the
    feed-forward layers read time."""

    def __init__(self, window: int, horizon: int, head: str = "point",
                 quantiles: tuple = (0.1, 0.5, 0.9), d_model: int = 64, heads: int = 4,
                 layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.out = ForecastHead(horizon, head, quantiles)
        self.embed = nn.Linear(window, d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, 2 * d_model, dropout,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.project = nn.Linear(d_model, self.out.size)

    def forecast(self, x: torch.Tensor, future=None) -> torch.Tensor:
        tokens = self.encoder(self.embed(x.transpose(1, 2)))     # [B, C, d]
        return self.out(self.project(tokens[:, 0]))              # the target's token
'''

TSMIXER = '''\
class TSMixer(Forecaster):
    """TSMixer (Chen et al., 2023): all-MLP; each block mixes along time (shared across
    variables) and then across variables, with residuals."""

    def __init__(self, window: int, channels_in: int, horizon: int, head: str = "point",
                 quantiles: tuple = (0.1, 0.5, 0.9), blocks: int = 2, hidden: int = 64,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.out = ForecastHead(horizon, head, quantiles)
        self.time_norm = nn.ModuleList(nn.LayerNorm(channels_in) for _ in range(blocks))
        self.time = nn.ModuleList(nn.Linear(window, window) for _ in range(blocks))
        self.feat_norm = nn.ModuleList(nn.LayerNorm(channels_in) for _ in range(blocks))
        self.feat = nn.ModuleList(nn.Sequential(
            nn.Linear(channels_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, channels_in)) for _ in range(blocks))
        self.drop = nn.Dropout(dropout)
        self.project = nn.Linear(window, self.out.size)

    def forecast(self, x: torch.Tensor, future=None) -> torch.Tensor:
        for tn, t, fn, f in zip(self.time_norm, self.time, self.feat_norm, self.feat):
            mixed = torch.relu(t(tn(x).transpose(1, 2))).transpose(1, 2)
            x = x + self.drop(mixed)
            x = x + self.drop(f(fn(x)))
        return self.out(self.project(x[..., 0]))
'''

TIMESNET = '''\
class TimesNet(Forecaster):
    """TimesNet (Wu et al., 2023), compact: the strongest periods are found with an FFT,
    the series is folded into a 2-D grid per period (cycles x phase), and 2-D
    convolutions read changes within and across cycles; periods are weighted by their
    strength. Training picks each batch's periods and keeps a running spectrum (like
    BatchNorm's running statistics); prediction uses the periods that spectrum fixes, so
    a forecast never depends on the other series in its batch, and exports trace it."""

    def __init__(self, window: int, channels_in: int, horizon: int, head: str = "point",
                 quantiles: tuple = (0.1, 0.5, 0.9), d_model: int = 32, hidden: int = 32,
                 layers: int = 2, top_k: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.window, self.top_k = window, max(1, min(top_k, window // 2))
        self.out = ForecastHead(horizon, head, quantiles)
        self.embed = nn.Linear(channels_in, d_model)
        self.position = nn.Parameter(torch.zeros(window, d_model))
        self.convs = nn.ModuleList(nn.Sequential(
            nn.Conv2d(d_model, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, d_model, 3, padding=1)) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(layers))
        self.drop = nn.Dropout(dropout)
        self.project = nn.Linear(window * d_model, self.out.size)
        self.register_buffer("spectrum", torch.zeros(layers, window // 2 + 1))
        self._fixed = None
        self._copies: list = []

    def train(self, mode: bool = True):
        out = super().train(mode)
        self._fixed = None if mode else self._freeze()
        return out

    def _load_from_state_dict(self, *args, **kwargs) -> None:
        super()._load_from_state_dict(*args, **kwargs)
        self._fixed = None

    def _freeze(self) -> list:
        """Periods from the running spectrum (plain ints, worked out eagerly so a traced
        or exported model holds them as constants), with cosine / sine bases that measure
        their strength without an FFT."""
        fixed = []
        t = torch.arange(self.window, dtype=torch.float32, device=self.spectrum.device)
        for row in self.spectrum.detach().cpu():
            freqs = [int(f) + 1 for f in row[1:].topk(self.top_k).indices]
            angle = 2 * math.pi * t[:, None] * torch.tensor(freqs, device=t.device) / self.window
            fixed.append(([max(1, self.window // f) for f in freqs],
                          torch.cos(angle), torch.sin(angle)))
        # one copy of each layer's convolutions per period: a module used at two call
        # sites gives an int8 XNNPACK program that ExecuTorch can't load
        self._copies = [[copy.deepcopy(conv) for _ in range(self.top_k)]
                        for conv in self.convs]
        return fixed

    def _frozen(self, layer: int):
        if self._fixed is None or self._copies[0][0][0].weight.device != self.spectrum.device:
            self._fixed = self._freeze()     # weights loaded, or moved, while in eval mode
        return self._fixed[layer]

    def _periods(self, z: torch.Tensor, layer: int):
        if not self.training:
            periods, cos, sin = self._frozen(layer)
            re, im = torch.einsum("bnd,nk->bkd", z, cos.to(z.dtype)), torch.einsum(
                "bnd,nk->bkd", z, sin.to(z.dtype))
            return periods, (re * re + im * im + 1e-12).sqrt().mean(-1)
        amp = torch.fft.rfft(z, dim=1).abs().mean(-1)            # [B, L//2 + 1]
        amp[:, 0] = 0
        with torch.no_grad():
            self.spectrum[layer].mul_(0.9).add_(0.1 * amp.mean(0))
        top = amp.mean(0).topk(self.top_k).indices
        return [max(1, self.window // max(int(f), 1)) for f in top.tolist()], amp[:, top]

    def _fold(self, z: torch.Tensor, conv: nn.Module, period: int) -> torch.Tensor:
        b, n, d = z.shape
        length = -(-n // period) * period
        padded = nn.functional.pad(z, (0, 0, 0, length - n)) if length > n else z
        grid = padded.reshape(b, length // period, period, d).permute(0, 3, 1, 2)
        out = conv(grid).permute(0, 2, 3, 1).reshape(b, length, d)
        return out[:, :n] if length > n else out

    def forecast(self, x: torch.Tensor, future=None) -> torch.Tensor:
        z = self.embed(x) + self.position
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            periods, weight = self._periods(z, i)
            convs = [conv] * len(periods) if self.training else self._copies[i]
            outs = torch.stack([self._fold(z, c, p) for c, p in zip(convs, periods)], -1)
            mixed = (outs * weight.softmax(-1)[:, None, None, :]).sum(-1)
            z = norm(z + self.drop(mixed))
        return self.out(self.project(z.flatten(1)))
'''

_MATH = "import math\n\n\n"
_COPY_MATH = "import copy\nimport math\n\n\n"
FORECAST_HELPERS: dict[str, str] = {
    "ForecastHead": HEAD, "DLinear": DLINEAR, "NBEATS": NBEATS, "NHITS": _MATH + NHITS,
    "TCNForecaster": TCN_FORECASTER, "PatchTST": PATCHTST, "TiDE": TIDE,
    "RNNForecaster": RNN_FORECASTER, "TransformerForecaster": TRANSFORMER_FORECASTER,
    "TFTLite": TFT, "NLinear": NLINEAR, "ITransformer": ITRANSFORMER, "TSMixer": TSMIXER,
    "TimesNet": _COPY_MATH + TIMESNET,
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


def _grn(din: int, hidden: int, dout: int) -> int:
    return (_lin(din, hidden) + _lin(hidden, hidden) + _lin(hidden, 2 * dout)
            + (_lin(din, dout) if din != dout else 0) + 2 * dout)


def _vsn(variables: int, d: int) -> int:
    return 2 * variables * d + _grn(variables * d, d, variables) + variables * _grn(d, d, d)


def tft_cost(window: int, cin: int, horizon: int, k: int, d: int,
             future: int) -> tuple[int, int]:
    """(parameters, MACs) of ``TFTLite``."""
    lstm = 8 * d * d + 8 * d
    gate = _lin(d, 2 * d) + 2 * d                  # GLU gate + LayerNorm
    attention = 4 * d * d + 4 * d
    grn = _grn(d, d, d)
    params = (_vsn(cin, d) + (_vsn(future, d) if future else horizon * d) + 2 * lstm
              + 3 * gate + 2 * grn + attention + _lin(d, k))
    steps = window + horizon
    macs = (window * _vsn(cin, d) + horizon * (_vsn(future, d) if future else 0)
            + steps * (lstm + gate + grn) + steps * 2 * d * d + horizon * 2 * d * d
            + 2 * horizon * steps * d + horizon * (2 * gate + grn + d * k))
    return params, macs


def nlinear_cost(window: int, out: int) -> tuple[int, int]:
    params = _lin(window, out)
    return params, params


def itransformer_cost(window: int, cin: int, out: int, d: int, layers: int) -> tuple[int, int]:
    params = _lin(window, d) + layers * encoder_layer_params(d) + _lin(d, out)
    macs = (cin * (window * d + layers * 8 * d * d) + layers * 2 * cin * cin * d + d * out)
    return params, macs


def tsmixer_cost(window: int, cin: int, out: int, blocks: int, hidden: int) -> tuple[int, int]:
    per = 4 * cin + _lin(window, window) + _lin(cin, hidden) + _lin(hidden, cin)
    params = blocks * per + _lin(window, out)
    macs = blocks * (cin * window * window + window * 2 * cin * hidden) + window * out
    return params, macs


def timesnet_cost(window: int, cin: int, out: int, d: int, hidden: int, layers: int,
                  top_k: int) -> tuple[int, int]:
    conv = (9 * d * hidden + hidden) + (9 * hidden * d + d)
    params = _lin(cin, d) + window * d + layers * (conv + 2 * d) + _lin(window * d, out)
    k = max(1, min(top_k, window // 2))
    macs = window * cin * d + layers * k * window * 18 * d * hidden + window * d * out
    return params, macs
