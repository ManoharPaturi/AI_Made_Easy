"""PyTorch helpers for normalizing flows, emitted into generated model code.

Every flow layer maps data towards noise (``forward(x) -> z``), stores the log-determinant
of its Jacobian per sample in ``self.logdet`` and can run backwards (``inverse(z) -> x``).
The training script adds the layers' log-determinants to the base log-density; sampling
runs the layers in reverse order. Conditioner networks start at zero, so every layer starts
as the identity.
"""
from __future__ import annotations

FLOW_BASE = '''\
class _FlowLayer(nn.Module):
    """Invertible layer: forward(x) -> z sets self.logdet [N]; inverse(z) -> x."""

    is_flow = True

    def __init__(self) -> None:
        super().__init__()
        self.logdet = None


def _conditioner(n_in: int, n_out: int, hidden: int, layers: int) -> nn.Sequential:
    mods, width = [], n_in
    for _ in range(layers):
        mods += [nn.Linear(width, hidden), nn.SiLU()]
        width = hidden
    last = nn.Linear(width, n_out)
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    return nn.Sequential(*mods, last)


def _parity_mask(dim: int, parity: str) -> torch.Tensor:
    """1 for the dimensions that condition (pass through), 0 for those transformed."""
    keep = 0 if parity == "even" else 1
    return (torch.arange(dim) % 2 == keep).float()
'''

AFFINE = '''\
class AffineCoupling(_FlowLayer):
    """RealNVP affine coupling (Dinh et al., 2017): half the dimensions are scaled and
    shifted by functions of the other half."""

    def __init__(self, dim: int, hidden: int = 64, layers: int = 2,
                 parity: str = "even") -> None:
        super().__init__()
        self.register_buffer("mask", _parity_mask(dim, parity))
        self.net = _conditioner(dim, 2 * dim, hidden, layers)

    def _st(self, x_cond: torch.Tensor):
        s, t = self.net(x_cond).chunk(2, dim=-1)
        s = torch.tanh(s) * (1 - self.mask)          # bounded scales keep training stable
        return s, t * (1 - self.mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s, t = self._st(x * self.mask)
        self.logdet = s.sum(-1)
        return x * self.mask + (1 - self.mask) * (x * torch.exp(s) + t)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        s, t = self._st(z * self.mask)
        return z * self.mask + (1 - self.mask) * ((z - t) * torch.exp(-s))
'''

MAF = '''\
class _MaskedLinear(nn.Linear):
    def __init__(self, n_in: int, n_out: int, mask: torch.Tensor) -> None:
        super().__init__(n_in, n_out)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight * self.mask, self.bias)


class MAFLayer(_FlowLayer):
    """Masked autoregressive flow layer (Papamakarios et al., 2017): a MADE network gives
    every dimension a shift and log-scale from the dimensions before it. Density is one
    pass; sampling is sequential."""

    def __init__(self, dim: int, hidden: int = 64, layers: int = 2,
                 reverse: bool = False) -> None:
        super().__init__()
        order = torch.arange(dim).flip(0) if reverse else torch.arange(dim)
        self.register_buffer("order", order)
        degrees_in = torch.empty(dim, dtype=torch.long)
        degrees_in[order] = torch.arange(dim)
        degrees, mods, width = degrees_in, [], dim
        for _ in range(layers):
            hidden_deg = torch.arange(hidden) % max(dim - 1, 1)
            mask = (hidden_deg[:, None] >= degrees[None, :]).float()
            mods += [_MaskedLinear(width, hidden, mask), nn.SiLU()]
            degrees, width = hidden_deg, hidden
        out_mask = (degrees_in[:, None] > degrees[None, :]).float()
        last = _MaskedLinear(width, 2 * dim, out_mask.repeat(2, 1))
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.net = nn.Sequential(*mods, last)

    def _params(self, x: torch.Tensor):
        mu, alpha = self.net(x).chunk(2, dim=-1)
        return mu, 3.0 * torch.tanh(alpha / 3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, alpha = self._params(x)
        self.logdet = -alpha.sum(-1)
        return (x - mu) * torch.exp(-alpha)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        x = torch.zeros_like(z)
        for i in self.order.tolist():
            mu, alpha = self._params(x)
            x[:, i] = z[:, i] * torch.exp(alpha[:, i]) + mu[:, i]
        return x
'''

SPLINE = '''\
def rq_spline(x: torch.Tensor, w: torch.Tensor, h: torch.Tensor, d: torch.Tensor,
              bound: float, inverse: bool = False):
    """Monotonic rational-quadratic spline on [-bound, bound] with identity tails
    (Durkan et al., 2019). x [M]; w, h [M, K]; d [M, K - 1]. Returns (y, log|dy/dx|)."""
    out, logdet = x.clone(), torch.zeros_like(x)
    inside = (x > -bound) & (x < bound)
    if not inside.any():
        return out, logdet
    x, w, h, d = x[inside], w[inside], h[inside], d[inside]
    k, min_size, min_d = w.shape[-1], 1e-3, 1e-3
    widths = min_size + (1 - min_size * k) * torch.softmax(w, -1)
    heights = min_size + (1 - min_size * k) * torch.softmax(h, -1)
    cw = nn.functional.pad(torch.cumsum(widths, -1), (1, 0)) * 2 * bound - bound
    ch = nn.functional.pad(torch.cumsum(heights, -1), (1, 0)) * 2 * bound - bound
    cw[:, 0], cw[:, -1], ch[:, 0], ch[:, -1] = -bound, bound, -bound, bound
    widths, heights = cw[:, 1:] - cw[:, :-1], ch[:, 1:] - ch[:, :-1]
    shift = float(torch.log(torch.expm1(torch.tensor(1 - min_d))))  # zero params: slope 1
    derivs = nn.functional.pad(min_d + nn.functional.softplus(d + shift), (1, 1), value=1.0)
    edges = (ch if inverse else cw).clone()
    edges[:, -1] += 1e-6
    idx = ((x[:, None] >= edges).sum(-1) - 1).clamp(0, k - 1)[:, None]
    xk, wk = cw.gather(-1, idx)[:, 0], widths.gather(-1, idx)[:, 0]
    yk, hk = ch.gather(-1, idx)[:, 0], heights.gather(-1, idx)[:, 0]
    dk, dk1 = derivs.gather(-1, idx)[:, 0], derivs.gather(-1, idx + 1)[:, 0]
    sk = hk / wk
    if inverse:
        dy = x - yk
        a = hk * (sk - dk) + dy * (dk1 + dk - 2 * sk)
        b = hk * dk - dy * (dk1 + dk - 2 * sk)
        c = -sk * dy
        theta = (2 * c) / (-b - torch.sqrt((b ** 2 - 4 * a * c).clamp_min(0)))
        y = theta * wk + xk
    else:
        theta = (x - xk) / wk
        y = yk + hk * (sk * theta ** 2 + dk * theta * (1 - theta)) / (
            sk + (dk1 + dk - 2 * sk) * theta * (1 - theta))
    t1 = theta * (1 - theta)
    den = sk + (dk1 + dk - 2 * sk) * t1
    ld = torch.log(sk ** 2 * (dk1 * theta ** 2 + 2 * sk * t1 + dk * (1 - theta) ** 2)) \\
        - 2 * torch.log(den)
    out[inside] = y
    logdet[inside] = -ld if inverse else ld
    return out, logdet


class SplineCoupling(_FlowLayer):
    """Neural spline flow coupling (Durkan et al., 2019): half the dimensions go through
    monotonic rational-quadratic splines whose knots are functions of the other half."""

    def __init__(self, dim: int, hidden: int = 64, layers: int = 2, bins: int = 8,
                 bound: float = 4.0, parity: str = "even") -> None:
        super().__init__()
        self.dim, self.bins, self.bound = dim, bins, bound
        self.register_buffer("mask", _parity_mask(dim, parity))
        self.net = _conditioner(dim, dim * (3 * bins - 1), hidden, layers)

    def _run(self, x: torch.Tensor, inverse: bool):
        params = self.net(x * self.mask).reshape(len(x), self.dim, 3 * self.bins - 1)
        k = self.bins
        y, ld = rq_spline(x.reshape(-1), params[..., :k].reshape(-1, k),
                          params[..., k:2 * k].reshape(-1, k),
                          params[..., 2 * k:].reshape(-1, k - 1), self.bound, inverse)
        y, ld = y.reshape(x.shape), ld.reshape(x.shape)
        keep = self.mask
        return x * keep + y * (1 - keep), (ld * (1 - keep)).sum(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, self.logdet = self._run(x, inverse=False)
        return z

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        return self._run(z, inverse=True)[0]
'''

ACTNORM = '''\
class ActNorm(_FlowLayer):
    """Per-dimension (vectors) or per-channel (images [C, H, W]) scale and shift,
    initialised from the first batch so its output has zero mean and unit variance
    (Kingma & Dhariwal, 2018)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.register_buffer("initialized", torch.tensor(False))

    def _view(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return t.view(1, -1, 1, 1) if x.dim() == 4 else t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and not bool(self.initialized) and len(x) > 1:
            with torch.no_grad():
                dims = (0, 2, 3) if x.dim() == 4 else (0,)
                self.bias.copy_(-x.mean(dims))
                self.log_scale.copy_(-torch.log(x.std(dims) + 1e-6))
                self.initialized.fill_(True)
        positions = x.shape[2] * x.shape[3] if x.dim() == 4 else 1
        self.logdet = (positions * self.log_scale.sum()).expand(len(x))
        return (x + self._view(self.bias, x)) * torch.exp(self._view(self.log_scale, x))

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        return z * torch.exp(-self._view(self.log_scale, z)) - self._view(self.bias, z)
'''


GLOW = '''\
class InvConv1x1(_FlowLayer):
    """Invertible 1×1 convolution with an LU-decomposed weight (Glow): a learned channel
    permutation whose log-determinant is H·W·sum(log|s|)."""

    def __init__(self, channels: int, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        w = torch.linalg.qr(torch.randn(channels, channels, generator=g))[0]
        p, lower, upper = torch.linalg.lu(w)
        s = torch.diagonal(upper)
        self.register_buffer("p", p)
        self.register_buffer("sign_s", torch.sign(s))
        self.register_buffer("l_mask", torch.tril(torch.ones(channels, channels), -1))
        self.register_buffer("eye", torch.eye(channels))
        self.lower = nn.Parameter(lower)
        self.upper = nn.Parameter(torch.triu(upper, 1))
        self.log_s = nn.Parameter(torch.log(torch.abs(s)))

    def weight(self) -> torch.Tensor:
        lower = self.lower * self.l_mask + self.eye
        upper = self.upper * self.l_mask.T + torch.diag(self.sign_s * torch.exp(self.log_s))
        return self.p @ lower @ upper

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight()
        self.logdet = (x.shape[2] * x.shape[3] * self.log_s.sum()).expand(len(x))
        return nn.functional.conv2d(x, w.view(*w.shape, 1, 1))

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        w = torch.linalg.inv(self.weight())
        return nn.functional.conv2d(z, w.view(*w.shape, 1, 1))


def _zero_conv(n_in: int, n_out: int) -> nn.Conv2d:
    conv = nn.Conv2d(n_in, n_out, 3, padding=1)
    nn.init.zeros_(conv.weight)
    nn.init.zeros_(conv.bias)
    return conv


class ConvCoupling(_FlowLayer):
    """Glow's affine coupling: half the channels are scaled (sigmoid, so the layer stays
    stable) and shifted by a small CNN of the other half."""

    def __init__(self, channels: int, hidden: int = 64, parity: str = "even") -> None:
        super().__init__()
        self.keep = channels // 2          # conditioning channels
        self.parity = parity
        n_cond, n_out = self.keep, channels - self.keep
        self.net = nn.Sequential(nn.Conv2d(n_cond, hidden, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(hidden, hidden, 1), nn.ReLU(),
                                 _zero_conv(hidden, 2 * n_out))

    def _st(self, cond: torch.Tensor):
        h = self.net(cond)
        shift, raw = h[:, 0::2], h[:, 1::2]
        return shift, torch.sigmoid(raw + 2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.parity == "even":
            cond, moved = x[:, :self.keep], x[:, self.keep:]
        else:
            moved, cond = x[:, :x.shape[1] - self.keep], x[:, x.shape[1] - self.keep:]
        shift, scale = self._st(cond)
        self.logdet = torch.log(scale).flatten(1).sum(-1)
        out = (moved + shift) * scale
        return torch.cat([cond, out] if self.parity == "even" else [out, cond], 1)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        if self.parity == "even":
            cond, moved = z[:, :self.keep], z[:, self.keep:]
        else:
            moved, cond = z[:, :z.shape[1] - self.keep], z[:, z.shape[1] - self.keep:]
        shift, scale = self._st(cond)
        out = moved / scale - shift
        return torch.cat([cond, out] if self.parity == "even" else [out, cond], 1)


class FlowSqueeze(_FlowLayer):
    """[C, H, W] -> [4C, H/2, W/2]: every 2×2 patch becomes channels (volume-preserving)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        self.logdet = x.new_zeros(n)
        x = x.view(n, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4)
        return x.reshape(n, 4 * c, h // 2, w // 2)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        n, c, h, w = z.shape
        z = z.view(n, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)
        return z.reshape(n, c // 4, 2 * h, 2 * w)


class FlowSplit(_FlowLayer):
    """Glow's multi-scale split: half the channels leave the flow here and are scored by
    a Gaussian whose mean and scale come from the other half (a learned prior); the rest
    continue. Sampling draws the factored half from that prior (``temperature``)."""

    temperature = 1.0

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.keep = channels // 2
        self.prior = _zero_conv(self.keep, 2 * (channels - self.keep))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z1, z2 = x[:, :self.keep], x[:, self.keep:]
        mean, log_scale = self.prior(z1).chunk(2, 1)
        logp = -0.5 * (((z2 - mean) * torch.exp(-log_scale)) ** 2 + 2 * log_scale
                       + 1.8378770664093453)       # log(2π)
        self.logdet = logp.flatten(1).sum(-1)
        return z1

    def inverse(self, z1: torch.Tensor) -> torch.Tensor:
        mean, log_scale = self.prior(z1).chunk(2, 1)
        z2 = mean + torch.exp(log_scale) * torch.randn_like(mean) * self.temperature
        return torch.cat([z1, z2], 1)
'''

PERMUTE = '''\
class FlowPermute(_FlowLayer):
    """Reorders the dimensions (reverse or a fixed random order) so the next layer
    transforms different ones; volume-preserving."""

    def __init__(self, dim: int, kind: str = "reverse", seed: int = 0) -> None:
        super().__init__()
        if kind == "reverse":
            perm = torch.arange(dim).flip(0)
        else:
            perm = torch.randperm(dim, generator=torch.Generator().manual_seed(seed))
        self.register_buffer("perm", perm)
        self.register_buffer("inv", torch.argsort(perm))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.logdet = x.new_zeros(len(x))
        return x[:, self.perm]

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        return z[:, self.inv]
'''

FLOW_HELPERS = {"FlowBase": FLOW_BASE, "AffineCoupling": AFFINE, "MAFLayer": MAF,
                "SplineCoupling": SPLINE, "ActNorm": ACTNORM, "FlowPermute": PERMUTE,
                "Glow": GLOW}


def register() -> None:
    from ai_made_easy.core.codegen.helpers import PYTORCH_HELPERS

    PYTORCH_HELPERS.update(FLOW_HELPERS)
