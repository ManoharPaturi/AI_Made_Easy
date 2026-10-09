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
    """Per-dimension scale and shift, initialised from the first batch so its output has
    zero mean and unit variance (Kingma & Dhariwal, 2018)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.register_buffer("initialized", torch.tensor(False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not bool(self.initialized) and self.training and len(x) > 1:
            with torch.no_grad():
                self.bias.copy_(-x.mean(0))
                self.log_scale.copy_(-torch.log(x.std(0) + 1e-6))
                self.initialized.fill_(True)
        self.logdet = self.log_scale.sum().expand(len(x))
        return (x + self.bias) * torch.exp(self.log_scale)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        return z * torch.exp(-self.log_scale) - self.bias
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
                "SplineCoupling": SPLINE, "ActNorm": ACTNORM, "FlowPermute": PERMUTE}


def register() -> None:
    from ai_made_easy.core.codegen.helpers import PYTORCH_HELPERS

    PYTORCH_HELPERS.update(FLOW_HELPERS)
