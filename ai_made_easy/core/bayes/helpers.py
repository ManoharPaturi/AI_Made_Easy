"""PyTorch helpers for Bayesian deep learning, emitted into generated model code.

``stochastic`` modules give a different output on every pass (the training script averages
``samples`` passes); ``kl_divergence()`` is added to the loss of Bayes-by-Backprop layers.
"""
from __future__ import annotations

MC_DROPOUT = '''\
class MCDropout(nn.Module):
    """Dropout that stays on when predicting: repeated passes sample the predictive
    distribution (Gal & Ghahramani, 2016)."""

    stochastic = True

    def __init__(self, p: float = 0.1, samples: int = 30) -> None:
        super().__init__()
        self.p, self.samples = p, samples

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.dropout(x, self.p, training=True)
'''

BAYES_LAYERS = '''\
class _BayesBase(nn.Module):
    """Gaussian weights N(mu, softplus(rho)^2) with a N(0, prior_sigma^2) prior."""

    stochastic = True

    def _init(self, shape: tuple, fan_in: int, prior_sigma: float, samples: int) -> None:
        bound = (1.0 / max(fan_in, 1)) ** 0.5
        self.weight_mu = nn.Parameter(torch.empty(shape).uniform_(-bound, bound))
        self.weight_rho = nn.Parameter(torch.full(shape, -5.0))
        self.bias_mu = nn.Parameter(torch.zeros(shape[0]))
        self.bias_rho = nn.Parameter(torch.full((shape[0],), -5.0))
        self.prior_sigma, self.samples = prior_sigma, samples

    def _weights(self):
        w = self.weight_mu + nn.functional.softplus(self.weight_rho) * \\
            torch.randn_like(self.weight_mu)
        b = self.bias_mu + nn.functional.softplus(self.bias_rho) * torch.randn_like(self.bias_mu)
        return w, b

    def kl_divergence(self) -> torch.Tensor:
        """KL(q || prior), summed over every weight (closed form for Gaussians)."""
        total = 0.0
        p = self.prior_sigma
        for mu, rho in ((self.weight_mu, self.weight_rho), (self.bias_mu, self.bias_rho)):
            s = nn.functional.softplus(rho) + 1e-8
            total = total + (torch.log(p / s) + (s ** 2 + mu ** 2) / (2 * p ** 2) - 0.5).sum()
        return total


class BayesLinear(_BayesBase):
    """Bayes-by-Backprop linear layer (Blundell et al., 2015)."""

    def __init__(self, in_features: int, out_features: int, prior_sigma: float = 1.0,
                 samples: int = 30) -> None:
        super().__init__()
        self._init((out_features, in_features), in_features, prior_sigma, samples)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w, b = self._weights()
        return nn.functional.linear(x, w, b)


class BayesConv2d(_BayesBase):
    """Bayes-by-Backprop 2-D convolution."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, prior_sigma: float = 1.0,
                 samples: int = 30) -> None:
        super().__init__()
        self.stride, self.padding = stride, padding
        self._init((out_channels, in_channels, kernel_size, kernel_size),
                   in_channels * kernel_size * kernel_size, prior_sigma, samples)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w, b = self._weights()
        return nn.functional.conv2d(x, w, b, self.stride, self.padding)
'''

MDN = '''\
class MDNHead(nn.Module):
    """Mixture density network head (Bishop, 1994): K Gaussian components per target.

    Output [B, K * (1 + 2 D)]: mixture logits, means and log standard deviations.
    """

    is_mdn = True

    def __init__(self, in_features: int, components: int = 5, targets: int = 1) -> None:
        super().__init__()
        self.k, self.d = components, targets
        self.proj = nn.Linear(in_features, components * (1 + 2 * targets))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

    def split(self, out: torch.Tensor):
        k, d = self.k, self.d
        logits = out[:, :k]
        mu = out[:, k:k + k * d].reshape(-1, k, d)
        log_sigma = out[:, k + k * d:].reshape(-1, k, d).clamp(-7.0, 7.0)
        return logits, mu, log_sigma
'''

BAYES_HELPERS = {"MCDropout": MC_DROPOUT, "BayesLinear": BAYES_LAYERS,
                 "BayesConv2d": BAYES_LAYERS, "MDNHead": MDN}


def register() -> None:
    from ai_made_easy.core.codegen.helpers import PYTORCH_HELPERS

    PYTORCH_HELPERS.update(BAYES_HELPERS)
