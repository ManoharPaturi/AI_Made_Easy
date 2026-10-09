"""PyTorch helpers for tabular deep learning, emitted into generated model code.

Inputs are the preprocessed feature vector: ordinal-encoded categorical columns sit at
the ``categorical`` positions as codes 0..card-1 (-1: unseen), every other position is a
number. Categorical codes are shifted by one so unseen values get their own embedding.
"""
from __future__ import annotations

SPLIT = '''\
class _TabularSplit(nn.Module):
    """Separates numeric columns from categorical codes (positions fixed at design time)."""

    def __init__(self, n_features: int, categorical: list, cardinalities: list) -> None:
        super().__init__()
        cat = torch.tensor(categorical, dtype=torch.long)
        num = torch.tensor([i for i in range(n_features) if i not in set(categorical)],
                           dtype=torch.long)
        self.register_buffer("cat_idx", cat)
        self.register_buffer("num_idx", num)
        self.cards = list(cardinalities)

    def split(self, x: torch.Tensor):
        codes = x[:, self.cat_idx].round().long() + 1          # -1 (unseen) -> 0
        limits = torch.tensor(self.cards, device=x.device)
        codes = torch.minimum(codes.clamp_min(0), limits)
        return x[:, self.num_idx], codes
'''

EMBED = '''\
class CategoricalEmbedding(_TabularSplit):
    """Replaces each categorical code with a learned vector (entity embeddings)."""

    def __init__(self, n_features: int, categorical: list, cardinalities: list,
                 dims: list) -> None:
        super().__init__(n_features, categorical, cardinalities)
        self.embeddings = nn.ModuleList(nn.Embedding(c + 1, d)
                                        for c, d in zip(cardinalities, dims))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num, codes = self.split(x)
        parts = [num] + [emb(codes[:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.cat(parts, dim=-1)
'''

RESNET = '''\
class TabularResNet(nn.Module):
    """ResNet for tables (Gorishniy et al., 2021): residual blocks of
    BatchNorm -> Linear -> ReLU -> Dropout -> Linear -> Dropout."""

    def __init__(self, n_features: int, d: int = 128, blocks: int = 3, hidden_factor: float = 2.0,
                 dropout: float = 0.1) -> None:
        super().__init__()
        hidden = max(1, int(d * hidden_factor))
        self.first = nn.Linear(n_features, d)
        self.blocks = nn.ModuleList(nn.Sequential(
            nn.BatchNorm1d(d), nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, d), nn.Dropout(dropout)) for _ in range(blocks))
        self.last = nn.Sequential(nn.BatchNorm1d(d), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.first(x)
        for block in self.blocks:
            h = h + block(h)
        return self.last(h)
'''

FT = '''\
class FTTransformer(_TabularSplit):
    """FT-Transformer (Gorishniy et al., 2021): every feature becomes a token (numbers
    through a per-feature linear map, categories through embeddings), a [CLS] token reads
    them with self-attention; returns the [CLS] representation."""

    def __init__(self, n_features: int, categorical: list, cardinalities: list, d: int = 64,
                 layers: int = 3, heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__(n_features, categorical, cardinalities)
        n_num = n_features - len(categorical)
        self.num_weight = nn.Parameter(torch.randn(n_num, d) / d ** 0.5)
        self.num_bias = nn.Parameter(torch.zeros(n_num, d))
        self.embeddings = nn.ModuleList(nn.Embedding(c + 1, d) for c in cardinalities)
        self.cls = nn.Parameter(torch.randn(1, 1, d) / d ** 0.5)
        layer = nn.TransformerEncoderLayer(d, heads, int(d * 4 / 3), dropout,
                                           activation="gelu", batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num, codes = self.split(x)
        tokens = [num.unsqueeze(-1) * self.num_weight + self.num_bias]
        if len(self.embeddings):
            tokens.append(torch.stack([emb(codes[:, i]) for i, emb in
                                       enumerate(self.embeddings)], dim=1))
        h = torch.cat([self.cls.expand(len(x), -1, -1), *tokens], dim=1)
        return self.head(self.encoder(h)[:, 0])
'''

TAB_TRANSFORMER = '''\
class TabTransformer(_TabularSplit):
    """TabTransformer (Huang et al., 2020): self-attention over categorical embeddings,
    concatenated with the normalised numbers and passed through an MLP."""

    def __init__(self, n_features: int, categorical: list, cardinalities: list, d: int = 32,
                 layers: int = 3, heads: int = 4, hidden: int = 128,
                 dropout: float = 0.1) -> None:
        super().__init__(n_features, categorical, cardinalities)
        n_num = n_features - len(categorical)
        self.embeddings = nn.ModuleList(nn.Embedding(c + 1, d) for c in cardinalities)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.num_norm = nn.LayerNorm(n_num) if n_num else None
        width = len(cardinalities) * d + n_num
        self.mlp = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num, codes = self.split(x)
        parts = []
        if len(self.embeddings):
            tokens = torch.stack([emb(codes[:, i]) for i, emb in enumerate(self.embeddings)], 1)
            parts.append(self.encoder(tokens).flatten(1))
        if self.num_norm is not None:
            parts.append(self.num_norm(num))
        return self.mlp(torch.cat(parts, dim=-1))
'''

TABNET = '''\
def sparsemax(z: torch.Tensor) -> torch.Tensor:
    """Euclidean projection onto the simplex (Martins & Astudillo, 2016): sparse weights."""
    zs, _ = torch.sort(z, dim=-1, descending=True)
    k = torch.arange(1, z.shape[-1] + 1, device=z.device, dtype=z.dtype)
    cumsum = zs.cumsum(-1)
    support = (1 + k * zs) > cumsum
    k_z = support.sum(-1, keepdim=True).clamp_min(1)
    tau = (cumsum.gather(-1, k_z - 1) - 1) / k_z.to(z.dtype)
    return (z - tau).clamp_min(0)


class _GLUBlock(nn.Module):
    def __init__(self, n_in: int, n_out: int, residual: bool) -> None:
        super().__init__()
        self.fc = nn.Linear(n_in, 2 * n_out, bias=False)
        self.bn = nn.BatchNorm1d(2 * n_out, momentum=0.02)
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = nn.functional.glu(self.bn(self.fc(x)), dim=-1)
        return (x + h) * 0.5 ** 0.5 if self.residual else h


class TabNet(nn.Module):
    """TabNet (Arik & Pfister, 2021): sequential attention chooses which features each
    decision step reads (sparsemax masks); returns the summed decision outputs. The mask
    entropy is added to the loss (aux_loss) to keep the selection sparse."""

    def __init__(self, n_features: int, n_d: int = 16, n_a: int = 16, steps: int = 3,
                 gamma: float = 1.3, sparsity: float = 1e-3) -> None:
        super().__init__()
        self.n_d, self.steps, self.gamma, self.sparsity = n_d, steps, gamma, sparsity
        width = n_d + n_a
        self.bn = nn.BatchNorm1d(n_features, momentum=0.02)
        self.shared = nn.ModuleList([_GLUBlock(n_features, width, False),
                                     _GLUBlock(width, width, True)])
        self.step_blocks = nn.ModuleList(
            nn.Sequential(_GLUBlock(width, width, True), _GLUBlock(width, width, True))
            for _ in range(steps + 1))
        self.attention = nn.ModuleList(
            nn.Sequential(nn.Linear(n_a, n_features, bias=False),
                          nn.BatchNorm1d(n_features, momentum=0.02))
            for _ in range(steps))
        self._entropy = None

    def _transform(self, x: torch.Tensor, i: int) -> torch.Tensor:
        for block in self.shared:
            x = block(x)
        return self.step_blocks[i](x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        prior = torch.ones_like(x)
        a = self._transform(x, 0)[:, self.n_d:]
        out = torch.zeros(len(x), self.n_d, device=x.device, dtype=x.dtype)
        entropy = 0.0
        for i in range(self.steps):
            mask = sparsemax(self.attention[i](a) * prior)
            entropy = entropy + (-mask * torch.log(mask + 1e-10)).sum(-1).mean()
            prior = prior * (self.gamma - mask)
            h = self._transform(x * mask, i + 1)
            out = out + torch.relu(h[:, :self.n_d])
            a = h[:, self.n_d:]
        self._entropy = entropy / self.steps
        return out

    def aux_loss(self) -> torch.Tensor:
        return self.sparsity * self._entropy if self._entropy is not None else 0.0
'''

TABULAR_HELPERS = {"TabularSplit": SPLIT, "CategoricalEmbedding": EMBED,
                   "TabularResNet": RESNET, "FTTransformer": FT,
                   "TabTransformer": TAB_TRANSFORMER, "TabNet": TABNET}


def register() -> None:
    from ai_made_easy.core.codegen.helpers import PYTORCH_HELPERS

    PYTORCH_HELPERS.update(TABULAR_HELPERS)
