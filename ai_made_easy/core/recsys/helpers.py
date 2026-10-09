"""PyTorch helpers for recommenders, emitted into generated model code.

Every model reads rows ``[user, item, *dense features]`` (ids as numbers) and returns one
score per row; higher means "more likely to interact / higher rating".
"""
from __future__ import annotations

IDS = '''\
def rec_ids(x: torch.Tensor, users: nn.Embedding, items: nn.Embedding):
    """User / item ids from the first two columns (clamped into the embedding tables)."""
    u = x[:, 0].round().long().clamp(0, users.num_embeddings - 1)
    i = x[:, 1].round().long().clamp(0, items.num_embeddings - 1)
    return u, i
'''

MF = '''\
class MatrixFactorization(nn.Module):
    """score = user · item (+ user bias + item bias + global bias)."""

    def __init__(self, n_users: int, n_items: int, dim: int = 32, biases: bool = True) -> None:
        super().__init__()
        self.user = nn.Embedding(n_users, dim)
        self.item = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user.weight, std=0.1)
        nn.init.normal_(self.item.weight, std=0.1)
        self.biases = biases
        if biases:
            self.user_bias = nn.Embedding(n_users, 1)
            self.item_bias = nn.Embedding(n_items, 1)
            nn.init.zeros_(self.user_bias.weight)
            nn.init.zeros_(self.item_bias.weight)
            self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, i = rec_ids(x, self.user, self.item)
        score = (self.user(u) * self.item(i)).sum(-1, keepdim=True)
        if self.biases:
            score = score + self.user_bias(u) + self.item_bias(i) + self.bias
        return score
'''

NCF = '''\
def _mlp(sizes: list, last_activation: bool = True) -> nn.Sequential:
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*(layers if last_activation else layers[:-1]))


class NeuralCF(nn.Module):
    """Neural collaborative filtering (He et al., 2017): a generalised matrix factorisation
    path and an MLP path over user / item embeddings, fused by a linear layer."""

    def __init__(self, n_users: int, n_items: int, dim: int = 32, hidden: list = (64, 32)) -> None:
        super().__init__()
        self.gmf_user, self.gmf_item = nn.Embedding(n_users, dim), nn.Embedding(n_items, dim)
        self.mlp_user, self.mlp_item = nn.Embedding(n_users, dim), nn.Embedding(n_items, dim)
        for emb in (self.gmf_user, self.gmf_item, self.mlp_user, self.mlp_item):
            nn.init.normal_(emb.weight, std=0.1)
        self.mlp = _mlp([2 * dim, *hidden])
        self.out = nn.Linear(dim + hidden[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, i = rec_ids(x, self.gmf_user, self.gmf_item)
        gmf = self.gmf_user(u) * self.gmf_item(i)
        mlp = self.mlp(torch.cat([self.mlp_user(u), self.mlp_item(i)], dim=-1))
        return self.out(torch.cat([gmf, mlp], dim=-1))
'''

TWO_TOWER = '''\
class TwoTower(nn.Module):
    """Two-tower retrieval: separate user and item towers (embedding -> MLP) whose
    normalised outputs are compared by a dot product; item vectors can be indexed for fast
    nearest-neighbour search."""

    def __init__(self, n_users: int, n_items: int, dim: int = 32, hidden: int = 64,
                 temperature: float = 0.1) -> None:
        super().__init__()
        self.user_emb, self.item_emb = nn.Embedding(n_users, hidden), nn.Embedding(n_items, hidden)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        self.user_tower = nn.Sequential(nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                        nn.Linear(hidden, dim))
        self.item_tower = nn.Sequential(nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                        nn.Linear(hidden, dim))
        self.temperature = temperature

    def users(self, u: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.user_tower(self.user_emb(u)), dim=-1)

    def items(self, i: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.item_tower(self.item_emb(i)), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, i = rec_ids(x, self.user_emb, self.item_emb)
        return (self.users(u) * self.items(i)).sum(-1, keepdim=True) / self.temperature
'''

DLRM = '''\
class DLRM(nn.Module):
    """DLRM-lite (Naumov et al., 2019): dense features through a bottom MLP, user and item
    embeddings, pairwise dot-product interactions, then a top MLP."""

    def __init__(self, n_users: int, n_items: int, n_dense: int, dim: int = 16,
                 bottom: list = (64,), top: list = (64, 32)) -> None:
        super().__init__()
        self.user, self.item = nn.Embedding(n_users, dim), nn.Embedding(n_items, dim)
        nn.init.normal_(self.user.weight, std=0.1)
        nn.init.normal_(self.item.weight, std=0.1)
        self.n_dense = n_dense
        self.bottom = _mlp([n_dense, *bottom, dim]) if n_dense else None
        n_vec = 3 if n_dense else 2
        pairs = n_vec * (n_vec - 1) // 2
        self.top = nn.Sequential(_mlp([dim + pairs, *top]), nn.Linear(top[-1], 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, i = rec_ids(x, self.user, self.item)
        vecs = [self.user(u), self.item(i)]
        if self.bottom is not None:
            vecs.insert(0, self.bottom(x[:, 2:2 + self.n_dense]))
        stack = torch.stack(vecs, dim=1)                              # [B, n, dim]
        inter = stack @ stack.transpose(1, 2)
        rows, cols = torch.triu_indices(len(vecs), len(vecs), 1)
        return self.top(torch.cat([vecs[0], inter[:, rows, cols]], dim=-1))
'''

REC_HELPERS = {"RecIds": IDS, "MatrixFactorization": MF, "NeuralCF": NCF,
               "TwoTower": TWO_TOWER, "DLRM": DLRM}


def register() -> None:
    from ai_made_easy.core.codegen.helpers import PYTORCH_HELPERS

    PYTORCH_HELPERS.update(REC_HELPERS)
