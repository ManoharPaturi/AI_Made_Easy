"""Models used by the import tests (also importable by the subprocess importer)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextLSTM(nn.Module):
    def __init__(self, vocab: int = 1000, classes: int = 5):
        super().__init__()
        self.emb = nn.Embedding(vocab, 32, padding_idx=0)
        self.lstm = nn.LSTM(32, 64, num_layers=2, batch_first=True)
        self.head = nn.Linear(64, classes)

    def forward(self, x):
        _, (h, _) = self.lstm(self.emb(x))
        return self.head(h[-1])


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(16, 64)
        layer = nn.TransformerEncoderLayer(64, 4, 128, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, 2)
        self.out = nn.Linear(64, 3)

    def forward(self, x):
        return self.out(self.enc(self.proj(x)).mean(dim=1))


class SkipUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Conv2d(1, 8, 3, padding=1)
        self.p = nn.MaxPool2d(2)
        self.b = nn.Conv2d(8, 8, 3, padding="same")
        self.u = nn.Upsample(scale_factor=2)
        self.c = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        a = torch.relu(self.a(x))
        b = self.u(torch.relu(self.b(self.p(a))))
        return torch.sigmoid(self.c(torch.cat([a, b], dim=1)))


class FunctionalMLP(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.a = nn.Linear(20, hidden)
        self.b = nn.Linear(hidden, 4)

    def forward(self, x):
        x = F.dropout(F.gelu(self.a(x)), 0.1, self.training)
        return F.log_softmax(self.b(x) * 2.0 + 1.0, dim=-1)


class Bilinear(nn.Module):
    """Uses a matrix product of two run-time tensors (no block for it)."""

    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 8)

    def forward(self, x):
        h = self.a(x)
        return torch.matmul(h.unsqueeze(2), h.unsqueeze(1)).flatten(1)


class Branchy(nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x
        return -x
