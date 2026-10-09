"""Reusable helper definitions emitted into generated files on demand.

A block lists the helpers it needs in ``pytorch_helpers`` / ``keras_helpers``;
the generator includes each helper's source exactly once, in a stable order.
Helpers must be self-contained (they rely only on the template imports:
``torch``/``nn``/``F`` for PyTorch, ``keras``/``layers``/``ops`` for Keras).
"""
from __future__ import annotations

PYTORCH_HELPERS: dict[str, str] = {
    "Lambda": '''\
class Lambda(nn.Module):
    """Wrap a tensor expression as a module."""

    def __init__(self, fn) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)
''',
    "RMSNorm": '''\
class RMSNorm(nn.Module):
    """Root-mean-square layer normalization over the last dimension."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight
''',
    "SqueezeExcite": '''\
class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al., 2018)."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(2, x.dim()))
        s = x.mean(dim=dims)
        s = torch.sigmoid(self.fc2(torch.relu(self.fc1(s))))
        return x * s.view(*s.shape, *([1] * len(dims)))
''',
    "PretrainedBackbone": '''\
class PretrainedBackbone(nn.Module):
    """torchvision backbone with its classifier removed: image -> pooled features."""

    def __init__(self, name: str, pretrained: bool = True, freeze: bool = True) -> None:
        super().__init__()
        import torchvision

        weights = "DEFAULT" if pretrained else None
        model = getattr(torchvision.models, name)(weights=weights)
        if name.startswith("vgg"):
            model.avgpool = nn.AdaptiveAvgPool2d(1)
            model.classifier = nn.Flatten()
        elif hasattr(model, "fc"):  # resnet / resnext / wide_resnet / regnet / shufflenet
            model.fc = nn.Identity()
        elif hasattr(model, "heads"):  # vit
            model.heads = nn.Identity()
        elif hasattr(model, "head"):  # swin
            model.head = nn.Identity()
        elif name.startswith(("convnext", "maxvit")):  # pooling lives in the classifier
            model.classifier[-1] = nn.Identity()
        else:  # mobilenet / efficientnet / densenet / mnasnet
            model.classifier = nn.Identity()
        self.model = model
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
''',
    "HFTextEncoder": '''\
class HFTextEncoder(nn.Module):
    """Hugging Face transformer encoder over token ids (0 = padding)."""

    def __init__(self, model_id: str, pooling: str = "cls", freeze: bool = True) -> None:
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(model_id)
        self.pooling = pooling
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        mask = (ids != 0).long()
        hidden = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        if self.pooling == "cls":
            return hidden[:, 0]
        if self.pooling == "mean":
            m = mask.unsqueeze(-1).to(hidden.dtype)
            return (hidden * m).sum(1) / m.sum(1).clamp_min(1.0)
        return hidden
''',
    "GaussianNoise": '''\
class GaussianNoise(nn.Module):
    """Additive zero-mean Gaussian noise, active only in training mode."""

    def __init__(self, stddev: float) -> None:
        super().__init__()
        self.stddev = stddev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.stddev > 0:
            return x + torch.randn_like(x) * self.stddev
        return x
''',
    "GaussianDropout": '''\
class GaussianDropout(nn.Module):
    """Multiplicative 1-centred Gaussian noise, active only in training mode."""

    def __init__(self, p: float) -> None:
        super().__init__()
        self.std = (p / (1.0 - p)) ** 0.5 if p < 1.0 else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.std > 0:
            return x * (1.0 + torch.randn_like(x) * self.std)
        return x
''',
    "PositionalEncoding": '''\
class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding added to [batch, length, dim]."""

    def __init__(self, length: int, dim: int) -> None:
        super().__init__()
        position = torch.arange(length).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * (-torch.log(torch.tensor(10000.0)) / dim))
        pe = torch.zeros(length, dim)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div[: dim // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]
''',
    "LearnedPositionalEmbedding": '''\
class LearnedPositionalEmbedding(nn.Module):
    """Trainable position embeddings added to [batch, length, dim]."""

    def __init__(self, length: int, dim: int) -> None:
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, length, dim))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]
''',
}

KERAS_HELPERS: dict[str, str] = {
    "pretrained_backbone": '''\
def pretrained_backbone(x, name: str, weights=None, trainable: bool = False):
    """keras.applications backbone without its top: image -> pooled features."""
    base = getattr(keras.applications, name)(include_top=False, weights=weights,
                                             input_shape=tuple(x.shape[1:]), pooling="avg")
    base.trainable = trainable
    return base(x)
''',
    "transformer_encoder": '''\
def transformer_encoder(x, num_heads: int, ff_dim: int, dropout: float = 0.1,
                        epsilon: float = 1e-5, activation: str = "relu",
                        norm_first: bool = False):
    """Transformer encoder block equivalent to torch.nn.TransformerEncoderLayer."""
    d_model = x.shape[-1]

    def attention(t):
        t = layers.MultiHeadAttention(num_heads=num_heads,
                                      key_dim=d_model // num_heads,
                                      dropout=dropout)(t, t)
        return layers.Dropout(dropout)(t)

    def feed_forward(t):
        t = layers.Dense(ff_dim, activation=activation)(t)
        t = layers.Dropout(dropout)(t)
        t = layers.Dense(d_model)(t)
        return layers.Dropout(dropout)(t)

    if norm_first:
        x = layers.Add()([x, attention(layers.LayerNormalization(epsilon=epsilon)(x))])
        return layers.Add()([x, feed_forward(layers.LayerNormalization(epsilon=epsilon)(x))])
    x = layers.LayerNormalization(epsilon=epsilon)(layers.Add()([x, attention(x)]))
    return layers.LayerNormalization(epsilon=epsilon)(layers.Add()([x, feed_forward(x)]))
''',
    "transformer_decoder": '''\
def transformer_decoder(x, memory, num_heads: int, ff_dim: int, dropout: float = 0.1,
                        epsilon: float = 1e-5, activation: str = "relu"):
    """Transformer decoder block equivalent to torch.nn.TransformerDecoderLayer."""
    d_model = x.shape[-1]
    key_dim = d_model // num_heads
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim,
                                     dropout=dropout)(x, x)
    x = layers.LayerNormalization(epsilon=epsilon)(
        layers.Add()([x, layers.Dropout(dropout)(attn)]))
    cross = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim,
                                      dropout=dropout)(x, memory)
    x = layers.LayerNormalization(epsilon=epsilon)(
        layers.Add()([x, layers.Dropout(dropout)(cross)]))
    ff = layers.Dense(ff_dim, activation=activation)(x)
    ff = layers.Dense(d_model)(layers.Dropout(dropout)(ff))
    return layers.LayerNormalization(epsilon=epsilon)(
        layers.Add()([x, layers.Dropout(dropout)(ff)]))
''',
    "pad_spatial": '''\
def pad_spatial(x, padding: int, mode: str):
    """Reflect / replicate / circular padding of every spatial axis (channels-last)."""
    rank = len(x.shape)
    for axis in range(1, rank - 1):
        n = x.shape[axis]

        def take(start, stop, t=x, axis=axis):
            index = [slice(None)] * rank
            index[axis] = slice(start, stop)
            return t[tuple(index)]

        if mode == "reflect":
            left = ops.flip(take(1, padding + 1), axis=axis)
            right = ops.flip(take(n - padding - 1, n - 1), axis=axis)
        elif mode == "edge":
            left = ops.repeat(take(0, 1), padding, axis=axis)
            right = ops.repeat(take(n - 1, n), padding, axis=axis)
        else:  # circular
            left, right = take(n - padding, n), take(0, padding)
        x = ops.concatenate([left, x, right], axis=axis)
    return x
''',
    "squeeze_excite": '''\
def squeeze_excite(x, reduction: int = 16):
    """Squeeze-and-Excitation channel attention on a channels-last tensor."""
    channels = x.shape[-1]
    rank = len(x.shape) - 2
    pool = layers.GlobalAveragePooling2D() if rank == 2 else (
        layers.GlobalAveragePooling1D() if rank == 1 else layers.GlobalAveragePooling3D())
    s = pool(x)
    s = layers.Dense(max(channels // reduction, 1), activation="relu")(s)
    s = layers.Dense(channels, activation="sigmoid")(s)
    s = layers.Reshape((1,) * rank + (channels,))(s)
    return layers.Multiply()([x, s])
''',
    "rms_norm": '''\
class RMSNorm(layers.Layer):
    """Root-mean-square layer normalization over the last axis."""

    def __init__(self, epsilon: float = 1e-6, **kwargs) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape) -> None:
        self.scale = self.add_weight(shape=(input_shape[-1],), initializer="ones")

    def call(self, x):
        rms = ops.rsqrt(ops.mean(ops.square(x), axis=-1, keepdims=True) + self.epsilon)
        return x * rms * self.scale
''',
    "positional_encoding": '''\
class PositionalEncoding(layers.Layer):
    """Fixed sinusoidal positional encoding added to [batch, length, dim]."""

    def call(self, x):
        length, dim = x.shape[1], x.shape[2]
        position = ops.expand_dims(ops.arange(length, dtype="float32"), 1)
        div = ops.exp(ops.arange(0, dim, 2, dtype="float32") * (-ops.log(10000.0) / dim))
        angles = position * div
        pe = ops.reshape(ops.stack([ops.sin(angles), ops.cos(angles)], axis=-1),
                         (length, -1))[:, :dim]
        return x + ops.cast(pe, x.dtype)
''',
    "learned_positional_embedding": '''\
class LearnedPositionalEmbedding(layers.Layer):
    """Trainable position embeddings added to [batch, length, dim]."""

    def build(self, input_shape) -> None:
        self.pe = self.add_weight(shape=(1, input_shape[1], input_shape[2]),
                                  initializer=keras.initializers.TruncatedNormal(stddev=0.02))

    def call(self, x):
        return x + self.pe
''',
}


def render_helpers(names: list[str], table: dict[str, str]) -> list[str]:
    """Helper sources for ``names`` in first-seen order, deduplicated."""
    sources: list[str] = []
    for name in names:
        if name not in table:
            raise KeyError(f"unknown codegen helper {name!r}")
        if table[name] not in sources:   # several names may share one source
            sources.append(table[name])
    return sources
