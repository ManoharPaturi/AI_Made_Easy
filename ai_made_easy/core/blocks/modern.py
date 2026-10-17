"""Modern network parts (3.3): the pieces of Llama-style decoders, Vision Transformers,
ConvNeXt, Swin and Conformer, plus mixture-of-experts, FiLM conditioning and LoRA.

Each block's PyTorch helper is defined here and added to the shared helper table; the
simpler ones (rotary embedding, gated feed-forward, LayerScale, DropPath) also have a
Keras form. Helpers use ``nn.functional`` (not ``F``) because the generated file
imports ``F`` only when a layer line uses it.
"""
from __future__ import annotations

from dataclasses import replace

from ai_made_easy.core.blocks._dsl import P, ShapeError, nn_block
from ai_made_easy.core.codegen.helpers import KERAS_HELPERS, PYTORCH_HELPERS
from ai_made_easy.core.registry import get_registry

TRANSFORMER = "Transformer Parts"   # attention family colour
VISION = "Vision Blocks"            # convolution family colour
SEQUENCE = "Sequence Models"        # recurrent family colour (shared with TCN, Mamba)

# ------------------------------------------------------------------ PyTorch helpers

TORCH = {
    "RotaryEmbedding": '''\
class RotaryEmbedding(nn.Module):
    """Rotary position embedding (Su et al., 2021): rotates feature pairs by position."""

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        inv = 1.0 / base ** (torch.arange(0, dim, 2).float() / dim)
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [..., L, D]
        pos = torch.arange(x.size(-2), device=x.device, dtype=self.inv_freq.dtype)
        angles = torch.outer(pos, self.inv_freq)
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        pairs = x.unflatten(-1, (-1, 2))
        x1, x2 = pairs[..., 0], pairs[..., 1]
        return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
''',
    "GatedFeedForward": '''\
class GatedFeedForward(nn.Module):
    """Gated feed-forward (Shazeer, 2020): SwiGLU with SiLU, GeGLU with GELU."""

    def __init__(self, dim: int, hidden: int, act: str = "silu", dropout: float = 0.0) -> None:
        super().__init__()
        self.gate_up = nn.Linear(dim, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.act = nn.SiLU() if act == "silu" else nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(self.drop(self.act(gate) * up))
''',
    "GroupedQueryAttention": '''\
class GroupedQueryAttention(nn.Module):
    """Self-attention where groups of query heads share key/value heads (Ainslie et al.,
    2023); kv_heads = heads is ordinary multi-head attention, kv_heads = 1 multi-query."""

    def __init__(self, dim: int, heads: int, kv_heads: int, causal: bool = True,
                 rope: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, dim // heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, 2 * kv_heads * self.head_dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.causal, self.dropout = causal, dropout
        self.rope = RotaryEmbedding(self.head_dim) if rope else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, L, D]
        b, n, _ = x.shape
        q = self.q(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(x).view(b, n, 2, self.kv_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)
        repeat = self.heads // self.kv_heads
        k, v = k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1)
        y = nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=self.causal, dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(b, n, -1))
''',
    "LlamaBlock": '''\
class LlamaBlock(nn.Module):
    """Pre-norm decoder block: RMSNorm, grouped-query attention with rotary positions,
    RMSNorm, SwiGLU feed-forward, each with a residual (Llama / Mistral style)."""

    def __init__(self, dim: int, heads: int, kv_heads: int, hidden: int,
                 causal: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.attn = GroupedQueryAttention(dim, heads, kv_heads, causal, True, dropout)
        self.ffn = GatedFeedForward(dim, hidden, "silu", dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))
''',
    "MixtureOfExperts": '''\
class MixtureOfExperts(nn.Module):
    """Top-k routed SwiGLU experts with a load-balancing loss (Switch / Mixtral style).
    Each token runs through its k best experts only; aux_loss() keeps the load even."""

    def __init__(self, dim: int, hidden: int, experts: int = 8, top_k: int = 2,
                 balance: float = 0.01) -> None:
        super().__init__()
        self.router = nn.Linear(dim, experts, bias=False)
        self.experts = nn.ModuleList(GatedFeedForward(dim, hidden) for _ in range(experts))
        self.top_k, self.balance = top_k, balance
        self._aux = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, x.size(-1))
        probs = self.router(flat).float().softmax(dim=-1)
        weight, index = probs.topk(self.top_k, dim=-1)
        weight = weight / weight.sum(dim=-1, keepdim=True)
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            rows, slot = (index == e).nonzero(as_tuple=True)
            if rows.numel():
                y = expert(flat[rows]) * weight[rows, slot, None].to(flat.dtype)
                out.index_add_(0, rows, y.to(out.dtype))
        counts = torch.zeros(probs.size(1), device=x.device)
        counts.scatter_add_(0, index.reshape(-1), torch.ones(index.numel(), device=x.device))
        self._aux = self.balance * probs.size(1) * (counts / index.numel() * probs.mean(0)).sum()
        return out.reshape(x.shape)

    def aux_loss(self) -> torch.Tensor:
        return self._aux if self._aux is not None else torch.zeros(())
''',
    "LayerScale": '''\
class LayerScale(nn.Module):
    """Learnable per-feature scale, initialised small (Touvron et al., 2021)."""

    def __init__(self, dim: int, init: float = 0.1) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), float(init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma
''',
    "DropPath": '''\
class DropPath(nn.Module):
    """Stochastic depth (Huang et al., 2016): drops the whole input per sample."""

    def __init__(self, p: float = 0.1) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.size(0),) + (1,) * (x.dim() - 1)).bernoulli_(keep)
        return x * mask / keep
''',
    "FiLM": '''\
class FiLM(nn.Module):
    """Feature-wise linear modulation (Perez et al., 2018): x * (1 + gamma(c)) + beta(c).
    Starts as the identity (zero-initialised projection)."""

    def __init__(self, cond_dim: int, features: int, channels_first: bool) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * features)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.channels_first = channels_first

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(c.flatten(1)).chunk(2, dim=-1)
        extra = (1,) * (x.dim() - 2)
        shape = (x.size(0), -1, *extra) if self.channels_first else (x.size(0), *extra, -1)
        return x * (1 + gamma.view(shape)) + beta.view(shape)
''',
    "LoRALinear": '''\
class LoRALinear(nn.Module):
    """A Dense layer plus a low-rank adapter (Hu et al., 2021): base(x) + (alpha / r) B A x.
    With freeze_base the base weights stay fixed and only the adapter trains."""

    def __init__(self, fin: int, fout: int, rank: int = 8, alpha: float = 16.0,
                 freeze_base: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = nn.Linear(fin, fout)
        self.base.requires_grad_(not freeze_base)
        self.lora_a = nn.Parameter(torch.randn(rank, fin) / rank)
        self.lora_b = nn.Parameter(torch.zeros(fout, rank))
        self.scale, self.drop = alpha / rank, nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (self.drop(x) @ self.lora_a.t() @ self.lora_b.t()) * self.scale
''',
    "PatchEmbedding": '''\
class PatchEmbedding(nn.Module):
    """ViT patch embedding (Dosovitskiy et al., 2021): [C, H, W] -> [tokens, dim], with an
    optional class token first and learned position embeddings."""

    def __init__(self, cin: int, dim: int, patch: int, tokens: int,
                 cls_token: bool = True) -> None:
        super().__init__()
        self.proj = nn.Conv2d(cin, dim, patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim)) if cls_token else None
        self.pos = nn.Parameter(torch.zeros(1, tokens + int(cls_token), dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.cls is not None:
            x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], dim=1)
        return x + self.pos
''',
    "ConvNeXtBlock": '''\
class ConvNeXtBlock(nn.Module):
    """ConvNeXt block (Liu et al., 2022): 7x7 depthwise conv, LayerNorm, 4x MLP with GELU,
    LayerScale and stochastic depth, residual. [C, H, W] -> [C, H, W]."""

    def __init__(self, dim: int, expansion: int = 4, layer_scale: float = 1e-6,
                 drop_path: float = 0.0) -> None:
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1, self.pw2 = nn.Linear(dim, expansion * dim), nn.Linear(expansion * dim, dim)
        self.act = nn.GELU()
        self.gamma = nn.Parameter(torch.full((dim,), layer_scale)) if layer_scale > 0 else None
        self.drop = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pw2(self.act(self.pw1(self.norm(self.dw(x).permute(0, 2, 3, 1)))))
        if self.gamma is not None:
            y = y * self.gamma
        return x + self.drop(y.permute(0, 3, 1, 2))
''',
    "SwinBlock": '''\
class SwinBlock(nn.Module):
    """Swin Transformer block (Liu et al., 2021): attention inside (shifted) windows with a
    relative position bias, then an MLP; pre-norm residuals. [C, H, W] -> [C, H, W]."""

    def __init__(self, dim: int, heads: int, window: int = 7, shift: bool = False,
                 mlp_ratio: float = 4.0, drop_path: float = 0.0) -> None:
        super().__init__()
        self.window, self.shift, self.heads = window, window // 2 if shift else 0, heads
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.bias = nn.Parameter(torch.zeros(heads, (2 * window - 1) ** 2))
        grid = torch.stack(torch.meshgrid(torch.arange(window), torch.arange(window),
                                          indexing="ij")).flatten(1)
        rel = (grid[:, :, None] - grid[:, None, :]).permute(1, 2, 0) + window - 1
        self.register_buffer("rel_index", rel[..., 0] * (2 * window - 1) + rel[..., 1],
                             persistent=False)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.drop = DropPath(drop_path)

    def _windows(self, t: torch.Tensor) -> torch.Tensor:  # [B, H, W, C] -> [B*nW, w*w, C]
        b, h, w, c = t.shape
        ws = self.window
        t = t.view(b, h // ws, ws, w // ws, ws, c).permute(0, 1, 3, 2, 4, 5)
        return t.reshape(-1, ws * ws, c)

    def _mask(self, h: int, w: int, device) -> torch.Tensor:
        ws, s = self.window, self.shift
        region = torch.zeros(1, h, w, 1, device=device)
        count = 0
        for rows in (slice(0, -ws), slice(-ws, -s), slice(-s, None)):
            for cols in (slice(0, -ws), slice(-ws, -s), slice(-s, None)):
                region[:, rows, cols, :] = count
                count += 1
        ids = self._windows(region)[..., 0]
        diff = ids.unsqueeze(1) - ids.unsqueeze(2)
        return torch.zeros_like(diff).masked_fill(diff != 0, float("-inf"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, C, H, W]
        b, c, h, w = x.shape
        ws, heads = self.window, self.heads
        t = x.permute(0, 2, 3, 1)
        y = self.norm1(t)
        if self.shift:
            y = torch.roll(y, (-self.shift, -self.shift), dims=(1, 2))
        win = self._windows(y)
        n = win.size(1)
        q, k, v = self.qkv(win).view(-1, n, 3, heads, c // heads).permute(2, 0, 3, 1, 4)
        bias = self.bias[:, self.rel_index].unsqueeze(0)  # [1, heads, n, n]
        if self.shift:
            mask = self._mask(h, w, x.device)
            bias = bias + mask.repeat(b, 1, 1).unsqueeze(1)
        y = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        y = self.proj(y.transpose(1, 2).reshape(-1, n, c))
        y = y.view(b, h // ws, w // ws, ws, ws, c).permute(0, 1, 3, 2, 4, 5).reshape(b, h, w, c)
        if self.shift:
            y = torch.roll(y, (self.shift, self.shift), dims=(1, 2))
        t = t + self.drop(y)
        t = t + self.drop(self.mlp(self.norm2(t)))
        return t.permute(0, 3, 1, 2)
''',
    "ConformerBlock": '''\
class ConformerBlock(nn.Module):
    """Conformer block (Gulati et al., 2020): half feed-forward, self-attention, a gated
    depthwise convolution module, half feed-forward, LayerNorm. [L, D] -> [L, D]."""

    def __init__(self, dim: int, heads: int = 4, ff_mult: int = 4, kernel: int = 31,
                 dropout: float = 0.1) -> None:
        super().__init__()

        def feed_forward() -> nn.Module:
            return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, ff_mult * dim), nn.SiLU(),
                                 nn.Dropout(dropout), nn.Linear(ff_mult * dim, dim),
                                 nn.Dropout(dropout))

        self.ff1, self.ff2 = feed_forward(), feed_forward()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.conv_norm = nn.LayerNorm(dim)
        self.pw1 = nn.Conv1d(dim, 2 * dim, 1)
        self.dw = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.bn = nn.BatchNorm1d(dim)
        self.pw2 = nn.Conv1d(dim, dim, 1)
        self.drop = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, L, D]
        x = x + 0.5 * self.ff1(x)
        y = self.attn_norm(x)
        x = x + self.drop(self.attn(y, y, y, need_weights=False)[0])
        y = nn.functional.glu(self.pw1(self.conv_norm(x).transpose(1, 2)), dim=1)
        y = self.pw2(nn.functional.silu(self.bn(self.dw(y))))
        x = x + self.drop(y.transpose(1, 2))
        return self.out_norm(x + 0.5 * self.ff2(x))
''',
}

KERAS = {
    "rotary_embedding": '''\
class RotaryEmbedding(layers.Layer):
    """Rotary position embedding over [batch, length, dim] (dim even)."""

    def __init__(self, base: float = 10000.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.base = base

    def call(self, x):
        length, dim = x.shape[1], x.shape[2]
        inv = 1.0 / ops.power(self.base, ops.arange(0, dim, 2, dtype="float32") / dim)
        angles = ops.expand_dims(ops.arange(length, dtype="float32"), 1) * inv
        cos, sin = ops.cast(ops.cos(angles), x.dtype), ops.cast(ops.sin(angles), x.dtype)
        pairs = ops.reshape(x, (-1, length, dim // 2, 2))
        x1, x2 = pairs[..., 0], pairs[..., 1]
        out = ops.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
        return ops.reshape(out, (-1, length, dim))
''',
    "gated_feed_forward": '''\
def gated_feed_forward(x, hidden: int, activation: str = "silu", dropout: float = 0.0):
    """SwiGLU / GeGLU feed-forward over the last axis."""
    dim = x.shape[-1]
    gate = layers.Dense(hidden, use_bias=False, activation=activation)(x)
    up = layers.Dense(hidden, use_bias=False)(x)
    h = layers.Dropout(dropout)(layers.Multiply()([gate, up]))
    return layers.Dense(dim, use_bias=False)(h)
''',
    "layer_scale": '''\
class LayerScale(layers.Layer):
    """Learnable per-feature scale over the last axis."""

    def __init__(self, init: float = 0.1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.init = init

    def build(self, input_shape) -> None:
        self.gamma = self.add_weight(shape=(input_shape[-1],),
                                     initializer=keras.initializers.Constant(self.init))

    def call(self, x):
        return x * self.gamma
''',
    "drop_path": '''\
class DropPath(layers.Layer):
    """Stochastic depth: drops the whole input per sample while training."""

    def __init__(self, rate: float = 0.1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.rate = rate
        self.seed = keras.random.SeedGenerator(0)

    def call(self, x, training=None):
        if not training or self.rate == 0.0:
            return x
        keep = 1.0 - self.rate
        shape = (ops.shape(x)[0],) + (1,) * (len(x.shape) - 1)
        mask = ops.floor(keep + keras.random.uniform(shape, seed=self.seed))
        return x * mask / keep
''',
}

PYTORCH_HELPERS.update(TORCH)
KERAS_HELPERS.update(KERAS)


# ------------------------------------------------------------------ shapes and costs

def _seq(s, name: str) -> None:
    if len(s) != 2:
        raise ShapeError(f"{name} expects a batch-first sequence [L, D], got {s}")


def _heads(d: int, heads: int, name: str) -> None:
    if heads < 1 or d % heads:
        raise ShapeError(f"{name}: the width D={d} is not divisible by {heads} heads")


def _hidden(d: int, p) -> int:
    """0 = the Llama default, 8/3 x D rounded up to a multiple of 8."""
    h = int(p.get("hidden") or 0)
    return h or -(-(8 * d // 3) // 8) * 8


def _rope_shape(in_shapes, p):
    s = in_shapes[0]
    _seq(s, "Rotary Embedding")
    if s[1] % 2:
        raise ShapeError(f"Rotary Embedding rotates feature pairs: D={s[1]} must be even")
    return list(s)


def _gqa_check(s, p, name: str) -> None:
    _seq(s, name)
    heads, kv = int(p["heads"]), int(p["kv_heads"])
    _heads(s[1], heads, name)
    if kv < 1 or heads % kv:
        raise ShapeError(f"{name}: kv_heads={kv} must divide heads={heads}")
    if (s[1] // heads) % 2:
        raise ShapeError(f"{name}: the head width {s[1] // heads} must be even for rotary "
                         "positions")


def _gqa_params(d: int, p) -> int:
    hd = d // int(p["heads"])
    return 2 * d * d + 2 * d * int(p["kv_heads"]) * hd


def _gqa_shape(in_shapes, p):
    _gqa_check(in_shapes[0], p, "Grouped-Query Attention")
    return list(in_shapes[0])


def _attn_macs(s, params: int) -> int:
    n, d = s
    return n * params + 2 * n * n * d


def _llama_shape(in_shapes, p):
    _gqa_check(in_shapes[0], p, "Llama Block")
    return list(in_shapes[0])


def _llama_params(s, p) -> int:
    d = s[-1]
    return _gqa_params(d, p) + 3 * d * _hidden(d, p) + 2 * d


def _last_dim_shape(name: str, ranks=(1, 2)):
    def fn(in_shapes, p):
        s = in_shapes[0]
        if len(s) not in ranks:
            raise ShapeError(f"{name} expects [D] or [L, D], got {s}")
        return list(s)

    return fn


def _moe_shape(in_shapes, p):
    s = _last_dim_shape("Mixture of Experts")(in_shapes, p)
    if not 1 <= int(p["top_k"]) <= int(p["experts"]):
        raise ShapeError(f"top_k={p['top_k']} must be between 1 and experts={p['experts']}")
    return s


def _moe_params(s, p) -> int:
    d = s[0][-1]
    return d * int(p["experts"]) + int(p["experts"]) * 3 * d * _hidden(d, p)


def _moe_cost(s, p):
    d, tokens = s[0][-1], (s[0][0] if len(s[0]) == 2 else 1)
    active = int(p["top_k"]) * 3 * d * _hidden(d, p) + d * int(p["experts"])
    return _moe_params(s, p), 2 * tokens * active


def _film_shape(in_shapes, p):
    x, c = in_shapes
    if len(x) not in (1, 2, 3, 4):
        raise ShapeError(f"FiLM modulates [D], [L, D], [C, H, W] or [C, D, H, W], got {x}")
    if not c:
        raise ShapeError("FiLM needs a condition vector on its second input")
    return list(x)


def _film_features(x) -> tuple[int, bool]:
    if len(x) >= 3:
        return x[0], True      # channels-first images / volumes
    return x[-1], False         # vectors and sequences: the last axis


def _film_torch(c):
    x, cond = c["in_shapes"]
    features, first = _film_features(x)
    width = 1
    for d in cond:
        width *= int(d)
    return f"FiLM({width}, {features}, channels_first={first})"


def _film_params(s, p) -> int:
    x, cond = s
    width = 1
    for d in cond:
        width *= int(d)
    return (width + 1) * 2 * _film_features(x)[0]


def _lora_shape(in_shapes, p):
    s = in_shapes[0]
    if len(s) not in (1, 2):
        raise ShapeError(f"LoRA Dense expects [F] or [L, F], got {s}")
    return [*s[:-1], int(p["units"])]


def _lora_params(s, p) -> int:
    fin, fout, r = s[0][-1], int(p["units"]), int(p["rank"])
    return fin * fout + fout + r * (fin + fout)


def _image(s, name: str) -> None:
    if len(s) != 3:
        raise ShapeError(f"{name} expects an image [C, H, W], got {s}")


def _patch_shape(in_shapes, p):
    s = in_shapes[0]
    _image(s, "Patch Embedding")
    patch = int(p["patch"])
    if s[1] % patch or s[2] % patch:
        raise ShapeError(f"Patch Embedding: {s[1]}×{s[2]} is not divisible into "
                         f"{patch}×{patch} patches")
    tokens = (s[1] // patch) * (s[2] // patch) + int(bool(p["cls_token"]))
    return [tokens, int(p["dim"])]


def _patch_torch(c):
    s, patch = c["input_shape"], int(c["patch"])
    tokens = (s[1] // patch) * (s[2] // patch)
    return (f"PatchEmbedding({s[0]}, {int(c['dim'])}, {patch}, {tokens}, "
            f"cls_token={bool(c['cls_token'])})")


def _patch_cost(s, p):
    c, h, w = s[0]
    patch, dim = int(p["patch"]), int(p["dim"])
    tokens = (h // patch) * (w // patch)
    conv = c * patch * patch * dim
    params = conv + dim + (tokens + int(bool(p["cls_token"]))) * dim + (
        dim if p["cls_token"] else 0)
    return params, 2 * tokens * conv


def _convnext_shape(in_shapes, p):
    _image(in_shapes[0], "ConvNeXt Block")
    return list(in_shapes[0])


def _convnext_cost(s, p):
    c, h, w = s[0]
    e = int(p["expansion"])
    params = c * 49 + c + 2 * c + (c * e * c + e * c) + (e * c * c + c) + c
    return params, 2 * h * w * (49 * c + 2 * e * c * c)


def _swin_shape(in_shapes, p):
    s = in_shapes[0]
    _image(s, "Swin Block")
    window = int(p["window"])
    if s[1] % window or s[2] % window:
        raise ShapeError(f"Swin Block: {s[1]}×{s[2]} is not divisible into {window}×{window} "
                         "windows")
    if p["shift"] and window >= min(s[1], s[2]):
        raise ShapeError("Swin Block: shifted windows need more than one window per side; "
                         "turn shift off or use a smaller window")
    _heads(s[0], int(p["heads"]), "Swin Block")
    return list(s)


def _swin_cost(s, p):
    c, h, w = s[0]
    hidden, window = int(c * float(p["mlp_ratio"])), int(p["window"])
    params = 4 * c + (3 * c * c + 3 * c) + (c * c + c) + 2 * c * hidden + hidden + c + int(
        p["heads"]) * (2 * window - 1) ** 2
    n = h * w
    return params, 2 * (n * (4 * c * c + 2 * c * hidden) + 2 * n * window * window * c)


def _conformer_shape(in_shapes, p):
    s = in_shapes[0]
    _seq(s, "Conformer Block")
    _heads(s[1], int(p["heads"]), "Conformer Block")
    if int(p["kernel_size"]) % 2 == 0:
        raise ShapeError("Conformer Block: kernel_size must be odd (same-length output)")
    return list(s)


def _conformer_cost(s, p):
    n, d = s[0]
    m, k = int(p["ff_mult"]), int(p["kernel_size"])
    ff = 2 * (2 * d + d * m * d + m * d + m * d * d + d)
    attn = 2 * d + 4 * d * d + 4 * d
    conv = 2 * d + (d * 2 * d + 2 * d) + (d * k + d) + 2 * d + (d * d + d)
    params = ff + attn + conv + 2 * d
    return params, 2 * n * (2 * m * d * d * 2 + 4 * d * d + 3 * d * d + k * d) + 4 * n * n * d


def _cost(defn, cost):
    return replace(defn, meta={**(defn.meta or {}), "cost": cost})


def _seq_cost(param_fn):
    def fn(s, p):
        params = param_fn(s, p)
        n = s[0][0] if len(s[0]) == 2 else 1
        return params, 2 * n * params
    return fn


# ------------------------------------------------------------------ blocks

def _blocks() -> list:
    rope = nn_block(
        "core.rotary_embedding", "Rotary Embedding (RoPE)", TRANSFORMER, family="attention",
        params=(P("base", "float", 10000.0, lo=1.0, help="Frequency base; larger = longer "
                                                         "contexts"),),
        shape=_rope_shape, layout="ir", param_fn=lambda s, p: 0,
        torch=lambda c: f"RotaryEmbedding({c['input_size']}, base={float(c['base'])})",
        torch_helpers=("RotaryEmbedding",),
        keras_expr=lambda c: f"RotaryEmbedding(base={float(c['base'])})({c['i0']})",
        keras_helpers=("rotary_embedding",),
        desc="Encodes each position by rotating feature pairs ([L, D] → [L, D]). Relative "
             "positions fall out of the attention dot product; no learned table.")
    gqa = nn_block(
        "core.gqa_attention", "Grouped-Query Attention", TRANSFORMER, family="attention",
        params=(P("heads", "int", 8, lo=1), P("kv_heads", "int", 2, lo=1,
                                                help="Key/value heads; must divide heads "
                                                     "(1 = multi-query)"),
                P("causal", "bool", True, help="Each position sees only earlier ones"),
                P("rope", "bool", True, help="Rotary positions on queries and keys"),
                P("dropout", "float", 0.0, lo=0.0, hi=1.0)),
        shape=_gqa_shape, layout="ir", param_fn=lambda s, p: _gqa_params(s[0][-1], p),
        torch=lambda c: (f"GroupedQueryAttention({c['input_size']}, {int(c['heads'])}, "
                         f"{int(c['kv_heads'])}, causal={bool(c['causal'])}, "
                         f"rope={bool(c['rope'])}, dropout={float(c['dropout'])})"),
        torch_helpers=("RotaryEmbedding", "GroupedQueryAttention"),
        desc="Self-attention where several query heads share each key/value head: most of "
             "multi-head quality with a fraction of the key/value cache (Llama 2/3, Mistral).")
    ffn = nn_block(
        "core.gated_ffn", "Gated Feed-Forward (SwiGLU)", TRANSFORMER, family="attention",
        params=(P("hidden", "int", 0, lo=0, help="0 = 8/3 × D, rounded to a multiple of 8"),
                P("activation", "enum", "silu", options=("silu", "gelu"),
                  help="silu = SwiGLU, gelu = GeGLU"),
                P("dropout", "float", 0.0, lo=0.0, hi=1.0)),
        shape=_last_dim_shape("Gated Feed-Forward"), layout="ir",
        param_fn=lambda s, p: 3 * s[0][-1] * _hidden(s[0][-1], p),
        torch=lambda c: (f"GatedFeedForward({c['input_size']}, {_hidden(c['input_size'], c)}, "
                         f"act=\"{c['activation']}\", dropout={float(c['dropout'])})"),
        torch_helpers=("GatedFeedForward",),
        keras_expr=lambda c: (f"gated_feed_forward({c['i0']}, {_hidden(c['input_size'], c)}, "
                              f"activation=\"{c['activation']}\", "
                              f"dropout={float(c['dropout'])})"),
        keras_helpers=("gated_feed_forward",),
        desc="The transformer feed-forward of modern LLMs: a gate branch times an up branch, "
             "then down to D ([.., D] → [.., D]).")
    llama = nn_block(
        "core.llama_block", "Llama Decoder Block", TRANSFORMER, family="attention",
        params=(P("heads", "int", 4, lo=1), P("kv_heads", "int", 2, lo=1),
                P("hidden", "int", 0, lo=0, help="Feed-forward width; 0 = 8/3 × D"),
                P("causal", "bool", True), P("dropout", "float", 0.0, lo=0.0, hi=1.0)),
        shape=_llama_shape, layout="ir", param_fn=lambda s, p: _llama_params(s[0], p),
        torch=lambda c: (f"LlamaBlock({c['input_size']}, {int(c['heads'])}, "
                         f"{int(c['kv_heads'])}, {_hidden(c['input_size'], c)}, "
                         f"causal={bool(c['causal'])}, dropout={float(c['dropout'])})"),
        torch_helpers=("RMSNorm", "RotaryEmbedding", "GatedFeedForward",
                       "GroupedQueryAttention", "LlamaBlock"),
        desc="One Llama-style decoder layer: RMSNorm → grouped-query attention with rotary "
             "positions → RMSNorm → SwiGLU, with residuals ([L, D] → [L, D]). Stack a few "
             "after an Embedding for a small modern language model.")
    moe = nn_block(
        "core.mixture_of_experts", "Mixture of Experts", TRANSFORMER, family="attention",
        params=(P("experts", "int", 8, lo=1, hi=128), P("top_k", "int", 2, lo=1,
                                                        help="Experts each token uses"),
                P("hidden", "int", 0, lo=0, help="Each expert's width; 0 = 8/3 × D"),
                P("balance", "float", 0.01, lo=0.0, hi=1.0,
                  help="Weight of the load-balancing loss")),
        shape=_moe_shape, layout="ir", param_fn=lambda s, p: _moe_params(s, p),
        torch=lambda c: (f"MixtureOfExperts({c['input_size']}, {_hidden(c['input_size'], c)}, "
                         f"experts={int(c['experts'])}, top_k={int(c['top_k'])}, "
                         f"balance={float(c['balance'])})"),
        torch_helpers=("GatedFeedForward", "MixtureOfExperts"),
        desc="A router sends each token to its top-k of N SwiGLU experts: many more "
             "parameters for the same compute per token. Adds a load-balancing loss so every "
             "expert is used ([.., D] → [.., D]).")
    scale = nn_block(
        "core.layer_scale", "LayerScale", TRANSFORMER, family="attention",
        params=(P("init", "float", 0.1, lo=0.0, help="Initial scale (1e-5 for deep "
                                                     "networks)"),),
        shape=_last_dim_shape("LayerScale"), layout="ir", param_fn=lambda s, p: s[0][-1],
        torch=lambda c: f"LayerScale({c['input_size']}, init={float(c['init'])})",
        torch_helpers=("LayerScale",),
        keras_expr=lambda c: f"LayerScale(init={float(c['init'])})({c['i0']})",
        keras_helpers=("layer_scale",),
        desc="Multiplies each feature by a learned scale that starts small, so a residual "
             "branch begins close to the identity (CaiT, DeiT III).")
    film = nn_block(
        "core.film", "FiLM Conditioning", TRANSFORMER, family="attention",
        inputs=("in", "condition"), shape=_film_shape, param_fn=_film_params,
        torch=_film_torch, torch_expr="self.{self_var}({i0}, {i1})", torch_helpers=("FiLM",),
        desc="Modulates features with a condition vector: x · (1 + γ(c)) + β(c), per "
             "channel. Conditions a network on a class, a time step or any side input; "
             "starts as the identity.")
    drop_path = nn_block(
        "core.drop_path", "DropPath (Stochastic Depth)", "Regularization",
        family="regularization",
        params=(P("p", "float", 0.1, lo=0.0, hi=0.99, help="Probability a sample's path is "
                                                           "dropped"),),
        shape=lambda s, p: list(s[0]), param_fn=lambda s, p: 0,
        torch=lambda c: f"DropPath({float(c['p'])})", torch_helpers=("DropPath",),
        keras_expr=lambda c: f"DropPath({float(c['p'])})({c['i0']})",
        keras_helpers=("drop_path",),
        desc="Drops a residual branch for whole samples while training (and rescales the "
             "rest). Put it on a branch before the Add; regularises deep ViTs and ConvNeXts.")
    lora = nn_block(
        "core.lora_dense", "LoRA Dense", "Linear", family="model",
        params=(P("units", "int", 64, lo=1), P("rank", "int", 8, lo=1, hi=512),
                P("alpha", "float", 16.0, lo=0.0),
                P("freeze_base", "bool", True, help="Train only the low-rank adapter"),
                P("dropout", "float", 0.0, lo=0.0, hi=1.0)),
        shape=_lora_shape, param_fn=_lora_params,
        torch=lambda c: (f"LoRALinear({c['input_size']}, {int(c['units'])}, "
                         f"rank={int(c['rank'])}, alpha={float(c['alpha'])}, "
                         f"freeze_base={bool(c['freeze_base'])}, "
                         f"dropout={float(c['dropout'])})"),
        torch_helpers=("LoRALinear",),
        desc="A Dense layer plus a trainable low-rank update. In a pipeline fine-tune stage "
             "the base keeps its loaded weights and only r × (in + out) numbers train.")
    patch = nn_block(
        "vision.patch_embedding", "Patch Embedding (ViT)", VISION, family="conv",
        params=(P("patch", "int", 4, lo=1, help="Patch side in pixels"),
                P("dim", "int", 128, lo=1, help="Token width"),
                P("cls_token", "bool", True, help="Prepend a learned class token")),
        shape=_patch_shape, param_fn=lambda s, p: _patch_cost(s, p)[0],
        torch=_patch_torch, torch_helpers=("PatchEmbedding",),
        desc="Cuts an image into patches and embeds each as a token, with learned "
             "positions: [C, H, W] → [tokens, dim]. Follow with transformer layers for a "
             "Vision Transformer.")
    convnext = nn_block(
        "vision.convnext_block", "ConvNeXt Block", VISION, family="conv",
        params=(P("expansion", "int", 4, lo=1), P("layer_scale", "float", 1e-6, lo=0.0),
                P("drop_path", "float", 0.0, lo=0.0, hi=0.99)),
        shape=_convnext_shape, param_fn=lambda s, p: _convnext_cost(s, p)[0],
        torch=lambda c: (f"ConvNeXtBlock({c['in_channels']}, expansion={int(c['expansion'])}, "
                         f"layer_scale={float(c['layer_scale'])}, "
                         f"drop_path={float(c['drop_path'])})"),
        torch_helpers=("DropPath", "ConvNeXtBlock"),
        desc="A modern residual conv block: 7×7 depthwise conv, LayerNorm, an inverted MLP "
             "with GELU, LayerScale ([C, H, W] → [C, H, W]).")
    swin = nn_block(
        "vision.swin_block", "Swin Transformer Block", VISION, family="conv",
        params=(P("heads", "int", 4, lo=1), P("window", "int", 4, lo=1),
                P("shift", "bool", False, help="Shifted windows: alternate off / on between "
                                               "blocks so windows talk to each other"),
                P("mlp_ratio", "float", 4.0, lo=0.5), P("drop_path", "float", 0.0, lo=0.0,
                                                         hi=0.99)),
        shape=_swin_shape, param_fn=lambda s, p: _swin_cost(s, p)[0],
        torch=lambda c: (f"SwinBlock({c['in_channels']}, {int(c['heads'])}, "
                         f"window={int(c['window'])}, shift={bool(c['shift'])}, "
                         f"mlp_ratio={float(c['mlp_ratio'])}, "
                         f"drop_path={float(c['drop_path'])})"),
        torch_helpers=("DropPath", "SwinBlock"),
        desc="Self-attention inside local windows with a relative position bias, then an MLP "
             "([C, H, W] → [C, H, W]). Linear in image size, unlike a plain ViT.")
    conformer = nn_block(
        "seq.conformer", "Conformer Block", SEQUENCE, family="recurrent",
        params=(P("heads", "int", 4, lo=1), P("ff_mult", "int", 4, lo=1),
                P("kernel_size", "int", 15, lo=1, help="Depthwise convolution width (odd)"),
                P("dropout", "float", 0.1, lo=0.0, hi=1.0)),
        shape=_conformer_shape, layout="ir", param_fn=lambda s, p: _conformer_cost(s, p)[0],
        torch=lambda c: (f"ConformerBlock({c['input_size']}, heads={int(c['heads'])}, "
                         f"ff_mult={int(c['ff_mult'])}, kernel={int(c['kernel_size'])}, "
                         f"dropout={float(c['dropout'])})"),
        torch_helpers=("ConformerBlock",),
        desc="Attention for global context plus a convolution module for local detail "
             "([L, D] → [L, D]): the standard speech-recognition encoder layer.")
    return [
        _cost(rope, lambda s, p: (0, 3 * s[0][0] * s[0][1])),
        _cost(gqa, lambda s, p: (_gqa_params(s[0][-1], p),
                                 2 * _attn_macs(s[0], _gqa_params(s[0][-1], p)))),
        _cost(ffn, _seq_cost(lambda s, p: 3 * s[0][-1] * _hidden(s[0][-1], p))),
        _cost(llama, lambda s, p: (_llama_params(s[0], p),
                                   2 * _attn_macs(s[0], _llama_params(s[0], p)))),
        replace(moe, meta={"cost": _moe_cost, "aux_loss": True}),
        _cost(scale, lambda s, p: (s[0][-1], 0)),
        _cost(film, lambda s, p: (_film_params(s, p), 2 * _film_params(s, p))),
        _cost(drop_path, lambda s, p: (0, 0)),
        _cost(lora, _seq_cost(lambda s, p: _lora_params(s, p))),
        _cost(patch, _patch_cost),
        _cost(convnext, _convnext_cost),
        _cost(swin, _swin_cost),
        _cost(conformer, _conformer_cost),
    ]


def register_all() -> None:
    reg = get_registry()
    for defn in _blocks():
        reg.register(defn)


register_all()
