"""More layers (3.8): attention from recent language models, linear-time sequence mixers,
and mobile vision blocks.

Multi-head latent attention, QK-norm attention and gated attention; Mamba-2, Gated
DeltaNet, gated linear attention and Griffin's RG-LRU block; MobileNetV4's universal
inverted bottleneck and StarNet's star block. Every one is checked by the codegen harness
(designed shape = real shape) and against PyTorch's parameter count. The sequence mixers
run their recurrence as a loop over time: clear and exact, and fine for the lengths a
design on the canvas trains on. Helpers use ``nn.functional`` (not ``F``) because the
generated file imports ``F`` only when a layer line uses it.
"""
from __future__ import annotations

from ai_made_easy.core.blocks._dsl import P, ShapeError, nn_block
from ai_made_easy.core.codegen.helpers import KERAS_HELPERS, PYTORCH_HELPERS
from ai_made_easy.core.registry import get_registry

reg = get_registry()
_BEFORE = {b.type_id for b in reg.all()}

TRANSFORMER = "Transformer Parts"
SEQUENCE = "Sequence Models"
VISION = "Vision Blocks"

_RMS = '''\
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def causal_mask(n: int, device) -> torch.Tensor:
    return torch.ones(n, n, dtype=torch.bool, device=device).tril()
'''

_CHUNK_SCAN = '''\
def to_chunks(x: torch.Tensor, chunk: int) -> torch.Tensor:
    """[B, L, ...] -> [B, L / chunk, chunk, ...], zero-padded at the end (causal layers:
    padding never reaches earlier steps)."""
    n = x.shape[1]
    pad = -n % chunk
    if pad:
        x = torch.cat([x, x.new_zeros(x.shape[0], pad, *x.shape[2:])], 1)
    return x.reshape(x.shape[0], -1, chunk, *x.shape[2:])


def from_chunks(x: torch.Tensor, n: int) -> torch.Tensor:
    return x.reshape(x.shape[0], -1, *x.shape[3:])[:, :n]


def decay_matrix(logs: torch.Tensor) -> torch.Tensor:
    """Cumulative log decays [..., C] -> [..., C, C] with exp(L_t - L_j) for j <= t, else 0."""
    c = logs.shape[-1]
    keep = torch.ones(c, c, dtype=torch.bool, device=logs.device).tril()
    diff = logs[..., :, None] - logs[..., None, :]
    return torch.exp(diff.masked_fill(~keep, float("-inf")))


def linear_scan(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """h_t = a_t * h_(t-1) + x_t along dim 1 (h_(-1) = 0), elementwise, in log2(L) steps
    (a parallel prefix scan: each step composes every position with the one d back)."""
    n, d = x.shape[1], 1
    while d < n:
        x = torch.cat([x[:, :d], x[:, d:] + a[:, d:] * x[:, :-d]], 1)
        a = torch.cat([a[:, :d], a[:, d:] * a[:, :-d]], 1)
        d *= 2
    return x
'''

TORCH = {
    "rms_norm": _RMS,
    "chunk_scan": _CHUNK_SCAN,
    "LatentAttention": '''\
class LatentAttention(nn.Module):
    """Multi-head latent attention (DeepSeek-V2, 2024): keys and values are rebuilt from one
    small shared latent per position, so a decoder caches the latent instead of every
    head's keys and values (without the decoupled rotary part)."""

    def __init__(self, dim: int, heads: int = 4, latent: int = 32, causal: bool = True) -> None:
        super().__init__()
        self.heads, self.causal = heads, causal
        self.q = nn.Linear(dim, dim, bias=False)
        self.down = nn.Linear(dim, latent, bias=False)
        self.norm = nn.Parameter(torch.ones(latent))
        self.up_k = nn.Linear(latent, dim, bias=False)
        self.up_v = nn.Linear(latent, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        c = rms_norm(self.down(x), self.norm)                    # the cached latent [B, L, r]

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(b, n, self.heads, d // self.heads).transpose(1, 2)

        h = nn.functional.scaled_dot_product_attention(
            split(self.q(x)), split(self.up_k(c)), split(self.up_v(c)), is_causal=self.causal)
        return self.out(h.transpose(1, 2).reshape(b, n, d))
''',
    "QKNormAttention": '''\
class QKNormAttention(nn.Module):
    """Self-attention with RMS-normalized queries and keys per head (OLMo 2, Gemma 3,
    Qwen3): attention logits can't blow up, so training stays stable at higher learning
    rates."""

    def __init__(self, dim: int, heads: int = 4, causal: bool = True) -> None:
        super().__init__()
        self.heads, self.causal = heads, causal
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.q_norm = nn.Parameter(torch.ones(dim // heads))
        self.k_norm = nn.Parameter(torch.ones(dim // heads))
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        q, k = rms_norm(q, self.q_norm), rms_norm(k, self.k_norm)
        h = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        return self.out(h.transpose(1, 2).reshape(b, n, d))
''',
    "GatedAttention": '''\
class GatedAttention(nn.Module):
    """Gated attention (Qiu et al., 2025; Qwen3-Next): each head's output is multiplied by a
    sigmoid gate computed from the input, which removes attention sinks and steadies
    training."""

    def __init__(self, dim: int, heads: int = 4, causal: bool = True) -> None:
        super().__init__()
        self.heads, self.causal = heads, causal
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.gate = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        h = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        h = h.transpose(1, 2).reshape(b, n, d)
        return self.out(h * torch.sigmoid(self.gate(x)))
''',
    "Mamba2Block": '''\
class Mamba2Block(nn.Module):
    """Mamba-2 (Dao & Gu, 2024): a selective state-space layer whose decay is one scalar
    per head (structured state-space duality); a short causal convolution, an input-
    dependent step size, a gate and an RMS norm around it."""

    def __init__(self, dim: int, state: int = 16, expand: int = 2, heads: int = 4,
                 conv: int = 4) -> None:
        super().__init__()
        inner = expand * dim
        self.inner, self.state, self.heads, self.chunk = inner, state, heads, 32
        self.in_proj = nn.Linear(dim, 2 * inner + 2 * state + heads, bias=False)
        self.conv = nn.Conv1d(inner, inner, conv, groups=inner, padding=conv - 1)
        self.a_log = nn.Parameter(torch.log(torch.linspace(1.0, 16.0, heads)))
        self.skip = nn.Parameter(torch.ones(heads))
        self.dt_bias = nn.Parameter(torch.zeros(heads))
        self.norm = nn.Parameter(torch.ones(inner))
        self.out = nn.Linear(inner, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        e, s, h = self.inner, self.state, self.heads
        u, z, bb, cc, dt = self.in_proj(x).split([e, e, s, s, h], dim=-1)
        u = nn.functional.silu(self.conv(u.transpose(1, 2))[..., :n].transpose(1, 2))
        dt = nn.functional.softplus(dt + self.dt_bias)                   # [B, L, H]
        log_decay = -torch.exp(self.a_log) * dt                          # [B, L, H]
        u = u.reshape(b, n, h, e // h)
        y = self.scan(log_decay, dt[..., None] * u, bb, cc, self.chunk)
        y = (y + self.skip[:, None] * u).reshape(b, n, e)
        return self.out(rms_norm(y * nn.functional.silu(z), self.norm))

    @staticmethod
    def scan(log_decay, x, bb, cc, chunk: int) -> torch.Tensor:
        """state_t = decay_t state_(t-1) + x_t b_t^T, y_t = state_t c_t, in chunks (state-
        space duality): within a chunk the outputs are one masked product, and only the
        state carried between chunks loops. log_decay [B, L, H], x [B, L, H, P],
        bb, cc [B, L, S] -> [B, L, H, P]."""
        b, n, h, p = x.shape
        logs = to_chunks(log_decay, chunk).cumsum(2)                     # [B, N, C, H]
        xc, bc, ccc = to_chunks(x, chunk), to_chunks(bb, chunk), to_chunks(cc, chunk)
        w = decay_matrix(logs.transpose(2, 3))                           # [B, N, H, C, C]
        y = torch.einsum("bnhtj,bntj,bnjhp->bnthp", w, ccc @ bc.transpose(2, 3), xc)
        to_end = torch.exp(logs[:, :, -1:] - logs)                       # [B, N, C, H]
        adds = torch.einsum("bnjh,bnjhp,bnjs->bnhps", to_end, xc, bc)
        total = torch.exp(logs[:, :, -1])                                # [B, N, H]
        state, starts = x.new_zeros(b, h, p, bb.shape[-1]), []
        for i in range(logs.shape[1]):
            starts.append(state)
            state = total[:, i, :, None, None] * state + adds[:, i]
        carried = torch.einsum("bnhps,bnts->bnthp", torch.stack(starts, 1), ccc)
        return from_chunks(y + torch.exp(logs)[..., None] * carried, n)
''',
    "GatedDeltaNet": '''\
class GatedDeltaNet(nn.Module):
    """Gated DeltaNet (Yang et al., 2025): linear attention whose memory is updated by the
    delta rule (overwrite what a key used to store, at rate beta) and decayed by a learned
    gate; strong at in-context recall for a fixed-size state."""

    def __init__(self, dim: int, heads: int = 4) -> None:
        super().__init__()
        self.heads, self.chunk = heads, 32
        self.q, self.k, self.v = (nn.Linear(dim, dim, bias=False) for _ in range(3))
        self.beta = nn.Linear(dim, heads)
        self.alpha = nn.Linear(dim, heads)
        self.gate = nn.Linear(dim, dim, bias=False)
        self.norm = nn.Parameter(torch.ones(dim // heads))
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        h, hd = self.heads, d // self.heads
        q = nn.functional.normalize(self.q(x).reshape(b, n, h, hd), dim=-1)
        k = nn.functional.normalize(self.k(x).reshape(b, n, h, hd), dim=-1)
        v = self.v(x).reshape(b, n, h, hd)
        beta = torch.sigmoid(self.beta(x))                               # [B, L, H]
        log_alpha = -nn.functional.softplus(self.alpha(x))               # decay in (0, 1)
        o = rms_norm(self.recur(q, k, v, log_alpha, beta, self.chunk), self.norm)
        o = o.reshape(b, n, d)
        return self.out(o * nn.functional.silu(self.gate(x)))

    @staticmethod
    def recur(q, k, v, log_alpha, beta, chunk: int = 32) -> torch.Tensor:
        """memory_t = alpha_t memory_(t-1) (I - beta_t k_t k_t^T) + beta_t v_t k_t^T and
        output memory_t q_t, in chunks: the writes within a chunk come from one small
        triangular system (solved for every chunk at once), and only the memory carried
        between chunks loops. q, k, v [B, L, H, hd], log_alpha, beta [B, L, H] ->
        [B, L, H, hd]."""
        b, n, h, d = k.shape
        logs = to_chunks(log_alpha, chunk).cumsum(2).transpose(2, 3)    # [B, N, H, C]
        qc, kc, vc = (to_chunks(t, chunk).transpose(2, 3) for t in (q, k, v))
        bc = to_chunks(beta, chunk).transpose(2, 3)[..., None]          # [B, N, H, C, 1]
        c = logs.shape[-1]
        w = decay_matrix(logs)                                           # [B, N, H, C, C]
        lower = bc * (w * (1 - torch.eye(c, device=q.device))) * (kc @ kc.transpose(-1, -2))
        g = torch.exp(logs)[..., None]                                   # [B, N, H, C, 1]
        rhs = torch.cat([bc * vc, bc * g * kc], -1)
        rows = []
        for t in range(c):                                               # (I + lower) X = rhs
            row = rhs[..., t, :]
            if t:
                row = row - (lower[..., t, :t, None] * torch.stack(rows, -2)).sum(-2)
            rows.append(row)
        solved = torch.stack(rows, -2)
        new_v, new_k = solved[..., :d], solved[..., d:]                 # writes: v - k M^T
        attn = w * (qc @ kc.transpose(-1, -2))
        to_end = torch.exp(logs[..., -1:] - logs)[..., None] * kc
        total = torch.exp(logs[..., -1])                                 # [B, N, H]
        memory, outs = k.new_zeros(b, h, d, d), []                       # values x keys
        for i in range(logs.shape[1]):
            written = new_v[:, i] - new_k[:, i] @ memory.transpose(-1, -2)
            outs.append(g[:, i] * (qc[:, i] @ memory.transpose(-1, -2)) + attn[:, i] @ written)
            memory = (total[:, i, :, None, None] * memory
                      + written.transpose(-1, -2) @ to_end[:, i])
        return from_chunks(torch.stack(outs, 1).transpose(2, 3), n)
''',
    "GatedLinearAttention": '''\
class GatedLinearAttention(nn.Module):
    """Gated linear attention (Yang et al., 2024): a linear-attention memory with a
    data-dependent forget gate per key channel; trains on long sequences in linear time."""

    def __init__(self, dim: int, heads: int = 4, gate_rank: int = 16) -> None:
        super().__init__()
        self.heads, self.chunk = heads, 16
        self.q, self.k, self.v = (nn.Linear(dim, dim, bias=False) for _ in range(3))
        self.forget = nn.Sequential(nn.Linear(dim, gate_rank, bias=False),
                                    nn.Linear(gate_rank, dim))
        self.gate = nn.Linear(dim, dim, bias=False)
        self.norm = nn.Parameter(torch.ones(dim // heads))
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        h, hd = self.heads, d // self.heads
        q = self.q(x).reshape(b, n, h, hd) * hd ** -0.5
        k, v = self.k(x).reshape(b, n, h, hd), self.v(x).reshape(b, n, h, hd)
        # log sigmoid as -softplus(-f): the same, and Core ML converts softplus
        log_keep = (-nn.functional.softplus(-self.forget(x)) / 16).reshape(b, n, h, hd)
        o = rms_norm(self.scan(q, k, v, log_keep, self.chunk), self.norm).reshape(b, n, d)
        return self.out(o * nn.functional.silu(self.gate(x)))

    @staticmethod
    def scan(q, k, v, log_keep, chunk: int) -> torch.Tensor:
        """memory_t = keep_t * memory_(t-1) + k_t v_t^T (keep per key channel), output
        q_t memory_t, in chunks: within a chunk one masked product, and only the memory
        carried between chunks loops. q, k, v, log_keep [B, L, H, D] -> [B, L, H, D]."""
        b, n, h, d = k.shape
        logs = to_chunks(log_keep, chunk).cumsum(2)                      # [B, N, C, H, D]
        qc, kc, vc = to_chunks(q, chunk), to_chunks(k, chunk), to_chunks(v, chunk)
        nc, c = logs.shape[1], logs.shape[2]
        flat = logs.reshape(b * nc, c, h, d)          # rank <= 5 below (a Core ML limit)
        lower = torch.ones(c, c, dtype=torch.bool, device=q.device).tril()[:, :, None, None]
        diff = flat[:, :, None] - flat[:, None]                          # [BN, t, j, H, D]
        w = torch.exp(diff.masked_fill(~lower, float("-inf")))
        scores = torch.einsum("xthd,xjhd,xtjhd->xhtj", qc.reshape(b * nc, c, h, d),
                              kc.reshape(b * nc, c, h, d), w)
        o = torch.einsum("xhtj,xjhd->xthd", scores, vc.reshape(b * nc, c, h, d))
        o = o.reshape(b, nc, c, h, d)
        adds = torch.einsum("bnjhk,bnjhv->bnhkv", kc * torch.exp(logs[:, :, -1:] - logs), vc)
        total = torch.exp(logs[:, :, -1])                                # [B, N, H, D]
        memory, starts = q.new_zeros(b, h, d, d), []                     # keys x values
        for i in range(logs.shape[1]):
            starts.append(memory)
            memory = total[:, i, :, :, None] * memory + adds[:, i]
        o = o + torch.einsum("bnthk,bnhkv->bnthv", qc * torch.exp(logs), torch.stack(starts, 1))
        return from_chunks(o, n)
''',
    "GriffinBlock": '''\
class GriffinBlock(nn.Module):
    """Griffin's recurrent block (De et al., 2024; Hawk, RecurrentGemma): a GELU branch
    multiplies a branch that runs a short causal convolution and the real-gated linear
    recurrent unit (RG-LRU), whose decay each step chooses from the input."""

    def __init__(self, dim: int, width: int = 0, conv: int = 4) -> None:
        super().__init__()
        width = width or dim
        self.gate_branch = nn.Linear(dim, width, bias=False)
        self.rec_branch = nn.Linear(dim, width, bias=False)
        self.conv = nn.Conv1d(width, width, conv, groups=width, padding=conv - 1)
        self.recurrence_gate = nn.Linear(width, width)
        self.input_gate = nn.Linear(width, width)
        self.decay = nn.Parameter(torch.linspace(2.0, 6.0, width))    # sigmoid -> 0.88..0.998
        self.out = nn.Linear(width, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[1]
        g = nn.functional.gelu(self.gate_branch(x))
        u = self.conv(self.rec_branch(x).transpose(1, 2))[..., :n].transpose(1, 2)
        r, i = torch.sigmoid(self.recurrence_gate(u)), torch.sigmoid(self.input_gate(u))
        a = torch.exp(-8.0 * r * nn.functional.softplus(-self.decay))   # sigmoid(decay)^(8 r)
        scaled = torch.sqrt(1 - a * a) * (i * u)
        return self.out(g * linear_scan(a, scaled))
''',
    "UIB": '''\
class UIB(nn.Module):
    """MobileNetV4's universal inverted bottleneck (Qin et al., 2024): an optional depthwise
    convolution before the 1x1 expansion and an optional one after it; with neither it is
    a feed-forward block, with only the second it is MobileNetV2's inverted residual."""

    def __init__(self, cin: int, cout: int, expand: int = 4, start_kernel: int = 3,
                 mid_kernel: int = 3, stride: int = 1) -> None:
        super().__init__()
        hidden = cin * expand
        self.residual = stride == 1 and cin == cout
        layers = []
        if start_kernel:
            layers += [nn.Conv2d(cin, cin, start_kernel, stride if not mid_kernel else 1,
                                 start_kernel // 2, groups=cin, bias=False), nn.BatchNorm2d(cin)]
        layers += [nn.Conv2d(cin, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU()]
        if mid_kernel:
            layers += [nn.Conv2d(hidden, hidden, mid_kernel, stride, mid_kernel // 2,
                                 groups=hidden, bias=False), nn.BatchNorm2d(hidden), nn.ReLU()]
        elif not start_kernel and stride > 1:
            layers += [nn.AvgPool2d(stride, ceil_mode=True)]
        layers += [nn.Conv2d(hidden, cout, 1, bias=False), nn.BatchNorm2d(cout)]
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.body(x)
        return x + y if self.residual else y
''',
    "StarBlock": '''\
class StarBlock(nn.Module):
    """StarNet's block (Ma et al., 2024): two 1x1 branches multiplied element-wise ("star"),
    which maps features to a high-dimensional space without widening the network, between
    two 7x7 depthwise convolutions."""

    def __init__(self, dim: int, mlp_ratio: int = 3) -> None:
        super().__init__()
        self.dw1 = nn.Sequential(nn.Conv2d(dim, dim, 7, 1, 3, groups=dim), nn.BatchNorm2d(dim))
        self.f1 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.f2 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.g = nn.Sequential(nn.Conv2d(mlp_ratio * dim, dim, 1), nn.BatchNorm2d(dim))
        self.dw2 = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw1(x)
        y = nn.functional.relu6(self.f1(y)) * self.f2(y)
        return x + self.dw2(self.g(y))
''',
}

PYTORCH_HELPERS.update(TORCH)


# 3.9: Keras forms (TorchLayout comes from library37: the same weight layout as PyTorch)
KERAS = {
    "keras_chunk_scan": '''\
def to_chunks(x, chunk: int):
    """[B, L, ...] -> [B, L / chunk, chunk, ...], zero-padded at the end."""
    n = x.shape[1]
    pad = -n % chunk
    if pad:
        x = ops.pad(x, ((0, 0), (0, pad)) + ((0, 0),) * (len(x.shape) - 2))
    return ops.reshape(x, (-1, (n + pad) // chunk, chunk) + tuple(x.shape[2:]))


def from_chunks(x, n: int):
    return ops.reshape(x, (-1, x.shape[1] * x.shape[2]) + tuple(x.shape[3:]))[:, :n]


def decay_matrix(logs):
    """Cumulative log decays [..., C] -> [..., C, C]: exp(L_t - L_j) for j <= t, else 0."""
    c = logs.shape[-1]
    i = ops.arange(c)
    keep = ops.expand_dims(i, 1) >= ops.expand_dims(i, 0)
    diff = ops.expand_dims(logs, -1) - ops.expand_dims(logs, -2)
    return ops.where(keep, ops.exp(ops.where(keep, diff, 0.0)), 0.0)


def linear_scan(a, x):
    """h_t = a_t * h_(t-1) + x_t along axis 1, as a parallel prefix scan."""
    n, d = x.shape[1], 1
    while d < n:
        x = ops.concatenate([x[:, :d], x[:, d:] + a[:, d:] * x[:, :-d]], 1)
        a = ops.concatenate([a[:, :d], a[:, d:] * a[:, :-d]], 1)
        d *= 2
    return x
''',
    "LatentAttention": '''\
class LatentAttention(TorchLayout):
    """Multi-head latent attention: keys and values rebuilt from one shared latent."""

    def __init__(self, heads: int, latent: int, causal: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.heads, self.latent, self.causal = heads, latent, causal

    def build(self, input_shape):
        d, r = input_shape[-1], self.latent
        self.norm = self.vector("norm", (r,), "ones")
        self.q = self.linear("q", d, d, bias=False)
        self.down = self.linear("down", d, r, bias=False)
        self.up_k = self.linear("up_k", r, d, bias=False)
        self.up_v = self.linear("up_v", r, d, bias=False)
        self.out = self.linear("out", d, d, bias=False)

    def call(self, x):
        c = self.rms(self.dense(x, self.down), self.norm)
        q, k, v = (self.heads_of(t, self.heads) for t in (
            self.dense(x, self.q), self.dense(c, self.up_k), self.dense(c, self.up_v)))
        return self.dense(self.merge(self.attend(q, k, v, self.causal)), self.out)
''',
    "QKNormAttention": '''\
class QKNormAttention(TorchLayout):
    """Self-attention with RMS-normalized queries and keys per head."""

    def __init__(self, heads: int, causal: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.heads, self.causal = heads, causal

    def build(self, input_shape):
        d = input_shape[-1]
        self.q_norm = self.vector("q_norm", (d // self.heads,), "ones")
        self.k_norm = self.vector("k_norm", (d // self.heads,), "ones")
        self.qkv = self.linear("qkv", d, 3 * d, bias=False)
        self.out = self.linear("out", d, d, bias=False)

    def call(self, x):
        d = x.shape[-1]
        qkv = self.dense(x, self.qkv)
        q, k, v = (self.heads_of(qkv[..., i * d:(i + 1) * d], self.heads) for i in range(3))
        q, k = self.rms(q, self.q_norm), self.rms(k, self.k_norm)
        return self.dense(self.merge(self.attend(q, k, v, self.causal)), self.out)
''',
    "GatedAttention": '''\
class GatedAttention(TorchLayout):
    """Self-attention whose head outputs pass a sigmoid gate computed from the input."""

    def __init__(self, heads: int, causal: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.heads, self.causal = heads, causal

    def build(self, input_shape):
        d = input_shape[-1]
        self.qkv = self.linear("qkv", d, 3 * d, bias=False)
        self.gate = self.linear("gate", d, d, bias=False)
        self.out = self.linear("out", d, d, bias=False)

    def call(self, x):
        d = x.shape[-1]
        qkv = self.dense(x, self.qkv)
        q, k, v = (self.heads_of(qkv[..., i * d:(i + 1) * d], self.heads) for i in range(3))
        h = self.merge(self.attend(q, k, v, self.causal))
        return self.dense(h * ops.sigmoid(self.dense(x, self.gate)), self.out)
''',
    "Mamba2Block": '''\
class Mamba2Block(TorchLayout):
    """Mamba-2: a selective state-space layer with one decay per head, in chunks."""

    def __init__(self, state: int = 16, expand: int = 2, heads: int = 4, conv: int = 4,
                 chunk: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.state, self.expand, self.heads, self.k, self.chunk = state, expand, heads, conv, chunk

    def build(self, input_shape):
        d = input_shape[-1]
        e, s, h = self.expand * d, self.state, self.heads
        self.inner = e
        self.a_log = self.vector("a_log", (h,), ops.log(ops.linspace(1.0, 16.0, h)))
        self.skip = self.vector("skip", (h,), "ones")
        self.dt_bias = self.vector("dt_bias", (h,), "zeros")
        self.norm = self.vector("norm", (e,), "ones")
        self.in_proj = self.linear("in_proj", d, 2 * e + 2 * s + h, bias=False)
        self.conv = self.depthwise("conv", e, self.k)
        self.out = self.linear("out", e, d, bias=False)

    def call(self, x):
        n = x.shape[1]
        e, s, h = self.inner, self.state, self.heads
        p = self.dense(x, self.in_proj)
        u, z = p[..., :e], p[..., e:2 * e]
        bb, cc, dt = p[..., 2 * e:2 * e + s], p[..., 2 * e + s:2 * e + 2 * s], p[..., 2 * e + 2 * s:]
        u = ops.silu(self.causal_conv(u, self.conv))
        dt = ops.softplus(dt + self.dt_bias)
        log_decay = -ops.exp(self.a_log) * dt
        u = ops.reshape(u, (-1, n, h, e // h))
        y = self.scan(log_decay, ops.expand_dims(dt, -1) * u, bb, cc, self.chunk)
        y = ops.reshape(y + ops.expand_dims(self.skip, -1) * u, (-1, n, e))
        return self.dense(self.rms(y * ops.silu(z), self.norm), self.out)

    @staticmethod
    def scan(log_decay, x, bb, cc, chunk: int):
        n, h, p = x.shape[1], x.shape[2], x.shape[3]
        logs = ops.cumsum(to_chunks(log_decay, chunk), axis=2)              # [B, N, C, H]
        xc, bc, ccc = to_chunks(x, chunk), to_chunks(bb, chunk), to_chunks(cc, chunk)
        w = decay_matrix(ops.transpose(logs, (0, 1, 3, 2)))                 # [B, N, H, C, C]
        y = ops.einsum("bnhtj,bntj,bnjhp->bnthp", w,
                       ops.matmul(ccc, ops.transpose(bc, (0, 1, 3, 2))), xc)
        to_end = ops.exp(logs[:, :, -1:] - logs)
        adds = ops.einsum("bnjh,bnjhp,bnjs->bnhps", to_end, xc, bc)
        total = ops.exp(logs[:, :, -1])
        state = ops.zeros_like(adds[:, 0])
        starts = []
        for i in range(logs.shape[1]):
            starts.append(state)
            state = total[:, i, :, None, None] * state + adds[:, i]
        carried = ops.einsum("bnhps,bnts->bnthp", ops.stack(starts, 1), ccc)
        return from_chunks(y + ops.expand_dims(ops.exp(logs), -1) * carried, n)
''',
    "GatedDeltaNet": '''\
class GatedDeltaNet(TorchLayout):
    """Gated DeltaNet: a delta-rule memory with a learned decay, in chunks."""

    def __init__(self, heads: int = 4, chunk: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.heads, self.chunk = heads, chunk

    def build(self, input_shape):
        d = input_shape[-1]
        h = self.heads
        self.norm = self.vector("norm", (d // h,), "ones")
        self.q = self.linear("q", d, d, bias=False)
        self.k = self.linear("k", d, d, bias=False)
        self.v = self.linear("v", d, d, bias=False)
        self.beta = self.linear("beta", d, h)
        self.alpha = self.linear("alpha", d, h)
        self.gate = self.linear("gate", d, d, bias=False)
        self.out = self.linear("out", d, d, bias=False)

    def call(self, x):
        n, d = x.shape[1], x.shape[2]
        h = self.heads

        def split(t):
            return ops.reshape(t, (-1, n, h, d // h))

        def unit(t):
            return t / ops.maximum(ops.sqrt(ops.sum(ops.square(t), axis=-1, keepdims=True)), 1e-12)

        q, k = unit(split(self.dense(x, self.q))), unit(split(self.dense(x, self.k)))
        v = split(self.dense(x, self.v))
        beta = ops.sigmoid(self.dense(x, self.beta))
        log_alpha = -ops.softplus(self.dense(x, self.alpha))
        o = ops.reshape(self.rms(self.recur(q, k, v, log_alpha, beta, self.chunk), self.norm),
                        (-1, n, d))
        return self.dense(o * ops.silu(self.dense(x, self.gate)), self.out)

    @staticmethod
    def recur(q, k, v, log_alpha, beta, chunk: int):
        n, d = k.shape[1], k.shape[3]
        tr = (0, 1, 3, 2, 4)
        logs = ops.transpose(ops.cumsum(to_chunks(log_alpha, chunk), axis=2), (0, 1, 3, 2))
        qc, kc, vc = (ops.transpose(to_chunks(t, chunk), tr) for t in (q, k, v))
        bc = ops.expand_dims(ops.transpose(to_chunks(beta, chunk), (0, 1, 3, 2)), -1)
        c = logs.shape[-1]
        w = decay_matrix(logs)
        kk = ops.matmul(kc, ops.transpose(kc, (0, 1, 2, 4, 3)))
        lower = bc * (w * (1 - ops.eye(c))) * kk
        g = ops.expand_dims(ops.exp(logs), -1)
        rhs = ops.concatenate([bc * vc, bc * g * kc], -1)
        rows = []
        for t in range(c):
            row = rhs[..., t, :]
            if t:
                row = row - ops.sum(ops.expand_dims(lower[..., t, :t], -1)
                                    * ops.stack(rows, -2), axis=-2)
            rows.append(row)
        solved = ops.stack(rows, -2)
        new_v, new_k = solved[..., :d], solved[..., d:]
        attn = w * ops.matmul(qc, ops.transpose(kc, (0, 1, 2, 4, 3)))
        to_end = ops.expand_dims(ops.exp(logs[..., -1:] - logs), -1) * kc
        total = ops.exp(logs[..., -1])
        memory = ops.zeros_like(ops.matmul(ops.transpose(vc[:, 0], (0, 1, 3, 2)), kc[:, 0]))
        outs = []
        for i in range(logs.shape[1]):
            mt = ops.transpose(memory, (0, 1, 3, 2))
            written = new_v[:, i] - ops.matmul(new_k[:, i], mt)
            outs.append(g[:, i] * ops.matmul(qc[:, i], mt) + ops.matmul(attn[:, i], written))
            memory = (total[:, i, :, None, None] * memory
                      + ops.matmul(ops.transpose(written, (0, 1, 3, 2)), to_end[:, i]))
        return from_chunks(ops.transpose(ops.stack(outs, 1), tr), n)
''',
    "GatedLinearAttention": '''\
class GatedLinearAttention(TorchLayout):
    """Gated linear attention: a memory with a forget gate per key channel, in chunks."""

    def __init__(self, heads: int = 4, gate_rank: int = 16, chunk: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.heads, self.rank, self.chunk = heads, gate_rank, chunk

    def build(self, input_shape):
        d = input_shape[-1]
        self.norm = self.vector("norm", (d // self.heads,), "ones")
        self.q = self.linear("q", d, d, bias=False)
        self.k = self.linear("k", d, d, bias=False)
        self.v = self.linear("v", d, d, bias=False)
        self.forget_0 = self.linear("forget_0", d, self.rank, bias=False)
        self.forget_1 = self.linear("forget_1", self.rank, d)
        self.gate = self.linear("gate", d, d, bias=False)
        self.out = self.linear("out", d, d, bias=False)

    def call(self, x):
        n, d = x.shape[1], x.shape[2]
        h, hd = self.heads, d // self.heads

        def split(t):
            return ops.reshape(t, (-1, n, h, hd))

        q = split(self.dense(x, self.q)) * hd ** -0.5
        k, v = split(self.dense(x, self.k)), split(self.dense(x, self.v))
        f = self.dense(self.dense(x, self.forget_0), self.forget_1)
        log_keep = split(-ops.softplus(-f) / 16)
        o = ops.reshape(self.rms(self.scan(q, k, v, log_keep, self.chunk), self.norm), (-1, n, d))
        return self.dense(o * ops.silu(self.dense(x, self.gate)), self.out)

    @staticmethod
    def scan(q, k, v, log_keep, chunk: int):
        n, h, d = k.shape[1], k.shape[2], k.shape[3]
        logs = ops.cumsum(to_chunks(log_keep, chunk), axis=2)              # [B, N, C, H, D]
        qc, kc, vc = to_chunks(q, chunk), to_chunks(k, chunk), to_chunks(v, chunk)
        nc, c = logs.shape[1], logs.shape[2]
        flat = ops.reshape(logs, (-1, c, h, d))
        i = ops.arange(c)
        lower = ops.reshape(ops.expand_dims(i, 1) >= ops.expand_dims(i, 0), (1, c, c, 1, 1))
        diff = ops.expand_dims(flat, 2) - ops.expand_dims(flat, 1)
        w = ops.where(lower, ops.exp(ops.where(lower, diff, 0.0)), 0.0)
        scores = ops.einsum("xthd,xjhd,xtjhd->xhtj", ops.reshape(qc, (-1, c, h, d)),
                            ops.reshape(kc, (-1, c, h, d)), w)
        o = ops.einsum("xhtj,xjhd->xthd", scores, ops.reshape(vc, (-1, c, h, d)))
        o = ops.reshape(o, (-1, nc, c, h, d))
        adds = ops.einsum("bnjhk,bnjhv->bnhkv", kc * ops.exp(logs[:, :, -1:] - logs), vc)
        total = ops.exp(logs[:, :, -1])
        memory = ops.zeros_like(adds[:, 0])
        starts = []
        for j in range(nc):
            starts.append(memory)
            memory = ops.expand_dims(total[:, j], -1) * memory + adds[:, j]
        o = o + ops.einsum("bnthk,bnhkv->bnthv", qc * ops.exp(logs), ops.stack(starts, 1))
        return from_chunks(o, n)
''',
    "GriffinBlock": '''\
class GriffinBlock(TorchLayout):
    """Griffin's recurrent block: a GELU branch times a causal conv + RG-LRU branch."""

    def __init__(self, width: int = 0, conv: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.width, self.k = width, conv

    def build(self, input_shape):
        d = input_shape[-1]
        w = self.width or d
        self.decay = self.vector("decay", (w,), ops.linspace(2.0, 6.0, w))
        self.gate_branch = self.linear("gate_branch", d, w, bias=False)
        self.rec_branch = self.linear("rec_branch", d, w, bias=False)
        self.conv = self.depthwise("conv", w, self.k)
        self.recurrence_gate = self.linear("recurrence_gate", w, w)
        self.input_gate = self.linear("input_gate", w, w)
        self.out = self.linear("out", w, d, bias=False)

    def call(self, x):
        g = ops.gelu(self.dense(x, self.gate_branch), approximate=False)
        u = self.causal_conv(self.dense(x, self.rec_branch), self.conv)
        r = ops.sigmoid(self.dense(u, self.recurrence_gate))
        i = ops.sigmoid(self.dense(u, self.input_gate))
        a = ops.exp(-8.0 * r * ops.softplus(-self.decay))
        scaled = ops.sqrt(1 - a * a) * (i * u)
        return self.dense(g * linear_scan(a, scaled), self.out)
''',
    "uib": '''\
def uib(x, out_channels: int, expand_ratio: int = 4, start_kernel: int = 3,
        mid_kernel: int = 3, stride: int = 1):
    """MobileNetV4's universal inverted bottleneck (channels-last)."""
    cin = x.shape[-1]
    hidden = cin * expand_ratio

    def norm(h):
        return layers.BatchNormalization(epsilon=1e-5, momentum=0.9)(h)

    h = x
    if start_kernel:
        h = norm(layers.DepthwiseConv2D(start_kernel, 1 if mid_kernel else stride,
                                        padding="same", use_bias=False)(h))
    h = layers.ReLU()(norm(layers.Conv2D(hidden, 1, use_bias=False)(h)))
    if mid_kernel:
        h = layers.ReLU()(norm(layers.DepthwiseConv2D(mid_kernel, stride, padding="same",
                                                      use_bias=False)(h)))
    elif not start_kernel and stride > 1:
        h = layers.AveragePooling2D(stride, padding="same")(h)
    h = norm(layers.Conv2D(out_channels, 1, use_bias=False)(h))
    return x + h if stride == 1 and cin == out_channels else h
''',
    "star_block": '''\
def star_block(x, mlp_ratio: int = 3):
    """StarNet's block (channels-last): two 1x1 branches multiplied, between 7x7
    depthwise convolutions."""
    d = x.shape[-1]
    y = layers.BatchNormalization(epsilon=1e-5, momentum=0.9)(
        layers.DepthwiseConv2D(7, padding="same")(x))
    y = ops.relu6(layers.Conv2D(mlp_ratio * d, 1)(y)) * layers.Conv2D(mlp_ratio * d, 1)(y)
    y = layers.BatchNormalization(epsilon=1e-5, momentum=0.9)(layers.Conv2D(d, 1)(y))
    return x + layers.DepthwiseConv2D(7, padding="same")(y)
''',
}
KERAS_HELPERS.update(KERAS)

# ------------------------------------------------------------------ shape helpers

def _seq_shape(name: str, check=None):
    def fn(in_shapes, p):
        s = in_shapes[0]
        if len(s) != 2:
            raise ShapeError(f"{name} expects a batch-first sequence [L, D], got {s}")
        if "heads" in p and (int(p["heads"]) < 1 or s[1] % int(p["heads"])):
            raise ShapeError(f"{name}: the width D={s[1]} is not divisible by "
                             f"{p['heads']} heads")
        if check:
            check(s, p)
        return list(s)

    return fn


def _image(s, name: str) -> None:
    if len(s) != 3:
        raise ShapeError(f"{name} expects feature maps [C, H, W], got {s}")


_HEADS = P("heads", "int", 4, lo=1, help="Attention heads; must divide the width D")
_CAUSAL = P("causal", "bool", True, help="Each position sees only itself and earlier ones")


def _latent(d: int, p) -> int:
    return int(p["latent"]) or max(1, d // 2)


# ------------------------------------------------------------------ attention

reg.register(nn_block(
    "core.latent_attention", "Multi-Head Latent Attention", TRANSFORMER, family="attention",
    params=(_HEADS, P("latent", "int", 0, lo=0, help="Width of the shared key/value latent; "
                                                     "0 = D / 2"), _CAUSAL),
    shape=_seq_shape("Multi-Head Latent Attention"), layout="ir",
    param_fn=lambda s, p: (2 * s[0][-1] ** 2 + 3 * s[0][-1] * _latent(s[0][-1], p)
                           + _latent(s[0][-1], p)),
    torch=lambda c: (f"LatentAttention({c['input_size']}, {int(c['heads'])}, "
                     f"latent={_latent(c['input_size'], c)}, causal={bool(c['causal'])})"),
    torch_helpers=("rms_norm", "LatentAttention"),
    keras_expr=lambda c: (f"LatentAttention({int(c['heads'])}, "
                          f"{_latent(c['input_size'], c)}, causal={bool(c['causal'])})({c['i0']})"),
    keras_helpers=("TorchLayout", "LatentAttention"),
    desc="DeepSeek's multi-head latent attention: keys and values are rebuilt from one small "
         "latent per position, so a decoder's cache shrinks several times with little loss "
         "in quality ([L, D] → [L, D]).",
))

reg.register(nn_block(
    "core.qk_norm_attention", "QK-Norm Attention", TRANSFORMER, family="attention",
    params=(_HEADS, _CAUSAL),
    shape=_seq_shape("QK-Norm Attention"), layout="ir",
    param_fn=lambda s, p: 4 * s[0][-1] ** 2 + 2 * (s[0][-1] // int(p["heads"])),
    torch=lambda c: (f"QKNormAttention({c['input_size']}, {int(c['heads'])}, "
                     f"causal={bool(c['causal'])})"),
    torch_helpers=("rms_norm", "QKNormAttention"),
    keras_expr=lambda c: (f"QKNormAttention({int(c['heads'])}, "
                          f"causal={bool(c['causal'])})({c['i0']})"),
    keras_helpers=("TorchLayout", "QKNormAttention"),
    desc="Self-attention with queries and keys RMS-normalized per head (OLMo 2, Gemma 3, "
         "Qwen3): the logits stay bounded, so training is stable at higher learning rates "
         "([L, D] → [L, D]).",
))

reg.register(nn_block(
    "core.gated_attention", "Gated Attention", TRANSFORMER, family="attention",
    params=(_HEADS, _CAUSAL),
    shape=_seq_shape("Gated Attention"), layout="ir",
    param_fn=lambda s, p: 5 * s[0][-1] ** 2,
    torch=lambda c: (f"GatedAttention({c['input_size']}, {int(c['heads'])}, "
                     f"causal={bool(c['causal'])})"),
    torch_helpers=("GatedAttention",),
    keras_expr=lambda c: (f"GatedAttention({int(c['heads'])}, "
                          f"causal={bool(c['causal'])})({c['i0']})"),
    keras_helpers=("TorchLayout", "GatedAttention"),
    desc="Attention whose output is scaled by a sigmoid gate read from the input (Qiu et al., "
         "2025; Qwen3-Next): no attention sinks, steadier training, better on long "
         "contexts ([L, D] → [L, D]).",
))


# ------------------------------------------------------------------ sequence mixers

def _mamba2_check(s, p):
    if (int(p["expand"]) * s[1]) % int(p["heads"]):
        raise ShapeError(f"Mamba-2: the inner width {int(p['expand']) * s[1]} is not divisible "
                         f"by {p['heads']} heads")


def _mamba2_params(s, p) -> int:
    d = s[0][-1]
    e, n, h, k = int(p["expand"]) * d, int(p["state"]), int(p["heads"]), int(p["conv"])
    return d * (2 * e + 2 * n + h) + (e * k + e) + 3 * h + e + e * d


reg.register(nn_block(
    "seq.mamba2", "Mamba-2 Block", SEQUENCE, family="recurrent",
    params=(P("state", "int", 16, lo=1, hi=256, help="State size N"),
            P("expand", "int", 2, lo=1, hi=4, help="Inner width = expand × D"),
            P("heads", "int", 4, lo=1, help="Heads; each has one decay"),
            P("conv", "int", 4, lo=1, hi=8, help="Causal convolution width")),
    shape=_seq_shape("Mamba-2", _mamba2_check), layout="ir", param_fn=_mamba2_params,
    torch=lambda c: (f"Mamba2Block({c['input_size']}, state={int(c['state'])}, "
                     f"expand={int(c['expand'])}, heads={int(c['heads'])}, "
                     f"conv={int(c['conv'])})"),
    torch_helpers=("rms_norm", "chunk_scan", "Mamba2Block"),
    keras_expr=lambda c: (f"Mamba2Block(state={int(c['state'])}, expand={int(c['expand'])}, "
                          f"heads={int(c['heads'])}, conv={int(c['conv'])})({c['i0']})"),
    keras_helpers=("TorchLayout", "keras_chunk_scan", "Mamba2Block"),
    desc="Mamba-2 (Dao & Gu, 2024): a selective state-space layer with one decay per head, "
         "the form that makes it as fast as attention to train. Causal; stack a few after "
         "an Embedding for a language model ([L, D] → [L, D]).",
))

reg.register(nn_block(
    "seq.gated_deltanet", "Gated DeltaNet", SEQUENCE, family="recurrent",
    params=(_HEADS,),
    shape=_seq_shape("Gated DeltaNet"), layout="ir",
    param_fn=lambda s, p: (5 * s[0][-1] ** 2 + 2 * (s[0][-1] * int(p["heads"]) + int(p["heads"]))
                           + s[0][-1] // int(p["heads"])),
    torch=lambda c: f"GatedDeltaNet({c['input_size']}, {int(c['heads'])})",
    torch_helpers=("rms_norm", "chunk_scan", "GatedDeltaNet"),
    keras_expr=lambda c: f"GatedDeltaNet({int(c['heads'])})({c['i0']})",
    keras_helpers=("TorchLayout", "keras_chunk_scan", "GatedDeltaNet"),
    desc="Gated DeltaNet (Yang et al., 2025; Qwen3-Next's linear layers): a fixed-size memory "
         "updated by the delta rule, so a new value replaces what a key stored, with a "
         "learned forgetting gate. Causal ([L, D] → [L, D]).",
))

reg.register(nn_block(
    "seq.gla", "Gated Linear Attention", SEQUENCE, family="recurrent",
    params=(_HEADS,),
    shape=_seq_shape("Gated Linear Attention"), layout="ir",
    param_fn=lambda s, p: (5 * s[0][-1] ** 2 + 32 * s[0][-1] + s[0][-1]
                           + s[0][-1] // int(p["heads"])),
    torch=lambda c: f"GatedLinearAttention({c['input_size']}, {int(c['heads'])})",
    torch_helpers=("rms_norm", "chunk_scan", "GatedLinearAttention"),
    keras_expr=lambda c: f"GatedLinearAttention({int(c['heads'])})({c['i0']})",
    keras_helpers=("TorchLayout", "keras_chunk_scan", "GatedLinearAttention"),
    desc="Gated linear attention (Yang et al., 2024): linear attention with a forget gate "
         "per key channel that depends on the input; linear time in the length. Causal "
         "([L, D] → [L, D]).",
))


def _griffin_width(d: int, p) -> int:
    return int(p["width"]) or d


def _griffin_params(s, p) -> int:
    d = s[0][-1]
    w, k = _griffin_width(d, p), int(p["conv"])
    return 2 * d * w + (w * k + w) + 2 * (w * w + w) + w + w * d


reg.register(nn_block(
    "seq.griffin", "Griffin Recurrent Block (RG-LRU)", SEQUENCE, family="recurrent",
    params=(P("width", "int", 0, lo=0, help="Recurrence width; 0 = D"),
            P("conv", "int", 4, lo=1, hi=8, help="Causal convolution width")),
    shape=_seq_shape("Griffin Recurrent Block"), layout="ir", param_fn=_griffin_params,
    torch=lambda c: (f"GriffinBlock({c['input_size']}, {_griffin_width(c['input_size'], c)}, "
                     f"conv={int(c['conv'])})"),
    torch_helpers=("chunk_scan", "GriffinBlock"),
    keras_expr=lambda c: (f"GriffinBlock({_griffin_width(c['input_size'], c)}, "
                          f"conv={int(c['conv'])})({c['i0']})"),
    keras_helpers=("TorchLayout", "keras_chunk_scan", "GriffinBlock"),
    desc="The recurrent block of Griffin and Hawk (De et al., 2024; RecurrentGemma): a gated "
         "linear recurrence (RG-LRU) after a short convolution, multiplied by a GELU branch. "
         "Mix with local attention for long sequences. Causal ([L, D] → [L, D]).",
))


# ------------------------------------------------------------------ vision

def _odd_or_zero(k: int, name: str) -> None:
    if k and k % 2 == 0:
        raise ShapeError(f"{name}: use an odd kernel size (3, 5, 7) or 0 to leave it out")


def _uib_shape(in_shapes, p):
    s = in_shapes[0]
    _image(s, "Universal Inverted Bottleneck")
    _odd_or_zero(int(p["start_kernel"]), "Universal Inverted Bottleneck")
    _odd_or_zero(int(p["mid_kernel"]), "Universal Inverted Bottleneck")
    stride = int(p["stride"])
    return [int(p["out_channels"]), -(-s[1] // stride), -(-s[2] // stride)]


def _uib_params(s, p) -> int:
    c = s[0][0]
    o, e = int(p["out_channels"]), int(p["expand_ratio"])
    k1, k2, hidden = int(p["start_kernel"]), int(p["mid_kernel"]), c * e
    total = c * hidden + 2 * hidden + hidden * o + 2 * o
    if k1:
        total += c * k1 * k1 + 2 * c
    if k2:
        total += hidden * k2 * k2 + 2 * hidden
    return total


reg.register(nn_block(
    "vision.uib", "Universal Inverted Bottleneck (MobileNetV4)", VISION, family="conv",
    params=(P("out_channels", "int", 32, lo=1),
            P("expand_ratio", "int", 4, lo=1, hi=8),
            P("start_kernel", "int", 3, lo=0, hi=7, help="Depthwise kernel before the "
                                                         "expansion; 0 = none"),
            P("mid_kernel", "int", 3, lo=0, hi=7, help="Depthwise kernel after the "
                                                       "expansion; 0 = none"),
            P("stride", "int", 1, lo=1, hi=2)),
    shape=_uib_shape, layout="cl", param_fn=_uib_params,
    torch=lambda c: (f"UIB({c['in_channels']}, {int(c['out_channels'])}, "
                     f"expand={int(c['expand_ratio'])}, start_kernel={int(c['start_kernel'])}, "
                     f"mid_kernel={int(c['mid_kernel'])}, stride={int(c['stride'])})"),
    torch_helpers=("UIB",),
    keras_expr=lambda c: (f"uib({c['i0']}, {int(c['out_channels'])}, "
                          f"expand_ratio={int(c['expand_ratio'])}, "
                          f"start_kernel={int(c['start_kernel'])}, "
                          f"mid_kernel={int(c['mid_kernel'])}, stride={int(c['stride'])})"),
    keras_helpers=("uib",),
    desc="MobileNetV4's universal inverted bottleneck: optional depthwise convolutions "
         "before and after a 1×1 expansion. Both on: ExtraDW; only the second: an inverted "
         "residual; neither: a feed-forward block ([C, H, W] → [out, H/stride, W/stride]).",
))


def _star_shape(in_shapes, p):
    _image(in_shapes[0], "Star Block")
    return list(in_shapes[0])


def _star_params(s, p) -> int:
    c, r = s[0][0], int(p["mlp_ratio"])
    return (49 * c + c + 2 * c) + 2 * (c * r * c + r * c) + (r * c * c + c + 2 * c) + (49 * c + c)


reg.register(nn_block(
    "vision.star_block", "Star Block (StarNet)", VISION, family="conv",
    params=(P("mlp_ratio", "int", 3, lo=1, hi=8),),
    shape=_star_shape, layout="cl", param_fn=_star_params,
    torch=lambda c: f"StarBlock({c['in_channels']}, {int(c['mlp_ratio'])})",
    torch_helpers=("StarBlock",),
    keras_expr=lambda c: f"star_block({c['i0']}, {int(c['mlp_ratio'])})",
    keras_helpers=("star_block",),
    desc="StarNet's block (Ma et al., 2024): two 1×1 branches multiplied element-wise "
         "between 7×7 depthwise convolutions; the product reaches a much richer feature "
         "space at no extra width. A fast backbone block for small devices ([C, H, W] → "
         "same).",
))


LIBRARY38_IDS = tuple(sorted({b.type_id for b in reg.all()} - _BEFORE))
