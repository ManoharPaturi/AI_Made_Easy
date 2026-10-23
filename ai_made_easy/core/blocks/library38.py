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
from ai_made_easy.core.codegen.helpers import PYTORCH_HELPERS
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

TORCH = {
    "rms_norm": _RMS,
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
        self.inner, self.state, self.heads = inner, state, heads
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
        decay = torch.exp(-torch.exp(self.a_log) * dt)                   # [B, L, H]
        u = u.reshape(b, n, h, e // h)
        state = x.new_zeros(b, h, e // h, s)
        ys = []
        for t in range(n):
            state = (decay[:, t, :, None, None] * state
                     + (dt[:, t, :, None] * u[:, t])[..., None] * bb[:, t, None, None, :])
            ys.append((state * cc[:, t, None, None, :]).sum(-1) + self.skip[:, None] * u[:, t])
        y = torch.stack(ys, 1).reshape(b, n, e)
        return self.out(rms_norm(y * nn.functional.silu(z), self.norm))
''',
    "GatedDeltaNet": '''\
class GatedDeltaNet(nn.Module):
    """Gated DeltaNet (Yang et al., 2025): linear attention whose memory is updated by the
    delta rule (overwrite what a key used to store, at rate beta) and decayed by a learned
    gate; strong at in-context recall for a fixed-size state."""

    def __init__(self, dim: int, heads: int = 4) -> None:
        super().__init__()
        self.heads = heads
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
        alpha = torch.exp(-nn.functional.softplus(self.alpha(x)))        # decay in (0, 1)
        o = rms_norm(self.recur(q, k, v, alpha, beta), self.norm).reshape(b, n, d)
        return self.out(o * nn.functional.silu(self.gate(x)))

    @staticmethod
    def recur(q, k, v, alpha, beta) -> torch.Tensor:
        """q, k, v [B, L, H, hd], alpha, beta [B, L, H] -> outputs [B, L, H, hd]."""
        b, n, h, hd = k.shape
        memory = k.new_zeros(b, h, hd, hd)                               # values x keys
        outs = []
        for t in range(n):
            kt, vt = k[:, t], v[:, t]
            memory = alpha[:, t, :, None, None] * memory
            recalled = (memory @ kt[..., None])[..., 0]
            memory = memory + (beta[:, t, :, None] * (vt - recalled))[..., None] * kt[:, :, None]
            outs.append((memory @ q[:, t, :, :, None])[..., 0])
        return torch.stack(outs, 1)
''',
    "GatedLinearAttention": '''\
class GatedLinearAttention(nn.Module):
    """Gated linear attention (Yang et al., 2024): a linear-attention memory with a
    data-dependent forget gate per key channel; trains on long sequences in linear time."""

    def __init__(self, dim: int, heads: int = 4, gate_rank: int = 16) -> None:
        super().__init__()
        self.heads = heads
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
        keep = torch.sigmoid(self.forget(x)).pow(1 / 16).reshape(b, n, h, hd)
        memory = x.new_zeros(b, h, hd, hd)                               # keys x values
        outs = []
        for t in range(n):
            memory = keep[:, t, :, :, None] * memory + k[:, t, :, :, None] * v[:, t, :, None, :]
            outs.append((q[:, t, :, None, :] @ memory)[:, :, 0])
        o = rms_norm(torch.stack(outs, 1), self.norm).reshape(b, n, d)
        return self.out(o * nn.functional.silu(self.gate(x)))
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
        state, ys = torch.zeros_like(u[:, 0]), []
        for t in range(n):
            state = a[:, t] * state + scaled[:, t]
            ys.append(state)
        return self.out(g * torch.stack(ys, 1))
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
    torch_helpers=("rms_norm", "Mamba2Block"),
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
    torch_helpers=("rms_norm", "GatedDeltaNet"),
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
    torch_helpers=("rms_norm", "GatedLinearAttention"),
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
    torch_helpers=("GriffinBlock",),
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
    desc="StarNet's block (Ma et al., 2024): two 1×1 branches multiplied element-wise "
         "between 7×7 depthwise convolutions; the product reaches a much richer feature "
         "space at no extra width. A fast backbone block for small devices ([C, H, W] → "
         "same).",
))


LIBRARY38_IDS = tuple(sorted({b.type_id for b in reg.all()} - _BEFORE))
