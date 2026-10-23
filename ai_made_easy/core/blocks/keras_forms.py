"""Keras forms of the 3.7 and 3.8 layers (3.9).

The sequence and attention layers subclass ``TorchLayout``: their weights keep PyTorch's
layout and names, so a PyTorch layer's weights copy over one-to-one and both compute the
same function (tests/test_keras_parity.py). The vision blocks are built from Keras' own
layers, channels-last. The blocks in library37 / library38 name these helpers.
"""
from __future__ import annotations

from ai_made_easy.core.codegen.helpers import KERAS_HELPERS

KERAS_FORMS = {
    "TorchLayout": '''\
class TorchLayout(layers.Layer):
    """A layer whose weights keep PyTorch's layout and names ([out, in] linear maps,
    [C, 1, k] depthwise filters), so a PyTorch layer's weights copy over one-to-one and the
    two frameworks compute the same function. Linear maps start as PyTorch's do."""

    def linear(self, name: str, n_in: int, n_out: int, bias: bool = True):
        bound = 1 / n_in ** 0.5
        init = keras.initializers.RandomUniform(-bound, bound)
        w = self.add_weight(name=f"{name}_weight", shape=(n_out, n_in), initializer=init)
        b = self.add_weight(name=f"{name}_bias", shape=(n_out,), initializer=init) if bias \\
            else None
        return w, b

    def vector(self, name: str, shape, init):
        if not callable(init) and not isinstance(init, str):
            init = keras.initializers.Constant(init)
        return self.add_weight(name=name, shape=tuple(shape), initializer=init)

    def depthwise(self, name: str, channels: int, k: int):
        bound = 1 / k ** 0.5
        init = keras.initializers.RandomUniform(-bound, bound)
        return (self.add_weight(name=f"{name}_weight", shape=(channels, 1, k), initializer=init),
                self.add_weight(name=f"{name}_bias", shape=(channels,), initializer=init))

    @staticmethod
    def dense(x, wb):
        w, b = wb
        y = ops.matmul(x, ops.transpose(w))
        return y if b is None else y + b

    @staticmethod
    def causal_conv(x, wb):
        """Depthwise causal convolution over [B, L, C] (PyTorch: Conv1d with groups=C and
        padding=k - 1, first L outputs)."""
        w, b = wb
        k, n = w.shape[-1], x.shape[1]
        xp = ops.pad(x, ((0, 0), (k - 1, 0), (0, 0)))
        y = b
        for j in range(k):
            y = y + xp[:, j:j + n] * w[:, 0, j]
        return y

    @staticmethod
    def rms(x, weight, eps: float = 1e-6):
        return x * ops.rsqrt(ops.mean(ops.square(x), axis=-1, keepdims=True) + eps) * weight

    @staticmethod
    def layer_norm(x, weight, bias, eps: float = 1e-5):
        mean = ops.mean(x, axis=-1, keepdims=True)
        var = ops.mean(ops.square(x - mean), axis=-1, keepdims=True)
        return (x - mean) * ops.rsqrt(var + eps) * weight + bias

    @staticmethod
    def attend(q, k, v, causal: bool):
        """Softmax attention over [B, H, L, hd] (PyTorch's scaled_dot_product_attention)."""
        s = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) / q.shape[-1] ** 0.5
        if causal:
            n = q.shape[2]
            i = ops.arange(n)
            s = ops.where(ops.expand_dims(i, 1) >= ops.expand_dims(i, 0), s, -1e30)
        return ops.matmul(ops.softmax(s, axis=-1), v)

    @staticmethod
    def heads_of(t, heads: int):
        n, d = t.shape[1], t.shape[2]
        return ops.transpose(ops.reshape(t, (-1, n, heads, d // heads)), (0, 2, 1, 3))

    @staticmethod
    def merge(t):
        h, n, hd = t.shape[1], t.shape[2], t.shape[3]
        return ops.reshape(ops.transpose(t, (0, 2, 1, 3)), (-1, n, h * hd))
''',
    "DiffAttention": '''\
class DiffAttention(TorchLayout):
    """Differential attention: the difference of two softmax maps."""

    def __init__(self, heads: int = 4, causal: bool = True, lambda_init: float = 0.8,
                 **kwargs):
        super().__init__(**kwargs)
        self.heads, self.causal, self.lambda_init = heads, causal, lambda_init

    def build(self, input_shape):
        d = input_shape[-1]
        hd = d // self.heads
        self.lambdas = self.vector("lambdas", (4, hd // 2),
                                   keras.initializers.RandomNormal(stddev=0.1))
        self.norm = self.vector("norm", (hd,), "ones")
        self.qkv = self.linear("qkv", d, 3 * d)
        self.out = self.linear("out", d, d)

    def call(self, x):
        n, d = x.shape[1], x.shape[2]
        h, hd = self.heads, d // self.heads
        qkv = self.dense(x, self.qkv)
        q, k, v = (self.heads_of(qkv[..., i * d:(i + 1) * d], h) for i in range(3))
        half = hd // 2
        scale = half ** -0.5
        s1 = ops.matmul(q[..., :half], ops.transpose(k[..., :half], (0, 1, 3, 2))) * scale
        s2 = ops.matmul(q[..., half:], ops.transpose(k[..., half:], (0, 1, 3, 2))) * scale
        if self.causal:
            i = ops.arange(n)
            allowed = ops.expand_dims(i, 1) >= ops.expand_dims(i, 0)
            s1, s2 = ops.where(allowed, s1, -1e30), ops.where(allowed, s2, -1e30)
        e = ops.exp(ops.sum(self.lambdas[0::2] * self.lambdas[1::2], axis=-1))
        lam = e[0] - e[1] + self.lambda_init
        o = ops.matmul(ops.softmax(s1, axis=-1) - lam * ops.softmax(s2, axis=-1), v)
        o = self.rms(o, self.norm) * (1 - self.lambda_init)
        return self.dense(self.merge(o), self.out)
''',
    "LinformerAttention": '''\
class LinformerAttention(TorchLayout):
    """Linformer self-attention: keys and values projected from length L down to k."""

    def __init__(self, k: int = 32, heads: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.k, self.heads = k, heads

    def build(self, input_shape):
        n, d = input_shape[1], input_shape[-1]
        self.qkv = self.linear("qkv", d, 3 * d)
        self.proj_k = self.linear("proj_k", n, self.k, bias=False)
        self.proj_v = self.linear("proj_v", n, self.k, bias=False)
        self.out = self.linear("out", d, d)

    def call(self, x):
        d = x.shape[-1]
        qkv = self.dense(x, self.qkv)
        q, k, v = qkv[..., :d], qkv[..., d:2 * d], qkv[..., 2 * d:]
        k = ops.transpose(self.dense(ops.transpose(k, (0, 2, 1)), self.proj_k), (0, 2, 1))
        v = ops.transpose(self.dense(ops.transpose(v, (0, 2, 1)), self.proj_v), (0, 2, 1))
        h = self.attend(*(self.heads_of(t, self.heads) for t in (q, k, v)), causal=False)
        return self.dense(self.merge(h), self.out)
''',
    "Retention": '''\
class Retention(TorchLayout):
    """RetNet's multi-scale retention, parallel form."""

    def __init__(self, heads: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.heads = heads

    def build(self, input_shape):
        d = input_shape[-1]
        self.q = self.linear("q", d, d, bias=False)
        self.k = self.linear("k", d, d, bias=False)
        self.v = self.linear("v", d, d, bias=False)
        self.gate = self.linear("gate", d, d, bias=False)
        self.out = self.linear("out", d, d, bias=False)
        self.norm_weight = self.vector("norm_weight", (d,), "ones")
        self.norm_bias = self.vector("norm_bias", (d,), "zeros")

    def call(self, x):
        n, d = x.shape[1], x.shape[2]
        h, hd = self.heads, d // self.heads
        q, k, v = (self.heads_of(self.dense(x, w), h) for w in (self.q, self.k, self.v))
        i = ops.cast(ops.arange(n), x.dtype)
        gap = ops.expand_dims(i, 1) - ops.expand_dims(i, 0)
        gamma = 1 - ops.power(2.0, -5.0 - ops.cast(ops.arange(h), x.dtype))
        decay = ops.where(gap >= 0, ops.power(ops.reshape(gamma, (h, 1, 1)), ops.maximum(gap, 0)),
                          0.0)
        y = ops.matmul(ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * hd ** -0.5 * decay, v)
        y = ops.reshape(ops.transpose(y, (0, 2, 1, 3)), (-1, n, h, hd))   # group norm per head
        mean = ops.mean(y, axis=-1, keepdims=True)
        var = ops.mean(ops.square(y - mean), axis=-1, keepdims=True)
        y = ops.reshape((y - mean) * ops.rsqrt(var + 1e-5), (-1, n, d)) * self.norm_weight \\
            + self.norm_bias
        return self.dense(ops.silu(self.dense(x, self.gate)) * y, self.out)
''',
    "RWKVBlock": '''\
class RWKVBlock(TorchLayout):
    """One RWKV-4 layer: time mixing (a recurrence with a learned decay) and channel mixing."""

    def __init__(self, hidden: int, **kwargs):
        super().__init__(**kwargs)
        self.hidden = hidden

    def build(self, input_shape):
        d = input_shape[-1]
        half = keras.initializers.Constant(0.5)
        self.mix_k, self.mix_v, self.mix_r = (self.vector(f"mix_{c}", (d,), half) for c in "kvr")
        self.decay = self.vector("decay", (d,), ops.linspace(-5.0, 1.0, d))
        self.first = self.vector("first", (d,), 0.3)
        self.cmix_k = self.vector("cmix_k", (d,), half)
        self.cmix_r = self.vector("cmix_r", (d,), half)
        self.ln1 = (self.vector("ln1_weight", (d,), "ones"), self.vector("ln1_bias", (d,), "zeros"))
        self.ln2 = (self.vector("ln2_weight", (d,), "ones"), self.vector("ln2_bias", (d,), "zeros"))
        self.key = self.linear("key", d, d, bias=False)
        self.value = self.linear("value", d, d, bias=False)
        self.receptance = self.linear("receptance", d, d, bias=False)
        self.output_ = self.linear("output", d, d, bias=False)   # Layer.output is taken
        self.ckey = self.linear("ckey", d, self.hidden, bias=False)
        self.cvalue = self.linear("cvalue", self.hidden, d, bias=False)
        self.creceptance = self.linear("creceptance", d, d, bias=False)

    @staticmethod
    def shift(x):
        return ops.pad(x, ((0, 0), (1, 0), (0, 0)))[:, :-1]

    def wkv(self, k, v):
        w, u = -ops.exp(self.decay), self.first
        num = ops.zeros_like(k[:, 0])
        den = ops.zeros_like(k[:, 0])
        top = ops.zeros_like(k[:, 0]) - 1e38
        out = []
        for t in range(k.shape[1]):
            kt, vt = k[:, t], v[:, t]
            ww = u + kt
            p = ops.maximum(top, ww)
            e1, e2 = ops.exp(top - p), ops.exp(ww - p)
            out.append((e1 * num + e2 * vt) / (e1 * den + e2))
            ww = top + w
            p = ops.maximum(ww, kt)
            e1, e2 = ops.exp(ww - p), ops.exp(kt - p)
            num, den, top = e1 * num + e2 * vt, e1 * den + e2, p
        return ops.stack(out, 1)

    def call(self, x):
        h = self.layer_norm(x, *self.ln1)
        prev = self.shift(h)
        k = self.dense(h * self.mix_k + prev * (1 - self.mix_k), self.key)
        v = self.dense(h * self.mix_v + prev * (1 - self.mix_v), self.value)
        r = ops.sigmoid(self.dense(h * self.mix_r + prev * (1 - self.mix_r), self.receptance))
        x = x + self.dense(r * self.wkv(k, v), self.output_)
        h = self.layer_norm(x, *self.ln2)
        prev = self.shift(h)
        k = ops.square(ops.relu(self.dense(h * self.cmix_k + prev * (1 - self.cmix_k), self.ckey)))
        r = ops.sigmoid(self.dense(h * self.cmix_r + prev * (1 - self.cmix_r), self.creceptance))
        return x + r * self.dense(k, self.cvalue)
''',
    "HyenaOperator": '''\
class HyenaOperator(TorchLayout):
    """Hyena, order 2: a position-generated long causal convolution between two gates
    (computed directly as a lower-triangular product; PyTorch uses FFTs)."""

    def __init__(self, filter_hidden: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.filter_hidden = filter_hidden

    def build(self, input_shape):
        d = input_shape[-1]
        self.bias = self.vector("bias", (d,), "zeros")
        self.in_proj = self.linear("in_proj", d, 3 * d)
        self.short = self.depthwise("short", 3 * d, 3)
        self.filter_0 = self.linear("filter_0", 3, self.filter_hidden)
        self.filter_2 = self.linear("filter_2", self.filter_hidden, d)
        self.out = self.linear("out", d, d)

    def call(self, x):
        n, d = x.shape[1], x.shape[2]
        u = self.causal_conv(self.dense(x, self.in_proj), self.short)
        x1, x2, v = u[..., :d], u[..., d:2 * d], u[..., 2 * d:]
        t = ops.expand_dims(ops.linspace(0.0, 1.0, n), 1)
        z = ops.concatenate([t, ops.sin(6.283 * t), ops.cos(6.283 * t)], -1)
        filt = self.dense(ops.silu(self.dense(z, self.filter_0)), self.filter_2) * ops.exp(-4 * t)
        i = ops.arange(n)
        gap = ops.expand_dims(i, 1) - ops.expand_dims(i, 0)                # t - j
        toeplitz = ops.take(filt, ops.maximum(gap, 0), axis=0)               # [n, n, d]
        toeplitz = ops.where(ops.expand_dims(gap >= 0, -1), toeplitz, 0.0)
        g = v * x1
        y = ops.einsum("tjd,bjd->btd", toeplitz, g)
        return self.dense((y + g * self.bias) * x2, self.out)
''',
    "involution": '''\
def involution(x, kernel_size: int = 7, groups: int = 4, reduction: int = 4):
    """Involution (channels-last): a kxk kernel generated at every pixel, shared across the
    channels of a group."""
    h, w, c = x.shape[1], x.shape[2], x.shape[3]
    k, g = kernel_size, groups
    r = layers.Conv2D(max(c // reduction, 1), 1, use_bias=False)(x)
    r = layers.ReLU()(layers.BatchNormalization(epsilon=1e-5, momentum=0.9)(r))
    kernel = ops.reshape(layers.Conv2D(k * k * g, 1)(r), (-1, h, w, g, k * k))
    patches = ops.image.extract_patches(x, (k, k), strides=1, padding="same")
    patches = ops.reshape(patches, (-1, h, w, k * k, g, c // g))
    return ops.reshape(ops.einsum("bhwgt,bhwtgc->bhwgc", kernel, patches), (-1, h, w, c))
''',
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

KERAS_HELPERS.update(KERAS_FORMS)
