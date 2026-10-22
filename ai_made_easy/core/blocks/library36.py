"""More layers (3.6): the building blocks of well-known CNNs, and a few layers people reach
for from papers and Keras / timm code.

ResNet's basic and bottleneck blocks, MobileNet / EfficientNet's MBConv, CBAM, DenseNet's
dense block, CoordConv, a Kolmogorov-Arnold (KAN) layer, highway and maxout layers, and
the sparsemax and Snake activations. Every one is checked by the codegen harness
(designed shape = real shape), against PyTorch's parameter count, and in Keras where it
has a Keras form. Helpers use ``nn.functional`` (not ``F``) because the generated file
imports ``F`` only when a layer line uses it.
"""
from __future__ import annotations

from ai_made_easy.core.blocks._dsl import P, ShapeError, nn_block
from ai_made_easy.core.codegen.helpers import KERAS_HELPERS, PYTORCH_HELPERS
from ai_made_easy.core.registry import get_registry

reg = get_registry()
_BEFORE = {b.type_id for b in reg.all()}

TORCH = {
    "ResidualBlock": '''\
class ResidualBlock(nn.Module):
    """ResNet's basic block (He et al., 2016): two 3x3 conv-BN layers and a shortcut
    (a 1x1 conv-BN projection when the shape changes)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.shortcut = nn.Identity() if stride == 1 and in_channels == out_channels else \\
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                          nn.BatchNorm2d(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.body(x) + self.shortcut(x))
''',
    "BottleneckBlock": '''\
class BottleneckBlock(nn.Module):
    """ResNet's bottleneck block: 1x1 reduce, 3x3, 1x1 expand (x expansion), plus a
    shortcut (projected when the shape changes)."""

    def __init__(self, in_channels: int, width: int, stride: int = 1,
                 expansion: int = 4) -> None:
        super().__init__()
        out = width * expansion
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, width, 1, bias=False), nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, stride, 1, bias=False), nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, out, 1, bias=False), nn.BatchNorm2d(out))
        self.shortcut = nn.Identity() if stride == 1 and in_channels == out else \\
            nn.Sequential(nn.Conv2d(in_channels, out, 1, stride, bias=False),
                          nn.BatchNorm2d(out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.body(x) + self.shortcut(x))
''',
    "MBConv": '''\
class MBConv(nn.Module):
    """Inverted residual (MobileNetV2 / EfficientNet): 1x1 expand, depthwise kxk,
    squeeze-and-excitation, 1x1 project; a residual when the shape is kept."""

    def __init__(self, in_channels: int, out_channels: int, expand_ratio: int = 4,
                 kernel_size: int = 3, stride: int = 1, se_ratio: float = 0.25) -> None:
        super().__init__()
        mid = in_channels * expand_ratio
        layers = []
        if expand_ratio != 1:
            layers += [nn.Conv2d(in_channels, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.SiLU()]
        layers += [nn.Conv2d(mid, mid, kernel_size, stride, kernel_size // 2, groups=mid,
                             bias=False), nn.BatchNorm2d(mid), nn.SiLU()]
        self.expand = nn.Sequential(*layers)
        squeeze = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(mid, squeeze, 1), nn.SiLU(),
                                nn.Conv2d(squeeze, mid, 1), nn.Sigmoid()) if se_ratio > 0 else None
        self.project = nn.Sequential(nn.Conv2d(mid, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels))
        self.residual = stride == 1 and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.expand(x)
        if self.se is not None:
            h = h * self.se(h)
        h = self.project(h)
        return x + h if self.residual else h
''',
    "CBAM": '''\
class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al., 2018): a channel gate from the
    average- and max-pooled features, then a spatial gate from a kxk convolution."""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(),
                                 nn.Linear(hidden, channels))
        self.spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.mlp(x.mean((2, 3))) + self.mlp(x.amax((2, 3))))
        x = x * gate[:, :, None, None]
        pooled = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.spatial(pooled))
''',
    "DenseBlock": '''\
class DenseBlock(nn.Module):
    """DenseNet's dense block (Huang et al., 2017): each layer (BN, ReLU, 3x3 conv) sees
    every earlier feature map and adds growth_rate channels."""

    def __init__(self, in_channels: int, layers: int = 4, growth_rate: int = 12) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Sequential(nn.BatchNorm2d(in_channels + i * growth_rate), nn.ReLU(),
                          nn.Conv2d(in_channels + i * growth_rate, growth_rate, 3, padding=1,
                                    bias=False))
            for i in range(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.cat([x, layer(x)], dim=1)
        return x
''',
    "CoordConv2d": '''\
class CoordConv2d(nn.Module):
    """CoordConv (Liu et al., 2018): a Conv2d that also sees each pixel's y and x
    position (two channels from -1 to 1)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, stride, padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        ys = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype)
        grid = torch.stack(torch.meshgrid(ys, xs, indexing="ij")).expand(b, 2, h, w)
        return self.conv(torch.cat([x, grid], dim=1))
''',
    "KANLinear": '''\
class KANLinear(nn.Module):
    """A Kolmogorov-Arnold layer (Liu et al., 2024) with Gaussian radial basis functions
    (FastKAN, Li 2024): every input-output edge learns its own 1-D function, plus a
    SiLU base path."""

    def __init__(self, in_features: int, units: int, grid_size: int = 8,
                 grid_range: float = 2.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.register_buffer("grid", torch.linspace(-grid_range, grid_range, grid_size))
        self.width = 2 * grid_range / max(grid_size - 1, 1)
        self.spline = nn.Linear(in_features * grid_size, units, bias=False)
        self.base = nn.Linear(in_features, units)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        basis = torch.exp(-((self.norm(x)[..., None] - self.grid) / self.width) ** 2)
        return self.spline(basis.flatten(-2)) + self.base(nn.functional.silu(x))
''',
    "Highway": '''\
class Highway(nn.Module):
    """Highway layer (Srivastava et al., 2015): a learned gate mixes a transformed
    input with the input itself."""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.transform = nn.Linear(features, features)
        self.gate = nn.Linear(features, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = torch.sigmoid(self.gate(x))
        return t * torch.relu(self.transform(x)) + (1 - t) * x
''',
    "Maxout": '''\
class Maxout(nn.Module):
    """Maxout (Goodfellow et al., 2013): the maximum of several linear pieces."""

    def __init__(self, in_features: int, units: int, pieces: int = 2) -> None:
        super().__init__()
        self.units, self.pieces = units, pieces
        self.linear = nn.Linear(in_features, units * pieces)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).unflatten(-1, (self.units, self.pieces)).amax(-1)
''',
    "Sparsemax": '''\
class Sparsemax(nn.Module):
    """Sparsemax (Martins & Astudillo, 2016) over the last axis: like softmax, but many
    outputs are exactly zero."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, _ = torch.sort(x, dim=-1, descending=True)
        k = torch.arange(1, x.shape[-1] + 1, device=x.device, dtype=x.dtype)
        support = 1 + k * z > z.cumsum(-1)
        count = support.sum(-1, keepdim=True)
        tau = ((z * support).sum(-1, keepdim=True) - 1) / count.to(x.dtype)
        return torch.clamp(x - tau, min=0)
''',
    "Snake": '''\
class Snake(nn.Module):
    """Snake activation (Ziyin et al., 2020): x + sin^2(a x) / a with a learned a per
    channel; periodic, so it suits audio and other oscillating signals."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.alpha.view(1, -1, *([1] * (x.ndim - 2)))
        return x + torch.sin(a * x) ** 2 / (a + 1e-9)
''',
}

KERAS = {
    "residual_block": '''\
def residual_block(x, filters: int, stride: int = 1):
    """ResNet basic block (channels-last)."""
    h = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
    h = layers.ReLU()(layers.BatchNormalization()(h))
    h = layers.BatchNormalization()(layers.Conv2D(filters, 3, padding="same", use_bias=False)(h))
    if stride != 1 or x.shape[-1] != filters:
        x = layers.BatchNormalization()(layers.Conv2D(filters, 1, strides=stride, use_bias=False)(x))
    return layers.ReLU()(h + x)
''',
    "bottleneck_block": '''\
def bottleneck_block(x, width: int, stride: int = 1, expansion: int = 4):
    """ResNet bottleneck block (channels-last)."""
    out = width * expansion
    h = layers.ReLU()(layers.BatchNormalization()(layers.Conv2D(width, 1, use_bias=False)(x)))
    h = layers.Conv2D(width, 3, strides=stride, padding="same", use_bias=False)(h)
    h = layers.ReLU()(layers.BatchNormalization()(h))
    h = layers.BatchNormalization()(layers.Conv2D(out, 1, use_bias=False)(h))
    if stride != 1 or x.shape[-1] != out:
        x = layers.BatchNormalization()(layers.Conv2D(out, 1, strides=stride, use_bias=False)(x))
    return layers.ReLU()(h + x)
''',
    "mbconv": '''\
def mbconv(x, filters: int, expand_ratio: int = 4, kernel_size: int = 3, stride: int = 1,
           se_ratio: float = 0.25):
    """MobileNetV2 / EfficientNet inverted residual (channels-last)."""
    c = x.shape[-1]
    h = x
    if expand_ratio != 1:
        h = layers.Activation("silu")(layers.BatchNormalization()(
            layers.Conv2D(c * expand_ratio, 1, use_bias=False)(h)))
    h = layers.DepthwiseConv2D(kernel_size, strides=stride, padding="same", use_bias=False)(h)
    h = layers.Activation("silu")(layers.BatchNormalization()(h))
    if se_ratio > 0:
        s = layers.GlobalAveragePooling2D(keepdims=True)(h)
        s = layers.Conv2D(max(1, int(c * se_ratio)), 1, activation="silu")(s)
        h = h * layers.Conv2D(h.shape[-1], 1, activation="sigmoid")(s)
    h = layers.BatchNormalization()(layers.Conv2D(filters, 1, use_bias=False)(h))
    return x + h if stride == 1 and c == filters else h
''',
    "cbam": '''\
def cbam(x, reduction: int = 16, kernel_size: int = 7):
    """CBAM channel then spatial attention (channels-last)."""
    c = x.shape[-1]
    hidden, out = layers.Dense(max(c // reduction, 1), activation="relu"), layers.Dense(c)
    avg, mx = ops.mean(x, axis=(1, 2)), ops.max(x, axis=(1, 2))
    gate = ops.sigmoid(out(hidden(avg)) + out(hidden(mx)))
    x = x * gate[:, None, None, :]
    pooled = ops.concatenate([ops.mean(x, axis=-1, keepdims=True),
                              ops.max(x, axis=-1, keepdims=True)], axis=-1)
    return x * layers.Conv2D(1, kernel_size, padding="same", activation="sigmoid")(pooled)
''',
    "dense_block": '''\
def dense_block(x, n_layers: int = 4, growth_rate: int = 12):
    """DenseNet dense block (channels-last)."""
    for _ in range(n_layers):
        h = layers.ReLU()(layers.BatchNormalization()(x))
        h = layers.Conv2D(growth_rate, 3, padding="same", use_bias=False)(h)
        x = layers.Concatenate()([x, h])
    return x
''',
    "coord_conv2d": '''\
def coord_conv2d(x, filters: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
    """CoordConv (channels-last): two position channels, then a Conv2D."""
    h, w = x.shape[1], x.shape[2]
    ys, xs = ops.meshgrid(ops.linspace(-1.0, 1.0, h), ops.linspace(-1.0, 1.0, w), indexing="ij")
    grid = ops.stack([ys, xs], axis=-1)[None]
    grid = ops.zeros_like(x[..., :1]) + ops.cast(grid, x.dtype)   # one copy per sample
    x = ops.concatenate([x, grid], axis=-1)
    if padding:
        x = layers.ZeroPadding2D(padding)(x)
    return layers.Conv2D(filters, kernel_size, strides=stride)(x)
''',
    "kan_dense": '''\
def kan_dense(x, units: int, grid_size: int = 8, grid_range: float = 2.0):
    """FastKAN layer: Gaussian radial basis on each normalized feature, plus SiLU base."""
    grid = ops.linspace(-grid_range, grid_range, grid_size)
    width = 2 * grid_range / max(grid_size - 1, 1)
    z = layers.LayerNormalization(epsilon=1e-5)(x)
    basis = ops.exp(-ops.square((z[..., None] - grid) / width))
    basis = ops.reshape(basis, (-1, *x.shape[1:-1], x.shape[-1] * grid_size))
    return layers.Dense(units, use_bias=False)(basis) + layers.Dense(units)(ops.silu(x))
''',
    "highway": '''\
def highway(x):
    """Highway layer: a sigmoid gate mixes relu(Wx + b) with x."""
    t = layers.Dense(x.shape[-1], activation="sigmoid")(x)
    return t * layers.Dense(x.shape[-1], activation="relu")(x) + (1 - t) * x
''',
    "maxout": '''\
def maxout(x, units: int, pieces: int = 2):
    """Maxout: the maximum of several linear pieces."""
    h = layers.Dense(units * pieces)(x)
    h = ops.reshape(h, (-1, *x.shape[1:-1], units, pieces))
    return ops.max(h, axis=-1)
''',
    "sparsemax": '''\
def sparsemax(x):
    """Sparsemax over the last axis."""
    z = ops.flip(ops.sort(x, axis=-1), axis=-1)
    k = ops.cast(ops.arange(1, x.shape[-1] + 1), x.dtype)
    support = ops.cast(1 + k * z > ops.cumsum(z, axis=-1), x.dtype)
    tau = (ops.sum(z * support, axis=-1, keepdims=True) - 1) / ops.sum(support, axis=-1,
                                                                     keepdims=True)
    return ops.maximum(x - tau, 0)
''',
    "Snake": '''\
class Snake(layers.Layer):
    """Snake activation with a learned alpha per channel (channels-last)."""

    def build(self, input_shape):
        self.alpha = self.add_weight(shape=(input_shape[-1],), initializer="ones")

    def call(self, x):
        return x + ops.square(ops.sin(self.alpha * x)) / (self.alpha + 1e-9)
''',
}

PYTORCH_HELPERS.update(TORCH)
KERAS_HELPERS.update(KERAS)


def _image(s, name: str) -> None:
    if len(s) != 3:
        raise ShapeError(f"{name} expects feature maps [C, H, W], got {s}")


def _down(d: int, stride: int) -> int:
    return (d - 1) // stride + 1          # a 3x3 / 1x1 conv with "same" padding


def _bn(c: int) -> int:
    return 2 * c


# ------------------------------------------------------------------ ResNet blocks

def _residual_shape(in_shapes, p):
    s = in_shapes[0]
    _image(s, "Residual Block")
    st = int(p["stride"])
    return [int(p["out_channels"]), _down(s[1], st), _down(s[2], st)]


def _residual_params(in_shapes, p) -> int:
    c, o, st = in_shapes[0][0], int(p["out_channels"]), int(p["stride"])
    n = c * o * 9 + _bn(o) + o * o * 9 + _bn(o)
    return n + (c * o + _bn(o) if st != 1 or c != o else 0)


reg.register(nn_block(
    "vision.residual_block", "Residual Block (ResNet)", "Vision Blocks", family="conv",
    params=(P("out_channels", "int", 64, lo=1), P("stride", "int", 1, lo=1, hi=4,
                                                    help="2 halves the height and width")),
    shape=_residual_shape, param_fn=_residual_params, layout="cl",
    torch="ResidualBlock({in_channels}, {out_channels}, stride={stride})",
    torch_helpers=("ResidualBlock",),
    keras_expr=lambda c: (f"residual_block({c['i0']}, {int(c['out_channels'])}, "
                          f"stride={int(c['stride'])})"),
    keras_helpers=("residual_block",),
    desc="ResNet's basic block (He et al., 2016): two 3×3 convolutions with batch norm "
         "and a shortcut around them, so deep stacks still train. Stack several to build "
         "a ResNet-18 / 34; stride 2 downsamples.",
))


def _bottleneck_shape(in_shapes, p):
    s = in_shapes[0]
    _image(s, "Bottleneck Block")
    st = int(p["stride"])
    return [int(p["width"]) * int(p["expansion"]), _down(s[1], st), _down(s[2], st)]


def _bottleneck_params(in_shapes, p) -> int:
    c, w, st = in_shapes[0][0], int(p["width"]), int(p["stride"])
    o = w * int(p["expansion"])
    n = c * w + _bn(w) + w * w * 9 + _bn(w) + w * o + _bn(o)
    return n + (c * o + _bn(o) if st != 1 or c != o else 0)


reg.register(nn_block(
    "vision.bottleneck_block", "Bottleneck Block (ResNet)", "Vision Blocks", family="conv",
    params=(P("width", "int", 64, lo=1, help="Channels of the inner 3×3 convolution"),
            P("expansion", "int", 4, lo=1, hi=8, help="Output channels = width × expansion"),
            P("stride", "int", 1, lo=1, hi=4)),
    shape=_bottleneck_shape, param_fn=_bottleneck_params, layout="cl",
    torch="BottleneckBlock({in_channels}, {width}, stride={stride}, expansion={expansion})",
    torch_helpers=("BottleneckBlock",),
    keras_expr=lambda c: (f"bottleneck_block({c['i0']}, {int(c['width'])}, "
                          f"stride={int(c['stride'])}, expansion={int(c['expansion'])})"),
    keras_helpers=("bottleneck_block",),
    desc="ResNet-50's bottleneck: a 1×1 convolution narrows the channels, a 3×3 works on "
         "them, a 1×1 widens them again (×4), with a shortcut around it all.",
))


# ------------------------------------------------------------------ MBConv

def _mbconv_shape(in_shapes, p):
    s = in_shapes[0]
    _image(s, "MBConv")
    k, st = int(p["kernel_size"]), int(p["stride"])
    if k % 2 == 0:
        raise ShapeError("MBConv: use an odd kernel size (3, 5, 7)")
    return [int(p["out_channels"]), _down(s[1], st), _down(s[2], st)]


def _mbconv_params(in_shapes, p) -> int:
    c, o, e = in_shapes[0][0], int(p["out_channels"]), int(p["expand_ratio"])
    k, se = int(p["kernel_size"]), float(p["se_ratio"])
    mid = c * e
    n = (c * mid + _bn(mid) if e != 1 else 0) + mid * k * k + _bn(mid)
    if se > 0:
        sq = max(1, int(c * se))
        n += mid * sq + sq + sq * mid + mid
    return n + mid * o + _bn(o)


reg.register(nn_block(
    "vision.mbconv", "MBConv (inverted residual)", "Vision Blocks", family="conv",
    params=(P("out_channels", "int", 32, lo=1), P("expand_ratio", "int", 4, lo=1, hi=8),
            P("kernel_size", "int", 3, lo=1, hi=9), P("stride", "int", 1, lo=1, hi=4),
            P("se_ratio", "float", 0.25, lo=0.0, hi=1.0,
              help="Squeeze-and-excitation width as a fraction of the input (0 = off)")),
    shape=_mbconv_shape, param_fn=_mbconv_params, layout="cl",
    torch=("MBConv({in_channels}, {out_channels}, expand_ratio={expand_ratio}, "
           "kernel_size={kernel_size}, stride={stride}, se_ratio={se_ratio})"),
    torch_helpers=("MBConv",),
    keras_expr=lambda c: (f"mbconv({c['i0']}, {int(c['out_channels'])}, "
                          f"expand_ratio={int(c['expand_ratio'])}, "
                          f"kernel_size={int(c['kernel_size'])}, stride={int(c['stride'])}, "
                          f"se_ratio={float(c['se_ratio'])})"),
    keras_helpers=("mbconv",),
    desc="The block of MobileNetV2 and EfficientNet: widen with a 1×1 convolution, filter "
         "each channel with a depthwise k×k, reweight channels (squeeze-and-excitation), "
         "narrow again; a shortcut when the shape stays. Cheap and accurate on phones.",
))


# ------------------------------------------------------------------ CBAM, dense block

def _cbam_shape(in_shapes, p):
    _image(in_shapes[0], "CBAM")
    if int(p["kernel_size"]) % 2 == 0:
        raise ShapeError("CBAM: use an odd kernel size (3 or 7)")
    return list(in_shapes[0])


def _cbam_params(in_shapes, p) -> int:
    c = in_shapes[0][0]
    h, k = max(c // int(p["reduction"]), 1), int(p["kernel_size"])
    return c * h + h + h * c + c + 2 * k * k + 1


reg.register(nn_block(
    "vision.cbam", "CBAM (channel + spatial attention)", "Attention", family="attention",
    params=(P("reduction", "int", 16, lo=1, help="Channel reduction ratio"),
            P("kernel_size", "int", 7, lo=1, hi=11, help="Spatial gate's kernel")),
    shape=_cbam_shape, param_fn=_cbam_params, layout="cl",
    torch="CBAM({in_channels}, reduction={reduction}, kernel_size={kernel_size})",
    torch_helpers=("CBAM",),
    keras_expr=lambda c: (f"cbam({c['i0']}, reduction={int(c['reduction'])}, "
                          f"kernel_size={int(c['kernel_size'])})"),
    keras_helpers=("cbam",),
    desc="Convolutional Block Attention Module (Woo et al., 2018): which channels matter "
         "(from average and max pooling), then where in the image (a k×k convolution). "
         "Drop it after any convolution; the shape stays.",
))


def _dense_shape(in_shapes, p):
    s = in_shapes[0]
    _image(s, "Dense Block")
    return [s[0] + int(p["layers"]) * int(p["growth_rate"]), s[1], s[2]]


def _dense_params(in_shapes, p) -> int:
    c, n, g = in_shapes[0][0], int(p["layers"]), int(p["growth_rate"])
    return sum(_bn(c + i * g) + (c + i * g) * g * 9 for i in range(n))


reg.register(nn_block(
    "vision.dense_block", "Dense Block (DenseNet)", "Vision Blocks", family="conv",
    params=(P("layers", "int", 4, lo=1, hi=48), P("growth_rate", "int", 12, lo=1, hi=128,
                                                    help="Channels each layer adds")),
    shape=_dense_shape, param_fn=_dense_params, layout="cl",
    torch="DenseBlock({in_channels}, layers={layers}, growth_rate={growth_rate})",
    torch_helpers=("DenseBlock",),
    keras_expr=lambda c: (f"dense_block({c['i0']}, n_layers={int(c['layers'])}, "
                          f"growth_rate={int(c['growth_rate'])})"),
    keras_helpers=("dense_block",),
    desc="DenseNet's block (Huang et al., 2017): every layer reads all the feature maps "
         "before it and adds growth-rate new ones ([C, H, W] → [C + layers × growth, H, W]). "
         "Follow it with a 1×1 convolution and pooling as a transition.",
))


# ------------------------------------------------------------------ CoordConv

def _coord_shape(in_shapes, p):
    s = in_shapes[0]
    _image(s, "CoordConv2D")
    k, st, pad = int(p["kernel_size"]), int(p["stride"]), int(p["padding"])
    h, w = (s[1] + 2 * pad - k) // st + 1, (s[2] + 2 * pad - k) // st + 1
    if h < 1 or w < 1:
        raise ShapeError("CoordConv2D: the kernel is larger than the padded input")
    return [int(p["out_channels"]), h, w]


reg.register(nn_block(
    "core.coord_conv2d", "CoordConv2D", "Convolution", family="conv",
    params=(P("out_channels", "int", 32, lo=1), P("kernel_size", "int", 3, lo=1),
            P("stride", "int", 1, lo=1), P("padding", "int", 1, lo=0)),
    shape=_coord_shape, layout="cl",
    param_fn=lambda s, p: (s[0][0] + 2) * int(p["out_channels"]) * int(p["kernel_size"]) ** 2
    + int(p["out_channels"]),
    torch=("CoordConv2d({in_channels}, {out_channels}, {kernel_size}, stride={stride}, "
           "padding={padding})"),
    torch_helpers=("CoordConv2d",),
    keras_expr=lambda c: (f"coord_conv2d({c['i0']}, {int(c['out_channels'])}, "
                          f"{int(c['kernel_size'])}, stride={int(c['stride'])}, "
                          f"padding={int(c['padding'])})"),
    keras_helpers=("coord_conv2d",),
    desc="A Conv2D that also sees each pixel's position (Liu et al., 2018): two extra "
         "channels hold y and x. Helps when where something is matters (localization, "
         "generating images, games).",
))


# ------------------------------------------------------------------ dense-style layers

def _features(name: str):
    def fn(in_shapes, p):
        s = in_shapes[0]
        if not s:
            raise ShapeError(f"{name} needs features on the last axis")
        return [*s[:-1], int(p["units"])] if "units" in p else list(s)

    return fn


reg.register(nn_block(
    "core.kan_dense", "KAN Layer (Kolmogorov-Arnold)", "Linear", family="model",
    params=(P("units", "int", 32, lo=1),
            P("grid_size", "int", 8, lo=2, hi=64, help="Basis functions per input"),
            P("grid_range", "float", 2.0, lo=0.1, hi=10.0,
              help="The basis covers −range…range of the normalized input")),
    shape=_features("KAN Layer"), layout="ir",
    param_fn=lambda s, p: (2 * s[0][-1] + s[0][-1] * int(p["grid_size"]) * int(p["units"])
                           + s[0][-1] * int(p["units"]) + int(p["units"])),
    torch=("KANLinear({features}, {units}, grid_size={grid_size}, "
           "grid_range={grid_range})"),
    torch_helpers=("KANLinear",),
    keras_expr=lambda c: (f"kan_dense({c['i0']}, {int(c['units'])}, "
                          f"grid_size={int(c['grid_size'])}, "
                          f"grid_range={float(c['grid_range'])})"),
    keras_helpers=("kan_dense",),
    desc="A Kolmogorov-Arnold Network layer (Liu et al., 2024; FastKAN's radial basis): "
         "instead of a weight, every input-output connection learns a 1-D curve. "
         "Interpretable fits of smooth formulas; a drop-in for Dense ([…, F] → […, units]).",
))

reg.register(nn_block(
    "core.highway", "Highway", "Linear", family="model",
    shape=_features("Highway"), layout="ir",
    param_fn=lambda s, p: 2 * (s[0][-1] ** 2 + s[0][-1]),
    torch="Highway({features})", torch_helpers=("Highway",),
    keras_expr="highway({i0})", keras_helpers=("highway",),
    desc="Highway layer (Srivastava et al., 2015): a learned gate decides, per feature, "
         "how much of a transformed input and how much of the input itself passes on — "
         "the idea residual networks simplified ([…, F] → same).",
))

reg.register(nn_block(
    "core.maxout", "Maxout", "Linear", family="model",
    params=(P("units", "int", 32, lo=1), P("pieces", "int", 2, lo=2, hi=16,
                                             help="Linear pieces per unit")),
    shape=_features("Maxout"), layout="ir",
    param_fn=lambda s, p: (s[0][-1] + 1) * int(p["units"]) * int(p["pieces"]),
    torch="Maxout({features}, {units}, pieces={pieces})", torch_helpers=("Maxout",),
    keras_expr=lambda c: f"maxout({c['i0']}, {int(c['units'])}, pieces={int(c['pieces'])})",
    keras_helpers=("maxout",),
    desc="Maxout (Goodfellow et al., 2013): each unit is the largest of several linear "
         "pieces, a learned piecewise-linear activation ([…, F] → […, units]).",
))


# ------------------------------------------------------------------ activations

reg.register(nn_block(
    "core.sparsemax", "Sparsemax", "Activations", family="activation",
    shape=_features("Sparsemax"), layout="ir",
    torch="Sparsemax()", torch_helpers=("Sparsemax",),
    keras_expr="sparsemax({i0})", keras_helpers=("sparsemax",),
    desc="Like softmax over the last axis, but many outputs are exactly zero (Martins & "
         "Astudillo, 2016): sparse attention weights and readable class scores.",
))


def _snake_shape(in_shapes, p):
    s = in_shapes[0]
    if not s:
        raise ShapeError("Snake needs a channel axis")
    return list(s)


reg.register(nn_block(
    "core.snake", "Snake", "Activations", family="activation",
    shape=_snake_shape, layout="cl",
    param_fn=lambda s, p: s[0][0],
    torch="Snake({in_channels})", torch_helpers=("Snake",),
    keras_expr="Snake()({i0})", keras_helpers=("Snake",),
    desc="x + sin²(αx)/α with a learned α per channel (Ziyin et al., 2020; BigVGAN): a "
         "periodic activation for audio, music and other signals that oscillate.",
))


LIBRARY36_IDS = tuple(sorted({b.type_id for b in reg.all()} - _BEFORE))
