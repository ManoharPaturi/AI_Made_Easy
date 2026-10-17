"""Normalizing-flow layers (RealNVP, MAF, neural spline flows, ActNorm, permutations) and
the 2-D density datasets they are tried on."""
from __future__ import annotations

from dataclasses import replace

from ai_made_easy.core.blocks._dsl import P, ShapeError, nn_block
from ai_made_easy.core.palette import family_color
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import BlockDefinition

CATEGORY = "Normalizing Flows"
GLOW_LAYERS = ("flow.invconv1x1", "flow.conv_coupling", "flow.squeeze", "flow.split")
FLOW_LAYERS = ("flow.affine_coupling", "flow.maf", "flow.spline_coupling", "flow.actnorm",
               "flow.permute", *GLOW_LAYERS)
COUPLINGS = ("flow.affine_coupling", "flow.spline_coupling", "flow.conv_coupling")
DENSITY_DATA = ("data.density_2d", "data.density_images")
IMAGE_DATA = ("data.density_images", "data.image_folder", "data.torchvision")
IMAGE_KINDS = ("shapes", "bars", "blobs")
DENSITY_KINDS = ("moons", "circles", "eight_gaussians", "checkerboard", "spiral", "pinwheel")

HIDDEN = P("hidden", "int", 64, lo=4, hi=4096, help="Width of the conditioner network")
LAYERS = P("layers", "int", 2, lo=1, hi=8, help="Hidden layers of the conditioner network")
PARITY = P("parity", "enum", "even", options=("even", "odd"),
           help="Which half passes through unchanged: alternate it between coupling layers")


def _vector(in_shapes, p):
    s = in_shapes[0]
    if len(s) != 1:
        raise ShapeError(f"flows here transform feature vectors [D], got {s}")
    return list(s)


def _coupling_shape(in_shapes, p):
    s = _vector(in_shapes, p)
    if s[0] < 2:
        raise ShapeError("a coupling layer needs at least 2 dimensions (it splits them)")
    return s


def _mlp_params(n_in: int, n_out: int, hidden: int, layers: int) -> int:
    total, width = 0, n_in
    for _ in range(layers):
        total += width * hidden + hidden
        width = hidden
    return total + width * n_out + n_out


def _affine_params(in_shapes, p):
    d = in_shapes[0][0]
    return _mlp_params(d, 2 * d, int(p["hidden"]), int(p["layers"]))


def _spline_params(in_shapes, p):
    d = in_shapes[0][0]
    return _mlp_params(d, d * (3 * int(p["bins"]) - 1), int(p["hidden"]), int(p["layers"]))


def _image(in_shapes, p):
    s = in_shapes[0]
    if len(s) != 3:
        raise ShapeError(f"Glow layers transform images [C, H, W], got {s}")
    return list(s)


def _vector_or_image(in_shapes, p):
    s = in_shapes[0]
    if len(s) not in (1, 3):
        raise ShapeError(f"ActNorm takes feature vectors [D] or images [C, H, W], got {s}")
    return list(s)


def _two_channels(in_shapes, p):
    s = _image(in_shapes, p)
    if s[0] < 2:
        raise ShapeError("a coupling layer needs at least 2 channels: put a Squeeze before "
                         "it")
    return s


def _squeeze(in_shapes, p):
    c, h, w = _image(in_shapes, p)
    if h % 2 or w % 2:
        raise ShapeError(f"Squeeze needs an even height and width, got {h}×{w}")
    return [4 * c, h // 2, w // 2]


def _split(in_shapes, p):
    c, h, w = _image(in_shapes, p)
    if c < 2:
        raise ShapeError("Split needs at least 2 channels: put a Squeeze before it")
    return [c // 2, h, w]


def _conv_coupling_params(in_shapes, p):
    c, hidden = in_shapes[0][0], int(p["hidden"])
    keep = c // 2
    return (keep * hidden * 9 + hidden) + (hidden * hidden + hidden) + \
        (hidden * 2 * (c - keep) * 9 + 2 * (c - keep))


def _glow_layers() -> list[BlockDefinition]:
    glow = ("FlowBase", "Glow")
    return [
        nn_block("flow.invconv1x1", "Invertible 1×1 Conv (Glow)", CATEGORY, family="model",
                 params=(P("seed", "int", 0, lo=0),), shape=_image,
                 param_fn=lambda s, p: s[0][0] * s[0][0],
                 torch=lambda c: f"InvConv1x1({c['input_shape'][0]}, {int(c['seed'])})",
                 torch_helpers=glow,
                 desc="A learned, invertible mixing of the channels (LU-decomposed, so its "
                      "log-determinant is cheap): Glow's replacement for fixed permutations."),
        nn_block("flow.conv_coupling", "Affine Coupling (Glow, conv)", CATEGORY,
                 family="model",
                 params=(P("hidden", "int", 64, lo=4, hi=1024,
                           help="Channels of the conditioner CNN"), PARITY),
                 shape=_two_channels, param_fn=_conv_coupling_params,
                 torch=lambda c: (f"ConvCoupling({c['input_shape'][0]}, {int(c['hidden'])}, "
                                  f"{c['parity']!r})"),
                 torch_helpers=glow,
                 desc="Scales and shifts half of the channels by a small CNN of the other "
                      "half: the image version of the RealNVP coupling (Glow)."),
        nn_block("flow.squeeze", "Squeeze (Flow)", CATEGORY, family="model", shape=_squeeze,
                 param_fn=lambda s, p: 0, torch=lambda c: "FlowSqueeze()",
                 torch_helpers=glow,
                 desc="[C, H, W] → [4C, H/2, W/2]: folds each 2×2 patch into channels so the "
                      "next layers see a larger neighbourhood."),
        nn_block("flow.split", "Split (multi-scale)", CATEGORY, family="model", shape=_split,
                 param_fn=lambda s, p: s[0][0] // 2 * 9 * 2 * (s[0][0] - s[0][0] // 2)
                 + 2 * (s[0][0] - s[0][0] // 2),
                 torch=lambda c: f"FlowSplit({c['input_shape'][0]})", torch_helpers=glow,
                 desc="Half of the channels leave the flow here, scored by a Gaussian "
                      "predicted from the other half: later layers work on less (Glow's "
                      "multi-scale architecture)."),
    ]


def _layers() -> list[BlockDefinition]:
    base = ("FlowBase",)
    blocks = [
        nn_block("flow.affine_coupling", "Affine Coupling (RealNVP)", CATEGORY,
                 family="model", params=(HIDDEN, LAYERS, PARITY), shape=_coupling_shape,
                 param_fn=_affine_params,
                 torch=lambda c: (f"AffineCoupling({c['input_shape'][0]}, {int(c['hidden'])}, "
                                  f"{int(c['layers'])}, {c['parity']!r})"),
                 torch_helpers=(*base, "AffineCoupling"),
                 desc="Scales and shifts half of the dimensions by functions of the other "
                      "half: invertible with a cheap log-determinant (RealNVP)."),
        nn_block("flow.maf", "Masked Autoregressive Layer (MAF)", CATEGORY, family="model",
                 params=(HIDDEN, LAYERS,
                         P("reverse", "bool", False,
                           help="Autoregressive order last-to-first (alternate between "
                                "layers)")),
                 shape=_vector,
                 param_fn=lambda s, p: _mlp_params(s[0][0], 2 * s[0][0], int(p["hidden"]),
                                                   int(p["layers"])),
                 torch=lambda c: (f"MAFLayer({c['input_shape'][0]}, {int(c['hidden'])}, "
                                  f"{int(c['layers'])}, {bool(c['reverse'])})"),
                 torch_helpers=(*base, "MAFLayer"),
                 desc="Each dimension is shifted and scaled by a function of the dimensions "
                      "before it (MADE). Exact density in one pass; sampling is sequential."),
        nn_block("flow.spline_coupling", "Spline Coupling (NSF)", CATEGORY, family="model",
                 params=(HIDDEN, LAYERS,
                         P("bins", "int", 8, lo=2, hi=64, help="Spline segments"),
                         P("bound", "float", 4.0, lo=0.5, hi=50.0,
                           help="Splines act on [-bound, bound]; identity outside (data is "
                                "standardised)"), PARITY),
                 shape=_coupling_shape, param_fn=_spline_params,
                 torch=lambda c: (f"SplineCoupling({c['input_shape'][0]}, {int(c['hidden'])}, "
                                  f"{int(c['layers'])}, {int(c['bins'])}, "
                                  f"{float(c['bound'])}, {c['parity']!r})"),
                 torch_helpers=(*base, "SplineCoupling"),
                 desc="Monotonic rational-quadratic splines for half of the dimensions: far "
                      "more flexible than affine couplings (neural spline flows)."),
        nn_block("flow.actnorm", "ActNorm", CATEGORY, family="model",
                 shape=_vector_or_image, param_fn=lambda s, p: 2 * s[0][0],
                 torch=lambda c: f"ActNorm({c['input_shape'][0]})",
                 torch_helpers=(*base, "ActNorm"),
                 desc="Learned per-dimension (or, on images, per-channel) scale and shift, "
                      "initialised from the first batch (Glow)."),
        nn_block("flow.permute", "Flow Permutation", CATEGORY, family="model",
                 params=(P("kind", "enum", "reverse", options=("reverse", "random")),
                         P("seed", "int", 0, lo=0)),
                 shape=_vector, param_fn=lambda s, p: 0,
                 torch=lambda c: (f"FlowPermute({c['input_shape'][0]}, {c['kind']!r}, "
                                  f"{int(c['seed'])})"),
                 torch_helpers=(*base, "FlowPermute"),
                 desc="Reorders the dimensions so the next layer transforms different ones."),
        *_glow_layers(),
    ]
    return [replace(b, meta={**(b.meta or {}), "flow": True}) for b in blocks]


def _data() -> BlockDefinition:
    return BlockDefinition(
        type_id="data.density_2d", display_name="2-D Density Data", category="Data",
        color=family_color("data"), library="PyTorch",
        params=(P("kind", "enum", "moons", options=DENSITY_KINDS),
                P("n_samples", "int", 5000, lo=100), P("noise", "float", 0.05, lo=0.0, hi=1.0),
                P("seed", "int", 0, lo=0),
                P("val_fraction", "float", 0.1, lo=0.0, hi=0.5),
                P("test_fraction", "float", 0.1, lo=0.0, hi=0.5)),
        description="Points from a 2-D distribution (moons, rings, a mixture of eight "
                    "Gaussians, a checkerboard, a spiral or a pinwheel) for learning densities "
                    "you can see.",
        meta={"modality": "tabular", "density": True})


def _image_data() -> BlockDefinition:
    return BlockDefinition(
        type_id="data.density_images", display_name="Density Images", category="Data",
        color=family_color("data"), library="PyTorch",
        params=(P("kind", "enum", "shapes", options=IMAGE_KINDS),
                P("size", "enum", "16", options=("8", "16", "32"),
                  help="Height and width in pixels"),
                P("n_samples", "int", 4000, lo=100), P("seed", "int", 0, lo=0),
                P("val_fraction", "float", 0.1, lo=0.0, hi=0.5),
                P("test_fraction", "float", 0.1, lo=0.0, hi=0.5)),
        description="Generated 8-bit greyscale images (rectangles and discs, bars, or soft "
                    "blobs) for image flows: likelihoods are reported in bits per dimension.",
        meta={"modality": "image", "density": True})


def register_all() -> None:
    reg = get_registry()
    for defn in (*_layers(), _data(), _image_data()):
        reg.register(defn)
