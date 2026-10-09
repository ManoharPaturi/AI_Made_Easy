"""Normalizing-flow layers (RealNVP, MAF, neural spline flows, ActNorm, permutations) and
the 2-D density datasets they are tried on."""
from __future__ import annotations

from dataclasses import replace

from ai_made_easy.core.blocks._dsl import P, ShapeError, nn_block
from ai_made_easy.core.palette import family_color
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import BlockDefinition

CATEGORY = "Normalizing Flows"
FLOW_LAYERS = ("flow.affine_coupling", "flow.maf", "flow.spline_coupling", "flow.actnorm",
               "flow.permute")
COUPLINGS = ("flow.affine_coupling", "flow.spline_coupling")
DENSITY_DATA = ("data.density_2d",)
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
                 shape=_vector, param_fn=lambda s, p: 2 * s[0][0],
                 torch=lambda c: f"ActNorm({c['input_shape'][0]})",
                 torch_helpers=(*base, "ActNorm"),
                 desc="Learned per-dimension scale and shift, initialised from the first "
                      "batch (Glow)."),
        nn_block("flow.permute", "Flow Permutation", CATEGORY, family="model",
                 params=(P("kind", "enum", "reverse", options=("reverse", "random")),
                         P("seed", "int", 0, lo=0)),
                 shape=_vector, param_fn=lambda s, p: 0,
                 torch=lambda c: (f"FlowPermute({c['input_shape'][0]}, {c['kind']!r}, "
                                  f"{int(c['seed'])})"),
                 torch_helpers=(*base, "FlowPermute"),
                 desc="Reorders the dimensions so the next layer transforms different ones."),
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


def register_all() -> None:
    reg = get_registry()
    for defn in (*_layers(), _data()):
        reg.register(defn)
