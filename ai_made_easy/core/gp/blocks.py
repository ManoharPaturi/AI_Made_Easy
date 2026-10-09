"""Gaussian-process blocks: kernels combined with Sum / Product, the GP model and data."""
from __future__ import annotations

from ai_made_easy.core.blocks._dsl import P
from ai_made_easy.core.palette import family_color
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import BlockDefinition, PortSpec

CATEGORY = "Gaussian Processes"
KERNEL_OUT = PortSpec("out", dtype="config", role="kernel")
# kernel type -> (label, params, description, libraries that support it)
KERNELS: dict[str, tuple] = {
    "gp.rbf": ("RBF Kernel", (P("lengthscale", "float", 1.0, lo=1e-6,
                                help="Initial distance over which values stay correlated"),),
               "Smooth functions (squared exponential).", ("gpytorch", "sklearn")),
    "gp.matern": ("Matérn Kernel", (P("nu", "enum", "2.5", options=("0.5", "1.5", "2.5")),
                                    P("lengthscale", "float", 1.0, lo=1e-6)),
                  "Rougher than RBF: nu 0.5 is jagged, 2.5 twice differentiable.",
                  ("gpytorch", "sklearn")),
    "gp.periodic": ("Periodic Kernel", (P("period", "float", 1.0, lo=1e-6,
                                          help="Initial period (in input units)"),
                                        P("lengthscale", "float", 1.0, lo=1e-6)),
                    "Patterns that repeat (seasonality).", ("gpytorch", "sklearn")),
    "gp.linear": ("Linear Kernel", (), "Linear trends (Bayesian linear regression).",
                  ("gpytorch", "sklearn")),
    "gp.rational_quadratic": ("Rational Quadratic Kernel",
                              (P("lengthscale", "float", 1.0, lo=1e-6),),
                              "A mixture of RBF kernels: variation at several scales.",
                              ("gpytorch", "sklearn")),
    "gp.spectral_mixture": ("Spectral Mixture Kernel",
                            (P("mixtures", "int", 4, lo=1, hi=50),),
                            "Learns the spectrum of the data: complex quasi-periodic "
                            "patterns (GPyTorch).", ("gpytorch",)),
    "gp.white": ("White Noise Kernel", (P("noise_level", "float", 0.1, lo=1e-9),),
                 "Independent noise (scikit-learn; GPyTorch learns noise in the likelihood).",
                 ("sklearn",)),
    "gp.constant": ("Constant Kernel", (P("value", "float", 1.0, lo=1e-9),),
                    "A constant offset shared by all points.", ("gpytorch", "sklearn")),
}
COMBINERS = ("gp.sum", "gp.product")
GP_DATA = ("data.synthetic_function",)


def _kernel(type_id: str) -> BlockDefinition:
    label, params, desc, libs = KERNELS[type_id]
    scale = () if type_id in ("gp.white", "gp.constant") else (
        P("scale", "bool", True, help="Learn an output scale (amplitude) for this kernel"),)
    columns = () if type_id in ("gp.white", "gp.constant") else (
        P("columns", "str", "", help="Input columns this kernel looks at (empty: all; "
                                     "GPyTorch only)"),
        P("ard", "bool", False, help="One lengthscale per input column (automatic "
                                     "relevance determination)"))
    return BlockDefinition(
        type_id=type_id, display_name=label, category=CATEGORY, color=family_color("model"),
        params=(*params, *scale, *columns), outputs=(KERNEL_OUT,),
        library=" · ".join({"gpytorch": "GPyTorch", "sklearn": "scikit-learn"}[x]
                           for x in libs),
        description=desc, meta={"gp": "kernel", "libraries": list(libs)})


def _combiner(type_id: str, label: str, desc: str) -> BlockDefinition:
    return BlockDefinition(
        type_id=type_id, display_name=label, category=CATEGORY, color=family_color("model"),
        inputs=(PortSpec("kernels", dtype="config", multi=True, role="kernel"),),
        outputs=(KERNEL_OUT,), library="GPyTorch · scikit-learn", description=desc,
        meta={"gp": "combiner"})


def _model() -> BlockDefinition:
    def checks(p: dict) -> list:
        out = []
        if p["likelihood"] != "gaussian" and p["kind"] == "exact" and p["library"] == "gpytorch":
            out.append(("error", f"a {p['likelihood']} likelihood needs a variational GP: set "
                                 "kind to svgp"))
        if p["library"] == "sklearn" and p["likelihood"] in ("poisson", "student_t"):
            out.append(("error", f"scikit-learn has no {p['likelihood']} likelihood: use the "
                                 "gpytorch library"))
        if p["library"] == "sklearn" and p["kind"] == "svgp":
            out.append(("info", "scikit-learn fits exact GPs; kind svgp applies to GPyTorch"))
        return out

    return BlockDefinition(
        type_id="gp.model", display_name="Gaussian Process", category=CATEGORY,
        color=family_color("model"), checks_fn=checks, library="GPyTorch · scikit-learn",
        inputs=(PortSpec("kernel", dtype="config", role="kernel"),),
        params=(P("target_column", "str", "y"),
                P("feature_columns", "str", "", help="Input columns (empty: every numeric "
                                                     "column except the target)"),
                P("library", "enum", "gpytorch", options=("gpytorch", "sklearn")),
                P("kind", "enum", "exact", options=("exact", "svgp"),
                  help="exact: all points (best up to a few thousand rows); svgp: sparse "
                       "variational GP with inducing points (large data, non-Gaussian "
                       "likelihoods)"),
                P("likelihood", "enum", "gaussian",
                  options=("gaussian", "student_t", "bernoulli", "poisson"),
                  help="gaussian / student_t: real values; bernoulli: 0 / 1 classes; "
                       "poisson: counts"),
                P("inducing_points", "int", 64, lo=4, hi=10000),
                P("iterations", "int", 200, lo=1, hi=100000),
                P("lr", "float", 0.05, lo=1e-6, hi=10.0),
                P("restarts", "int", 2, lo=0, hi=50,
                  help="scikit-learn: optimiser restarts for the kernel hyperparameters"),
                P("test_fraction", "float", 0.2, lo=0.0, hi=0.9),
                P("seed", "int", 0, lo=0)),
        description="A distribution over functions defined by its kernel: predictions with "
                    "calibrated uncertainty. Wire a kernel (or a Sum / Product of kernels) "
                    "into it.",
        meta={"gp": "model"})


def _datasets() -> list[BlockDefinition]:
    return [BlockDefinition(
        type_id="data.synthetic_function", display_name="Synthetic Function", category="Data",
        color=family_color("data"), library="NumPy", meta={"modality": "tabular", "gp": "data"},
        params=(P("kind", "enum", "trend_periodic",
                  options=("smooth", "periodic", "trend_periodic", "step", "classes_2d")),
                P("n_points", "int", 200, lo=10, hi=1_000_000),
                P("noise", "float", 0.2, lo=0.0), P("seed", "int", 0, lo=0)),
        description="Noisy samples of a known function (columns x, y; x1, x2, y for "
                    "classes_2d): compare kernels and see the uncertainty grow away from "
                    "the data.")]


def register_all() -> None:
    reg = get_registry()
    for defn in (*(_kernel(t) for t in KERNELS),
                 _combiner("gp.sum", "Kernel Sum", "Adds kernels: a function made of "
                                                   "independent parts (trend + seasonality)."),
                 _combiner("gp.product", "Kernel Product",
                           "Multiplies kernels: one pattern modulating another (a seasonal "
                           "shape that changes slowly)."),
                 _model(), *_datasets()):
        reg.register(defn)
