"""Probabilistic-programming blocks (PyMC): distributions whose parameters are constants or
wired parent variables, deterministic expressions, the sampler and generated data."""
from __future__ import annotations

from dataclasses import dataclass

from ai_made_easy.core.blocks._dsl import P
from ai_made_easy.core.palette import family_color
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import BlockDefinition, PortSpec

CATEGORY = "Probabilistic Programs"
# supports: real | positive | unit (0..1) | count (0, 1, 2, ...) | binary | category | vector
POSITIVE_PARAMS = {"sigma", "lam", "alpha", "beta", "b", "nu", "eta", "sd"}


@dataclass(frozen=True)
class Dist:
    type_id: str
    label: str
    pm: str                                    # pymc class
    params: tuple[tuple[str, float, str], ...]  # (name, default, support of the parameter)
    support: str                               # support of the variable itself
    desc: str


DISTRIBUTIONS: tuple[Dist, ...] = (
    Dist("ppl.normal", "Normal", "Normal", (("mu", 0.0, "real"), ("sigma", 1.0, "positive")),
         "real", "Bell curve: location mu, spread sigma. The default for unbounded effects."),
    Dist("ppl.student_t", "Student-t", "StudentT",
         (("nu", 4.0, "positive"), ("mu", 0.0, "real"), ("sigma", 1.0, "positive")), "real",
         "Heavy-tailed bell curve: robust to outliers (small nu = heavier tails)."),
    Dist("ppl.laplace", "Laplace", "Laplace", (("mu", 0.0, "real"), ("b", 1.0, "positive")),
         "real", "Sharp peak with exponential tails (a sparsity-favouring prior)."),
    Dist("ppl.cauchy", "Cauchy", "Cauchy", (("alpha", 0.0, "real"), ("beta", 1.0, "positive")),
         "real", "Very heavy tails: no mean or variance."),
    Dist("ppl.uniform", "Uniform", "Uniform", (("lower", 0.0, "real"), ("upper", 1.0, "real")),
         "real", "Equally likely between lower and upper."),
    Dist("ppl.half_normal", "Half-Normal", "HalfNormal", (("sigma", 1.0, "positive"),),
         "positive", "Positive values near zero: the usual prior for a scale (sigma)."),
    Dist("ppl.half_cauchy", "Half-Cauchy", "HalfCauchy", (("beta", 1.0, "positive"),),
         "positive", "Positive and heavy-tailed: a weakly informative prior for scales."),
    Dist("ppl.exponential", "Exponential", "Exponential", (("lam", 1.0, "positive"),),
         "positive", "Positive, mode at zero: waiting times, or a prior for scales."),
    Dist("ppl.log_normal", "Log-Normal", "LogNormal",
         (("mu", 0.0, "real"), ("sigma", 1.0, "positive")), "positive",
         "Positive and right-skewed: its logarithm is Normal (prices, durations)."),
    Dist("ppl.gamma", "Gamma", "Gamma", (("alpha", 2.0, "positive"), ("beta", 1.0, "positive")),
         "positive", "Positive and skewed: rates and precisions."),
    Dist("ppl.inverse_gamma", "Inverse-Gamma", "InverseGamma",
         (("alpha", 3.0, "positive"), ("beta", 1.0, "positive")), "positive",
         "Positive, keeps away from zero: a prior for variances."),
    Dist("ppl.weibull", "Weibull", "Weibull",
         (("alpha", 1.5, "positive"), ("beta", 1.0, "positive")), "positive",
         "Lifetimes and failure times."),
    Dist("ppl.beta", "Beta", "Beta", (("alpha", 1.0, "positive"), ("beta", 1.0, "positive")),
         "unit", "A probability between 0 and 1 (conversion rates, proportions)."),
    Dist("ppl.bernoulli", "Bernoulli", "Bernoulli", (("p", 0.5, "unit"),), "binary",
         "Yes / no outcomes with probability p (logistic regression likelihood)."),
    Dist("ppl.binomial", "Binomial", "Binomial", (("n", 10.0, "count"), ("p", 0.5, "unit")),
         "count", "Successes out of n trials."),
    Dist("ppl.poisson", "Poisson", "Poisson", (("mu", 3.0, "positive"),), "count",
         "Counts with mean mu (events per interval)."),
    Dist("ppl.negative_binomial", "Negative Binomial", "NegativeBinomial",
         (("mu", 3.0, "positive"), ("alpha", 2.0, "positive")), "count",
         "Over-dispersed counts (more variable than Poisson)."),
    Dist("ppl.dirichlet", "Dirichlet", "Dirichlet", (("a", 1.0, "positive"),), "vector",
         "Probabilities over k categories that sum to 1 (set size)."),
    Dist("ppl.categorical", "Categorical", "Categorical", (("p", 0.0, "vector"),), "category",
         "One of k categories with probabilities p (wire a Dirichlet into p)."),
)
BY_ID = {d.type_id: d for d in DISTRIBUTIONS}
SAMPLER_METHODS = ("nuts", "advi", "smc", "map")
PPL_DATA = ("data.synthetic_groups",)


def _dist_checks(dist: Dist):
    def checks(p: dict) -> list:
        out = []
        for name, _default, support in dist.params:
            value = p.get(name)
            if support == "positive" and value is not None and float(value) <= 0:
                out.append(("error", f"{name} must be positive"))
            if support == "unit" and value is not None and not 0 <= float(value) <= 1:
                out.append(("error", f"{name} must be between 0 and 1"))
            if name in ("sigma", "beta", "b") and value is not None and float(value) >= 1000:
                out.append(("info", f"{name} = {value} is a very wide prior: the data alone "
                                    "decide (consider a scale that matches the data)"))
        if dist.type_id == "ppl.uniform" and float(p["lower"]) >= float(p["upper"]):
            out.append(("error", "lower must be below upper"))
        return out
    return checks


def _distribution(dist: Dist) -> BlockDefinition:
    ports = tuple(PortSpec(name, dtype="config", role="variable")
                  for name, _d, _s in dist.params)
    numeric = tuple(P(name, "float", default,
                      lo=1e-12 if support == "positive" else (0.0 if support == "unit" else None),
                      hi=1.0 if support == "unit" else None,
                      help="Used when nothing is wired into this port")
                    for name, default, support in dist.params if support != "vector")
    common = (P("name", "str", dist.label.split("-")[0].lower().replace(" ", "_"),
                help="Variable name (used in expressions and results)"),
              *numeric,
              P("observed", "str", "", help="Data column this variable explains (empty: a "
                                            "latent / prior variable)"),
              P("group", "str", "",
                help="Hierarchical: one value per level of this data column (partial "
                     "pooling across groups)"),
              *((P("size", "int", 3, lo=2, hi=1000, help="Number of categories"),)
                if dist.type_id == "ppl.dirichlet" else ()),
              *((P("non_centered", "bool", True,
                   help="Grouped with wired mu and sigma: sample standardised offsets and "
                        "scale them (avoids divergences when groups have little data)"),)
                if dist.type_id == "ppl.normal" else ()))
    return BlockDefinition(
        type_id=dist.type_id, display_name=dist.label, category=CATEGORY,
        color=family_color("model"), params=common, inputs=ports,
        outputs=(PortSpec("out", dtype="config", role="variable"),), library="PyMC",
        description=dist.desc, checks_fn=_dist_checks(dist),
        meta={"ppl": "distribution", "support": dist.support})


def _deterministic() -> BlockDefinition:
    return BlockDefinition(
        type_id="ppl.deterministic", display_name="Deterministic", category=CATEGORY,
        color=family_color("model"), library="PyMC",
        inputs=(PortSpec("inputs", dtype="config", multi=True, role="variable"),),
        outputs=(PortSpec("out", dtype="config", role="variable"),),
        params=(P("name", "str", "mu"),
                P("expression", "str", "intercept + slope * x",
                  help="Python-style arithmetic over the wired variables and data columns; "
                       "functions: exp, log, sqrt, abs, invlogit, logit, softplus, "
                       "invprobit, sigmoid"),
                P("group", "str", "", help="Only for expressions over grouped variables")),
        description="A value computed from other variables and data columns, e.g. a linear "
                    "predictor 'intercept + slope * x' or a link 'invlogit(eta)'.",
        meta={"ppl": "deterministic", "support": "real"})


def _sampler() -> BlockDefinition:
    def checks(p: dict) -> list:
        out = []
        if p["method"] == "map":
            out.append(("info", "MAP finds the single most probable values: no posterior "
                                "uncertainty and no convergence diagnostics"))
        if p["method"] == "advi":
            out.append(("info", "ADVI is fast but approximate: it often underestimates "
                                "posterior uncertainty"))
        if p["method"] == "nuts" and int(p["chains"]) < 2:
            out.append(("warning", "R-hat needs at least two chains to check convergence"))
        return out

    return BlockDefinition(
        type_id="ppl.sampler", display_name="Sampler", category=CATEGORY,
        color=family_color("training"), library="PyMC", checks_fn=checks,
        params=(P("method", "enum", "nuts", options=SAMPLER_METHODS,
                  help="nuts: Hamiltonian Monte Carlo (exact, the default); advi: "
                       "variational approximation (fast); smc: sequential Monte Carlo "
                       "(multimodal posteriors); map: the most probable point"),
                P("draws", "int", 1000, lo=10, hi=1_000_000),
                P("tune", "int", 1000, lo=0, hi=1_000_000, help="Warm-up steps (discarded)"),
                P("chains", "int", 4, lo=1, hi=64),
                P("target_accept", "float", 0.9, lo=0.5, hi=0.999,
                  help="Higher = smaller steps: fewer divergences, slower"),
                P("advi_iterations", "int", 20000, lo=100, hi=10_000_000),
                P("backend", "enum", "pymc", options=("pymc", "numpyro"),
                  help="numpyro: JAX NUTS (often faster; needs numpyro installed)"),
                P("test_fraction", "float", 0.2, lo=0.0, hi=0.9,
                  help="Rows held out to check predictions"),
                P("seed", "int", 0, lo=0)),
        description="How the posterior is computed: NUTS sampling with convergence "
                    "diagnostics (R-hat, ESS, divergences), ADVI, SMC or MAP.",
        meta={"ppl": "config"})


def _datasets() -> list[BlockDefinition]:
    return [BlockDefinition(
        type_id="data.synthetic_groups", display_name="Synthetic Groups", category="Data",
        color=family_color("data"), library="PyMC", meta={"modality": "tabular", "ppl": "data"},
        params=(P("n_groups", "int", 8, lo=1, hi=1000),
                P("rows_per_group", "int", 25, lo=2, hi=100000,
                  help="Average rows per group (groups vary in size)"),
                P("intercept", "float", 1.0, help="Population mean intercept"),
                P("group_sd", "float", 0.8, lo=0.0, help="Spread of group intercepts"),
                P("slope", "float", 2.0), P("noise", "float", 0.5, lo=1e-6),
                P("outcome", "enum", "gaussian", options=("gaussian", "binary", "count")),
                P("seed", "int", 0, lo=0)),
        description="Generated grouped data (columns group, x, y) with known true "
                    "parameters: try hierarchical models and check they are recovered.")]


def register_all() -> None:
    reg = get_registry()
    for defn in (*(_distribution(d) for d in DISTRIBUTIONS), _deterministic(), _sampler(),
                 *_datasets()):
        reg.register(defn)
