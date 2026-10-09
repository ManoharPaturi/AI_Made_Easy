"""Graphical-model blocks (pgmpy / hmmlearn): random variables wired parent → child, the
model kind, structure / parameter learning, inference, queries, HMMs and datasets."""
from __future__ import annotations

from ai_made_easy.core.blocks._dsl import P
from ai_made_easy.core.palette import family_color
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import BlockDefinition, PortSpec

CATEGORY = "Graphical Models"
VARIABLES = ("pgm.variable", "pgm.gaussian")
CONFIG = ("pgm.model", "pgm.structure_learning", "pgm.parameter_learning", "pgm.inference")
KINDS = ("bayesian_network", "markov_network", "naive_bayes", "dynamic_bn")
PGM_DATA = ("data.network_sample", "data.regime_series")
PARENTS = PortSpec("parents", dtype="config", multi=True, role="variable")
OUT = PortSpec("out", dtype="config", role="variable")


def _names(value) -> list[str]:  # noqa: ANN001
    return [v.strip() for v in str(value or "").split(",") if v.strip()]


def _variable_checks(p: dict) -> list:
    out = []
    states = _names(p.get("states"))
    if len(states) != len(set(states)):
        out.append(("error", "states repeat a name"))
    if p.get("states") and len(states) < 2:
        out.append(("error", "a discrete variable needs at least two states"))
    if p.get("latent") and not states:
        out.append(("error", "a latent (hidden) variable needs its states: the data cannot "
                             "tell how many there are"))
    return out


def _variables() -> list[BlockDefinition]:
    color = family_color("model")
    discrete = BlockDefinition(
        type_id="pgm.variable", display_name="Discrete Variable", category=CATEGORY,
        color=color, inputs=(PARENTS,), outputs=(OUT,), library="pgmpy",
        params=(P("name", "str", "X", help="Variable name = the data column it reads"),
                P("states", "str", "", help="Comma-separated states (empty: from the data)"),
                P("cpd", "table", "",
                  help="Conditional probability table P(variable | parents); empty: learned "
                       "from the data"),
                P("latent", "bool", False,
                  help="Hidden: not in the data, learned with expectation-maximization"),
                P("lagged_parents", "str", "",
                  help="Dynamic networks: variables whose previous time step is a parent "
                       "(use the variable's own name for persistence)")),
        checks_fn=_variable_checks,
        description="A random variable with named states. Wire parents into it: its "
                    "probability table conditions on them.",
        meta={"pgm": "variable"})
    gaussian = BlockDefinition(
        type_id="pgm.gaussian", display_name="Gaussian Variable", category=CATEGORY,
        color=color, inputs=(PARENTS,), outputs=(OUT,), library="pgmpy",
        params=(P("name", "str", "Y"),
                P("intercept", "float", 0.0, help="Mean when the parents are 0"),
                P("coefficients", "str", "",
                  help="One weight per parent, comma-separated (empty: learned)"),
                P("variance", "float", 1.0, lo=1e-9)),
        description="A continuous variable: a linear function of its parents plus Gaussian "
                    "noise (linear-Gaussian Bayesian networks).",
        meta={"pgm": "variable"})
    return [discrete, gaussian]


def _config() -> list[BlockDefinition]:
    color = family_color("training")

    def block(type_id: str, name: str, params: tuple, desc: str, checks=None):
        return BlockDefinition(type_id=type_id, display_name=name, category=CATEGORY,
                               color=color, params=params, description=desc, library="pgmpy",
                               checks_fn=checks, meta={"pgm": "config"})

    def _structure_checks(p: dict) -> list:
        if p["method"] in ("pc", "tree") and p["score"] != "bic":
            return [("info", f"the {p['method']} method does not use a score")]
        return []

    return [
        block("pgm.model", "Graphical Model",
              (P("kind", "enum", "bayesian_network", options=KINDS,
                 help="bayesian_network: directed, CPD tables; markov_network: undirected "
                      "factors; naive_bayes: one class variable explains every feature; "
                      "dynamic_bn: variables over time"),
               P("class_variable", "str", "", help="Naive Bayes: the class variable"),
               P("test_fraction", "float", 0.2, lo=0.0, hi=0.9,
                 help="Rows held out to score the model (log-likelihood)")),
              "The kind of graphical model the wired variables form."),
        block("pgm.structure_learning", "Structure Learning",
              (P("method", "enum", "ges", options=("ges", "hill_climb", "pc", "tree"),
                 help="ges: greedy equivalence search (finds the best-scoring class); "
                      "hill_climb: fast local search (can stop at a local optimum); pc: "
                      "independence tests; tree: Chow-Liu tree"),
               P("score", "enum", "bic", options=("bic", "bdeu", "k2", "aic", "bds")),
               P("max_indegree", "int", 0, lo=0, hi=20, help="Most parents per variable "
                                                             "(0: no limit)"),
               P("significance", "float", 0.01, lo=1e-6, hi=0.5,
                 help="PC: independence-test level"),
               P("start_from_design", "bool", True,
                 help="Hill climbing starts from the wired edges")),
              "Learns which variables depend on which from the data; the learned edges are "
              "reported and compared to the design (edges whose direction the data cannot "
              "tell apart count as skeleton matches).",
              _structure_checks),
        block("pgm.parameter_learning", "Parameter Learning",
              (P("estimator", "enum", "mle", options=("mle", "bayesian", "em"),
                 help="mle: frequencies; bayesian: frequencies plus a Dirichlet prior "
                      "(robust with little data); em: expectation-maximization for hidden "
                      "variables or missing values"),
               P("prior", "enum", "BDeu", options=("BDeu", "K2")),
               P("equivalent_sample_size", "float", 5.0, lo=0.0, hi=1e6),
               P("iterations", "int", 100, lo=1, hi=10000)),
              "How the probability tables are estimated from the data."),
        block("pgm.inference", "Inference",
              (P("method", "enum", "variable_elimination",
                 options=("variable_elimination", "belief_propagation", "sampling")),
               P("samples", "int", 5000, lo=100, hi=10_000_000)),
              "How queries are answered: exact variable elimination or belief propagation, "
              "or approximate sampling for large networks."),
        BlockDefinition(
            type_id="pgm.query", display_name="Query", category=CATEGORY, color=color,
            library="pgmpy", meta={"pgm": "query"},
            params=(P("variables", "str", "", help="Comma-separated variables to ask about"),
                    P("evidence", "str", "", help="Observed values, e.g. G=C, S=high"),
                    P("kind", "enum", "marginal", options=("marginal", "map"),
                      help="marginal: probability of each state; map: the most likely joint "
                           "assignment")),
            description="A question for the model: P(variables | evidence) or the most "
                        "probable states."),
        BlockDefinition(
            type_id="pgm.hmm", display_name="Hidden Markov Model", category=CATEGORY,
            color=family_color("model"), library="hmmlearn", meta={"pgm": "hmm"},
            params=(P("observations", "str", "value",
                      help="Comma-separated observed columns"),
                    P("sequence_column", "str", "",
                      help="Column naming each sequence (empty: one sequence)"),
                    P("emission", "enum", "gaussian",
                      options=("gaussian", "gmm", "categorical", "poisson")),
                    P("n_states", "int", 3, lo=1, hi=100),
                    P("mixtures", "int", 2, lo=1, hi=20, help="GMM components per state"),
                    P("covariance", "enum", "diag", options=("diag", "full", "spherical")),
                    P("iterations", "int", 100, lo=1, hi=10000),
                    P("restarts", "int", 5, lo=1, hi=100,
                      help="EM runs from different starts; the most likely one is kept (EM "
                           "can stop in poor local optima)"),
                    P("test_fraction", "float", 0.2, lo=0.0, hi=0.9),
                    P("seed", "int", 0, lo=0)),
            description="Hidden regimes behind a sequence (e.g. calm / volatile markets): "
                        "learns transition and emission probabilities, then decodes the "
                        "most likely state at every step (Viterbi)."),
    ]


def _datasets() -> list[BlockDefinition]:
    color = family_color("data")
    return [
        BlockDefinition(
            type_id="data.network_sample", display_name="Network Sample", category="Data",
            color=color, library="pgmpy", meta={"modality": "tabular", "pgm": "data"},
            params=(P("n_rows", "int", 2000, lo=10, hi=10_000_000), P("seed", "int", 0, lo=0)),
            description="Rows sampled from the designed network's own probability tables: "
                        "check that structure learning recovers it."),
        BlockDefinition(
            type_id="data.regime_series", display_name="Regime Series", category="Data",
            color=color, library="hmmlearn", meta={"modality": "timeseries", "pgm": "data"},
            params=(P("n_steps", "int", 1500, lo=20), P("n_regimes", "int", 3, lo=1, hi=10),
                    P("stickiness", "float", 0.97, lo=0.0, hi=0.9999,
                      help="Probability of staying in the same regime"),
                    P("seed", "int", 0, lo=0)),
            description="A generated series that switches between regimes with different "
                        "means and volatility (column 'value', true regime in 'regime')."),
    ]


def register_all() -> None:
    reg = get_registry()
    for defn in (*_variables(), *_config(), *_datasets()):
        reg.register(defn)
