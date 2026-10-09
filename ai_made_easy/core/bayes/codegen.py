"""Context for the probabilistic section of supervised PyTorch scripts (None: not needed)."""
from __future__ import annotations

from ai_made_easy.core.bayes.blocks import CALIBRATION, KL_LAYERS, STOCHASTIC


def probabilistic_context(graph) -> dict | None:  # noqa: ANN001
    nodes = list(graph.nodes.values())
    types = {n.type_id for n in nodes}
    if not types & {*STOCHASTIC, "bayes.mdn", "bayes.laplace", *CALIBRATION}:
        return None

    def params(type_id: str) -> dict | None:
        node = next((n for n in nodes if n.type_id == type_id), None)
        return dict(node.resolved_params()) if node is not None else None

    samples = [int(n.resolved_params()["samples"]) for n in nodes if n.type_id in STOCHASTIC]
    mdn = params("bayes.mdn")
    ece = params("eval.ece")
    conformal = params("eval.conformal")
    return {"mc_samples": max(samples) if samples else 1,
            "kl": bool(types & set(KL_LAYERS)),
            "mdn": {"k": int(mdn["components"]), "d": int(mdn["targets"])} if mdn else None,
            "laplace": params("bayes.laplace"),
            "ece_bins": int(ece["bins"]) if ece else 15,
            "temperature": "eval.temperature_scaling" in types,
            "conformal": float(conformal["alpha"]) if conformal else None}
