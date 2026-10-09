"""The kernel wired into a GP model, as a GPyTorch or scikit-learn expression.

Kernels form a tree: leaves are kernel blocks, inner nodes are Kernel Sum / Product
blocks whose ``kernels`` port collects children. GPyTorch expressions call the script's
``dims([...])`` helper for kernels restricted to some input columns.
"""
from __future__ import annotations

from ai_made_easy.core.gp.blocks import COMBINERS, KERNELS


class KernelError(ValueError):
    """The wired kernel cannot be built (message for the user)."""


def _names(value) -> list[str]:  # noqa: ANN001
    return [v.strip() for v in str(value or "").split(",") if v.strip()]


def model_node(graph):  # noqa: ANN001, ANN201
    return next((n for n in graph.nodes.values() if n.type_id == "gp.model"), None)


def kernel_root(graph):  # noqa: ANN001, ANN201
    model = model_node(graph)
    if model is None:
        return None
    edge = next((e for e in graph.incoming(model.instance_id) if e.target_port == "kernel"),
                None)
    return graph.nodes.get(edge.source_id) if edge else None


def leaves(graph, node, seen=None) -> list:  # noqa: ANN001
    seen = seen or set()
    if node.instance_id in seen:
        raise KernelError("the kernels are wired in a loop")
    seen = seen | {node.instance_id}
    if node.type_id in COMBINERS:
        out = []
        for e in graph.incoming(node.instance_id):
            out += leaves(graph, graph.nodes[e.source_id], seen)
        return out
    return [node]


def _children(graph, node) -> list:  # noqa: ANN001
    kids = [graph.nodes[e.source_id] for e in graph.incoming(node.instance_id)
            if e.source_id in graph.nodes]
    if not kids:
        raise KernelError(f"{node.definition().display_name} has no kernels wired into it")
    return sorted(kids, key=lambda n: n.instance_id)


def gpytorch_expr(graph, node) -> str | None:  # noqa: ANN001
    """GPyTorch kernel expression (None for kernels GPyTorch models elsewhere: White)."""
    if node.type_id in COMBINERS:
        parts = [p for p in (gpytorch_expr(graph, k) for k in _children(graph, node)) if p]
        if not parts:
            return None
        op = " + " if node.type_id == "gp.sum" else " * "
        return "(" + op.join(parts) + ")" if len(parts) > 1 else parts[0]
    p = dict(node.resolved_params())
    if node.type_id == "gp.white":
        return None
    if node.type_id == "gp.constant":
        return f"init(gpytorch.kernels.ConstantKernel(), constant={float(p['value'])!r})"
    cols = _names(p.get("columns"))
    kwargs = []
    if cols:
        kwargs.append(f"active_dims=dims({cols!r})")
    if p.get("ard"):
        kwargs.append(f"ard_num_dims={'len(' + repr(cols) + ')' if cols else 'len(FEATURES)'}")
    args = ", ".join(kwargs)
    inits = {}
    if node.type_id == "gp.rbf":
        base = f"gpytorch.kernels.RBFKernel({args})"
        inits["lengthscale"] = p["lengthscale"]
    elif node.type_id == "gp.matern":
        base = f"gpytorch.kernels.MaternKernel(nu={float(p['nu'])}{', ' + args if args else ''})"
        inits["lengthscale"] = p["lengthscale"]
    elif node.type_id == "gp.periodic":
        base = f"gpytorch.kernels.PeriodicKernel({args})"
        inits.update(period_length=p["period"], lengthscale=p["lengthscale"])
    elif node.type_id == "gp.linear":
        base = f"gpytorch.kernels.LinearKernel({args})"
    elif node.type_id == "gp.rational_quadratic":
        base = f"gpytorch.kernels.RQKernel({args})"
        inits["lengthscale"] = p["lengthscale"]
    elif node.type_id == "gp.spectral_mixture":
        ard = f"ard_num_dims={'len(' + repr(cols) + ')' if cols else 'len(FEATURES)'}"
        extra = f", active_dims=dims({cols!r})" if cols else ""
        base = f"gpytorch.kernels.SpectralMixtureKernel(num_mixtures={int(p['mixtures'])}, " \
               f"{ard}{extra})"
    else:
        raise KernelError(f"unknown kernel {node.type_id}")
    if inits:
        base = f"init({base}, " + ", ".join(f"{k}={float(v)!r}" for k, v in inits.items()) + ")"
    if p.get("scale"):
        base = f"gpytorch.kernels.ScaleKernel({base})"
    return base


def sklearn_expr(graph, node) -> str:  # noqa: ANN001
    if node.type_id in COMBINERS:
        parts = [sklearn_expr(graph, k) for k in _children(graph, node)]
        op = " + " if node.type_id == "gp.sum" else " * "
        return "(" + op.join(parts) + ")" if len(parts) > 1 else parts[0]
    p = dict(node.resolved_params())
    if "sklearn" not in KERNELS[node.type_id][3]:
        raise KernelError(f"{node.definition().display_name} needs the gpytorch library")
    if _names(p.get("columns")):
        raise KernelError("kernels restricted to some columns need the gpytorch library")
    length = (f"np.full(len(FEATURES), {float(p['lengthscale'])!r})" if p.get("ard")
              else repr(float(p.get("lengthscale", 1.0))))
    if node.type_id == "gp.rbf":
        base = f"RBF(length_scale={length})"
    elif node.type_id == "gp.matern":
        base = f"Matern(length_scale={length}, nu={float(p['nu'])})"
    elif node.type_id == "gp.periodic":
        base = (f"ExpSineSquared(length_scale={float(p['lengthscale'])!r}, "
                f"periodicity={float(p['period'])!r})")
    elif node.type_id == "gp.linear":
        base = "DotProduct()"
    elif node.type_id == "gp.rational_quadratic":
        base = f"RationalQuadratic(length_scale={float(p['lengthscale'])!r})"
    elif node.type_id == "gp.white":
        return f"WhiteKernel(noise_level={float(p['noise_level'])!r})"
    elif node.type_id == "gp.constant":
        return f"ConstantKernel(constant_value={float(p['value'])!r})"
    else:
        raise KernelError(f"unknown kernel {node.type_id}")
    return f"ConstantKernel(1.0) * {base}" if p.get("scale") else base
