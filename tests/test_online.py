"""Pretrained weights, for real: every block that downloads weights builds with them and
gives its designed output shape. Opt-in (AIME_ONLINE=1); the nightly CI job runs it with
every extra installed. Offline runs use random weights (scripts/verify_codegen.py)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

if os.environ.get("AIME_ONLINE") != "1":
    pytest.skip("downloads pretrained weights: set AIME_ONLINE=1", allow_module_level=True)

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from verify_codegen import CONTEXTS, DEFAULT_OVERRIDES, _graph_for, block_cases  # noqa: E402

from ai_made_easy.core.codegen import class_name_for, generate  # noqa: E402
from ai_made_easy.core.registry import get_registry  # noqa: E402

PRETRAINED = sorted(t for t, o in DEFAULT_OVERRIDES.items() if o == {"weights": "none"})


def _build(graph, seed: int):  # noqa: ANN001, ANN202
    ns: dict = {"__name__": "aime_generated"}
    exec(compile(generate(graph, "pytorch"), f"<{graph.name}>", "exec"), ns)
    torch.manual_seed(seed)
    return ns[class_name_for(graph.name)]().eval()


@pytest.mark.parametrize("type_id", PRETRAINED)
def test_pretrained_block_downloads_and_runs(type_id):
    defn = get_registry().get(type_id)
    case = next((c for c in block_cases() if c[0] == type_id), None)
    if case is None:
        pytest.skip(f"{type_id}: its extra is not installed")
    _type, shape, params, _offline = case
    spec = next(p for p in defn.params if p.name == "weights")
    # the default, or the first real checkpoint when the default is random weights
    weights = spec.default if spec.default != "none" else \
        next(o for o in spec.options if o != "none")
    graph = _graph_for(type_id, shape, {**params, "weights": weights},
                       CONTEXTS.get(type_id))
    errors = [i.message for i in graph.validate() if i.severity == "error"]
    assert not errors, errors
    a, b = _build(graph, 0), _build(graph, 1)
    # pretrained weights do not depend on the seed; random initialisation would
    first = next(n for n, p in a.named_parameters() if p.dim() > 1)
    assert torch.equal(dict(a.named_parameters())[first], dict(b.named_parameters())[first])
    shapes = graph.infer_shapes()
    x = torch.randn(2, *shapes["inp"])
    with torch.no_grad():
        out = a(x)
    assert tuple(out.shape[1:]) == tuple(shapes["out"])
