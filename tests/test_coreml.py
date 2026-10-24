"""Core ML export of the 3.7 and 3.8 layers (3.9).

Each layer's harness design (scripts/verify_codegen.py) is exported to Core ML. On macOS
the converted program runs and must match PyTorch within the float16 tolerance
(``exporters.HALF_TOL``); elsewhere it must convert. A layer that can't is listed in the
export rules (``deploy/exportability.py``) with the reason, so users see it before
exporting.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("coremltools")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from test_export import _build  # noqa: E402
from test_library37 import VARIANTS as VARIANTS37  # noqa: E402
from test_library38 import VARIANTS as VARIANTS38  # noqa: E402
from verify_codegen import block_cases  # noqa: E402

from ai_made_easy.core.blocks.library37 import LIBRARY37_IDS  # noqa: E402
from ai_made_easy.core.blocks.library38 import LIBRARY38_IDS  # noqa: E402
from ai_made_easy.core.deploy import exporters as ex  # noqa: E402
from ai_made_easy.core.deploy.exportability import blockers  # noqa: E402

LAYERS = set(LIBRARY37_IDS) | set(LIBRARY38_IDS)


def _cases():
    seen, out = set(), []
    for case in block_cases({**VARIANTS37, **VARIANTS38}):
        if case[0] in LAYERS and case[0] not in seen:   # one design per layer
            seen.add(case[0])
            out.append(case)
    return out


@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.parametrize("case", _cases(), ids=lambda c: c[0])
def test_layer_exports_to_core_ml_or_says_why_not(case, tmp_path):
    type_id, _shape, _params, graph = case
    blocked = blockers(graph)
    model, samples = _build(graph)
    report = ex.export_all(model, "pytorch", ["coreml"], tmp_path, samples,
                           log=lambda _m: None)["coreml"]
    if "coreml" in blocked:
        assert blocked["coreml"][0], "a blocked format needs a reason"
        assert not report["ok"], f"{type_id} exports to Core ML now: lift its export rule"
        return
    assert report["ok"], f"{type_id}: {report.get('error')}"
    if sys.platform == "darwin":
        assert report["note"].startswith("matches PyTorch"), report["note"]
