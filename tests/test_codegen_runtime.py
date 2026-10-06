"""Generated code is executed, not just compiled: for every model-flow block
(and the risky parameter variants), the real PyTorch / Keras output shape must
equal the shape the designer inferred."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("KERAS_BACKEND", "torch")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from verify_codegen import VARIANTS, block_cases, run_keras, run_torch  # noqa: E402

from ai_made_easy.core.registry import get_registry  # noqa: E402

CASES = list(block_cases(VARIANTS))
IDS = [f"{t}-{'x'.join(map(str, s))}-{i}" for i, (t, s, _p, _g) in enumerate(CASES)]


def test_every_model_block_is_exercised():
    covered = {t for t, *_ in CASES}
    nn_blocks = {b.type_id for b in get_registry().all()
                 if b.shape_fn is not None and b.builder is None
                 and b.type_id not in ("core.input", "core.output")}
    assert nn_blocks - covered == set(), "blocks without a runnable test case"


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_pytorch_runtime_shape(case):
    pytest.importorskip("torch")
    _type_id, _shape, _params, graph = case
    assert run_torch(graph) == tuple(graph.infer_shapes()["out"])


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_keras_runtime_shape(case):
    pytest.importorskip("torch")
    pytest.importorskip("keras")
    type_id, _shape, _params, graph = case
    if not get_registry().get(type_id).supports("keras"):
        pytest.skip("PyTorch-only block")
    from ai_made_easy.core.codegen import CodegenError

    try:
        got, want = run_keras(graph)
    except CodegenError as exc:
        assert "no Keras equivalent" in str(exc)
        return
    assert got == want
