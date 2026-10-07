"""Generated training scripts run end to end for every modality and task."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from verify_training import SCENARIOS, run_scenario  # noqa: E402

CASES = [(sc, fw) for sc in SCENARIOS for fw in sc.frameworks]


@pytest.mark.parametrize("scenario,framework", CASES,
                         ids=[f"{sc.name}-{fw}" for sc, fw in CASES])
def test_training_script_runs(scenario, framework, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("pandas")
    if framework == "keras":
        pytest.importorskip("keras")
    if scenario.name.startswith("images"):
        pytest.importorskip("torchvision")
        pytest.importorskip("PIL")
    run_scenario(scenario, framework, tmp_path)
