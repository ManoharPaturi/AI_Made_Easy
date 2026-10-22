"""Release consistency: one version everywhere, extras that really provide what blocks and
recipes ask for, and a FAMILIES.md that matches the registries."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import ai_made_easy

ROOT = Path(__file__).resolve().parent.parent
tomllib = pytest.importorskip("tomllib")  # Python 3.11+


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_one_version_everywhere():
    version = _pyproject()["project"]["version"]
    assert ai_made_easy.__version__ == version
    assert json.loads((ROOT / "web/package.json").read_text(encoding="utf-8"))["version"] == version
    assert json.loads((ROOT / "web/package-lock.json").read_text(encoding="utf-8"))["version"] == version
    assert f'"CFBundleShortVersionString": "{version}"' in \
        (ROOT / "packaging/ai_made_easy.spec").read_text(encoding="utf-8")
    assert f"\n## {version}\n" in (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")


def _norm(name: str) -> str:
    return re.split(r"[<>=\[ ;]", name)[0].lower().replace("_", "-")


# import names that differ from the pip package name
PIP_NAMES = {"umap": "umap-learn", "imblearn": "imbalanced-learn"}


def test_extras_provide_what_blocks_and_recipes_require():
    from ai_made_easy.core.recipes import all_recipes
    from ai_made_easy.core.registry import get_registry

    extras = {k: {_norm(p) for p in v}
              for k, v in _pyproject()["project"]["optional-dependencies"].items()}
    items = [(b.type_id, b.requires, b.extra) for b in get_registry().all() if b.requires]
    items += [(r.id, r.requires, r.extra) for r in all_recipes() if r.requires]
    assert items
    for item, requires, extra in items:
        assert extra in extras, f"{item} names the extra {extra!r}, which does not exist"
        missing = [m for m in requires if _norm(PIP_NAMES.get(m, m)) not in extras[extra]]
        assert not missing, f"{item}: pip install ai-made-easy[{extra}] lacks {missing}"


def test_every_family_extra_exists():
    from ai_made_easy.core.families import all_families

    extras = _pyproject()["project"]["optional-dependencies"]
    for family in all_families():
        assert set(family.extras) <= set(extras), family.id


def test_families_doc_is_current():
    proc = subprocess.run([sys.executable, str(ROOT / "scripts/gen_families.py"), "--check"],
                          capture_output=True, text=True, cwd=ROOT, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_deploy_doc_is_current():
    proc = subprocess.run([sys.executable, str(ROOT / "scripts/gen_deploy.py"), "--check"],
                          capture_output=True, text=True, cwd=ROOT, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_families_doc_quotes_the_rules():
    sys.path.insert(0, str(ROOT / "scripts"))
    from gen_families import rule_messages

    pipeline = [m for _s, m in rule_messages(ROOT / "ai_made_easy/core/pipelines/rules.py")]
    assert "Ensemble: every member must do the same task" in pipeline
    assert "Distill: the student outputs … but the teacher …" in pipeline
    text = (ROOT / "docs/FAMILIES.md").read_text(encoding="utf-8")
    assert "## Graphical model" in text and "Ensemble: every member must do the same task" in text
