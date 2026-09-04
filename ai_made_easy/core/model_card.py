"""Model Report Card — the AI4K12 grades 3-5 capstone artifact.

A kid-worded subset of the Model Cards format (Mitchell et al. 2019 +
HuggingFace structure): YAML front-matter + five short sections, filled
automatically from the run's artifacts. Pure string building; the Qt side
lives in features/report_card.py.
"""
from __future__ import annotations

import json
from pathlib import Path


def _read(workdir: Path | None, name: str) -> list:
    if workdir is None:
        return []
    try:
        return json.loads((Path(workdir) / name).read_text())
    except (OSError, ValueError):
        return []


def _accuracy_of(predictions: list) -> float | None:
    scored = [p for p in predictions if len(p.get("probs", [])) > 1]
    if not scored:
        return None
    right = sum(1 for p in scored
                if max(range(len(p["probs"])),
                       key=lambda i: p["probs"][i]) == p["true"])
    return right / len(scored)


def _worst_confusions(mistakes: list) -> list[tuple[int, int, int]]:
    pairs: dict[tuple[int, int], int] = {}
    for m in mistakes:
        if len(m.get("probs", [])) < 2:
            continue
        guessed = max(range(len(m["probs"])), key=lambda i: m["probs"][i])
        pairs[(m["true"], guessed)] = pairs.get((m["true"], guessed), 0) + 1
    return [(true, guessed, n)
            for (true, guessed), n in
            sorted(pairs.items(), key=lambda kv: -kv[1])[:3]]


def build_card(name: str, dataset_comment: str, trainer_params: dict,
               workdir: Path | None = None,
               superpower: str = "", careful: str = "") -> str:
    """Assemble the markdown card from the run's artifacts."""
    predictions = _read(workdir, "predictions.json")
    mistakes = _read(workdir, "mistakes.json")

    accuracy = _accuracy_of(predictions)
    smart = (f"**{accuracy:.0%}** of the examples it was checked on"
             if accuracy is not None
             else "**?** (train with checks to fill this in)")

    confusion_lines = "\n".join(
        f"- it often says **class {guessed}** when the right answer was "
        f"**class {true}** ({n}×)"
        for true, guessed, n in _worst_confusions(mistakes))
    if not confusion_lines:
        confusion_lines = (
            "- nothing in the Mistake Museum — it got them all right 🎉"
            if predictions else
            "- unknown — train first, then this fills in from the Mistake "
            "Museum")

    accuracy_yaml = round(accuracy, 4) if accuracy is not None else "null"

    return f"""---
model_name: {name}
created_with: AI Made Easy
task: classification
metrics:
  accuracy: {accuracy_yaml}
epochs: {trainer_params.get("epochs", "—")}
batch_size: {trainer_params.get("batch_size", "—")}
---

## 🦸 My model's superpower
{superpower or "It can tell things apart by looking at examples."}

## 📚 What it learned from
{dataset_comment}.
A computer found number patterns in the examples — nobody told it rules.

## 📊 How accurate is it?
It got {smart} right.
(That's only the examples it was checked on — new ones can still trick it!)

## 🤔 When it gets confused
{confusion_lines}

## ⚠️ Be careful
{careful or "(write one thing this model should NOT be used for)"}
"""
