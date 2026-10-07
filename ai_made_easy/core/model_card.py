"""Model cards (Mitchell et al., 2019) generated from a training run.

Pure string building over run artefacts (metrics.json, predictions.json,
mistakes.json, classes.json) plus the training configuration; the dialog in
``ui/features/analysis.py`` lets users edit the free-text sections.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path


def read_artifact(workdir: Path | None, name: str, default=None):
    if workdir is None:
        return default
    try:
        return json.loads((Path(workdir) / name).read_text())
    except (OSError, ValueError):
        return default


def per_class_accuracy(predictions: list) -> dict[int, tuple[int, int]]:
    """class index -> (correct, total) over predictions with integer targets."""
    stats: dict[int, list[int]] = {}
    for p in predictions:
        if not isinstance(p.get("true"), int) or len(p.get("probs", [])) < 2:
            continue
        guess = max(range(len(p["probs"])), key=lambda i: p["probs"][i])
        entry = stats.setdefault(p["true"], [0, 0])
        entry[0] += int(guess == p["true"])
        entry[1] += 1
    return {c: (v[0], v[1]) for c, v in sorted(stats.items())}


def top_confusions(predictions: list, limit: int = 5) -> list[tuple[int, int, int]]:
    """(actual, predicted, count) for the most frequent misclassifications."""
    pairs: dict[tuple[int, int], int] = {}
    for p in predictions:
        if not isinstance(p.get("true"), int) or len(p.get("probs", [])) < 2:
            continue
        guess = max(range(len(p["probs"])), key=lambda i: p["probs"][i])
        if guess != p["true"]:
            pairs[(p["true"], guess)] = pairs.get((p["true"], guess), 0) + 1
    ranked = sorted(pairs.items(), key=lambda kv: -kv[1])[:limit]
    return [(a, b, n) for (a, b), n in ranked]


def build_card(name: str, dataset_comment: str, trainer_params: dict,
               workdir: Path | None = None, intended_use: str = "",
               limitations: str = "", details: dict | None = None) -> str:
    """Assemble a markdown model card."""
    details = details or {}
    metrics = read_artifact(workdir, "metrics.json", {}) or {}
    predictions = read_artifact(workdir, "predictions.json", []) or []
    classes = read_artifact(workdir, "classes.json", None)
    def label(c: int) -> str:
        return str(classes[c]) if classes and c < len(classes) else f"class {c}"

    scalar = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    metric_rows = "\n".join(f"| {k.replace('_', ' ')} | {v:.4f} |" for k, v in scalar.items())
    yaml_metrics = "\n".join(f"  {k}: {round(float(v), 6)}" for k, v in scalar.items()) or "  {}"

    per_class = per_class_accuracy(predictions)
    class_rows = "\n".join(f"| {label(c)} | {ok}/{n} | {ok / n:.1%} |"
                           for c, (ok, n) in per_class.items() if n)
    confusions = "\n".join(f"- {label(a)} predicted as {label(b)}: {n}"
                           for a, b, n in top_confusions(predictions))

    sections = [f"""---
model_name: {name}
library: {details.get('framework', 'PyTorch')}
task: {details.get('task', 'classification')}
created: {date.today().isoformat()}
metrics:
{yaml_metrics}
---

# Model card: {name}

## Model details
- **Architecture:** {details.get('architecture', 'see the exported model file')}
- **Parameters:** {details.get('parameters', 'n/a')}
- **Framework:** {details.get('framework', 'PyTorch')}
- **Generated with:** AI Made Easy

## Training data
- **Dataset:** {dataset_comment}

## Training procedure
- **Epochs:** {trainer_params.get('epochs', 'n/a')}
- **Batch size:** {trainer_params.get('batch_size', 'n/a')}
- **Optimizer:** {details.get('optimizer', 'n/a')}
- **Loss:** {details.get('loss', 'n/a')}
- **Seed:** {trainer_params.get('seed', 'n/a')}

## Evaluation (held-out test split)
"""]
    sections.append(f"| Metric | Value |\n|---|---|\n{metric_rows}\n" if metric_rows
                    else "_No test metrics recorded for this run._\n")
    if class_rows:
        sections.append(f"\n### Per-class accuracy (first {len(predictions)} test samples)\n"
                        f"| Class | Correct | Accuracy |\n|---|---|---|\n{class_rows}\n")
    if confusions:
        sections.append(f"\n### Most frequent errors\n{confusions}\n")
    sections.append(f"""
## Intended use
{intended_use.strip() or '_Describe the intended use cases and users._'}

## Limitations and risks
{limitations.strip() or '_Describe known limitations, failure modes and out-of-scope uses._'}
""")
    return "".join(sections)
