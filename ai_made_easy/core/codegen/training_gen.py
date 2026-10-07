"""Training-script generation (compatibility facade).

The implementation lives in :mod:`ai_made_easy.core.training`; this module keeps
the long-standing import path used by the UI, CLI, runner and tests.
"""
from __future__ import annotations

from ai_made_easy.core.training import catalog as _cat
from ai_made_easy.core.training import data_catalog as _dcat
from ai_made_easy.core.training.generate import generate_inspect, generate_training
from ai_made_easy.core.training.spec import TrainingSpec, collect_spec, dataset_comment

_LOSSES = _cat.LOSS_IDS
_OPTIMIZERS = _cat.OPTIMIZER_IDS
_SCHEDULERS = _cat.SCHEDULER_IDS
_DATASETS = _dcat.DATASET_IDS
METRIC_BLOCKS = {c.type_id: c.meta["key"] for c in _cat.METRICS}
_dataset_comment = dataset_comment

__all__ = ["TrainingSpec", "collect_spec", "dataset_comment", "generate_inspect",
           "generate_training", "METRIC_BLOCKS"]
