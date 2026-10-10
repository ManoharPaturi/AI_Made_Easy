"""Multi-stage pipelines: Train, Cross-validate, Fine-tune, Distill, Ensemble, Prune,
Quantize, Evaluate, Export and Register stages wired on the canvas. The ``pipeline``
family is validated by :mod:`.rules` and run by :class:`.runner.PipelineRunner`, which
records every compute stage as a child run in the run history."""
from ai_made_easy.core.pipelines import blocks as _blocks

_blocks.register_all()


def _register_family() -> None:
    from ai_made_easy.core.families import Family, register_family
    from ai_made_easy.core.pipelines.rules import pipeline_issues

    register_family(Family(
        "pipeline", "Pipeline",
        "Stages that train, fine-tune, distill, prune, quantize, evaluate, export and "
        "register models, in order.",
        detect=lambda types: any(t.startswith("pipeline.") for t in types), priority=100,
        validate=pipeline_issues, trainable=False))


_register_family()
