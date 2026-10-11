"""A pipeline's task: the task of the design its first Train (or Cross-validate) stage
trains, which every later stage keeps."""
from __future__ import annotations

from ai_made_easy.core.tasks import register_task_resolver


def pipeline_task(graph) -> str | None:  # noqa: ANN001
    if not any(n.type_id.startswith("pipeline.") for n in graph.nodes.values()):
        return None
    from ai_made_easy.core.pipelines.plan import build_plan
    from ai_made_easy.core.tasks import task_of

    try:
        plan = build_plan(graph)
    except Exception:  # noqa: BLE001 — an unreadable pipeline has no task yet
        return None
    for stage in plan:
        if stage.kind in ("pipeline.train", "pipeline.cross_validate") \
                and stage.design is not None:
            task = task_of(stage.design)
            return task.id if task else None
    return None


register_task_resolver(pipeline_task)
