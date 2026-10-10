"""Design rules for pipelines: wiring, the designs each stage trains, and the conditions
fine-tuning, distillation, ensembles, export and the budget impose."""
from __future__ import annotations

from ai_made_easy.core.lints import _issue
from ai_made_easy.core.pipelines import blocks as B
from ai_made_easy.core.pipelines.designs import data_signature, layers, output_shape, \
    torch_supervised
from ai_made_easy.core.pipelines.plan import build_plan


def _design_errors(design) -> list[str]:  # noqa: ANN001
    return [i.message for i in design.validate() if i.severity == "error"]


def _task(design):  # noqa: ANN001, ANN202
    from ai_made_easy.core.tasks import task_of

    return task_of(design)


def pipeline_issues(graph) -> list:  # noqa: ANN001
    plan = build_plan(graph)
    out = []
    if not plan.stages:
        return [_issue("error", "add a Train stage to start the pipeline")]
    for stage in plan:
        name = stage.label
        if stage.error:
            out.append(_issue("error", f"{name}: {stage.error}", stage.id))
            continue
        if stage.design is not None:
            errors = _design_errors(stage.design)
            if errors:
                out.append(_issue("error", f"{name}: the design {stage.design.name!r} has "
                                           f"{len(errors)} error(s): {errors[0]}", stage.id))
                continue
        needs = 2 if stage.kind == "pipeline.ensemble" else 1 \
            if stage.kind in B.NEEDS_INPUT else 0
        if len(stage.inputs) < needs:
            out.append(_issue("error", f"{name} needs {'two or more models' if needs == 2 else 'a model'}"
                                       ": connect a Train (or another model stage) to it",
                              stage.id))
            continue
        sources = [plan.stages[i] for i in stage.inputs]
        bad = [s for s in sources if not s.produces_model]
        if bad:
            what = "only reports cross-validation scores" \
                if bad[0].kind == "pipeline.cross_validate" else "does not pass on a model"
            out.append(_issue("error", f"{name}: {bad[0].label} {what}", stage.id))
            continue
        if stage.kind in B.TORCH_STAGES:
            for source in sources:
                why = torch_supervised(source.model) if source.model is not None else ""
                if why:
                    out.append(_issue("error", f"{name} works inside a supervised PyTorch "
                                               f"network; the input is {why}", stage.id))
                    break
        out += _stage_rules(stage, sources)
    out += _budget(graph, plan)
    return out


def _stage_rules(stage, sources) -> list:  # noqa: ANN001
    out = []
    first = sources[0].model if sources else None
    if stage.kind == "pipeline.distill" and first is not None and stage.design is not None:
        student = stage.design
        why = torch_supervised(student)
        if why:
            return [_issue("error", f"Distill trains a supervised PyTorch student; the "
                                    f"student is {why}", stage.id)]
        if data_signature(student) != data_signature(first):
            out.append(_issue("error", "Distill: the student must read the same dataset with "
                                       "the same preprocessing as the teacher", stage.id))
        if output_shape(student) != output_shape(first):
            out.append(_issue("error", f"Distill: the student outputs {output_shape(student)} "
                                       f"but the teacher {output_shape(first)}", stage.id))
        teacher_task, student_task = _task(first), _task(student)
        if teacher_task and student_task and teacher_task.id != student_task.id:
            out.append(_issue("error", f"Distill: the teacher does {teacher_task.label} but "
                                       f"the student {student_task.label}", stage.id))
    if stage.kind == "pipeline.finetune" and stage.design is not None and first is not None:
        new, old = layers(stage.design), layers(first)
        if new[:-1] != old[:-1]:
            out.append(_issue("error", "Fine-tune: the design's architecture differs from the "
                                       "input model's, so its weights cannot be loaded",
                              stage.id))
        elif new[-1:] != old[-1:]:
            out.append(_issue("info", "Fine-tune: the last layer differs from the input "
                                      "model's and starts from new weights", stage.id))
    if stage.kind == "pipeline.ensemble":
        models = [s.model for s in sources if s.model is not None]
        if len({str(data_signature(m)) for m in models}) > 1:
            out.append(_issue("error", "Ensemble: every member must read the same dataset with "
                                       "the same preprocessing (same test split)", stage.id))
        if len({str(output_shape(m)) for m in models}) > 1:
            out.append(_issue("error", "Ensemble: the members' outputs differ: "
                                       + ", ".join(str(output_shape(m)) for m in models),
                              stage.id))
        if len({(_task(m).id if _task(m) else None) for m in models}) > 1:
            out.append(_issue("error", "Ensemble: every member must do the same task",
                              stage.id))
    if stage.kind == "pipeline.export" and first is not None:
        from ai_made_easy.core.deploy.package import FORMATS
        from ai_made_easy.core.families import resolve_framework

        try:
            framework = resolve_framework(first)
        except ValueError:
            framework = "pytorch"
        wanted = [f.strip() for f in str(stage.params["formats"]).split(",") if f.strip()]
        unknown = [f for f in wanted if f not in FORMATS.get(framework, ())]
        if unknown:
            out.append(_issue("error", f"Export: {', '.join(unknown)} not available for "
                                       f"{framework} models; choose from "
                                       f"{', '.join(FORMATS.get(framework, ()))}", stage.id))
    return out


def _budget(graph, plan) -> list:  # noqa: ANN001
    """Every training stage must fit the pipeline's budget; an ensemble serves all its
    members, so their parameters add up."""
    from ai_made_easy.core import budget

    limits = budget.budget_of(graph)
    if not limits["device"] and not any(limits[k] for k in budget.BUDGET_KEYS[1:]):
        return []
    out = []
    for stage in plan:
        if stage.design is None or stage.kind == "pipeline.cross_validate":
            continue
        design = stage.design
        design.meta = {**(design.meta or {}), "budget": limits}
        try:
            over = [c for c in budget.check(design) if c.over]
        except Exception:  # noqa: BLE001 — no estimate for this family
            continue
        for c in over:
            out.append(_issue("error", f"{stage.label}: {c.kind.replace('_', ' ')} "
                                       f"{c.used:.3g} {c.unit} > the pipeline's {c.limit:g} "
                                       f"{c.unit}", stage.id))
    if limits["max_params_m"]:
        for stage in plan:
            if stage.kind != "pipeline.ensemble":
                continue
            total = 0
            for sid in stage.inputs:
                model = plan.stages[sid].model
                if model is not None:
                    try:
                        total += budget.estimate(model).params
                    except Exception:  # noqa: BLE001
                        pass
            if total / 1e6 > limits["max_params_m"]:
                out.append(_issue("error", f"Ensemble: the members have {total / 1e6:.3g} M "
                                           f"parameters together > the pipeline's "
                                           f"{limits['max_params_m']:g} M", stage.id))
    return out
