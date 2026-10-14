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


def _classic(design) -> bool:  # noqa: ANN001
    from ai_made_easy.core.families import family_of

    return design is not None and family_of(design).id == "classic"


def _datasets(design) -> list:  # noqa: ANN001
    return [row for row in data_signature(design) if row[0].startswith("data.")]


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
            what = "only reports cross-validation scores: set its refit to true to pass on " \
                "a model" if bad[0].kind == "pipeline.cross_validate" else \
                "does not pass on a model"
            out.append(_issue("error", f"{name}: {bad[0].label} {what}", stage.id))
            continue
        if stage.kind in B.TORCH_STAGES:
            for source in sources:
                if stage.kind == "pipeline.distill" and _classic(source.model):
                    continue            # scikit-learn / boosting teachers give soft labels
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
        if _classic(first):
            return out + _classic_teacher(stage, first, student)
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
    if stage.kind == "pipeline.finetune" and float(stage.params.get("lr_decay", 1.0)) < 1.0 \
            and stage.params["freeze"] == "all_but_last" and not stage.params["unfreeze_after"]:
        out.append(_issue("info", "Fine-tune: only the last layer trains, so lr_decay changes "
                                  "nothing: set unfreeze_after, or set freeze to none",
                          stage.id))
    if stage.kind == "pipeline.qat" and first is not None:
        inputs = [n for n in first.nodes.values() if n.type_id == "core.input"]
        if inputs and str(inputs[0].resolved_params().get("dtype", "")).startswith("int"):
            out.append(_issue("info", "QAT keeps embedding tables 8-bit with float scales "
                                      "(weight-only); the layers after them become int8",
                              stage.id))
        try:
            kinds = {n.type_id for n in first.model_nodes()}
        except Exception:  # noqa: BLE001 — not a network (reported above)
            kinds = set()
        if kinds & FLOAT_ONLY:
            out.append(_issue("info", "QAT quantizes linear and convolution layers; "
                                      "recurrent and attention layers stay in float",
                              stage.id))
    if stage.kind == "pipeline.finetune" and stage.design is not None and first is not None:
        new, old = layers(stage.design), layers(first)
        if new[:-1] != old[:-1]:
            out.append(_issue("error", "Fine-tune: the design's architecture differs from the "
                                       "input model's, so its weights cannot be loaded",
                              stage.id))
        elif new[-1:] != old[-1:]:
            out.append(_issue("info", "Fine-tune: the last layer differs from the input "
                                      "model's and starts from new weights", stage.id))
    if stage.kind == "pipeline.ensemble" and stage.params["method"] == "stack":
        out += _stacking(stage, sources)
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
        from ai_made_easy.core.deploy.exporters import ALIASES
        from ai_made_easy.core.deploy.exportability import blockers

        wanted = [ALIASES.get(f.strip(), f.strip())
                  for f in str(stage.params["formats"]).split(",") if f.strip()]
        unknown = [f for f in wanted if f not in FORMATS.get(framework, ())]
        if unknown:
            out.append(_issue("error", f"Export: {', '.join(unknown)} not available for "
                                       f"{framework} models; choose from "
                                       f"{', '.join(FORMATS.get(framework, ()))}", stage.id))
        blocked = blockers(first, framework)
        for fmt in wanted:
            if fmt in blocked:
                out.append(_issue("warning", f"Export: {fmt} will be skipped: "
                                             f"{blocked[fmt][0]}", stage.id))
    return out


FLOAT_ONLY = {"core.lstm", "core.gru", "core.rnn", "core.multihead_attention",
              "core.transformer_encoder", "core.transformer_decoder"}


def _classic_teacher(stage, teacher, student) -> list:  # noqa: ANN001
    """A scikit-learn / boosting teacher: a classifier or regressor on the same dataset."""
    out = []
    teacher_task, student_task = _task(teacher), _task(student)
    if teacher_task is None or teacher_task.id not in ("binary", "multiclass", "regression"):
        return [_issue("error", "Distill: a classic teacher must be a classifier or a "
                                "regressor", stage.id)]
    if student_task is not None and teacher_task.id != student_task.id and not (
            {teacher_task.id, student_task.id} <= {"binary", "multiclass"}):
        out.append(_issue("error", f"Distill: the teacher does {teacher_task.label} but the "
                                   f"student {student_task.label}", stage.id))
    if _datasets(student) != _datasets(teacher):
        out.append(_issue("error", "Distill: the student must read the same dataset as the "
                                   "classic teacher (the same rows get the teacher's soft "
                                   "labels)", stage.id))
    return out


def _stacking(stage, sources) -> list:  # noqa: ANN001
    import importlib.util

    out = []
    plain = [s for s in sources if s.kind != "pipeline.cross_validate"]
    if plain:
        out.append(_issue("error", f"Ensemble (stack) learns from out-of-fold predictions: "
                                   f"replace {plain[0].label} with a Cross-validate stage "
                                   "(refit on)", stage.id))
    if len({int(s.params.get("k", 0)) for s in sources
            if s.kind == "pipeline.cross_validate"}) > 1:
        out.append(_issue("error", "Ensemble (stack): give every Cross-validate stage the same "
                                   "number of folds so their out-of-fold predictions line up",
                          stage.id))
    if importlib.util.find_spec("sklearn") is None:
        out.append(_issue("error", "Ensemble (stack) fits its meta-learner with scikit-learn: "
                                   "pip install 'ai-made-easy[classic]'", stage.id))
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
