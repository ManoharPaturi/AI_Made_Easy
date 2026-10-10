"""A pipeline's plan: its stages in order, each stage's inputs, the design it trains and
the design behind the model it passes on. Shared by the rules and the runner."""
from __future__ import annotations

from dataclasses import dataclass, field

from ai_made_easy.core.pipelines import blocks as B
from ai_made_easy.core.pipelines.designs import DesignError, resolve

PASS_THROUGH = (*B.PASS_THROUGH, *B.SHIP)


@dataclass
class Stage:
    id: str
    kind: str                       # block type id
    params: dict
    inputs: list[str] = field(default_factory=list)
    design: object = None           # the Graph this stage trains (train, cv, distill, ...)
    model: object = None            # the Graph behind the model this stage passes on
    error: str = ""                 # a design that cannot be read

    @property
    def label(self) -> str:
        from ai_made_easy.core.registry import get_registry

        return get_registry().get(self.kind).display_name

    @property
    def produces_model(self) -> bool:
        return self.kind in B.MODEL_STAGES or self.kind in PASS_THROUGH


@dataclass
class Plan:
    stages: dict[str, Stage]
    order: list[str]

    def __iter__(self):
        return (self.stages[i] for i in self.order)


def build_plan(graph, base=None) -> Plan:  # noqa: ANN001
    stages: dict[str, Stage] = {}
    for node in graph.nodes.values():
        if node.type_id.startswith("pipeline."):
            stages[node.instance_id] = Stage(node.instance_id, node.type_id,
                                             node.resolved_params())
    for edge in graph.edges:
        if edge.target_id in stages and edge.source_id in stages:
            stages[edge.target_id].inputs.append(edge.source_id)
    try:
        order = [i for i in graph.topo_order() if i in stages]
    except Exception:  # noqa: BLE001 — cycles are reported by the generic validation
        order = list(stages)
    for sid in order:
        stage = stages[sid]
        key = {"pipeline.train": "design", "pipeline.cross_validate": "design",
               "pipeline.distill": "student", "pipeline.finetune": "design"}.get(stage.kind)
        value = str(stage.params.get(key) or "") if key else ""
        if key and (value or stage.kind != "pipeline.finetune"):
            try:
                stage.design = resolve(value, graph, base)
            except DesignError as exc:
                stage.error = str(exc)
        upstream = stages.get(stage.inputs[0]) if stage.inputs else None
        if stage.kind in ("pipeline.train", "pipeline.distill"):
            stage.model = stage.design
        elif stage.kind == "pipeline.finetune":
            stage.model = stage.design or (upstream.model if upstream else None)
        elif stage.kind in ("pipeline.prune", "pipeline.quantize", *PASS_THROUGH):
            stage.model = upstream.model if upstream else None
    return Plan(stages, order)
