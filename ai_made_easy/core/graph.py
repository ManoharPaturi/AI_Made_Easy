"""The framework-agnostic graph IR (intermediate representation).

Canonical conventions (framework quirks are resolved at codegen, never here):
  * shapes are channels-first, no batch dim, e.g. [C, H, W] or [F]
  * edges connect named ports; v1 supports feed-forward DAGs: branches are
    free, merges (Add/Concat) take explicit numbered input ports
  * config blocks (training, evaluation, data sources) carry no tensor ports
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import BlockDefinition, parse_shape


class GraphError(Exception):
    """Structural problem that prevents codegen (cycles, broken wiring)."""


@dataclass(frozen=True)
class ValidationIssue:
    severity: str  # "error" | "warning"
    message: str
    node_id: str | None = None

    def __str__(self) -> str:
        loc = f" [{self.node_id}]" if self.node_id else ""
        return f"{self.severity.upper()}{loc}: {self.message}"


@dataclass
class NodeInstance:
    """A block placed on the canvas."""

    instance_id: str
    type_id: str
    params: dict[str, Any] = field(default_factory=dict)
    position: tuple[float, float] = (0.0, 0.0)

    def definition(self) -> BlockDefinition:
        return get_registry().get(self.type_id)

    def resolved_params(self) -> dict[str, Any]:
        """User params merged over block defaults."""
        merged = self.definition().default_params()
        merged.update(self.params)
        return merged


@dataclass(frozen=True)
class Edge:
    source_id: str
    source_port: str
    target_id: str
    target_port: str


@dataclass
class Graph:
    """The canonical model: nodes + edges + metadata. Pure JSON in/out."""

    name: str = "untitled"
    nodes: dict[str, NodeInstance] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- build

    def add_node(self, node: NodeInstance) -> NodeInstance:
        if node.instance_id in self.nodes:
            raise GraphError(f"duplicate node id: {node.instance_id}")
        node.definition()  # raises if type unknown — fail fast
        self.nodes[node.instance_id] = node
        return node

    def add_edge(self, edge: Edge) -> Edge:
        for nid, port, want in (
            (edge.source_id, edge.source_port, "out"),
            (edge.target_id, edge.target_port, "in"),
        ):
            if nid not in self.nodes:
                raise GraphError(f"edge references unknown node: {nid}")
            defn = self.nodes[nid].definition()
            names = {p.name for p in defn.outputs if want == "out"} | {
                p.name for p in defn.inputs if want == "in"
            }
            if port not in names:
                raise GraphError(f"edge references unknown port {port!r} on {nid}")
        self.edges.append(edge)
        return edge

    # ------------------------------------------------------------- traverse

    def incoming(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.target_id == node_id]

    def outgoing(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.source_id == node_id]

    def topo_order(self) -> list[str]:
        """Kahn's algorithm; raises GraphError on cycles."""
        indeg = {nid: 0 for nid in self.nodes}
        for e in self.edges:
            indeg[e.target_id] += 1
        queue = sorted(nid for nid, d in indeg.items() if d == 0)
        order: list[str] = []
        while queue:
            nid = queue.pop(0)
            order.append(nid)
            for e in self.outgoing(nid):
                indeg[e.target_id] -= 1
                if indeg[e.target_id] == 0:
                    queue.append(e.target_id)
        if len(order) != len(self.nodes):
            cyclic = sorted(set(self.nodes) - set(order))
            raise GraphError(f"graph contains a cycle involving: {cyclic}")
        return order

    def input_edge_for(self, node_id: str, port_name: str) -> Edge | None:
        matches = [e for e in self.edges if e.target_id == node_id and e.target_port == port_name]
        return matches[0] if matches else None

    def model_nodes(self) -> list[NodeInstance]:
        """Topo-ordered tensor-flow nodes: everything on a tensor path from
        the single Input to the single Output, including branches/merges.

        Raises GraphError when the model flow is structurally broken.
        """
        inputs = [n for n in self.nodes.values() if n.type_id == "core.input"]
        outputs = [n for n in self.nodes.values() if n.type_id == "core.output"]
        if len(inputs) != 1 or len(outputs) != 1:
            raise GraphError(
                f"v1 requires exactly one Input and one Output "
                f"(found {len(inputs)} input(s), {len(outputs)} output(s))"
            )
        entry = inputs[0]
        # forward-reach from Input along tensor edges
        reach = {entry.instance_id}
        frontier = [entry.instance_id]
        while frontier:
            nid = frontier.pop()
            for e in self.outgoing(nid):
                src_def = self.nodes[e.source_id].definition()
                port = next((p for p in src_def.outputs if p.name == e.source_port), None)
                if port is not None and port.dtype == "tensor" and e.target_id not in reach:
                    reach.add(e.target_id)
                    frontier.append(e.target_id)
        if outputs[0].instance_id not in reach:
            raise GraphError("Output is not reachable from Input via tensor connections")
        order = self.topo_order()
        return [self.nodes[nid] for nid in order if nid in reach]

    # -------------------------------------------------------- shape + check

    def infer_shapes(self) -> dict[str, list[int]]:
        """Compute each model node's output shape by walking the DAG in
        topo order and calling each block's ``shape_fn``.

        Raises GraphError on the first problem (codegen contract); use
        ``infer_shapes_detailed()`` to collect issues per node instead.
        """
        shapes, issues = self.infer_shapes_detailed()
        if issues:
            raise GraphError(issues[0].message)
        return shapes

    def infer_shapes_detailed(self) -> tuple[dict[str, list[int]],
                                             list[ValidationIssue]]:
        """Shape walk that attributes failures to the failing node.

        Returns (shapes, issues): shapes for every node computed before the
        first failure; one issue per structural problem or per failing
        ``shape_fn`` (with ``node_id`` set, so the canvas can mark it red).
        """
        shapes: dict[str, list[int]] = {}
        issues: list[ValidationIssue] = []
        chain: list[NodeInstance]
        try:
            chain = self.model_nodes()
        except GraphError as exc:
            issues.append(ValidationIssue("error", str(exc)))
            return shapes, issues
        head = chain[0]
        shapes[head.instance_id] = parse_shape(head.resolved_params()["shape"])
        for node in chain[1:]:
            defn = node.definition()
            if node.type_id == "core.output":
                in_edge = self.input_edge_for(node.instance_id, defn.inputs[0].name)
                shapes[node.instance_id] = list(shapes[in_edge.source_id])
                continue
            if defn.builder is not None:
                issues.append(
                    ValidationIssue(
                        "error",
                        f"{node.instance_id} ({defn.display_name}) is an architecture "
                        "macro — select it and press ⤢ Expand to turn it into blocks",
                        node.instance_id,
                    )
                )
                break
            if defn.shape_fn is None:
                issues.append(
                    ValidationIssue(
                        "error",
                        f"{node.instance_id} ({defn.display_name}) is a config block; "
                        "it cannot sit in the model flow",
                        node.instance_id,
                    )
                )
                break
            in_shapes: list[list[int]] = []
            missing = False
            for port in defn.inputs:
                in_edge = self.input_edge_for(node.instance_id, port.name)
                if in_edge is None:
                    issues.append(
                        ValidationIssue(
                            "error",
                            f"{node.instance_id} ({defn.display_name}): input "
                            f"'{port.name}' is not connected",
                            node.instance_id,
                        )
                    )
                    missing = True
                else:
                    in_shapes.append(shapes.get(in_edge.source_id, []))
            if missing:
                break
            try:
                shapes[node.instance_id] = list(
                    defn.shape_fn(in_shapes, node.resolved_params()))
            except (GraphError, ValueError) as exc:
                issues.append(
                    ValidationIssue("error", str(exc), node.instance_id)
                )
                break  # downstream shapes depend on this one
        return shapes, issues

    # ---------------------------------------------------------- guardrails

    def _param_issues(self) -> list[ValidationIssue]:
        """Rules 1-3: every param inside its declared domain, plus each
        block's cross-param ``checks_fn``."""
        issues: list[ValidationIssue] = []
        for node in self.nodes.values():
            defn = node.definition()
            resolved = node.resolved_params()
            for spec in defn.params:
                value = resolved.get(spec.name)
                if value is None:
                    continue
                if spec.type in ("int", "float"):
                    try:
                        value = float(value)
                    except (TypeError, ValueError):
                        issues.append(ValidationIssue(
                            "error",
                            f"{spec.name} must be a number (got {value!r})",
                            node.instance_id))
                        continue
                    if spec.minimum is not None and value < float(spec.minimum):
                        issues.append(ValidationIssue(
                            "error",
                            f"{spec.name} = {value:g} is too small "
                            f"(minimum {spec.minimum:g})",
                            node.instance_id))
                    if spec.maximum is not None and value > float(spec.maximum):
                        issues.append(ValidationIssue(
                            "error",
                            f"{spec.name} = {value:g} is too big "
                            f"(maximum {spec.maximum:g})",
                            node.instance_id))
                elif spec.type == "enum" and spec.options and value not in spec.options:
                    issues.append(ValidationIssue(
                        "error",
                        f"{spec.name} = {value!r} is not one of "
                        f"{', '.join(map(str, spec.options))}",
                        node.instance_id))
            if defn.checks_fn is not None:
                for sev, msg in defn.checks_fn(resolved):
                    issues.append(ValidationIssue(sev, msg, node.instance_id))
        return issues

    def _flow_issues(self) -> tuple[list[ValidationIssue], set[str]]:
        """Rules 7-11: wire multiplicity, single Input/Output, off-path
        blocks, training-config completeness. Returns (issues, model_chain)
        — chain is empty when the tensor flow is structurally broken."""
        issues: list[ValidationIssue] = []
        reg = get_registry()

        # duplicate wires into a single-writer input port
        seen: dict[tuple[str, str], int] = {}
        for e in self.edges:
            seen[(e.target_id, e.target_port)] = (
                seen.get((e.target_id, e.target_port), 0) + 1)
        for (tid, port), count in seen.items():
            if count < 2 or tid not in self.nodes:
                continue
            defn = self.nodes[tid].definition()
            spec = next((p for p in defn.inputs if p.name == port), None)
            if spec is not None and not spec.multi:
                issues.append(ValidationIssue(
                    "error",
                    f"input '{port}' has {count} wires — a block input takes "
                    "one wire (use a Concat/Merge block to combine tensors)",
                    tid))

        inputs = [n for n in self.nodes.values() if n.type_id == "core.input"]
        outputs = [n for n in self.nodes.values() if n.type_id == "core.output"]
        for n in inputs[1:]:
            issues.append(ValidationIssue(
                "error", "only one Input block is allowed", n.instance_id))
        for n in outputs[1:]:
            issues.append(ValidationIssue(
                "error", "only one Output block is allowed", n.instance_id))
        if not inputs:
            issues.append(ValidationIssue(
                "error", "add an Input block — every model starts with one"))
        if not outputs:
            issues.append(ValidationIssue(
                "error", "add an Output block — every model ends with one"))

        chain: set[str] = set()
        if len(inputs) == 1 and len(outputs) == 1:
            try:
                chain = {n.instance_id for n in self.model_nodes()}
                if outputs[0].instance_id not in chain:
                    issues.append(ValidationIssue(
                        "error",
                        "the Output block is not reachable from the Input "
                        "block — connect your blocks into one chain",
                        outputs[0].instance_id))
            except GraphError:
                chain = set()
        # off-path tensor blocks (rule 10) — only when the flow is sound
        if chain:
            for node in self.nodes.values():
                if (reg.has(node.type_id)
                        and node.definition().shape_fn is not None
                        and node.instance_id not in chain
                        and node.type_id not in ("core.input", "core.output")):
                    issues.append(ValidationIssue(
                        "warning",
                        f"{node.definition().display_name} is not connected "
                        "to your model — wire it in (Input → … → Output) or "
                        "delete it",
                        node.instance_id))

        # training-config completeness (rule 11)
        trainers = [n for n in self.nodes.values() if n.type_id == "train.trainer"]
        if len(trainers) > 1:
            for n in trainers[1:]:
                issues.append(ValidationIssue(
                    "error", "only one Trainer block is allowed",
                    n.instance_id))
        optimizers = [n for n in self.nodes.values()
                      if n.type_id.startswith("train.")
                      and "loss" not in n.type_id
                      and n.type_id not in ("train.trainer",)
                      and "lr" not in n.type_id.split(".")[-1]]
        losses = [n for n in self.nodes.values()
                  if n.type_id.startswith("train.loss")]
        schedulers = [n for n in self.nodes.values()
                      if n.type_id.startswith("train.")
                      and n.type_id not in ("train.trainer",)
                      and not n.type_id.startswith("train.loss")
                      and n not in optimizers]
        if trainers:
            if not losses:
                issues.append(ValidationIssue(
                    "warning",
                    "no Loss block on the canvas — training will fall back "
                    "to a default loss (CrossEntropy)", trainers[0].instance_id))
            if not optimizers:
                issues.append(ValidationIssue(
                    "warning",
                    "no Optimizer block (Adam…) on the canvas — training "
                    "will fall back to a default optimizer",
                    trainers[0].instance_id))
        else:
            for n in schedulers:
                issues.append(ValidationIssue(
                    "warning",
                    f"{n.definition().display_name} has no Trainer block to "
                    "configure", n.instance_id))

        # LLM completeness: LLM blocks are useless without a model block
        llm_nodes = [n for n in self.nodes.values()
                     if n.type_id.startswith("llm.")]
        if llm_nodes and not any(n.type_id == "llm.model"
                                 for n in self.nodes.values()):
            issues.append(ValidationIssue(
                "warning",
                "every LLM workflow needs an 'HF Model' block — add one from "
                "the 💬 LLM palette", llm_nodes[0].instance_id))
        return issues, chain

    def _dataset_issues(self, chain: set[str]) -> list[ValidationIssue]:
        """Rule 12: a dataset block's feature count must match the Input
        block's shape."""
        from ai_made_easy.core.spec import shape_volume

        issues: list[ValidationIssue] = []
        input_nodes = [n for n in self.nodes.values() if n.type_id == "core.input"]
        if len(input_nodes) != 1 or not chain:
            return issues
        try:
            want = shape_volume(
                parse_shape(input_nodes[0].resolved_params()["shape"]))
        except ValueError:
            return issues
        for node in self.nodes.values():
            if not node.type_id.startswith("data."):
                continue
            resolved = node.resolved_params()
            features = resolved.get("features")
            if features is None:
                continue
            try:
                got = int(features)
            except (TypeError, ValueError):
                continue
            if got != want:
                issues.append(ValidationIssue(
                    "error",
                    f"{node.definition().display_name} has {got} features but "
                    f"the Input block's shape holds {want} — make them match",
                    input_nodes[0].instance_id))
        return issues

    def validate(self) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        reg = get_registry()
        for node in self.nodes.values():
            if not reg.has(node.type_id):
                issues.append(
                    ValidationIssue(
                        "error", f"unknown block type {node.type_id}", node.instance_id
                    )
                )
        # edge dtype compatibility
        for e in self.edges:
            if e.source_id not in self.nodes or e.target_id not in self.nodes:
                continue
            src_out = {p.name: p for p in self.nodes[e.source_id].definition().outputs}
            dst_in = {p.name: p for p in self.nodes[e.target_id].definition().inputs}
            sp, dp = src_out.get(e.source_port), dst_in.get(e.target_port)
            if sp is not None and dp is not None and sp.dtype != dp.dtype:
                issues.append(
                    ValidationIssue(
                        "error",
                        f"incompatible connection: {sp.dtype} -> {dp.dtype}",
                        e.target_id,
                    )
                )
        # disconnected tensor inputs
        for node in self.nodes.values():
            if not reg.has(node.type_id):
                continue
            if node.type_id == "core.input":
                continue
            for port in node.definition().inputs:
                if port.dtype != "tensor":
                    continue
                if self.input_edge_for(node.instance_id, port.name) is None:
                    issues.append(
                        ValidationIssue(
                            "error",
                            f"input '{port.name}' is not connected",
                            node.instance_id,
                        )
                    )
        # structure
        try:
            self.topo_order()
        except GraphError as exc:
            issues.append(ValidationIssue("error", str(exc)))

        issues += self._param_issues()
        flow_issues, chain = self._flow_issues()
        issues += flow_issues
        if chain:
            issues += self._dataset_issues(chain)
            _, shape_issues = self.infer_shapes_detailed()
            issues += shape_issues
        from ai_made_easy.core.suggestions import add_tips
        return add_tips(issues, self)

    # ----------------------------------------------------------------- JSON

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "name": self.name,
            "nodes": [
                {
                    "id": n.instance_id,
                    "type": n.type_id,
                    "params": n.params,
                    "position": list(n.position),
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {"from": f"{e.source_id}/{e.source_port}", "to": f"{e.target_id}/{e.target_port}"}
                for e in self.edges
            ],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Graph":
        g = cls(name=data.get("name", "untitled"), meta=data.get("meta", {}))
        for nd in data.get("nodes", []):
            pos = nd.get("position") or [0.0, 0.0]
            g.add_node(
                NodeInstance(
                    instance_id=nd["id"],
                    type_id=nd["type"],
                    params=nd.get("params", {}),
                    position=(float(pos[0]), float(pos[1])),
                )
            )
        for ed in data.get("edges", []):
            src, src_port = ed["from"].rsplit("/", 1)
            tgt, tgt_port = ed["to"].rsplit("/", 1)
            g.add_edge(Edge(src, src_port, tgt, tgt_port))
        return g
