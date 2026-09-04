"""Composite blocks: architecture macros that expand into primitive blocks.

A Fragment is a mini-Graph (nodes + edges + entry/exit ids) produced by a
block's ``builder``. Expanding replaces the composite node on the canvas
with the fragment's nodes, splicing external wires onto entry/exit.
Composite blocks are macros — codegen requires expansion first.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ai_made_easy.core.graph import Edge, Graph, NodeInstance
from ai_made_easy.core.registry import get_registry

# builders receive the composite's resolved params and return a Fragment
FragmentBuilder = Callable[[dict[str, Any]], "Fragment"]


@dataclass
class Fragment:
    nodes: list[dict] = field(default_factory=list)  # {"id","type","params","position"}
    edges: list[tuple[str, str, str, str]] = field(default_factory=list)
    entry: str = ""
    exit: str = ""


def fragment_to_dict(frag: Fragment, name: str) -> dict:
    return {
        "name": name,
        "nodes": frag.nodes,
        "edges": [
            {"from": f"{s}/{sp}", "to": f"{t}/{tp}"} for s, sp, t, tp in frag.edges
        ],
        "entry": frag.entry,
        "exit": frag.exit,
    }


def fragment_from_dict(data: dict) -> Fragment:
    frag = Fragment(
        nodes=data["nodes"],
        edges=[
            (*e["from"].rsplit("/", 1), *e["to"].rsplit("/", 1))
            for e in data["edges"]
        ],
        entry=data["entry"],
        exit=data["exit"],
    )
    return frag


def expand_in_graph(graph: Graph, node_id: str) -> Graph:
    """Pure-core expansion: replace a composite node with its fragment.

    Returns a NEW graph (the input graph is untouched); external edges are
    re-attached to the fragment's entry/exit and positions are offset from
    the replaced node. Fragment node ids are namespaced to stay unique.
    """
    node = graph.nodes[node_id]
    defn = node.definition()
    if defn is None or defn.builder is None:
        raise ValueError(f"{node_id} is not a composite block")
    frag = defn.builder(node.resolved_params())

    prefix = f"x{len(graph.nodes)}_"
    out = Graph(name=graph.name, meta=dict(graph.meta))
    for inst in graph.nodes.values():
        if inst.instance_id == node_id:
            continue
        out.add_node(NodeInstance(inst.instance_id, inst.type_id,
                                  dict(inst.params), inst.position))
    first_x = min(nd["position"][0] for nd in frag.nodes)
    min_y = min(nd["position"][1] for nd in frag.nodes)
    for nd in frag.nodes:
        out.add_node(NodeInstance(
            prefix + nd["id"], nd["type"], dict(nd.get("params", {})),
            (node.position[0] + nd["position"][0] - first_x,
             node.position[1] + nd["position"][1] - min_y),
        ))
    entry, exit_ = prefix + frag.entry, prefix + frag.exit
    for edge in graph.edges:
        src, sport = (edge.source_id, edge.source_port)
        dst, dport = (edge.target_id, edge.target_port)
        if src == node_id:
            src, sport = exit_, "out"
        if dst == node_id:
            dst, dport = entry, _first_input_port(graph, node_id)
        out.add_edge(Edge(src, sport, dst, dport))
    for s, sp, t, tp in frag.edges:
        out.add_edge(Edge(prefix + s, sp, prefix + t, tp))
    return out


def _first_input_port(graph: Graph, node_id: str) -> str:
    return graph.nodes[node_id].definition().inputs[0].name


def wrap_fragment(frag: Fragment, input_shape: list[int]) -> Graph:
    """Wrap a fragment in Input/Output for validation + codegen + testing."""
    g = Graph(name="fragment")
    g.add_node(NodeInstance("wrap_in", "core.input", {"shape": ",".join(map(str, input_shape))}))
    g.add_node(NodeInstance("wrap_out", "core.output", {}))
    for nd in frag.nodes:
        g.add_node(
            NodeInstance(nd["id"], nd["type"], dict(nd.get("params", {})),
                         tuple(nd.get("position", (0.0, 0.0))))
        )
    for s, sp, t, tp in frag.edges:
        g.add_edge(Edge(s, sp, t, tp))
    g.add_edge(Edge("wrap_in", "out", frag.entry,
                    get_registry().get(g.nodes[frag.entry].type_id).inputs[0].name))
    g.add_edge(Edge(frag.exit, "out", "wrap_out", "in"))
    return g
