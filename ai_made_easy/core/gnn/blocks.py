"""Graph neural network blocks (PyTorch Geometric): message-passing layers, pooling, the
link-prediction decoder and graph datasets.

Shapes are per node: an Input of [F] means F features per node. Message-passing layers
keep one row per node; global pooling turns nodes into one row per graph.
"""
from __future__ import annotations

from dataclasses import replace

from ai_made_easy.core.blocks._dsl import P, ShapeError, nn_block
from ai_made_easy.core.palette import family_color
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import BlockDefinition

CATEGORY = "Graph Neural Networks"
REQUIRES = ("torch_geometric",)
EXTRA = "graph"
CONV_LAYERS = ("graph.gcn", "graph.gat", "graph.sage", "graph.gin", "graph.graph_conv",
               "graph.edge_conv", "graph.transformer_conv")
GLOBAL_POOLS = ("graph.global_pool", "graph.attention_pool")
NODE_POOLS = ("graph.topk_pool", "graph.sag_pool")
GRAPH_BLOCKS = (*CONV_LAYERS, *GLOBAL_POOLS, *NODE_POOLS)
# dataset type -> "nodes" (one graph: node classification / link prediction) or "graphs"
GRAPH_DATA: dict[str, str] = {}
# known benchmark sizes: name -> (node features, classes)
PLANETOID = {"Cora": (1433, 7), "CiteSeer": (3703, 6), "PubMed": (500, 3)}
TU_DATASETS = {"MUTAG": (7, 2), "PROTEINS": (3, 2), "ENZYMES": (3, 6), "NCI1": (37, 2)}

E = "g['edge_index']"
OUT = P("out_channels", "int", 64, lo=1, hi=65536, help="Features per node after the layer")


def _nodes(in_shapes, p):
    s = in_shapes[0]
    if len(s) != 1:
        raise ShapeError(f"graph layers read node features [F], got {s}: flatten first")
    return s


def _width(units: str):
    def shape(in_shapes, p):
        _nodes(in_shapes, p)
        return [int(p[units])]
    return shape


def _heads_shape(in_shapes, p):
    _nodes(in_shapes, p)
    out = int(p["out_channels"])
    return [out * int(p["heads"]) if p["concat"] else out]


def _linear(f: int, out: int, bias: bool = True) -> int:
    return f * out + (out if bias else 0)


def _conv(type_id: str, name: str, params: tuple, shape, torch, cost, desc: str):
    return nn_block(type_id, name, CATEGORY, family="model", params=params, shape=shape,
                    torch=torch, torch_expr=lambda c: f"self.{c['self_var']}({c['i0']}, {E})",
                    torch_helpers=("GraphContext",), param_fn=cost, desc=desc)


def _layers() -> list[BlockDefinition]:
    heads = P("heads", "int", 4, lo=1, hi=64)
    concat = P("concat", "bool", True, help="Concatenate the heads (else average them)")
    aggr = P("aggr", "enum", "mean", options=("mean", "max", "add"),
             help="How neighbour messages are combined")

    def f(c):
        return c["input_shape"][0]

    blocks = [
        _conv("graph.gcn", "GCN Layer", (OUT, P("improved", "bool", False,
                                                help="Weight self-loops by 2")),
              _width("out_channels"),
              lambda c: f"pyg.GCNConv({f(c)}, {int(c['out_channels'])}, "
                        f"improved={bool(c['improved'])})",
              lambda s, p: _linear(s[0][0], int(p["out_channels"])),
              "Graph convolution (Kipf & Welling): averages neighbour features with "
              "degree normalisation."),
        _conv("graph.gat", "GAT Layer", (OUT, heads, concat,
                                         P("dropout", "float", 0.0, lo=0.0, hi=0.95),
                                         P("v2", "bool", True,
                                           help="GATv2: dynamic attention (usually better)")),
              _heads_shape,
              lambda c: (f"pyg.{'GATv2Conv' if c['v2'] else 'GATConv'}({f(c)}, "
                         f"{int(c['out_channels'])}, heads={int(c['heads'])}, "
                         f"concat={bool(c['concat'])}, dropout={float(c['dropout'])})"),
              lambda s, p: (int(p["heads"]) * int(p["out_channels"])
                            * (2 * s[0][0] + 3 if p["v2"] else s[0][0] + 2)
                            + (int(p["heads"]) * int(p["out_channels"]) if p["concat"]
                               else int(p["out_channels"]))),
              "Graph attention: each node weighs its neighbours with learned attention."),
        _conv("graph.sage", "GraphSAGE Layer", (OUT, aggr), _width("out_channels"),
              lambda c: f"pyg.SAGEConv({f(c)}, {int(c['out_channels'])}, aggr={c['aggr']!r})",
              lambda s, p: 2 * s[0][0] * int(p["out_channels"]) + int(p["out_channels"]),
              "GraphSAGE: combines a node with an aggregate of its neighbours "
              "(inductive: works on unseen graphs)."),
        _conv("graph.gin", "GIN Layer",
              (P("hidden", "int", 64, lo=1, hi=65536),
               P("train_eps", "bool", True, help="Learn the weight of the node itself")),
              _width("hidden"),
              lambda c: (f"pyg.GINConv(nn.Sequential(nn.Linear({f(c)}, {int(c['hidden'])}), "
                         f"nn.ReLU(), nn.Linear({int(c['hidden'])}, {int(c['hidden'])})), "
                         f"train_eps={bool(c['train_eps'])})"),
              lambda s, p: (_linear(s[0][0], int(p["hidden"]))
                            + _linear(int(p["hidden"]), int(p["hidden"]))
                            + (1 if p["train_eps"] else 0)),
              "Graph isomorphism network: sums neighbours and applies an MLP (as "
              "expressive as the Weisfeiler-Lehman test)."),
        _conv("graph.graph_conv", "GraphConv Layer", (OUT, aggr), _width("out_channels"),
              lambda c: (f"pyg.GraphConv({f(c)}, {int(c['out_channels'])}, "
                         f"aggr={c['aggr']!r})"),
              lambda s, p: 2 * s[0][0] * int(p["out_channels"]) + int(p["out_channels"]),
              "Higher-order graph convolution (Morris et al.): node and neighbour sums "
              "with separate weights."),
        _conv("graph.edge_conv", "EdgeConv Layer", (OUT, aggr), _width("out_channels"),
              lambda c: (f"pyg.EdgeConv(nn.Sequential(nn.Linear({2 * f(c)}, "
                         f"{int(c['out_channels'])}), nn.ReLU(), "
                         f"nn.Linear({int(c['out_channels'])}, {int(c['out_channels'])})), "
                         f"aggr={c['aggr']!r})"),
              lambda s, p: (_linear(2 * s[0][0], int(p["out_channels"]))
                            + _linear(int(p["out_channels"]), int(p["out_channels"]))),
              "Edge convolution (DGCNN): an MLP on each node and its neighbour's "
              "difference."),
        _conv("graph.transformer_conv", "Graph Transformer Layer", (OUT, heads, concat),
              _heads_shape,
              lambda c: (f"pyg.TransformerConv({f(c)}, {int(c['out_channels'])}, "
                         f"heads={int(c['heads'])}, concat={bool(c['concat'])})"),
              lambda s, p: (3 * _linear(s[0][0], int(p["heads"]) * int(p["out_channels"]))
                            + _linear(s[0][0], int(p["heads"]) * int(p["out_channels"])
                                      if p["concat"] else int(p["out_channels"]))),
              "Transformer attention over each node's neighbours (Shi et al., UniMP)."),
        nn_block("graph.global_pool", "Global Pooling", CATEGORY, family="model",
                 params=(P("mode", "enum", "mean", options=("mean", "max", "add")),),
                 shape=_nodes, param_fn=lambda s, p: 0,
                 torch_expr=lambda c: f"pyg.global_{c['mode']}_pool({c['i0']}, g['batch'])",
                 torch_helpers=("GraphContext",),
                 desc="One row per graph: the mean, max or sum of its node features "
                      "(graph classification)."),
        nn_block("graph.attention_pool", "Attention Pooling", CATEGORY, family="model",
                 shape=_nodes, param_fn=lambda s, p: s[0][0] + 1,
                 torch=lambda c: (f"pyg.aggr.AttentionalAggregation("
                                  f"nn.Linear({c['input_shape'][0]}, 1))"),
                 torch_expr=lambda c: f"self.{c['self_var']}({c['i0']}, g['batch'])",
                 torch_helpers=("GraphContext",),
                 desc="One row per graph: a learned attention-weighted sum of its nodes."),
    ]
    for kind, label, desc in (("topk", "TopK Pooling", "keeps the nodes with the highest "
                                                       "learned score"),
                              ("sag", "SAG Pooling", "keeps the nodes a graph convolution "
                                                     "scores highest (self-attention)")):
        blocks.append(nn_block(
            f"graph.{kind}_pool", label, CATEGORY, family="model",
            params=(P("ratio", "float", 0.5, lo=0.01, hi=1.0, help="Fraction of nodes kept"),),
            shape=_nodes,
            param_fn=lambda s, p, k=kind: 2 * s[0][0] + 2 if k == "sag" else s[0][0],
            torch=lambda c, k=kind: (f"GraphPool({k!r}, {c['input_shape'][0]}, "
                                     f"{float(c['ratio'])})"),
            torch_expr=lambda c: f"self.{c['self_var']}({c['i0']}, g)",
            torch_helpers=("GraphContext", "GraphPool"),
            desc=f"Hierarchical pooling: {desc} and the edges between them."))
    return [replace(b, requires=REQUIRES, extra=EXTRA, library="PyTorch Geometric",
                    meta={**(b.meta or {}), "graph": True}) for b in blocks]


def _config() -> list[BlockDefinition]:
    return [BlockDefinition(
        type_id="graph.link_decoder", display_name="Link Predictor", category=CATEGORY,
        color=family_color("model"), library="PyTorch Geometric", requires=REQUIRES,
        extra=EXTRA,
        params=(P("decoder", "enum", "dot", options=("dot", "mlp"),
                  help="dot: score = similarity of the two node embeddings; mlp: a small "
                       "network on their product"),
                P("hidden", "int", 64, lo=1, hi=65536, help="MLP decoder width"),
                P("negatives", "float", 1.0, lo=0.1, hi=20.0,
                  help="Random non-edges sampled per true edge")),
        description="Link prediction: the network embeds every node and this block scores "
                    "pairs of nodes (is there an edge?). Edges are split into train / "
                    "validation / test.",
        meta={"graph": "link"})]


def _data(type_id: str, name: str, kind: str, params: tuple, desc: str,
          requires: tuple = ()) -> BlockDefinition:
    GRAPH_DATA[type_id] = kind
    split = (P("val_fraction", "float", 0.1, lo=0.0, hi=0.5),
             P("test_fraction", "float", 0.2, lo=0.0, hi=0.5))
    return BlockDefinition(
        type_id=type_id, display_name=name, category="Data", color=family_color("data"),
        params=(*params, *split), description=desc, library="PyTorch Geometric",
        requires=requires, extra=EXTRA if requires else "",
        meta={"modality": "graph", "graph_data": kind})


def _datasets() -> list[BlockDefinition]:
    return [
        _data("data.synthetic_graph", "Synthetic Graph (communities)", "nodes",
              (P("n_nodes", "int", 600, lo=10, hi=1_000_000),
               P("communities", "int", 4, lo=2, hi=100, help="Node classes"),
               P("p_in", "float", 0.05, lo=0.0, hi=1.0, help="Edge probability inside a "
                                                              "community"),
               P("p_out", "float", 0.005, lo=0.0, hi=1.0, help="Edge probability across "
                                                                "communities"),
               P("feature_dim", "int", 16, lo=1, hi=10000),
               P("feature_signal", "float", 0.5, lo=0.0, hi=10.0,
                 help="How much the features alone reveal the class (0: only the graph "
                      "does)"),
               P("seed", "int", 0, lo=0)),
              "A stochastic block model: nodes in the same community link more often. "
              "Node classification or link prediction without downloads."),
        _data("data.synthetic_graphs", "Synthetic Graphs (motifs)", "graphs",
              (P("n_graphs", "int", 600, lo=10, hi=1_000_000),
               P("min_nodes", "int", 12, lo=4, hi=10000),
               P("max_nodes", "int", 24, lo=4, hi=10000),
               P("feature_dim", "int", 8, lo=1, hi=1000,
                 help="Node features: one-hot degree (capped) — the class is in the "
                      "structure"),
               P("seed", "int", 0, lo=0)),
              "Random trees with a planted motif — a cycle, a house or a star (3 classes): "
              "graph classification without downloads."),
        _data("data.graph_csv", "Graph CSV (nodes + edges)", "nodes",
              (P("nodes_path", "str", "nodes.csv", help="One row per node"),
               P("edges_path", "str", "edges.csv", help="One row per edge"),
               P("id_column", "str", "id"), P("label_column", "str", "label",
                                                help="Node class (empty: link prediction "
                                                     "only)"),
               P("feature_columns", "str", "", help="Node feature columns (empty: every "
                                                     "numeric column except id and label)"),
               P("source_column", "str", "source"), P("target_column", "str", "target"),
               P("directed", "bool", False, help="Keep edge direction (else both ways)")),
              "Your own graph: a node table (features, label) and an edge list."),
        _data("data.planetoid", "Planetoid Citations", "nodes",
              (P("name", "enum", "Cora", options=tuple(PLANETOID)),
               P("root", "str", "~/.cache/aime/planetoid")),
              "Cora / CiteSeer / PubMed citation graphs (downloaded once): papers, their "
              "words and their topic.", requires=REQUIRES),
        _data("data.tudataset", "TU Graph Benchmarks", "graphs",
              (P("name", "enum", "MUTAG", options=tuple(TU_DATASETS)),
               P("root", "str", "~/.cache/aime/tudataset")),
              "Molecule and protein graph-classification benchmarks (MUTAG, PROTEINS, "
              "ENZYMES, NCI1; downloaded once).", requires=REQUIRES),
    ]


def register_all() -> None:
    reg = get_registry()
    for defn in (*_layers(), *_config(), *_datasets()):
        reg.register(defn)
