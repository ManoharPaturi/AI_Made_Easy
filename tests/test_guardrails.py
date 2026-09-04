"""Guardrails: every constraint from the plan, pinned.

Rules 1-14 of docs/ARCHITECTURE.md "Guardrails": param domains, cross-param
checks, wire multiplicity, single Input/Output, off-path + training-config
warnings, dataset/shape match, node-attributed shape errors, 💡 tips, the
canvas wire guard, the properties-panel patches, and Train gating.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_made_easy.core.graph import (  # noqa: E402
    Edge,
    Graph,
    GraphError,
    NodeInstance,
)
from ai_made_easy.core.registry import get_registry  # noqa: E402


# ------------------------------------------------------------ fixtures

def build(*specs, name="t"):
    g = Graph(name=name)
    for spec in specs:
        nid, type_id, params = (spec + ({},))[:3]
        g.add_node(NodeInstance(nid, type_id, params))
    return g


def wire(g, *pairs):
    for src, dst in pairs:
        g.add_edge(Edge(src, "out", dst, "in"))
    return g


def make_mlp():
    return wire(build(("i", "core.input", {"shape": "784"}),
                      ("d", "core.dense", {"units": 32}),
                      ("r", "core.relu"),
                      ("o", "core.output")),
                ("i", "d"), ("d", "r"), ("r", "o"))


# ------------------------------------------------- rule 1/2: param domain

def test_every_numeric_param_is_bounded():
    missing = [(b.type_id, p.name) for b in get_registry().all()
               for p in b.params
               if p.type in ("int", "float") and p.minimum is None]
    assert not missing, missing


def test_every_default_lies_inside_its_bounds():
    bad = [(b.type_id, p.name, p.default, p.minimum, p.maximum)
           for b in get_registry().all() for p in b.params
           if p.type in ("int", "float") and p.minimum is not None
           and (float(p.default) < float(p.minimum)
               or (p.maximum is not None
                   and float(p.default) > float(p.maximum)))]
    assert not bad, bad


def test_enum_defaults_are_options():
    bad = [(b.type_id, p.name, p.default) for b in get_registry().all()
           for p in b.params
           if p.type == "enum" and p.options and p.default not in p.options]
    assert not bad, bad


def test_param_below_minimum_is_an_error_with_node():
    g = make_mlp()
    g.nodes["d"].params["units"] = -5
    issues = [i for i in g.validate() if i.node_id == "d"]
    assert any("units" in i.message and i.severity == "error" for i in issues)


def test_dropout_above_one_is_an_error():
    g = make_mlp()
    g.add_node(NodeInstance("drop", "core.dropout", {"p": 1.5}))
    g.add_edge(Edge("d", "out", "drop", "in"))
    g.add_edge(Edge("drop", "out", "o", "in"))
    g.nodes["r"] = NodeInstance("r", "core.relu")  # detach relu (off-path)
    issues = [i for i in g.validate() if i.node_id == "drop"]
    assert any("p" in i.message for i in issues)


# ------------------------------------------------- rule 3: checks_fn

def test_mha_embed_dim_must_divide_by_heads():
    g = wire(build(("i", "core.input", {"shape": "100,8"}),
                   ("m", "core.multihead_attention",
                    {"embed_dim": 100, "num_heads": 8}),
                   ("o", "core.output")),
             ("i", "m"), ("m", "o"))
    issues = [i for i in g.validate() if i.node_id == "m"]
    assert any("divisible" in i.message for i in issues)
    # auto mode (embed_dim = 0) must not trip the param check
    g.nodes["m"].params["embed_dim"] = 0
    assert not any("divisible" in i.message
                   for i in g.validate() if i.node_id == "m")


# ------------------------------------------- rules 7/8: wires + I/O count

def test_duplicate_wires_into_one_input_is_an_error():
    g = make_mlp()
    g.add_edge(Edge("i", "out", "d", "in"))  # second wire, same ports
    issues = [i for i in g.validate() if i.node_id == "d"]
    assert any("one wire" in i.message for i in issues)


def test_second_input_block_is_an_error_with_node_id():
    g = make_mlp()
    g.add_node(NodeInstance("i2", "core.input", {"shape": "4"}))
    issues = [i for i in g.validate() if i.node_id == "i2"]
    assert any("one Input" in i.message for i in issues)


# ------------------------------------------- rule 10: off-path warnings

def test_off_path_tensor_block_warns():
    g = make_mlp()
    g.add_node(NodeInstance("orphan", "core.relu"))
    warns = [i for i in g.validate()
             if i.node_id == "orphan" and i.severity == "warning"]
    assert warns and "not connected to your model" in warns[0].message


# ------------------------------------- rule 11: training completeness

def test_trainer_without_loss_and_optimizer_warns():
    g = make_mlp()
    g.add_node(NodeInstance("tr", "train.trainer", {"epochs": 3}))
    warns = [i for i in g.validate() if i.severity == "warning"
             and i.node_id == "tr"]
    assert any("Loss" in i.message for i in warns)
    assert any("Optimizer" in i.message for i in warns)


def test_two_trainers_is_an_error():
    g = make_mlp()
    g.add_node(NodeInstance("t1", "train.trainer", {"epochs": 1}))
    g.add_node(NodeInstance("t2", "train.trainer", {"epochs": 2}))
    issues = [i for i in g.validate() if i.node_id == "t2"]
    assert any("one Trainer" in i.message and i.severity == "error"
               for i in issues)


def test_scheduler_without_trainer_warns():
    g = make_mlp()
    g.add_node(NodeInstance("sch", "train.cosine_annealing_lr"))
    issues = [i for i in g.validate() if i.node_id == "sch"]
    assert issues and issues[0].severity == "warning"


# --------------------------------------------- rule 12: dataset match

def test_dataset_features_must_match_input_shape():
    g = make_mlp()  # input holds 784 numbers
    g.add_node(NodeInstance("ds", "data.synthetic",
                            {"features": 100, "classes": 10}))
    issues = [i for i in g.validate() if "features" in i.message]
    assert issues and issues[0].severity == "error"
    g.nodes["ds"].params["features"] = 784
    assert not [i for i in g.validate() if "features" in i.message]


# -------------------------- rule 6: node-attributed shape errors + tips

def test_shape_error_points_at_the_failing_block():
    # conv output [8,24,24] fed straight into Dense (needs rank 1)
    g = wire(build(("i", "core.input", {"shape": "1,28,28"}),
                   ("c", "core.conv2d", {"out_channels": 8}),
                   ("d", "core.dense", {"units": 10}),
                   ("o", "core.output")),
             ("i", "c"), ("c", "d"), ("d", "o"))
    issues = [i for i in g.validate() if i.node_id == "d"]
    assert issues, "shape error must carry the failing node id"
    assert any("Flatten" in i.message for i in issues)  # rule 13 tip


def test_infer_shapes_still_raises_for_codegen():
    g = wire(build(("i", "core.input", {"shape": "1,28,28"}),
                   ("c", "core.conv2d", {"out_channels": 8}),
                   ("d", "core.dense", {"units": 10}),
                   ("o", "core.output")),
             ("i", "c"), ("c", "d"), ("d", "o"))
    with pytest.raises(GraphError):
        g.infer_shapes()


def test_disconnected_input_gets_a_wire_tip():
    g = build(("d", "core.dense", {"units": 4}), ("o", "core.output"))
    g.add_edge(Edge("d", "out", "o", "in"))
    issues = [i for i in g.validate() if "not connected" in i.message]
    assert issues and "💡" in issues[0].message


def test_valid_mlp_has_no_issues():
    assert make_mlp().validate() == []


# ---------------------------------------- canvas guard (rule 4) + patches

def _canvas():
    pytest.importorskip("PySide6")
    from ai_made_easy.ui.app import _ensure_qt_plugin_path
    _ensure_qt_plugin_path()
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from ai_made_easy.ui.canvas.adapter import CanvasController
    return app, CanvasController()


class _StubPort:
    """Duck-typed OdenGraphQt port backed by a real canvas node."""

    def __init__(self, node, name, ptype, dtype, recorder):
        self._node, self._name = node, name
        self._ptype, self._dtype = ptype, dtype
        self._rec = recorder

    def node(self):
        return self._node

    def name(self):
        return self._name

    def type_(self):
        return self._ptype

    def disconnect_from(self, other):
        self._rec.append((self._name, other._name))


def test_wire_guard_undoes_dtype_mismatch():
    app, controller = _canvas()
    g = controller.node_graph
    in_node = g.create_node("aim.data.InputNode")
    dense = g.create_node("aim.layers.DenseLinearNode")
    rec: list = []
    # fabricate ports with mismatched dtypes (tensor out -> config in)
    out_port = _StubPort(in_node, "out", "out", "tensor", rec)
    in_port = _StubPort(dense, "in", "in", "config", rec)
    controller._port_dtype = lambda port: port._dtype
    messages: list = []
    controller.guard_notifiers.append(messages.append)
    controller._on_port_connected(in_port, out_port)
    assert rec == [("out", "in")], "mismatched wire must be undone"
    assert messages and "can't be wired" in messages[0]
    # matching dtypes are left alone
    rec.clear()
    ok_out = _StubPort(in_node, "out", "out", "tensor", rec)
    ok_in = _StubPort(dense, "in", "in", "tensor", rec)
    controller._on_port_connected(ok_in, ok_out)
    assert rec == []


def test_prop_spin_patches_fix_the_properties_bin():
    app, controller = _canvas()
    from OdenGraphQt import PropertiesBinWidget
    from OdenGraphQt.custom_widgets.properties_bin.prop_widgets_base import (
        PropDoubleSpinBox,
    )

    from ai_made_easy.ui.canvas.area import CanvasArea
    from ai_made_easy.ui.canvas.prop_widgets_patch import (
        install_prop_widget_patches,
    )

    install_prop_widget_patches()
    assert hasattr(PropDoubleSpinBox(), "set_min")

    area = CanvasArea(controller)
    prop_bin = area.make_properties_widget()
    node = controller.node_graph.create_node("aim.layers.DenseLinearNode")
    prop_bin.add_node(node)  # crashed with AttributeError before the patch
    w = prop_bin  # panel populated without raising
    assert w is not None

    box = PropDoubleSpinBox()
    box.set_value(1e-8)
    assert box.decimals() >= 10
    assert abs(box.get_value() - 1e-8) < 1e-12


# ------------------------------------------------------ Train gating

GUARD_SMOKE = """
import sys
from ai_made_easy.ui.app import _ensure_qt_plugin_path
_ensure_qt_plugin_path()
from PySide6 import QtCore, QtWidgets
QtCore.QCoreApplication.setOrganizationName("aime-tests")
QtCore.QCoreApplication.setApplicationName("smoke")
app = QtWidgets.QApplication([])
QtCore.QSettings().setValue("aime/pedagogy/predict_gate", False)
from ai_made_easy.ui.context import AppContext
from ai_made_easy.core.graph import ValidationIssue
ctx = AppContext()
calls = []
ctx.process_service.run_training = lambda g: calls.append(g)
# 1) invalid graph -> blocked
ctx.validation_store.update([ValidationIssue("error", "units too small", "d")])
QtWidgets.QMessageBox.warning = staticmethod(
    lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Ok)
ctx.act_train()
assert calls == [], "train must be blocked while errors exist"
# 2) valid graph -> runs
ctx.validation_store.update([])
ctx.act_train()
assert len(calls) == 1, "train should run once the graph is valid"
print("GUARD-OK", flush=True)
QtCore.QTimer.singleShot(0, app.quit)
app.exec()
"""


def test_train_is_gated_on_validation():
    pytest.importorskip("PySide6")
    import subprocess
    result = subprocess.run(
        [sys.executable, "-c", GUARD_SMOKE],
        cwd=str(Path(__file__).parent.parent),
        capture_output=True, text=True, timeout=180,
        env={"PYTHONPATH": str(Path(__file__).parent.parent),
             "PATH": "/usr/bin:/bin:/usr/local/bin"})
    assert "GUARD-OK" in result.stdout, \
        result.stdout[-800:] + result.stderr[-800:]
