"""Audit fixes: blocks ported from Orange3/Langflow + their guardrails.

Catalog gaps filled: ROC-AUC, K-Fold CV, Fill Missing Values (Orange3);
Document Loader, Text Splitter, Chat Memory, Output Parser (Langflow).
Safeguard ports: prompt-brace validation, compile-only Lambda checks,
LLM completeness warning, issue-bearing node tooltips.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_made_easy.core.graph import Edge, Graph, NodeInstance  # noqa: E402
from ai_made_easy.core.registry import get_registry  # noqa: E402


def _mlp(extra=()):
    g = Graph(name="audit")
    g.add_node(NodeInstance("i", "core.input", {"shape": "20"}))
    g.add_node(NodeInstance("d", "core.dense", {"units": 8}))
    g.add_node(NodeInstance("o", "core.output"))
    for a, b in (("i", "d"), ("d", "o")):
        g.add_edge(Edge(a, "out", b, "in"))
    for nid, tid, params in extra:
        g.add_node(NodeInstance(nid, tid, params))
    return g


# ------------------------------------------------------------ catalog

def test_audit_blocks_are_registered():
    reg = get_registry()
    for tid, name in [
        ("eval.roc_auc", "ROC-AUC"),
        ("train.kfold", "K-Fold Cross-Validation"),
        ("prep.impute", "Fill Missing Values"),
        ("llm.doc_loader", "Document Loader"),
        ("llm.text_splitter", "Text Splitter"),
        ("llm.chat_memory", "Chat Memory"),
        ("llm.output_parser", "Output Parser (JSON)"),
    ]:
        assert reg.has(tid), tid
        assert reg.get(tid).display_name == name
    assert len(reg.all()) >= 130


def test_audit_blocks_are_bounded_and_clean():
    for tid in ("eval.roc_auc", "train.kfold", "prep.impute",
                "llm.doc_loader", "llm.text_splitter", "llm.chat_memory",
                "llm.output_parser"):
        for p in get_registry().get(tid).params:
            if p.type in ("int", "float"):
                assert p.minimum is not None, (tid, p.name)
                assert float(p.default) >= float(p.minimum), (tid, p.name)
            if p.type == "enum":
                assert p.default in p.options, (tid, p.name)
    assert _mlp().validate() == [] or all(
        i.severity != "error" for i in _mlp().validate())


# -------------------------------------------------------- training codegen

def test_kfold_script_generates_and_parses():
    from ai_made_easy.core.codegen.training_gen import generate_training

    g = _mlp([("kf", "train.kfold", {"k": 3})])
    code = generate_training(g, "pytorch")
    ast.parse(code)
    assert "kfold_cross_validate" in code
    assert "fold {k + 1}/{K_FOLDS}" in code
    assert "STRATIFIED = True" in code


def test_roc_auc_metric_in_script():
    from ai_made_easy.core.codegen.training_gen import generate_training

    code = generate_training(_mlp([("auc", "eval.roc_auc", {})]), "pytorch")
    ast.parse(code)
    assert "_binary_auc" in code and 'metrics["roc_auc"]' in code


def test_impute_in_csv_scripts():
    from ai_made_easy.core.codegen.training_gen import generate_training

    g = _mlp([("ds", "data.csv", {"path": "x.csv", "target_column": "y"}),
              ("im", "prep.impute", {"strategy": "constant", "constant": 0.0})])
    for fw in ("pytorch", "keras"):
        code = generate_training(g, fw)
        ast.parse(code)
        assert "impute_missing" in code, fw


# ------------------------------------------------------------ LLM codegen

def _llm_graph(workflow="generate"):
    g = Graph(name="llmcheck")
    g.add_node(NodeInstance("m", "llm.model", {}))
    g.add_node(NodeInstance("p", "llm.prompt", {}))
    if workflow == "generate":
        g.add_node(NodeInstance("mem", "llm.chat_memory", {"n_messages": 4}))
        g.add_node(NodeInstance("par", "llm.output_parser", {}))
    else:
        for nid, tid in (("e", "llm.embedding_model"), ("s", "llm.vector_store"),
                         ("r", "llm.retriever"), ("rag", "llm.rag"),
                         ("ld", "llm.doc_loader"), ("sp", "llm.text_splitter")):
            g.add_node(NodeInstance(nid, tid, {}))
    return g


def test_generate_script_has_memory_and_parser():
    from ai_made_easy.core.codegen.llm_gen import generate_llm_script

    code = generate_llm_script(_llm_graph("generate"))
    ast.parse(code)
    assert "MEMORY_TURNS = 4" in code
    assert "history.append" in code
    assert "OUTPUT_SCHEMA" in code and "json.loads" in code


def test_rag_script_uses_loader_and_splitter():
    from ai_made_easy.core.codegen.llm_gen import generate_llm_script

    g = _llm_graph("rag")
    g.nodes["sp"].params["chunk_size"] = 222
    g.nodes["sp"].params["separator"] = "\n\n"
    code = generate_llm_script(g)
    ast.parse(code)
    assert "CHUNK_SIZE = 222" in code
    assert "CHUNK_SEPARATOR" in code
    assert "text.split(CHUNK_SEPARATOR)" in code


# ------------------------------------------------- ported guardrails

def test_prompt_template_brace_validation():
    g = Graph()
    g.add_node(NodeInstance("m", "llm.model", {}))
    g.add_node(NodeInstance("p", "llm.prompt",
                            {"template": "answer {input} with a { brace"}))
    issues = [i for i in g.validate() if i.node_id == "p"]
    assert any("stray {" in i.message and "error" == i.severity
               for i in issues)


def test_prompt_without_input_placeholder_warns():
    g = Graph()
    g.add_node(NodeInstance("m", "llm.model", {}))
    g.add_node(NodeInstance("p", "llm.prompt", {"template": "static text"}))
    issues = [i for i in g.validate() if i.node_id == "p"]
    assert any(i.severity == "warning" and "{input}" in i.message
               for i in issues)


def test_lambda_expression_compile_check():
    g = _mlp([("lam", "core.lambda", {"expression": "t * "} )])
    issues = [i for i in g.validate() if i.node_id == "lam"]
    assert any("Python typo" in i.message for i in issues)
    g.nodes["lam"].params["expression"] = "t.clamp(0, 1)"
    assert not any("Python typo" in i.message
                   for i in g.validate() if i.node_id == "lam")


def test_lambda_ignoring_tensor_warns():
    g = _mlp([("lam", "core.lambda", {"expression": "42"})])
    issues = [i for i in g.validate() if i.node_id == "lam"
              and "doesn't use 't'" in i.message]
    assert issues and issues[0].severity == "warning"


def test_splitter_overlap_larger_than_chunk_is_error():
    g = Graph()
    g.add_node(NodeInstance("m", "llm.model", {}))
    g.add_node(NodeInstance("sp", "llm.text_splitter",
                            {"chunk_size": 100, "chunk_overlap": 100}))
    issues = [i for i in g.validate() if i.node_id == "sp"]
    assert any("chunk_overlap" in i.message and i.severity == "error"
               for i in issues)


def test_llm_blocks_without_model_warn():
    g = Graph()
    g.add_node(NodeInstance("p", "llm.prompt", {}))
    issues = [i for i in g.validate() if "HF Model" in i.message]
    assert issues and issues[0].severity == "warning"


def test_node_tooltip_carries_issues():
    pytest.importorskip("PySide6")
    from ai_made_easy.ui.app import _ensure_qt_plugin_path
    _ensure_qt_plugin_path()
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from ai_made_easy.ui.canvas.adapter import CanvasController

    controller = CanvasController()
    g = controller.node_graph
    dense = g.create_node("aim.layers.DenseLinearNode")
    dense.set_property("units", -3)  # out of bounds → error issue
    ir = controller.to_ir()
    issues = ir.validate()
    err_ids = {i.node_id for i in issues if i.severity == "error" and i.node_id}
    controller.apply_validation(err_ids, set(), {}, issues)
    tip = dense.view.toolTip()
    assert "units" in tip and "too small" in tip
