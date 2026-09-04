"""Phase 6 tests: LLM workflow detection, spec defaults, script fragments.
Network-free — end-to-end LLM runs were verified manually (tiny-gpt2,
MiniLM) and can be re-run with AIME_LLM_E2E=1."""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ai_made_easy.core.codegen.llm_gen import collect_llm_spec, generate_llm_script
from ai_made_easy.core.graph import Edge, Graph, NodeInstance

SAMPLES = Path(__file__).parent.parent / "samples"


def llm_graph(name: str, blocks: list[tuple[str, str, dict]]) -> Graph:
    g = Graph(name=name)
    g.add_node(NodeInstance("in", "core.input", {"shape": "4"}))
    g.add_node(NodeInstance("d", "core.dense", {"units": 2}))
    g.add_node(NodeInstance("out", "core.output", {}))
    g.add_edge(Edge("in", "out", "d", "in"))
    g.add_edge(Edge("d", "out", "out", "in"))
    for nid, t, p in blocks:
        g.add_node(NodeInstance(nid, t, p))
    return g


def test_workflow_detection():
    model = ("m", "llm.model", {"repo_id": "x/y", "dtype": "float32",
                                 "trust_remote_code": False})
    spec = collect_llm_spec(llm_graph("g", [model]))
    assert spec.workflow == "generate"  # model alone defaults to generation

    spec = collect_llm_spec(llm_graph("g", [
        model,
        ("l", "llm.lora", {"mode": "lora", "r": 4, "alpha": 8, "dropout": 0.0,
                            "target_modules": "c_attn"}),
        ("ds", "llm.dataset", {"source": "text_file", "dataset_id": "",
                                "split": "train", "text_field": "text",
                                "file_path": "d.txt"}),
        ("s", "llm.sft", {"epochs": 1, "batch_size": 1, "lr": 1e-3,
                           "gradient_accumulation": 1, "warmup_ratio": 0.0,
                           "seed": 42}),
    ]))
    assert spec.workflow == "finetune"
    assert spec.tokenizer["repo_id"] == "x/y"  # defaults from model


def test_mixed_workflows_rejected():
    with pytest.raises(Exception, match="mixes workflows"):
        collect_llm_spec(llm_graph("g", [
            ("m", "llm.model", {"repo_id": "x/y", "dtype": "float32",
                                 "trust_remote_code": False}),
            ("l", "llm.lora", {"mode": "lora", "r": 4, "alpha": 8, "dropout": 0.0,
                                "target_modules": "c_attn"}),
            ("vs", "llm.vector_store", {"backend": "numpy", "persist_path": ""}),
        ]))


def test_missing_blocks_rejected():
    with pytest.raises(Exception, match="HF Model"):
        collect_llm_spec(llm_graph("g", []))
    with pytest.raises(Exception, match="SFT Dataset"):
        collect_llm_spec(llm_graph("g", [
            ("m", "llm.model", {"repo_id": "x/y", "dtype": "float32",
                                 "trust_remote_code": False}),
            ("l", "llm.lora", {"mode": "lora", "r": 4, "alpha": 8, "dropout": 0.0,
                                "target_modules": "c_attn"}),
        ]))


@pytest.mark.parametrize("sample,workflow", [
    ("llm_generation.json", "generate"),
    ("llm_lora_finetune.json", "finetune"),
    ("llm_rag_assistant.json", "rag"),
])
def test_samples_generate_valid_scripts(sample, workflow):
    import json

    graph = Graph.from_dict(json.loads((SAMPLES / sample).read_text()))
    assert collect_llm_spec(graph).workflow == workflow
    code = generate_llm_script(graph)
    ast.parse(code)


def test_generation_script_fragments():
    code = generate_llm_script(llm_graph("g", [
        ("m", "llm.model", {"repo_id": "org/model", "dtype": "float16",
                             "trust_remote_code": False}),
        ("pr", "llm.prompt", {"template": "Q: {input}\nA:"}),
        ("gn", "llm.generation", {"max_new_tokens": 42, "do_sample": False,
                                   "temperature": 0.1, "top_p": 0.9, "top_k": 50,
                                   "repetition_penalty": 1.0}),
    ]))
    assert 'MODEL_ID = \'org/model\'' in code
    assert '"float16"' in code
    assert "MAX_NEW_TOKENS = 42" in code
    assert "DO_SAMPLE = False" in code
    assert "PROMPT_TEMPLATE" in code and "{input}" in code


def test_finetune_script_fragments():
    code = generate_llm_script(llm_graph("g", [
        ("m", "llm.model", {"repo_id": "org/model", "dtype": "float32",
                             "trust_remote_code": False}),
        ("l", "llm.lora", {"mode": "qlora", "r": 16, "alpha": 32, "dropout": 0.05,
                            "target_modules": "q_proj,v_proj"}),
        ("ds", "llm.dataset", {"source": "hf_hub", "dataset_id": "hf/set",
                                "split": "train", "text_field": "text",
                                "file_path": ""}),
        ("s", "llm.sft", {"epochs": 2, "batch_size": 1, "lr": 1e-4,
                           "gradient_accumulation": 4, "warmup_ratio": 0.03,
                           "seed": 42}),
    ]))
    assert "LORA_R = 16" in code and "LORA_ALPHA = 32" in code
    assert "LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA" in code
    assert "TARGET_MODULES = ['q_proj', 'v_proj']" in code
    assert "load_dataset" in code and "hf/set" in code
    assert "BitsAndBytesConfig" in code  # qlora branch
    assert "save_pretrained" in code


def test_rag_script_fragments():
    code = generate_llm_script(llm_graph("g", [
        ("m", "llm.model", {"repo_id": "org/gen", "dtype": "float32",
                             "trust_remote_code": False}),
        ("emb", "llm.embedding_model", {"repo_id": "org/emb", "pooling": "cls",
                                         "batch_size": 4}),
        ("vs", "llm.vector_store", {"backend": "faiss", "persist_path": ""}),
        ("ret", "llm.retriever", {"top_k": 5}),
        ("r", "llm.rag", {"docs_path": "docs/*.md", "chunk_size": 300,
                           "chunk_overlap": 30}),
    ]))
    assert "import faiss" in code
    assert "EMBEDDER_ID = 'org/emb'" in code
    assert "TOP_K = 5" in code
    assert "DOCS_GLOB = 'docs/*.md'" in code
    assert "{context}" in code  # prompt gets a context-aware default


@pytest.mark.skipif(not os.environ.get("AIME_LLM_E2E"), reason="network + model downloads")
def test_e2e_generation_with_tiny_model(tmp_path: Path):
    script = tmp_path / "gen_llm.py"
    script.write_text(generate_llm_script(llm_graph("e2e", [
        ("m", "llm.model", {"repo_id": "sshleifer/tiny-gpt2", "dtype": "float32",
                             "trust_remote_code": False}),
        ("gn", "llm.generation", {"max_new_tokens": 8, "do_sample": False,
                                   "temperature": 0.7, "top_p": 0.9, "top_k": 50,
                                   "repetition_penalty": 1.0}),
    ])))
    result = subprocess.run([sys.executable, script.name, "Hello"], cwd=tmp_path,
                            capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stderr
