"""LLM blocks (config): HuggingFace models, tokenizers, LoRA/QLoRA
fine-tuning, generation, embeddings, vector stores, retrieval, RAG.

These configure the LLM script generator (core/codegen/llm_gen.py); they
carry no tensor ports. Heavy dependencies (transformers, peft, datasets)
stay optional — only the generated scripts import them.
"""
from __future__ import annotations

from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import BlockDefinition, ParamSpec
from ai_made_easy.core.blocks._palette import family_color

reg = get_registry()

_LLM_COLOR = family_color("llm")

# which workflow each block declares (shared blocks join whatever is active)
WORKFLOW_OF = {
    "llm.lora": "finetune",
    "llm.dataset": "finetune",
    "llm.sft": "finetune",
    "llm.embedding_model": "rag",
    "llm.vector_store": "rag",
    "llm.retriever": "rag",
    "llm.rag": "rag",
}


def _cfg(type_id: str, name: str, params: tuple) -> None:
    reg.register(BlockDefinition(
        type_id=type_id,
        display_name=name,
        category="LLM",
        color=_LLM_COLOR,
        params=params,
    ))


_cfg("llm.model", "HF Model", (
    ParamSpec(name="repo_id", type="str", default="sshleifer/tiny-gpt2",
              help="Any causal-LM repo id on the HuggingFace Hub"),
    ParamSpec(name="dtype", type="enum", default="float32",
              options=("float32", "float16", "bfloat16")),
    ParamSpec(name="trust_remote_code", type="bool", default=False),
))

_cfg("llm.tokenizer", "HF Tokenizer", (
    ParamSpec(name="repo_id", type="str", default="",
              help="Empty = use the model's repo"),
    ParamSpec(name="max_length", type="int", default=512, minimum=16),
))

def _prompt_checks(p: dict) -> list[tuple[str, str]]:
    """{placeholder} sanity: braces must pair and {input} should appear."""
    import string

    tmpl = str(p.get("template", ""))
    if not tmpl.strip():
        return [("error", "the prompt is empty — write what the AI should do")]
    try:
        fields = [f for _, f, _, _ in string.Formatter().parse(tmpl) if f]
    except ValueError:
        return [("error",
                 "the prompt has a stray { or } — write {{ or }} when you "
                 "want to show a brace")]
    if "input" not in fields:
        return [("warning",
                 "add {input} where the user's text goes — without it the "
                 "prompt never changes")]
    return []


reg.register(BlockDefinition(
    type_id="llm.prompt",
    display_name="Prompt Template",
    category="LLM",
    color=_LLM_COLOR,
    checks_fn=_prompt_checks,
    params=(ParamSpec(name="template", type="str",
                      default="You are a helpful assistant.\nUser: {input}\nAssistant:",
                      help="Use {input} where the user text goes"),),
))

_cfg("llm.generation", "Generation Params", (
    ParamSpec(name="max_new_tokens", type="int", default=128, minimum=1),
    ParamSpec(name="do_sample", type="bool", default=True),
    ParamSpec(name="temperature", type="float", default=0.7, minimum=0.0),
    ParamSpec(name="top_p", type="float", default=0.9, minimum=0.0, maximum=1.0),
    ParamSpec(name="top_k", type="int", default=50, minimum=1),
    ParamSpec(name="repetition_penalty", type="float", default=1.1, minimum=0.0),
))

_cfg("llm.lora", "LoRA / QLoRA Config", (
    ParamSpec(name="mode", type="enum", default="lora", options=("lora", "qlora")),
    ParamSpec(name="r", type="int", default=8, minimum=1),
    ParamSpec(name="alpha", type="int", default=16, minimum=1),
    ParamSpec(name="dropout", type="float", default=0.05,
              minimum=0.0, maximum=1.0),
    ParamSpec(name="target_modules", type="str", default="q_proj,k_proj,v_proj,o_proj",
              help="Comma-separated attention/MLP projection names"),
))

_cfg("llm.dataset", "SFT Dataset", (
    ParamSpec(name="source", type="enum", default="hf_hub",
              options=("hf_hub", "text_file")),
    ParamSpec(name="dataset_id", type="str", default="HuggingFaceTB/smoltalk",
              help="HF dataset id (source = hf_hub)"),
    ParamSpec(name="split", type="str", default="train"),
    ParamSpec(name="text_field", type="str", default="text"),
    ParamSpec(name="file_path", type="str", default="data.txt",
              help="One example per line (source = text_file)"),
))

_cfg("llm.sft", "Fine-tune (SFT)", (
    ParamSpec(name="epochs", type="int", default=1, minimum=1),
    ParamSpec(name="batch_size", type="int", default=1, minimum=1),
    ParamSpec(name="lr", type="float", default=2e-4, minimum=1e-8),
    ParamSpec(name="gradient_accumulation", type="int", default=8, minimum=1),
    ParamSpec(name="warmup_ratio", type="float", default=0.03,
              minimum=0.0, maximum=1.0),
    ParamSpec(name="seed", type="int", default=42, minimum=0),
))

_cfg("llm.embedding_model", "Embedding Model", (
    ParamSpec(name="repo_id", type="str", default="sentence-transformers/all-MiniLM-L6-v2"),
    ParamSpec(name="pooling", type="enum", default="mean", options=("mean", "cls")),
    ParamSpec(name="batch_size", type="int", default=32, minimum=1),
))

_cfg("llm.vector_store", "Vector Store", (
    ParamSpec(name="backend", type="enum", default="numpy",
              options=("numpy", "faiss"),
              help="numpy = zero-dependency in-memory cosine store"),
    ParamSpec(name="persist_path", type="str", default="",
              help="Optional path to save the index (faiss / .npz)"),
))

_cfg("llm.retriever", "Retriever", (
    ParamSpec(name="top_k", type="int", default=3, minimum=1),
))

_cfg("llm.rag", "RAG Pipeline", (
    ParamSpec(name="docs_path", type="str", default="docs/*.txt",
              help="Glob of documents to index (.txt, .md)"),
    ParamSpec(name="chunk_size", type="int", default=500, minimum=32),
    ParamSpec(name="chunk_overlap", type="int", default=50, minimum=0),
))

# ---- audit additions (ported from Langflow's component set) ----

_cfg("llm.doc_loader", "Document Loader", (
    ParamSpec(name="docs_path", type="str", default="docs/*.txt",
              help="Glob of files to read (.txt, .md, .json)"),
    ParamSpec(name="encoding", type="str", default="utf-8"),
))


def _splitter_checks(p: dict) -> list[tuple[str, str]]:
    if p.get("chunk_overlap", 0) >= max(p.get("chunk_size", 500), 1):
        return [("error",
                 f"chunk_overlap ({p['chunk_overlap']}) must be smaller than "
                 f"chunk_size ({p['chunk_size']}) — try half or less")]
    return []


reg.register(BlockDefinition(
    type_id="llm.text_splitter",
    display_name="Text Splitter",
    category="LLM",
    color=_LLM_COLOR,
    checks_fn=_splitter_checks,
    params=(
        ParamSpec(name="chunk_size", type="int", default=500, minimum=32),
        ParamSpec(name="chunk_overlap", type="int", default=50, minimum=0),
        ParamSpec(name="separator", type="str", default="\n\n",
                  help="Split on this text (paragraphs by default)"),
    ),
))

_cfg("llm.chat_memory", "Chat Memory", (
    ParamSpec(name="mode", type="enum", default="store",
              options=("store", "retrieve"),
              help="store = remember the chat, retrieve = replay it"),
    ParamSpec(name="n_messages", type="int", default=8, minimum=1, maximum=50,
              help="How many past messages to keep"),
))

_cfg("llm.output_parser", "Output Parser (JSON)", (
    ParamSpec(name="schema", type="str",
              default='{"answer": "short answer", "steps": ["how you got there"]}',
              help="Describe the JSON you want back — the model is told to "
                   "follow it exactly"),
    ParamSpec(name="strict", type="bool", default=True,
              help="Strict = the script fails loudly if the model's answer "
                   "isn't valid JSON"),
))
