"""LLM workflow tasks: generation, fine-tuning and retrieval-augmented answering.

LLM workflows are exported as scripts rather than trained in the app; the tasks name what
a workflow does for the wizard, Explain and the project list.
"""
from __future__ import annotations

from ai_made_easy.core.tasks import Task, register_task, register_task_resolver

_LLM = dict(family="llm", modalities=("text",), trainer_kind="language_model",
            output_role="tokens", meta={"losses": (), "metrics": ()})
register_task(Task(
    "text_generation", "Text generation (LLM)",
    "Prompt a pretrained language model and get text back.",
    target="none (prompted)", serving="text", **_LLM))
register_task(Task(
    "llm_finetuning", "LLM fine-tuning",
    "Adapt a pretrained language model to your examples (supervised fine-tuning, LoRA).",
    target="prompt / response pairs", serving="text", **_LLM))
register_task(Task(
    "retrieval_qa", "Retrieval-augmented answering (RAG)",
    "Answer questions from your documents: retrieve the relevant passages, then generate.",
    target="none (documents)", serving="text", **_LLM))


def llm_task(graph) -> str | None:  # noqa: ANN001
    types = {n.type_id for n in graph.nodes.values()}
    if not any(t.startswith("llm.") for t in types):
        return None
    if "llm.sft" in types or "llm.lora" in types:
        return "llm_finetuning"
    if types & {"llm.rag", "llm.retriever", "llm.vector_store"}:
        return "retrieval_qa"
    return "text_generation"


register_task_resolver(llm_task)
