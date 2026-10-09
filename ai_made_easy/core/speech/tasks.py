"""Speech tasks (keyword spotting, speech recognition, audio tagging) and their resolver."""
from __future__ import annotations

from ai_made_easy.core.speech.blocks import SPEECH_DATA, SPEECH_LOSSES, SPEECH_TASKS
from ai_made_easy.core.tasks import Task, register_task, register_task_resolver

_COMMON = {"default_optimizer": "AdamW (lr = 1e-3)"}

register_task(Task(
    "keyword_spotting", "Keyword spotting",
    "Recognise which short spoken command a clip contains.",
    target="keyword index", output_role="logits", classification=True, modalities=("audio",),
    trainer_kind="speech", serving="keyword", default_metrics=("eval.accuracy",),
    meta={**_COMMON, "loss_tasks": (), "losses": ["train.loss_cross_entropy"],
          "metrics": ["eval.accuracy", "eval.f1"], "default_loss": "train.loss_cross_entropy"}))
register_task(Task(
    "speech_recognition", "Speech recognition (CTC)",
    "Transcribe speech to text with per-frame character scores and CTC.",
    target="transcript", output_role="logits", modalities=("audio",), trainer_kind="speech",
    serving="transcript", default_metrics=("eval.cer", "eval.wer"),
    meta={**_COMMON, "loss_tasks": (), "losses": ["speech.loss_ctc"],
          "metrics": ["eval.cer", "eval.wer"], "default_loss": "speech.loss_ctc"}))
register_task(Task(
    "audio_tagging", "Audio tagging",
    "Detect every sound event present in a clip (multi-label).",
    target="tag set", output_role="logits", modalities=("audio",), trainer_kind="speech",
    serving="tags", default_metrics=("eval.tag_map",),
    meta={**_COMMON, "loss_tasks": (), "losses": ["train.loss_bce_logits"],
          "metrics": ["eval.tag_map"], "default_loss": "train.loss_bce_logits"}))


def dataset_of(graph):  # noqa: ANN001, ANN201
    return next((n for n in graph.nodes.values() if n.type_id in SPEECH_DATA), None)


def speech_task(graph) -> str | None:  # noqa: ANN001
    data = dataset_of(graph)
    if data is not None:
        return SPEECH_TASKS[data.resolved_params()["task"]]
    if any(n.type_id in SPEECH_LOSSES for n in graph.nodes.values()):
        return "speech_recognition"
    return None


register_task_resolver(speech_task)
