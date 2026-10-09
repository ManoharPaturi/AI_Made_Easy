"""Speech blocks: keyword / transcript / tag datasets, the CTC loss and speech metrics."""
from __future__ import annotations

from ai_made_easy.core.blocks._dsl import P
from ai_made_easy.core.palette import family_color
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import BlockDefinition

ALPHABET = "abcdefghijklmnopqrstuvwxyz' "
LEXICON = ("yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go")
SOUND_TAGS = ("tone", "chirp", "noise", "click", "hum")
SPEECH_TASKS = {"keywords": "keyword_spotting", "transcripts": "speech_recognition",
                "tags": "audio_tagging"}
SPEECH_DATA: dict[str, dict] = {}
SPEECH_LOSSES: dict[str, dict] = {}
SPEECH_METRICS: dict[str, dict] = {}
CHARS_PER_SECOND = 22   # fastest synthetic speech: one letter per 45 ms


def _checks(p: dict) -> list:
    out = []
    if float(p["val_fraction"]) + float(p["test_fraction"]) >= 0.9:
        out.append(("error", "validation + test take almost every clip: lower val_fraction / "
                             "test_fraction"))
    if p["task"] == "transcripts" and "alphabet" in p and len(set(p["alphabet"])) != \
            len(p["alphabet"]):
        out.append(("error", "the alphabet repeats a character"))
    return out


def _data(type_id: str, name: str, params: tuple, desc: str,
          duration: float = 1.0) -> BlockDefinition:
    SPEECH_DATA[type_id] = {}
    common = (P("task", "enum", "keywords", options=tuple(SPEECH_TASKS),
                help="keywords: one word per clip; transcripts: speech recognition (CTC); "
                     "tags: several sound events per clip"),
              P("sample_rate", "int", 16000, lo=1000, help="Clips are resampled to this rate"),
              P("duration", "float", duration, lo=0.1,
                help="Seconds; clips are zero-padded or trimmed to this length"),
              P("augment", "bool", True, help="Random gain, time shift and noise in training"),
              P("val_fraction", "float", 0.1, lo=0.0, hi=0.5),
              P("test_fraction", "float", 0.1, lo=0.0, hi=0.5))
    return BlockDefinition(
        type_id=type_id, display_name=name, category="Data", color=family_color("data"),
        params=(*params, *common), description=desc, library="PyTorch", checks_fn=_checks,
        meta={"modality": "audio", "speech": True})


def _datasets() -> list[BlockDefinition]:
    return [
        _data("data.speech_csv", "Speech Manifest",
              (P("path", "str", "clips.csv", help="CSV / TSV with one row per audio file"),
               P("audio_column", "str", "path",
                 help="WAV file paths, relative to the manifest"),
               P("label_column", "str", "label",
                 help="Keyword, transcript or tags (by task)"),
               P("tag_separator", "str", ";", help="Separates tags in one cell"),
               P("alphabet", "str", ALPHABET,
                 help="Characters the recognizer emits (transcripts are lower-cased and "
                      "other characters dropped)")),
              "Audio clips listed in a CSV manifest with a keyword, transcript or tags column."),
        _data("data.synthetic_speech", "Synthetic Speech",
              (P("n_clips", "int", 600, lo=10), P("n_keywords", "int", 6, lo=2, hi=10,
                                                   help="Words from yes / no / up / down / ..."),
               P("noise", "float", 0.03, lo=0.0), P("seed", "int", 0, lo=0)),
              "Tone-coded words, sentences and sound events (tone, chirp, noise, click, hum): "
              "try speech tasks without recordings."),
    ]


def _metric(type_id: str, name: str, key: str, tasks: tuple, desc: str) -> BlockDefinition:
    SPEECH_METRICS[type_id] = {"key": key, "tasks": tasks}
    return BlockDefinition(
        type_id=type_id, display_name=name, category="Metrics",
        color=family_color("evaluation"), description=desc, library="PyTorch",
        meta={"kind": "metrics", "key": key, "tasks": list(tasks), "speech": True})


def _losses_metrics() -> list[BlockDefinition]:
    SPEECH_LOSSES["speech.loss_ctc"] = {}
    ctc = BlockDefinition(
        type_id="speech.loss_ctc", display_name="CTC Loss", category="Loss",
        color=family_color("training"),
        params=(P("zero_infinity", "bool", True,
                  help="Ignore clips too short for their transcript instead of failing"),),
        description="Connectionist temporal classification: aligns per-frame character "
                    "scores [T, alphabet + 1] with transcripts (index 0 is the blank).",
        library="PyTorch", meta={"kind": "loss", "task": "speech_recognition", "speech": True})
    return [
        ctc,
        _metric("eval.cer", "Character Error Rate", "cer", ("speech_recognition",),
                "Character edits (insert / delete / substitute) per reference character."),
        _metric("eval.wer", "Word Error Rate", "wer", ("speech_recognition",),
                "Word edits per reference word."),
        _metric("eval.tag_map", "Tag mAP", "map", ("audio_tagging",),
                "Mean average precision over tags (each tag ranked across clips)."),
    ]


def register_all() -> None:
    reg = get_registry()
    for defn in (*_datasets(), *_losses_metrics()):
        reg.register(defn)
