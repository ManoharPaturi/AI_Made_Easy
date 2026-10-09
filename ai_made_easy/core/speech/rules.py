"""Design rules for speech tasks (registered as lints; Quick Fixes in core.fixes).

"set <param> to <value>" and "Set the Input shape to '…'" phrasing becomes a
one-click fix on the flagged block.
"""
from __future__ import annotations

from ai_made_easy.core.lints import LintContext, _issue, _name, register_rule
from ai_made_easy.core.speech.blocks import (
    ALPHABET,
    CHARS_PER_SECOND,
    LEXICON,
    SOUND_TAGS,
    SPEECH_LOSSES,
)
from ai_made_easy.core.speech.tasks import dataset_of, speech_task

MEL_BLOCKS = ("audio.mel_spectrogram", "audio.mfcc")


def data_info(node) -> dict:  # noqa: ANN001
    """Task, sample rate, clip samples, class count and alphabet of a speech dataset."""
    p = dict(node.resolved_params()) if node is not None else {
        "task": "transcripts", "sample_rate": 16000, "duration": 1.0, "n_keywords": 6}
    rate = int(p["sample_rate"])
    info = {"task": p["task"], "rate": rate, "samples": int(round(float(p["duration"]) * rate)),
            "classes": None, "alphabet": str(p.get("alphabet", ALPHABET)), "max_chars": None}
    synthetic = node is None or node.type_id == "data.synthetic_speech"
    if synthetic:
        info["classes"] = {"keywords": max(2, min(int(p["n_keywords"]), len(LEXICON))),
                           "tags": len(SOUND_TAGS)}.get(p["task"])
        info["max_chars"] = max(1, int((float(p["duration"]) - 0.1) / 0.07))
    elif p["task"] == "transcripts":
        info["max_chars"] = int(float(p["duration"]) * CHARS_PER_SECOND * 0.75)
    return info


def speech_rules(ctx: LintContext) -> list:
    if speech_task(ctx.graph) is None:
        return []
    data_node = dataset_of(ctx.graph)
    info = data_info(data_node)
    out: list = []
    inputs = ctx.nodes_of("core.input")
    have = ctx.shapes.get(inputs[0].instance_id) if inputs else None
    if have is not None and list(have) not in ([1, info["samples"]], [info["samples"]]):
        source = _name(data_node) if data_node is not None else "the generated speech"
        out.append(_issue("error", f"{source} gives {info['samples']:,}-sample waveforms "
                                   f"({info['samples'] / info['rate']:g} s at {info['rate']} Hz) "
                                   f"but the Input is {list(have)}. Set the Input shape to "
                                   f"'1, {info['samples']}'", inputs[0].instance_id))
    out += _output_rules(ctx, info)
    out += _rate_rules(ctx, info, data_node)
    for loss in ctx.nodes_of(*SPEECH_LOSSES):
        if info["task"] != "transcripts":
            out.append(_issue("error", f"{_name(loss)} trains speech recognition; this dataset's "
                                       f"task is {info['task']}: remove the loss or set task to "
                                       "transcripts", loss.instance_id))
    return out


def _output_rules(ctx: LintContext, info: dict) -> list:
    last = ctx.last_compute()
    shape = ctx.shapes.get(last.instance_id) if last is not None else None
    if shape is None:
        return []
    shape = list(shape)
    if info["task"] == "transcripts":
        want = len(info["alphabet"]) + 1
        if len(shape) != 2:
            return [_issue("error", f"speech recognition needs per-frame character scores "
                                    f"[frames, {want}] (alphabet + CTC blank) but the model "
                                    f"outputs {shape}: keep the time axis (sequence layout, "
                                    "RNN with return_sequences) and end with Dense",
                           last.instance_id)]
        out = []
        if shape[1] != want:
            out.append(_issue("error", f"the alphabet has {want - 1} characters plus the CTC "
                                       f"blank, but the model scores {shape[1]} per frame: set "
                                       f"units to {want}", last.instance_id))
        if info["max_chars"] and shape[0] < info["max_chars"]:
            out.append(_issue("warning", f"the model emits {shape[0]} frames per clip for "
                                         f"transcripts of up to ~{info['max_chars']} characters: "
                                         "CTC needs at least one frame per character (more for "
                                         "repeated letters); pool or stride less, or lower "
                                         "hop_length", last.instance_id))
        return out
    k = info["classes"]
    what = "keyword" if info["task"] == "keywords" else "tag"
    if len(shape) != 1:
        return [_issue("error", f"{what} models output one score per {what} but this one "
                                f"outputs {shape}: add global pooling or Flatten, then Dense",
                       last.instance_id)]
    out = []
    if k is not None and shape[0] != k:
        out.append(_issue("error", f"the dataset has {k} {what}s but the model outputs "
                                   f"{shape[0]} scores: set units to {k}", last.instance_id))
    if info["task"] == "tags" and last.type_id == "core.softmax":
        out.append(_issue("warning", "softmax makes tags compete; several tags can be present "
                                     "at once: remove the Softmax (training applies a sigmoid)",
                          last.instance_id))
    return out


def _rate_rules(ctx: LintContext, info: dict, data_node) -> list:  # noqa: ANN001
    out = []
    for node in ctx.nodes_of("audio.hf_encoder"):
        p = dict(node.resolved_params())
        if info["rate"] != 16000:
            target = data_node.instance_id if data_node is not None else node.instance_id
            out.append(_issue("error", f"{_name(node)} was pretrained on 16 kHz audio but the "
                                       f"clips are {info['rate']} Hz: set sample_rate to 16000",
                              target))
        if info["task"] == "transcripts" and p["pooling"] == "mean":
            out.append(_issue("error", f"{_name(node)} averages the frames away; speech "
                                       "recognition needs them: set pooling to none",
                              node.instance_id))
    for node in ctx.nodes_of(*MEL_BLOCKS):
        rate = int(node.resolved_params()["sample_rate"])
        if rate != info["rate"]:
            out.append(_issue("warning", f"{_name(node)} assumes {rate} Hz but the clips are "
                                         f"{info['rate']} Hz, so its mel bands are shifted: "
                                         f"set sample_rate to {info['rate']}",
                              node.instance_id))
    return out


register_rule(speech_rules)
