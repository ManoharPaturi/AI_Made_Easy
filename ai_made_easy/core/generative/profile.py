"""Data workspace profiles for generative datasets (images, text corpora, text pairs).

Uses the training script's own readers (``core.generative.runtime``) and reports
what matters for generation: image classes, corpus size and byte make-up, and the
source / target lengths that decide the Input and max_target_len of a seq2seq design.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from ai_made_easy.core.data.health import ClassCount, Finding
from ai_made_easy.core.data.lints import LOCAL_BLOCKS
from ai_made_easy.core.data.profile import DataProfile, register_profiler, resolve_path
from ai_made_easy.core.generative.blocks import GENERATIVE_DATA

MIN_CORPUS = 100_000    # bytes: below this, small models memorise the text


def _images(params: dict) -> DataProfile:
    from ai_made_easy.core.generative.runtime import namespace

    ns = namespace()
    n = min(int(params["n_images"]), 500)
    _x, labels, names = ns["synthetic_images"](n, int(params["image_size"]),
                                               int(params["channels"]), params["kind"],
                                               int(params["seed"]))
    counts = Counter(int(v) for v in labels)
    profile = DataProfile("data.synthetic_images", "synthetic", source="generated",
                          rows=int(params["n_images"]), task="generation")
    profile.classes = [ClassCount(names[k], v) for k, v in sorted(counts.items())]
    size = int(params["image_size"])
    profile.summary = (f"{int(params['n_images']):,} generated {params['kind']} images · "
                       f"{len(names)} class(es)")
    profile.details = [f"images [{params['channels']}, {size}, {size}] → Input "
                       f"[{params['channels']}, {size}, {size}] (VAE / diffusion) or the "
                       "generator's output (GAN)"]
    return profile


def _text(type_id: str, params: dict, base) -> DataProfile:
    from ai_made_easy.core.generative.runtime import namespace

    ns = namespace()
    p = {"block": type_id, **params}
    if type_id == "data.text_corpus":
        p["path"] = str(resolve_path(str(params["path"]), base))
        if not Path(p["path"]).exists():
            return DataProfile(type_id, "table", source=p["path"],
                               error=f"{p['path']} does not exist")
    try:
        text = ns["load_text"](p)
    except (OSError, ValueError) as exc:
        return DataProfile(type_id, "table", error=str(exc))
    data = ns["encode_bytes"](text)
    profile = DataProfile(type_id, "synthetic" if type_id == "data.synthetic_text" else "folder",
                          source=str(p.get("path") or "generated"), rows=len(data),
                          task="language_modeling")
    lines = text.splitlines()
    findings: list[Finding] = []
    if len(data) < MIN_CORPUS:
        findings.append(Finding("info", f"the corpus has {len(data):,} bytes",
                                "small corpora are memorised quickly: keep the model small or "
                                "add text (100 KB+)"))
    non_ascii = float(np.mean(data >= 128)) if len(data) else 0.0
    if non_ascii > 0.2:
        findings.append(Finding("info", f"{non_ascii:.0%} of bytes are non-ASCII",
                                "byte-level models spend several tokens per character: use a "
                                "longer context"))
    profile.summary = f"{len(data):,} bytes · {len(lines):,} lines · {len(set(data.tolist()))} " \
                      "distinct bytes"
    profile.details = [f"line length median {int(np.median([len(ln) for ln in lines] or [0]))} "
                       "characters", "byte-level tokens: vocab_size 256"]
    profile.findings = findings
    return profile


def _pairs(type_id: str, params: dict, base) -> DataProfile:
    from ai_made_easy.core.generative.runtime import namespace

    ns = namespace()
    p = {"block": type_id, **params}
    if type_id == "data.text_pairs":
        p["path"] = str(resolve_path(str(params["path"]), base))
        if not Path(p["path"]).exists():
            return DataProfile(type_id, "table", source=p["path"],
                               error=f"{p['path']} does not exist")
    try:
        pairs = ns["load_pairs"](p)
    except (OSError, ValueError, KeyError) as exc:
        return DataProfile(type_id, "table", error=str(exc))
    profile = DataProfile(type_id, "synthetic" if type_id == "data.synthetic_pairs" else "table",
                          source=str(p.get("path") or "generated"), rows=len(pairs),
                          task="sequence_to_sequence")
    if not pairs:
        profile.error = "no pairs found"
        return profile
    src = np.array([len(ns["encode_bytes"](s)) for s, _ in pairs])
    tgt = np.array([len(ns["encode_bytes"](t)) for _, t in pairs])
    findings: list[Finding] = []
    empty = int(((src == 0) | (tgt == 0)).sum())
    if empty:
        findings.append(Finding("warning", f"{empty} pair(s) have an empty source or target",
                                "drop them before training"))
    p95_src, p95_tgt = int(np.percentile(src, 95)), int(np.percentile(tgt, 95))
    profile.summary = f"{len(pairs):,} pairs · source ≤ {src.max()} bytes · target ≤ " \
                      f"{tgt.max()} bytes"
    profile.details = [f"source bytes median {int(np.median(src))}, 95% ≤ {p95_src} → Input "
                       f"[{src.max()}]",
                       f"target bytes median {int(np.median(tgt))}, 95% ≤ {p95_tgt} → "
                       f"max_target_len {tgt.max() + 1} (incl. EOS)"]
    profile.findings = findings
    return profile


def profile_generative(type_id: str, params: dict, base=None) -> DataProfile:
    kind = GENERATIVE_DATA[type_id]
    if kind == "images":
        return _images(params)
    if kind == "text":
        return _text(type_id, params, base)
    return _pairs(type_id, params, base)


def register() -> None:
    for type_id in GENERATIVE_DATA:
        register_profiler(type_id, lambda params, base, _t=type_id: profile_generative(
            _t, params, base))
        LOCAL_BLOCKS.add(type_id)
