"""Data workspace profiles for speech datasets (manifests and synthetic speech).

Reads WAV headers (not the audio) for manifests and reports what matters for
speech: clip count and durations, sample rates, missing files, clips the fixed
duration trims, keyword / tag balance and transcript lengths vs the alphabet.
"""
from __future__ import annotations

import csv
import wave
from collections import Counter
from pathlib import Path

import numpy as np

from ai_made_easy.core.data.health import ClassCount, Finding
from ai_made_easy.core.data.lints import LOCAL_BLOCKS
from ai_made_easy.core.data.profile import DataProfile, register_profiler, resolve_path
from ai_made_easy.core.speech.blocks import CHARS_PER_SECOND, SPEECH_DATA, SPEECH_TASKS

MAX_HEADERS = 2000   # WAV headers read for durations / sample rates


def _header(path: Path) -> tuple[float, int] | None:
    try:
        with wave.open(str(path)) as fh:
            return fh.getnframes() / fh.getframerate(), fh.getframerate()
    except (OSError, wave.Error, EOFError, ZeroDivisionError):
        return None


def _labels(profile: DataProfile, task: str, labels: list, findings: list[Finding],
            separator: str = ";", alphabet: str = "", duration: float = 1.0) -> None:
    if task == "keywords":
        counts = Counter(labels)
        profile.classes = [ClassCount(str(k), v) for k, v in sorted(counts.items())]
        if counts and max(counts.values()) > 5 * max(min(counts.values()), 1):
            findings.append(Finding("warning", "keyword counts are very unbalanced",
                                    "collect more clips of the rare keywords"))
    elif task == "tags":
        counts = Counter(t for tags in labels for t in tags)
        profile.classes = [ClassCount(str(k), v) for k, v in sorted(counts.items())]
        per_clip = np.mean([len(t) for t in labels]) if labels else 0
        profile.details.append(f"{per_clip:.1f} tags per clip on average")
    else:
        lengths = np.array([len(t) for t in labels]) if labels else np.zeros(1)
        profile.details.append(f"transcripts {lengths.min()}–{lengths.max()} characters "
                               f"(mean {lengths.mean():.0f})")
        if alphabet:
            dropped = Counter(c for t in labels for c in str(t).lower() if c not in alphabet)
            if dropped:
                shown = "".join(sorted(dropped)[:12])
                findings.append(Finding("info", f"{sum(dropped.values()):,} transcript "
                                                f"characters are outside the alphabet "
                                                f"({shown!r})",
                                        "they are dropped; add them to the alphabet to keep "
                                        "them"))
        if lengths.max() > duration * CHARS_PER_SECOND:
            findings.append(Finding("warning", f"the longest transcript has {lengths.max()} "
                                               f"characters for {duration:g} s of audio",
                                    "raise duration, or check the clips are not trimmed"))


def _profile_manifest(params: dict, base) -> DataProfile:
    path = resolve_path(str(params["path"]), base)
    if not path.exists():
        return DataProfile("data.speech_csv", "table", source=str(path),
                           error=f"{path} does not exist")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t" if path.suffix == ".tsv" else ","))
    profile = DataProfile("data.speech_csv", "table", source=str(path), rows=len(rows),
                          format=path.suffix.lstrip("."), task=SPEECH_TASKS[params["task"]])
    if not rows:
        profile.error = "the manifest has no rows"
        return profile
    audio, label = params["audio_column"], params["label_column"]
    missing_cols = [c for c in (audio, label) if c not in rows[0]]
    if missing_cols:
        profile.error = f"columns {missing_cols} not found; the manifest has {list(rows[0])}"
        return profile
    findings: list[Finding] = []
    files = [Path(r[audio]) if Path(r[audio]).is_absolute() else path.parent / r[audio]
             for r in rows]
    missing = [f for f in files if not f.exists()]
    if missing:
        findings.append(Finding("warning", f"{len(missing)} audio file(s) are missing "
                                           f"(e.g. {missing[0].name})",
                                "fix the paths (relative to the manifest) or drop the rows"))
    headers = [h for f in files[:MAX_HEADERS] if f.exists() and (h := _header(f))]
    unreadable = len([f for f in files[:MAX_HEADERS] if f.exists()]) - len(headers)
    if unreadable:
        findings.append(Finding("warning", f"{unreadable} file(s) are not PCM WAV",
                                "convert them to WAV, or install soundfile to read them"))
    duration, rate = float(params["duration"]), int(params["sample_rate"])
    if headers:
        seconds = np.array([h[0] for h in headers])
        rates = Counter(h[1] for h in headers)
        profile.details.append(f"clips {seconds.min():.2f}–{seconds.max():.2f} s "
                               f"(mean {seconds.mean():.2f} s)")
        profile.details.append("sample rates: " + ", ".join(f"{r} Hz × {n}"
                                                            for r, n in rates.most_common(4)))
        trimmed = int((seconds > duration + 1e-3).sum())
        if trimmed:
            findings.append(Finding("warning", f"{trimmed} clip(s) are longer than "
                                               f"duration ({duration:g} s) and get trimmed",
                                    f"set duration to {np.ceil(seconds.max() * 10) / 10:g}"))
        if set(rates) != {rate}:
            findings.append(Finding("info", f"clips are resampled to {rate} Hz",
                                    "fine for most models; pretrained speech encoders need "
                                    "16000 Hz"))
    task = params["task"]
    raw = [r[label] for r in rows]
    labels = ([[t.strip() for t in v.split(params["tag_separator"]) if t.strip()] for v in raw]
              if task == "tags" else [v.strip() for v in raw])
    _labels(profile, task, labels, findings, alphabet=params.get("alphabet", ""),
            duration=duration)
    profile.summary = (f"{len(rows):,} clips · {task}"
                       + (f" · {len(profile.classes)} classes" if profile.classes else ""))
    profile.findings = findings
    return profile


def _profile_synthetic(params: dict) -> DataProfile:
    from ai_made_easy.core.speech.runtime import namespace

    ns = namespace()
    n = min(int(params["n_clips"]), 300)
    records, names = ns["synthetic_speech"](params["task"], n, int(params["n_keywords"]),
                                            float(params["duration"]),
                                            int(params["sample_rate"]), float(params["noise"]),
                                            int(params["seed"]))
    task = params["task"]
    profile = DataProfile("data.synthetic_speech", "synthetic", source="generated",
                          rows=int(params["n_clips"]), task=SPEECH_TASKS[task])
    labels = [names[r["label"]] if task == "keywords" else
              [names[i] for i in r["label"]] if task == "tags" else r["label"]
              for r in records]
    findings: list[Finding] = []
    _labels(profile, task, labels, findings, duration=float(params["duration"]))
    profile.details.insert(0, f"{params['duration']} s clips at {params['sample_rate']} Hz → "
                              f"Input [1, {int(round(float(params['duration']) * int(params['sample_rate'])))}]")
    profile.summary = f"{int(params['n_clips']):,} generated clips · {task} · {len(names)} " \
                      f"{'characters' if task == 'transcripts' else 'classes'}"
    profile.findings = findings
    return profile


def profile_speech(type_id: str, params: dict, base=None) -> DataProfile:
    if type_id == "data.synthetic_speech":
        return _profile_synthetic(params)
    return _profile_manifest(params, base)


def register() -> None:
    for type_id in SPEECH_DATA:
        register_profiler(type_id, lambda params, base, _t=type_id: profile_speech(
            _t, params, base))
        LOCAL_BLOCKS.add(type_id)
