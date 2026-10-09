"""Runtime code shared by generated speech scripts and the app (numpy + stdlib ``wave``).

Like ``core.forecast.runtime``: the training script embeds these strings and the
app executes the same source for profiling and tests.
"""
from __future__ import annotations

DATA_CODE = r'''
# ======================================================================== speech data
ALPHABET = "abcdefghijklmnopqrstuvwxyz' "
LEXICON = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
SOUND_TAGS = ["tone", "chirp", "noise", "click", "hum"]
TASK_OF = {"keywords": "keyword_spotting", "transcripts": "speech_recognition",
           "tags": "audio_tagging"}


def resample(wave: np.ndarray, rate: int, target: int) -> np.ndarray:
    """Linear-interpolation resampling (good enough for speech features)."""
    if rate == target or len(wave) == 0:
        return wave.astype(np.float32)
    n = max(1, int(round(len(wave) * target / rate)))
    return np.interp(np.linspace(0, len(wave) - 1, n), np.arange(len(wave)),
                     wave).astype(np.float32)


def fit_length(wave: np.ndarray, n: int) -> np.ndarray:
    """Zero-pad or trim to exactly n samples."""
    out = np.zeros(n, np.float32)
    out[: min(n, len(wave))] = wave[:n]
    return out


def decode_wav_bytes(data: bytes) -> tuple:
    """(mono float32 waveform in [-1, 1], sample rate) from WAV bytes."""
    import io
    import wave as wavmod

    try:
        with wavmod.open(io.BytesIO(data)) as fh:
            rate, width, channels = fh.getframerate(), fh.getsampwidth(), fh.getnchannels()
            raw = fh.readframes(fh.getnframes())
    except (wavmod.Error, EOFError) as exc:
        try:
            import soundfile

            wave, rate = soundfile.read(io.BytesIO(data), dtype="float32", always_2d=True)
            return wave.mean(1), int(rate)
        except Exception:  # noqa: BLE001 — report the original problem
            raise ValueError(f"not a readable WAV file ({exc}); install soundfile for "
                             "FLAC / OGG / float WAV") from None
    if width == 1:
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128
    elif width == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
    elif width == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        x = ((b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)) << 8 >> 8).astype(np.float32) / 2**23
    else:
        x = np.frombuffer(raw, "<i4").astype(np.float32) / 2**31
    return x.reshape(-1, channels).mean(1), rate


def read_wav(path, sample_rate: int, n_samples: int) -> np.ndarray:
    wave, rate = decode_wav_bytes(Path(path).read_bytes())
    return fit_length(resample(wave, rate, sample_rate), n_samples)


def write_wav(path, wave: np.ndarray, sample_rate: int) -> None:
    import wave as wavmod

    pcm = (np.clip(wave, -1, 1) * 32767).astype("<i2")
    with wavmod.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(pcm.tobytes())


def normalize_text(text: str, alphabet: str = ALPHABET) -> str:
    keep = set(alphabet)
    text = " ".join(str(text).lower().split())
    return "".join(c for c in text if c in keep)


# ---------------------------------------------------------------- synthetic speech
def _tone(freq: float, seconds: float, sample_rate: int, rng) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    f = freq * rng.uniform(0.97, 1.03)
    wave = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)
    return (wave * np.hanning(len(t))).astype(np.float32)


def speak(text: str, sample_rate: int, rng) -> np.ndarray:
    """Tone-coded "speech": each letter is a short two-harmonic tone, spaces are pauses."""
    parts = []
    for c in text:
        if c == " ":
            parts.append(np.zeros(int(rng.uniform(0.05, 0.08) * sample_rate), np.float32))
        else:
            freq = 250 + 70 * ALPHABET.index(c)
            parts.append(_tone(freq, rng.uniform(0.045, 0.065), sample_rate, rng))
    return np.concatenate(parts) if parts else np.zeros(0, np.float32)


def _event(tag: str, sample_rate: int, rng) -> np.ndarray:
    n = int(rng.uniform(0.15, 0.35) * sample_rate)
    t = np.arange(n) / sample_rate
    if tag == "tone":
        wave = np.sin(2 * np.pi * rng.uniform(600, 900) * t)
    elif tag == "chirp":
        f0, f1 = rng.uniform(300, 600), rng.uniform(1500, 3000)
        wave = np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t ** 2 / (2 * t[-1])))
    elif tag == "noise":
        wave = rng.standard_normal(n) * 0.5
    elif tag == "click":
        wave = np.zeros(n)
        wave[:: int(sample_rate / rng.uniform(15, 30))] = 1.0
    else:   # hum: mains-like low harmonics
        f = rng.uniform(100, 130)
        wave = sum(np.sin(2 * np.pi * k * f * t) / k for k in (1, 2, 3))
    return (wave * np.hanning(n)).astype(np.float32)


def synthetic_speech(task: str, n_clips: int, n_keywords: int, duration: float,
                     sample_rate: int, noise: float, seed: int) -> tuple:
    """(records, names): records hold "wave" [samples] and "label" (class index, text or
    list of tag indices); names are keywords, tags or the alphabet."""
    rng = np.random.default_rng(seed)
    n = int(round(duration * sample_rate))
    k = max(2, min(int(n_keywords), len(LEXICON)))
    records = []
    for i in range(n_clips):
        clip = np.zeros(n, np.float32)
        if task == "tags":
            label = [j for j in range(len(SOUND_TAGS)) if rng.random() < 0.35] or \
                [int(rng.integers(len(SOUND_TAGS)))]
            for j in label:
                event = _event(SOUND_TAGS[j], sample_rate, rng)[:n]
                start = int(rng.integers(0, n - len(event) + 1))
                clip[start:start + len(event)] += event * rng.uniform(0.3, 0.8)
        else:
            if task == "keywords":
                label = i % k
                text = LEXICON[label]
            else:
                for _ in range(20):   # sentences that fit the clip
                    text = " ".join(rng.choice(LEXICON[:k], int(rng.integers(1, 4))))
                    if len(text) * 0.07 + 0.1 < duration:
                        break
                label = text
            voice = speak(text, sample_rate, rng)[:n]
            start = int(rng.integers(0, max(1, n - len(voice)) + 1))
            clip[start:start + len(voice)] = voice[: n - start] * rng.uniform(0.4, 0.9)
        clip += noise * rng.standard_normal(n).astype(np.float32)
        records.append({"wave": clip, "label": label, "id": f"clip_{i}"})
    names = SOUND_TAGS if task == "tags" else (LEXICON[:k] if task == "keywords"
                                                else list(ALPHABET))
    return records, list(names)


# ---------------------------------------------------------------- manifests
def read_manifest(path, task: str, audio_column: str, label_column: str, separator: str,
                  alphabet: str, sample_rate: int, duration: float) -> tuple:
    """Rows of a CSV / TSV manifest: audio file path (relative to the manifest) + label."""
    import csv

    path = Path(str(path)).expanduser()
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter=delimiter))
    if rows and (audio_column not in rows[0] or label_column not in rows[0]):
        raise ValueError(f"columns '{audio_column}' / '{label_column}' not found; the "
                         f"manifest has {list(rows[0])}")
    n = int(round(duration * sample_rate))
    if task == "keywords":
        names = sorted({r[label_column].strip() for r in rows})
    elif task == "tags":
        names = sorted({t.strip() for r in rows for t in r[label_column].split(separator)
                        if t.strip()})
    else:
        names = list(alphabet)
    index = {name: i for i, name in enumerate(names)}
    records = []
    for i, row in enumerate(rows):
        file = Path(row[audio_column])
        file = file if file.is_absolute() else path.parent / file
        raw = row[label_column]
        if task == "keywords":
            label = index[raw.strip()]
        elif task == "tags":
            label = sorted({index[t.strip()] for t in raw.split(separator) if t.strip()})
        else:
            label = normalize_text(raw, alphabet)
        records.append({"file": str(file), "label": label, "id": file.stem or f"row_{i}",
                        "wave": None})
    for r in records:
        r["wave"] = read_wav(r["file"], sample_rate, n)
    return records, names


def load_speech(dataset: dict) -> tuple:
    if dataset["block"] == "data.synthetic_speech":
        return synthetic_speech(dataset["task"], int(dataset["n_clips"]),
                                int(dataset["n_keywords"]), float(dataset["duration"]),
                                int(dataset["sample_rate"]), float(dataset["noise"]),
                                int(dataset["seed"]))
    return read_manifest(dataset["path"], dataset["task"], dataset["audio_column"],
                         dataset["label_column"], dataset["tag_separator"],
                         dataset["alphabet"], int(dataset["sample_rate"]),
                         float(dataset["duration"]))


def split_records(records: list, val_fraction: float, test_fraction: float,
                  seed: int) -> dict:
    order = np.random.default_rng(seed).permutation(len(records))
    n_test = int(round(len(records) * test_fraction))
    n_val = int(round(len(records) * val_fraction))
    pick = lambda idx: [records[i] for i in idx]  # noqa: E731
    return {"test": pick(order[:n_test]), "val": pick(order[n_test:n_test + n_val]),
            "train": pick(order[n_test + n_val:])}
'''

METRICS_CODE = r'''
# ======================================================================== speech metrics
def edit_distance(ref, hyp) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def error_rate(refs: list, hyps: list, words: bool) -> float:
    split = (lambda s: s.split()) if words else list
    errors = sum(edit_distance(split(r), split(h)) for r, h in zip(refs, hyps))
    return errors / max(1, sum(len(split(r)) for r in refs))


def ctc_greedy(ids, alphabet: str) -> str:
    """Collapse repeats, drop blanks (index 0); index i > 0 is alphabet[i - 1]."""
    out, last = [], 0
    for i in ids:
        i = int(i)
        if i != last and i != 0:
            out.append(alphabet[i - 1])
        last = i
    return " ".join("".join(out).split())


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-score)
    hits = y[order]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / hits.sum())


def classification_scores(y: np.ndarray, pred: np.ndarray, k: int) -> dict:
    confusion = np.zeros((k, k), int)
    np.add.at(confusion, (y, pred), 1)
    tp = np.diag(confusion).astype(float)
    precision = tp / np.maximum(confusion.sum(0), 1)
    recall = tp / np.maximum(confusion.sum(1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {"accuracy": float(tp.sum() / max(len(y), 1)), "macro_f1": float(f1.mean()),
            "per_class_accuracy": recall.round(4).tolist(), "confusion": confusion.tolist()}


def tagging_scores(y: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    ap = [average_precision(y[:, j], prob[:, j]) for j in range(y.shape[1])]
    pred = prob >= threshold
    tp = float((pred & (y > 0)).sum())
    f1 = 2 * tp / max(float(pred.sum() + (y > 0).sum()), 1.0)
    return {"map": float(np.nanmean(ap)) if not np.all(np.isnan(ap)) else 0.0, "f1": f1,
            "per_class_ap": [None if np.isnan(a) else round(a, 4) for a in ap]}
'''

PLOT_CODE = r'''
# ======================================================================== plots
def spectrogram_image(wave: np.ndarray, title: str, subtitle: str = "",
                      size=(480, 200)) -> "Image.Image":
    """Log-magnitude spectrogram (time → right, frequency ↑) with captions."""
    n_fft, hop = 512, 128
    frames = max(1, 1 + (len(wave) - n_fft) // hop)
    window = np.hanning(n_fft)
    spec = np.stack([np.abs(np.fft.rfft(wave[i * hop:i * hop + n_fft] * window,
                                        n=n_fft)) for i in range(frames)], 1)
    db = np.log10(spec[: n_fft // 4] + 1e-6)
    db = (db - db.min()) / max(float(db.max() - db.min()), 1e-6)
    stops = np.array([[20, 18, 40], [70, 40, 120], [200, 70, 90], [250, 200, 80]], float)
    pos = db[::-1] * (len(stops) - 1)
    lo = np.clip(pos.astype(int), 0, len(stops) - 2)
    frac = (pos - lo)[..., None]
    rgb = (stops[lo] * (1 - frac) + stops[lo + 1] * frac).astype(np.uint8)
    w, h = size
    img = Image.new("RGB", size, (24, 26, 30))
    img.paste(Image.fromarray(rgb).resize((w, h - 40)), (0, 40))
    draw = ImageDraw.Draw(img)
    draw.text((8, 4), title, fill=(230, 230, 230))
    draw.text((8, 22), subtitle, fill=(160, 200, 160))
    return img
'''


def namespace() -> dict:
    import json
    from pathlib import Path

    import numpy as np

    try:   # plots only; reading data and metrics work without pillow
        from PIL import Image, ImageDraw
    except ImportError:
        Image = ImageDraw = None
    ns: dict = {"np": np, "json": json, "Path": Path, "Image": Image, "ImageDraw": ImageDraw}
    for code in (DATA_CODE, METRICS_CODE, PLOT_CODE):
        exec(compile(code, "<speech-runtime>", "exec"), ns)  # noqa: S102 — our own source
    return ns
