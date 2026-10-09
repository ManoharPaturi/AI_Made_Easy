"""Runtime code shared by generated generative-model scripts and the app (numpy only).

Like ``core.forecast.runtime``: the training script embeds these strings and the
app executes the same source for profiling and tests.
"""
from __future__ import annotations

DATA_CODE = r'''
# ======================================================================== generative data
SHAPES = ["circle", "square", "triangle"]
PAD, BOS, EOS = 256, 257, 258        # sequence-to-sequence specials after the 256 bytes


def synthetic_images(n: int, size: int, channels: int, kind: str, seed: int) -> tuple:
    """(images [N, C, S, S] in [0, 1], labels [N], class names): one filled shape per
    image ("shapes") or soft Gaussian blobs ("blobs", one class)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) + 0.5
    images = np.zeros((n, channels, size, size), np.float32)
    labels = np.zeros(n, np.int64)
    for i in range(n):
        if kind == "blobs":
            img = np.zeros((size, size), np.float32)
            for _ in range(int(rng.integers(1, 4))):
                cx, cy = rng.uniform(0.2, 0.8, 2) * size
                r = rng.uniform(0.08, 0.2) * size
                img += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))
            mask = np.clip(img, 0, 1)
        else:
            label = int(rng.integers(len(SHAPES)))
            labels[i] = label
            r = rng.uniform(0.18, 0.32) * size
            cx, cy = rng.uniform(r, size - r, 2)
            if label == 0:
                mask = ((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r).astype(np.float32)
            elif label == 1:
                mask = ((np.abs(xx - cx) <= r * 0.85) & (np.abs(yy - cy) <= r * 0.85)).astype(
                    np.float32)
            else:
                top = cy - r
                mask = ((yy >= top) & (yy <= cy + r * 0.8)
                        & (np.abs(xx - cx) <= (yy - top) * 0.6)).astype(np.float32)
        color = rng.uniform(0.5, 1.0, channels) if channels > 1 else np.ones(1)
        images[i] = mask[None] * color[:, None, None].astype(np.float32)
    names = ["blob"] if kind == "blobs" else list(SHAPES)
    return images, labels, names


def read_image_folder(root, channels: int, height: int, width: int) -> tuple:
    """root/<class>/<image> (or images directly under root) resized to [C, H, W] in [0, 1]."""
    from PIL import Image

    root = Path(str(root)).expanduser()
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    groups = [(c, sorted((root / c).iterdir())) for c in classes] or [("image",
                                                                       sorted(root.iterdir()))]
    mode = "L" if channels == 1 else "RGB"
    images, labels = [], []
    for ci, (_name, files) in enumerate(groups):
        for file in files:
            if file.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"):
                continue
            arr = np.asarray(Image.open(file).convert(mode).resize((width, height)),
                             np.float32) / 255.0
            images.append(arr[None] if arr.ndim == 2 else arr.transpose(2, 0, 1))
            labels.append(ci)
    if not images:
        raise ValueError(f"no images under {root}")
    return np.stack(images), np.asarray(labels, np.int64), [c for c, _f in groups]


def load_images(dataset: dict, shape: list) -> tuple:
    channels, height, width = shape
    if dataset["block"] == "data.synthetic_images":
        return synthetic_images(int(dataset["n_images"]), int(dataset["image_size"]),
                                int(dataset["channels"]), dataset["kind"], int(dataset["seed"]))
    if dataset["block"] == "data.torchvision":
        import torch
        import torchvision

        cls = getattr(torchvision.datasets, dataset["dataset"])
        data = cls(str(Path(dataset.get("root") or "~/.aime/datasets").expanduser()),
                   train=True, download=True)
        x = torch.as_tensor(np.asarray(data.data)).float() / 255.0
        x = x[:, None] if x.dim() == 3 else x.permute(0, 3, 1, 2)
        if x.shape[1] != channels:
            x = x.mean(1, keepdim=True).repeat(1, channels, 1, 1)
        x = torch.nn.functional.interpolate(x, size=(height, width), mode="bilinear",
                                            antialias=True)
        return (x.clamp(0, 1).numpy(), np.asarray(data.targets, np.int64),
                list(getattr(data, "classes", [])))
    return read_image_folder(dataset["root"], channels, height, width)


# ---------------------------------------------------------------- text
def encode_bytes(text: str) -> np.ndarray:
    return np.frombuffer(text.encode("utf-8", errors="replace"), np.uint8).astype(np.int64)


def decode_bytes(ids) -> str:
    return bytes(int(i) for i in ids if 0 <= int(i) < 256).decode("utf-8", errors="replace")


_SUBJECTS = ["the cat", "a dog", "the bird", "my friend", "the robot", "a child", "the teacher"]
_VERBS = ["sees", "likes", "finds", "draws", "follows", "builds", "reads"]
_OBJECTS = ["a red ball", "the old map", "a small box", "the blue door", "a green leaf",
            "the tall tree", "a quiet song"]
_PLACES = ["in the park", "at home", "near the river", "after lunch", "every morning"]


def synthetic_text(n_sentences: int, seed: int) -> str:
    """Grammatical toy sentences ("the cat sees a red ball in the park.") with structure a
    small model can learn."""
    rng = np.random.default_rng(seed)
    lines = []
    for _ in range(n_sentences):
        parts = [rng.choice(_SUBJECTS), rng.choice(_VERBS), rng.choice(_OBJECTS)]
        if rng.random() < 0.5:
            parts.append(rng.choice(_PLACES))
        lines.append(" ".join(parts) + ".")
    return "\n".join(lines) + "\n"


def load_text(dataset: dict) -> str:
    if dataset["block"] == "data.synthetic_text":
        return synthetic_text(int(dataset["n_sentences"]), int(dataset["seed"]))
    path = Path(str(dataset["path"])).expanduser()
    files = sorted(path.rglob("*.txt")) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"no .txt files under {path}")
    return "\n".join(f.read_text(encoding="utf-8", errors="replace") for f in files)


_DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def synthetic_pairs(task: str, n: int, max_len: int, seed: int) -> list:
    """(source, target) strings: reverse / sort letters, copy, or spell digits as words."""
    rng = np.random.default_rng(seed)
    letters = "abcdefghij"
    pairs = []
    for _ in range(n):
        if task == "digits":
            k = int(rng.integers(1, max(2, max_len // 5) + 1))
            digits = "".join(str(int(d)) for d in rng.integers(0, 10, k))
            pairs.append((digits, " ".join(_DIGITS[int(d)] for d in digits)))
            continue
        word = "".join(rng.choice(list(letters), int(rng.integers(3, max_len + 1))))
        target = {"reverse": word[::-1], "sort": "".join(sorted(word)), "copy": word}[task]
        pairs.append((word, target))
    return pairs


def load_pairs(dataset: dict) -> list:
    if dataset["block"] == "data.synthetic_pairs":
        return synthetic_pairs(dataset["task"], int(dataset["n_pairs"]),
                               int(dataset["max_length"]), int(dataset["seed"]))
    import csv

    path = Path(str(dataset["path"])).expanduser()
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t" if path.suffix == ".tsv" else ","))
    src, tgt = dataset["source_column"], dataset["target_column"]
    if rows and (src not in rows[0] or tgt not in rows[0]):
        raise ValueError(f"columns '{src}' / '{tgt}' not found; the file has {list(rows[0])}")
    return [(r[src], r[tgt]) for r in rows]


def encode_pair(text: str, length: int, wrap: bool) -> np.ndarray:
    """Bytes padded with PAD to ``length``; wrap adds BOS ... EOS (targets)."""
    ids = list(encode_bytes(text)[: length - (2 if wrap else 0)])
    if wrap:
        ids = [BOS, *ids, EOS]
    return np.asarray(ids + [PAD] * (length - len(ids)), np.int64)


def split_indices(n: int, val_fraction: float, test_fraction: float, seed: int) -> dict:
    order = np.random.default_rng(seed).permutation(n)
    n_test, n_val = int(round(n * test_fraction)), int(round(n * val_fraction))
    return {"test": order[:n_test], "val": order[n_test:n_test + n_val],
            "train": order[n_test + n_val:]}
'''

METRICS_CODE = r'''
# ======================================================================== generative metrics
def _sqrtm_psd(m: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh((m + m.T) / 2)
    return (vecs * np.sqrt(np.clip(vals, 0, None))) @ vecs.T


def frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    """FID between two feature sets [N, D]: |mu_a - mu_b|^2 + Tr(Sa + Sb - 2 (Sa^1/2 Sb Sa^1/2)^1/2)."""
    mu_a, mu_b = a.mean(0), b.mean(0)
    sa, sb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    root = _sqrtm_psd(sa)
    cross = _sqrtm_psd(root @ sb @ root)
    return float(np.sum((mu_a - mu_b) ** 2) + np.trace(sa) + np.trace(sb) - 2 * np.trace(cross))


def kernel_distance(a: np.ndarray, b: np.ndarray) -> float:
    """KID: unbiased MMD^2 with the cubic polynomial kernel (x.y / d + 1)^3."""
    d = a.shape[1]
    kxx, kyy, kxy = ((a @ a.T / d + 1) ** 3, (b @ b.T / d + 1) ** 3, (a @ b.T / d + 1) ** 3)
    m, n = len(a), len(b)
    return float((kxx.sum() - np.trace(kxx)) / (m * (m - 1))
                 + (kyy.sum() - np.trace(kyy)) / (n * (n - 1)) - 2 * kxy.mean())


def pixel_features(images: np.ndarray, side: int = 16) -> np.ndarray:
    """Images [N, C, H, W] in [0, 1] -> area-averaged [N, C * side * side] features (offline)."""
    n, c, h, w = images.shape
    ys = (np.arange(side + 1) * h / side).astype(int)
    xs = (np.arange(side + 1) * w / side).astype(int)
    out = np.zeros((n, c, side, side), np.float32)
    for i in range(side):
        for j in range(side):
            out[:, :, i, j] = images[:, :, ys[i]:max(ys[i + 1], ys[i] + 1),
                                     xs[j]:max(xs[j + 1], xs[j] + 1)].mean((2, 3))
    return out.reshape(n, -1)


def _ngrams(tokens: list, n: int) -> dict:
    out: dict = {}
    for i in range(len(tokens) - n + 1):
        key = tuple(tokens[i:i + n])
        out[key] = out.get(key, 0) + 1
    return out


def bleu(refs: list, hyps: list, max_n: int = 4) -> float:
    """Corpus BLEU-4 (0-100) on whitespace tokens, with add-one smoothing for n > 1."""
    matches, totals = [0] * max_n, [0] * max_n
    ref_len = hyp_len = 0
    for ref, hyp in zip(refs, hyps):
        r, h = ref.split(), hyp.split()
        ref_len, hyp_len = ref_len + len(r), hyp_len + len(h)
        for n in range(1, max_n + 1):
            rc, hc = _ngrams(r, n), _ngrams(h, n)
            matches[n - 1] += sum(min(c, rc.get(g, 0)) for g, c in hc.items())
            totals[n - 1] += max(len(h) - n + 1, 0)
    if hyp_len == 0 or matches[0] == 0:
        return 0.0
    logs = [math.log(matches[0] / totals[0])]
    logs += [math.log((matches[n] + 1) / (totals[n] + 1)) for n in range(1, max_n)]
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    return 100 * bp * math.exp(sum(logs) / max_n)


def chrf(refs: list, hyps: list, n: int = 6, beta: float = 2.0) -> float:
    """chrF (0-100, as sacrebleu): character n-gram precision and recall averaged over the
    orders 1..6 that occur, then the recall-weighted F-score."""
    precisions, recalls = [], []
    for k in range(1, n + 1):
        match = ref_total = hyp_total = 0
        for ref, hyp in zip(refs, hyps):
            rc, hc = _ngrams(list(ref.replace(" ", "")), k), _ngrams(list(hyp.replace(" ", "")), k)
            match += sum(min(c, rc.get(g, 0)) for g, c in hc.items())
            ref_total += sum(rc.values())
            hyp_total += sum(hc.values())
        if ref_total == 0 and hyp_total == 0:
            continue   # no n-grams of this order on either side: not counted
        precisions.append(match / hyp_total if hyp_total else 0.0)
        recalls.append(match / ref_total if ref_total else 0.0)
    if not precisions:
        return 0.0
    p, r = float(np.mean(precisions)), float(np.mean(recalls))
    return 0.0 if p + r == 0 else 100 * (1 + beta ** 2) * p * r / (beta ** 2 * p + r)


def rouge_l(refs: list, hyps: list) -> float:
    """ROUGE-L F1 (0-100) from the longest common subsequence of words."""
    total = 0.0
    for ref, hyp in zip(refs, hyps):
        r, h = ref.split(), hyp.split()
        if not r or not h:
            continue
        prev = [0] * (len(h) + 1)
        for a in r:
            cur = [0]
            for j, b in enumerate(h, 1):
                cur.append(prev[j - 1] + 1 if a == b else max(prev[j], cur[j - 1]))
            prev = cur
        lcs = prev[-1]
        if lcs:
            p, rec = lcs / len(h), lcs / len(r)
            total += 2 * p * rec / (p + rec)
    return 100 * total / max(len(refs), 1)
'''

PLOT_CODE = r'''
# ======================================================================== plots
def image_grid(images: np.ndarray, columns: int = 8, scale: int = 0) -> "Image.Image":
    """Images [N, C, H, W] in [0, 1] -> one PNG-ready grid (upscaled when small)."""
    n, c, h, w = images.shape
    columns = min(columns, n)
    rows = (n + columns - 1) // columns
    grid = np.ones((rows * (h + 2) + 2, columns * (w + 2) + 2, 3), np.float32) * 0.12
    for i in range(n):
        r, col = divmod(i, columns)
        tile = images[i].transpose(1, 2, 0)
        tile = np.repeat(tile, 3, axis=2) if c == 1 else tile[..., :3]
        grid[2 + r * (h + 2):2 + r * (h + 2) + h, 2 + col * (w + 2):2 + col * (w + 2) + w] = tile
    img = Image.fromarray((np.clip(grid, 0, 1) * 255).astype(np.uint8))
    scale = scale or max(1, 256 // max(h, w))
    return img.resize((img.width * scale, img.height * scale), Image.NEAREST)
'''


def namespace() -> dict:
    import json
    import math
    from pathlib import Path

    import numpy as np

    try:   # plots only; data and metrics work without pillow
        from PIL import Image, ImageDraw
    except ImportError:
        Image = ImageDraw = None
    ns: dict = {"np": np, "json": json, "math": math, "Path": Path, "Image": Image,
                "ImageDraw": ImageDraw}
    for code in (DATA_CODE, METRICS_CODE, PLOT_CODE):
        exec(compile(code, "<generative-runtime>", "exec"), ns)  # noqa: S102 — our own source
    return ns
