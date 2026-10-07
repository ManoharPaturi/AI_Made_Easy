"""Dataset health checks for class-per-folder datasets (images, text, audio).

Pure pathlib + hashlib — no Qt, no torch. Findings feed the data preview
and the Problems panel (as warnings): empty / small classes, class imbalance,
exact duplicates, and background shortcuts in image datasets.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
TEXT_SUFFIXES = {".txt"}
AUDIO_SUFFIXES = {".wav"}

# classes with fewer samples than this rarely generalise
TOO_FEW = 10
# biggest/smallest class count ratio that triggers an imbalance warning
IMBALANCE_RATIO = 3.0


@dataclass
class ClassCount:
    name: str
    count: int


@dataclass
class Finding:
    severity: str  # "info" | "warning"
    message: str
    hint: str = ""  # suggested remedy
    classes: list[str] = field(default_factory=list)


@dataclass
class HealthReport:
    root: str
    classes: list[ClassCount] = field(default_factory=list)
    total: int = 0
    findings: list[Finding] = field(default_factory=list)

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warning"]


def _files_in(folder: Path, suffixes: set[str]) -> list[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in suffixes)


def _mean_rgb(path: Path):
    try:
        from PIL import Image

        with Image.open(path) as im:
            small = im.convert("RGB").resize((8, 8))
            data = list(small.getdata())
        n = len(data)
        return tuple(round(sum(c[i] for c in data) / n) for i in range(3))
    except Exception:
        return None


def background_shortcut(classes_means: dict[str, list[tuple]]) -> Finding | None:
    """Detect the classic confound: every class lives on its own background.

    classes_means maps class name → list of mean-RGB per image. If each
    class splits cleanly into ONE bright/dark bucket (luminance) and the
    buckets differ between classes, the background may be doing the
    recognising. Pure math — no fs.
    """
    if len(classes_means) < 2:
        return None
    bucket_share: dict[str, tuple[int, int]] = {}  # class -> (dark_n, light_n)
    class_bucket: dict[str, int] = {}
    for name, means in classes_means.items():
        means = [m for m in means if m]
        if len(means) < 6:
            return None
        dark = sum(1 for m in means if sum(m) / 3 < 128)
        light = len(means) - dark
        bucket_share[name] = (dark, light)
        class_bucket[name] = 0 if dark >= light else 1
    names = list(classes_means)
    for a, b in zip(names, names[1:], strict=False):
        if class_bucket[a] == class_bucket[b]:
            return None  # same dominant bucket — no class/background pairing
    shares = [max(*bucket_share[n]) / sum(bucket_share[n]) for n in names]
    if min(shares) < 0.9:
        return None
    return Finding(
        "warning",
        "possible background shortcut: each class has a distinct overall brightness ("
        + " vs ".join(names) + "); the model may learn the background instead of the "
        "object",
        "collect samples with varied backgrounds for every class")


def _hash_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_image_folder(root: Path) -> HealthReport:
    return scan_class_folder(root, IMAGE_SUFFIXES)


def scan_class_folder(root: Path, suffixes: set[str] = IMAGE_SUFFIXES) -> HealthReport:
    """Walk a class-per-subfolder dataset and report its health."""
    root = Path(root)
    report = HealthReport(root=str(root))
    images = suffixes == IMAGE_SUFFIXES
    if not root.exists():
        report.findings.append(Finding(
            "warning", f"dataset folder {root} does not exist",
            "create it with one subfolder per class"))
        return report

    hashes: dict[str, list[str]] = {}
    means: dict[str, list[tuple]] = {}
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        files = _files_in(sub, suffixes)
        report.classes.append(ClassCount(sub.name, len(files)))
        report.total += len(files)
        for f in files:
            try:
                hashes.setdefault(_hash_of(f), []).append(f.name)
            except OSError:
                pass
        if images and len(files) >= 6 and len(report.classes) <= 4:
            means[sub.name] = [m for m in (_mean_rgb(f) for f in files[:60])
                               if m is not None]
    shortcut = background_shortcut(means)
    if shortcut is not None:
        report.findings.append(shortcut)

    if not report.classes:
        report.findings.append(Finding(
            "warning", f"no class subfolders inside {root}",
            "create one subfolder per class"))
        return report

    for cc in report.classes:
        if cc.count == 0:
            report.findings.append(Finding(
                "warning", f"class '{cc.name}' contains no samples",
                "add samples or remove the folder", classes=[cc.name]))
        elif cc.count < TOO_FEW:
            report.findings.append(Finding(
                "warning",
                f"class '{cc.name}' has only {cc.count} sample(s)",
                f"collect at least {TOO_FEW}–20 varied samples per class",
                classes=[cc.name]))

    counts = [c.count for c in report.classes if c.count > 0]
    if len(counts) >= 2:
        ratio = max(counts) / max(min(counts), 1)
        if ratio >= IMBALANCE_RATIO:
            big = max(report.classes, key=lambda c: c.count)
            small = min((c for c in report.classes if c.count > 0),
                        key=lambda c: c.count)
            report.findings.append(Finding(
                "warning",
                f"class imbalance {ratio:.1f}:1 ('{big.name}' {big.count} vs "
                f"'{small.name}' {small.count})",
                "add samples to minority classes or use Class Balancing",
                classes=[big.name, small.name]))

    dups = {h: names for h, names in hashes.items() if len(names) > 1}
    if dups:
        n_groups = len(dups)
        example = sorted(next(iter(dups.values())))[0]
        report.findings.append(Finding(
            "warning",
            f"{n_groups} group(s) of exact duplicate files (e.g. {example})",
            "remove duplicates so they cannot leak across train/test splits"))

    if not report.findings:
        report.findings.append(Finding(
            "info", f"{report.total} samples across {len(report.classes)} class(es); "
            "no issues found"))
    return report
