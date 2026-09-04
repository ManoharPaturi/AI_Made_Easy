"""Dataset health checks for Image Folder datasets (the 🩺 meter).

Pure pathlib + hashlib — no Qt, no torch. Findings feed two places:
the DataPreviewDialog meter (bars + nudges) and the Summary Checks list
(as warnings), so imbalance surfaces where kids already look.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}

# below this a class is too thin to learn from (rule of thumb for kids)
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
    hint: str = ""  # the kid-worded nudge
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


def _images_in(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def _hash_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_image_folder(root: Path) -> HealthReport:
    """Walk one class-per-subfolder image dataset and judge its health."""
    root = Path(root)
    report = HealthReport(root=str(root))
    if not root.exists():
        report.findings.append(Finding(
            "info", f"folder {root} doesn't exist yet",
            "create it, then add one subfolder per thing you want the "
            "model to recognise"))
        return report

    hashes: dict[str, list[str]] = {}
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        files = _images_in(sub)
        report.classes.append(ClassCount(sub.name, len(files)))
        report.total += len(files)
        for f in files:
            try:
                hashes.setdefault(_hash_of(f), []).append(f.name)
            except OSError:
                pass

    if not report.classes:
        report.findings.append(Finding(
            "warning", f"no class subfolders inside {root}",
            "make one subfolder per class — e.g. cats/ and dogs/ — and "
            "put photos inside"))
        return report

    for cc in report.classes:
        if cc.count == 0:
            report.findings.append(Finding(
                "warning", f"class '{cc.name}' is empty",
                "add photos to it or delete the folder", classes=[cc.name]))
        elif cc.count < TOO_FEW:
            report.findings.append(Finding(
                "warning",
                f"class '{cc.name}' has only {cc.count} photo(s) — too few",
                "aim for at least 10–20 photos per class, from different "
                "places and angles", classes=[cc.name]))

    counts = [c.count for c in report.classes if c.count > 0]
    if len(counts) >= 2:
        ratio = max(counts) / max(min(counts), 1)
        if ratio >= IMBALANCE_RATIO:
            big = max(report.classes, key=lambda c: c.count)
            small = min((c for c in report.classes if c.count > 0),
                        key=lambda c: c.count)
            report.findings.append(Finding(
                "warning",
                f"unbalanced: '{big.name}' has {big.count} photos but "
                f"'{small.name}' only {small.count}",
                "models over-learn classes with more photos — add more "
                f"examples of '{small.name}'",
                classes=[big.name, small.name]))

    dups = {h: names for h, names in hashes.items() if len(names) > 1}
    if dups:
        n_groups = len(dups)
        example = sorted(next(iter(dups.values())))[0]
        report.findings.append(Finding(
            "warning",
            f"{n_groups} exact duplicate photo(s) (e.g. {example})",
            "identical copies make the model memorise instead of learn — "
            "delete the copies"))

    if not report.findings:
        report.findings.append(Finding(
            "info", f"{report.total} photos across {len(report.classes)} "
            f"class(es) — looks healthy ✅"))
    return report
