"""Dataset profiles: what a dataset block will feed the pipeline, and what is wrong with it.

``profile_dataset(type_id, params)`` reads the dataset the way the generated
training script will (same readers, same label encoding, same folder order) and
returns a :class:`DataProfile`: size, per-column statistics for tables, class
counts, image / text / audio statistics and *findings* — missing values,
constant and ID-like columns, target leakage, duplicates, class imbalance,
high-cardinality categories. Pure Python + pandas / numpy / PIL; no Qt.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ai_made_easy.core.data.health import (
    AUDIO_SUFFIXES,
    IMAGE_SUFFIXES,
    IMBALANCE_RATIO,
    TEXT_SUFFIXES,
    ClassCount,
    Finding,
    scan_class_folder,
)

BENCHMARKS = {
    "mnist": "Handwritten digits 0–9 · 60,000 train / 10,000 test · [1, 28, 28]",
    "fashion_mnist": "Zalando clothing (10 classes) · 60,000 / 10,000 · [1, 28, 28]",
    "kmnist": "Kuzushiji characters (10 classes) · 60,000 / 10,000 · [1, 28, 28]",
    "cifar10": "Natural images (10 classes) · 50,000 / 10,000 · [3, 32, 32]",
    "cifar100": "Natural images (100 classes) · 50,000 / 10,000 · [3, 32, 32]",
    "svhn": "Street-view house numbers · 73,257 / 26,032 · [3, 32, 32]",
    "stl10": "Natural images (10 classes) · 5,000 / 8,000 · [3, 96, 96]",
    "usps": "Handwritten digits · 7,291 / 2,007 · [1, 16, 16]",
    "iris": "Iris flowers · 150 samples · 4 features · 3 classes",
    "wine": "Wine cultivars · 178 samples · 13 features · 3 classes",
    "breast_cancer": "Breast cancer diagnosis · 569 samples · 30 features · 2 classes",
    "digits": "8×8 digits · 1,797 samples · 64 features · 10 classes",
    "diabetes": "Diabetes progression · 442 samples · 10 features · regression",
    "california_housing": "California housing · 20,640 samples · 8 features · regression",
}
TABLE_BLOCKS = ("data.csv", "data.text_csv", "data.timeseries_csv")
FOLDER_SUFFIXES = {"data.image_folder": IMAGE_SUFFIXES, "data.text_folder": TEXT_SUFFIXES,
                   "data.audio_folder": AUDIO_SUFFIXES}
# rows read for profiling; bigger tables are profiled on the first MAX_ROWS rows
MAX_ROWS = 200_000
# categorical columns with more distinct values are cut by one-hot encoding
MAX_CATEGORIES = 50
# image files opened to measure sizes / channels
IMAGE_SAMPLE = 400


@dataclass
class ColumnProfile:
    name: str
    kind: str          # numeric | categorical | text | boolean | datetime | empty
    dtype: str
    count: int         # non-missing values
    missing: int
    unique: int
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    median: float | None = None
    max: float | None = None
    skew: float | None = None
    top: list[tuple[str, int]] = field(default_factory=list)
    histogram: list[int] = field(default_factory=list)
    bin_edges: list[float] = field(default_factory=list)
    role: str = "feature"  # feature | target | text | label | ignored

    @property
    def missing_fraction(self) -> float:
        total = self.count + self.missing
        return self.missing / total if total else 0.0


@dataclass
class DataProfile:
    type_id: str
    kind: str                     # table | folder | arrays | records | benchmark | remote | synthetic
    summary: str = ""
    source: str = ""              # resolved path / dataset name
    format: str = ""              # table file format (csv, tsv, parquet, ...)
    rows: int = 0
    truncated: bool = False       # only the first MAX_ROWS rows were read
    columns: list[ColumnProfile] = field(default_factory=list)
    classes: list[ClassCount] = field(default_factory=list)
    task: str = ""                # classification | regression | "" (unknown)
    findings: list[Finding] = field(default_factory=list)
    details: list[str] = field(default_factory=list)
    samples: dict[str, list[str]] = field(default_factory=dict)  # class -> files (folders)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity in ("warning", "error")]

    def column(self, name: str) -> ColumnProfile | None:
        return next((c for c in self.columns if c.name == name), None)

    def text(self) -> str:
        """Plain-text report (CLI / tooltips)."""
        lines = [self.summary] if self.summary else []
        if self.error:
            lines.append(self.error)
        lines += self.details
        if self.classes:
            lines.append("classes: " + ", ".join(f"{c.name} {c.count:,}" for c in self.classes[:20])
                         + (" …" if len(self.classes) > 20 else ""))
        if self.columns:
            lines.append("")
            for c in self.columns:
                stats = (f"mean {c.mean:.4g} · std {c.std:.4g} · [{c.min:.4g}, {c.max:.4g}]"
                         if c.mean is not None else
                         ", ".join(f"{v} ({n})" for v, n in c.top[:3]))
                role = f" [{c.role}]" if c.role != "feature" else ""
                lines.append(f"  {c.name}{role}: {c.kind}, {c.missing_fraction:.0%} missing, "
                             f"{c.unique:,} unique — {stats}")
        if self.findings:
            lines.append("")
            lines += [f"{f.severity}: {f.message}" + (f" — {f.hint}" if f.hint else "")
                      for f in self.findings]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["ok"] = self.ok
        return json.loads(json.dumps(data, default=str))


# ===================================================================== paths

def resolve_path(raw: str, base: str | Path | None = None) -> Path:
    """Expand ``~`` and resolve relative paths against ``base`` (then the cwd)."""
    p = Path(str(raw)).expanduser()
    if p.is_absolute():
        return p
    for root in (base, Path.cwd()):
        if root is not None and (Path(root) / p).exists():
            return Path(root) / p
    return Path(base) / p if base is not None else p


def read_table(path: str | Path, fmt: str = "csv", nrows: int | None = None):
    """Read a table with the same readers as the generated scripts."""
    import pandas as pd

    path = Path(path)
    if fmt == "csv":
        return pd.read_csv(path, nrows=nrows)
    if fmt == "tsv":
        return pd.read_csv(path, sep="\t", nrows=nrows)
    if fmt == "parquet":
        frame = pd.read_parquet(path)
    elif fmt == "excel":
        frame = pd.read_excel(path, nrows=nrows)
    elif fmt == "json":
        lines = str(path).endswith(".jsonl")
        frame = pd.read_json(path, lines=lines, nrows=nrows) if lines and nrows else \
            pd.read_json(path, lines=lines)
    else:
        raise ValueError(f"unknown table format {fmt!r}")
    return frame.head(nrows) if nrows else frame


def guess_format(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    return {".tsv": "tsv", ".parquet": "parquet", ".pq": "parquet", ".xlsx": "excel",
            ".xls": "excel", ".json": "json", ".jsonl": "json"}.get(suffix, "csv")


def _csv_list(value) -> list[str]:
    return [v.strip() for v in str(value or "").split(",") if v.strip()]


# ===================================================================== columns

def _finite(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def profile_column(series, role: str = "feature") -> ColumnProfile:
    """Statistics for one pandas Series."""
    import numpy as np
    import pandas as pd

    missing = int(series.isna().sum())
    present = series.dropna()
    try:
        unique = int(present.nunique())
    except TypeError:  # unhashable cells (lists / dicts in JSON tables)
        present = present.astype(str)
        unique = int(present.nunique())
    if pd.api.types.is_bool_dtype(series):
        kind = "boolean"
    elif pd.api.types.is_numeric_dtype(series):
        kind = "numeric"
    elif pd.api.types.is_datetime64_any_dtype(series):
        kind = "datetime"
    elif len(present) and present.astype(str).str.len().mean() > 30 and unique > 0.5 * len(present):
        kind = "text"
    else:
        kind = "categorical"
    if not len(present):
        kind = "empty"
    col = ColumnProfile(name=str(series.name), kind=kind, dtype=str(series.dtype),
                        count=int(len(present)), missing=missing, unique=unique, role=role)
    if kind == "numeric":
        values = present.to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        if len(values):
            col.mean, col.std = _finite(values.mean()), _finite(values.std())
            col.min, col.max = _finite(values.min()), _finite(values.max())
            col.median = _finite(np.median(values))
            if len(values) > 2 and col.std:
                col.skew = _finite(((values - values.mean()) ** 3).mean() / values.std() ** 3)
            bins = min(20, max(1, unique))
            counts, edges = np.histogram(values, bins=bins)
            col.histogram, col.bin_edges = counts.tolist(), [float(e) for e in edges]
    if kind != "numeric" or unique <= 20:
        top = present.astype(str).value_counts().head(10)
        col.top = [(str(k), int(v)) for k, v in top.items()]
    return col


def _class_counts(values) -> list[ClassCount]:
    counts = values.astype(str).value_counts()
    return [ClassCount(str(k), int(v)) for k, v in sorted(counts.items())]


def _imbalance(classes: list[ClassCount]) -> Finding | None:
    counts = [c for c in classes if c.count > 0]
    if len(counts) < 2:
        return None
    big, small = max(counts, key=lambda c: c.count), min(counts, key=lambda c: c.count)
    ratio = big.count / max(small.count, 1)
    if ratio < IMBALANCE_RATIO:
        return None
    return Finding("warning", f"class imbalance {ratio:.1f}:1 ('{big.name}' {big.count:,} vs "
                              f"'{small.name}' {small.count:,})",
                   "add Class Balancing, or collect more samples of the minority classes",
                   classes=[big.name, small.name])


def table_findings(frame, target: str | None, columns: list[ColumnProfile]) -> list[Finding]:
    """Column-level problems: missing, constant, ID-like, leakage, cardinality, duplicates."""
    import numpy as np
    import pandas as pd

    rows = len(frame)
    out: list[Finding] = []
    features = [c for c in columns if c.role == "feature"]
    for c in features:
        if c.kind == "empty":
            out.append(Finding("warning", f"column '{c.name}' is entirely empty",
                               "drop it with Drop Columns", classes=[c.name]))
            continue
        if c.unique <= 1:
            out.append(Finding("warning", f"column '{c.name}' is constant — it carries no "
                                          "information", "drop it with Drop Columns",
                               classes=[c.name]))
        if c.missing_fraction >= 0.5:
            out.append(Finding("warning", f"column '{c.name}' is {c.missing_fraction:.0%} "
                                          "missing", "drop it, or impute the missing values",
                               classes=[c.name]))
        elif c.missing:
            out.append(Finding("info", f"column '{c.name}' has {c.missing:,} missing value(s)",
                               "Impute Missing Values fills them", classes=[c.name]))
        id_like = False
        if rows >= 20 and c.unique == c.count == rows and c.kind in ("numeric", "categorical"):
            series = frame[c.name]
            monotonic = c.kind == "numeric" and (series.is_monotonic_increasing
                                                 or series.is_monotonic_decreasing)
            id_like = c.kind == "categorical" or monotonic or "id" in c.name.lower()
            if id_like:
                out.append(Finding("warning", f"column '{c.name}' looks like an identifier "
                                              "(a different value in every row)",
                                   "drop it — models memorise IDs instead of learning",
                                   classes=[c.name]))
        if c.kind in ("categorical", "text") and c.unique > MAX_CATEGORIES and not id_like:
            out.append(Finding("warning", f"column '{c.name}' has {c.unique:,} categories; "
                                          f"one-hot encoding keeps only the first "
                                          f"{MAX_CATEGORIES}",
                               "use Ordinal Encode, or drop the column", classes=[c.name]))
        if c.kind == "numeric" and c.skew is not None and abs(c.skew) > 3 and (c.min or 0) >= 0:
            out.append(Finding("info", f"column '{c.name}' is strongly skewed "
                                       f"(skew {c.skew:.1f})", "Log Transform may help",
                               classes=[c.name]))
    if target and target in frame.columns:
        y = frame[target]
        for c in features:
            if c.kind == "empty" or c.unique <= 1:
                continue
            x = frame[c.name]
            leak = False
            if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
                both = pd.concat([x, y], axis=1).dropna()
                if len(both) > 10 and both.iloc[:, 0].std() > 0 and both.iloc[:, 1].std() > 0:
                    corr = float(np.corrcoef(both.iloc[:, 0], both.iloc[:, 1])[0, 1])
                    leak = abs(corr) > 0.995
            if not leak and c.unique < rows * 0.5:
                pair = pd.DataFrame({"x": x.astype(str), "y": y.astype(str)})
                leak = (pair.groupby("x")["y"].nunique().max() == 1
                        and pair.groupby("y")["x"].nunique().max() == 1)
            if leak:
                out.append(Finding("warning", f"column '{c.name}' predicts '{target}' perfectly "
                                              "— likely target leakage",
                                   "drop it unless it is really available at prediction time",
                                   classes=[c.name]))
        if y.isna().any():
            out.append(Finding("warning", f"target '{target}' has {int(y.isna().sum()):,} "
                                          "missing value(s)",
                               "remove those rows from the file before training"))
    try:
        dups = int(frame.duplicated().sum())
    except TypeError:
        dups = 0
    if dups:
        out.append(Finding("warning" if dups / max(rows, 1) > 0.01 else "info",
                           f"{dups:,} duplicate row(s) ({dups / max(rows, 1):.1%})",
                           "duplicates can land in both training and test splits and "
                           "inflate the scores"))
    return out


# ===================================================================== tables

def profile_frame(frame, *, target: str | None = None, features: list[str] | None = None,
                  task: str = "", extra_roles: dict[str, str] | None = None) -> DataProfile:
    """Profile an in-memory DataFrame (target / feature roles as the training script uses)."""
    import pandas as pd

    roles = dict(extra_roles or {})
    if target:
        roles[target] = "target"
    if features:
        for name in frame.columns:
            roles.setdefault(str(name), "feature" if str(name) in features else "ignored")
    profile = DataProfile(type_id="", kind="table", rows=len(frame))
    profile.columns = [profile_column(frame[c], roles.get(str(c), "feature"))
                       for c in frame.columns]
    profile.summary = (f"{len(frame):,} rows × {len(frame.columns)} columns · "
                       f"{int(frame.isna().sum().sum()):,} missing values")
    if target and target in frame.columns:
        y = frame[target]
        if not task:
            numeric = pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y)
            task = "regression" if numeric and y.nunique() > 20 else "classification"
        profile.task = task
        if task == "classification":
            profile.classes = _class_counts(y.dropna())
            imb = _imbalance(profile.classes)
            if imb:
                profile.findings.append(imb)
            small = [c for c in profile.classes if c.count < 2]
            if small:
                profile.findings.append(Finding(
                    "warning", f"{len(small)} class(es) have a single sample "
                               f"(e.g. '{small[0].name}')",
                    "a class needs samples in every split", classes=[c.name for c in small]))
    elif target:
        profile.findings.append(Finding(
            "error", f"target column '{target}' is not in the table",
            "columns: " + ", ".join(map(str, list(frame.columns)[:20]))))
    missing = [f for f in (features or []) if f not in frame.columns]
    if missing:
        profile.findings.append(Finding("error", "feature column(s) not in the table: "
                                        + ", ".join(missing)))
    profile.findings += table_findings(frame, target if target in frame.columns else None,
                                       profile.columns)
    return profile


def _load_frame(path: Path, fmt: str):
    frame = read_table(path, fmt, nrows=MAX_ROWS + 1)
    truncated = len(frame) > MAX_ROWS
    return frame.head(MAX_ROWS), truncated


def _profile_table(type_id: str, params: dict, base) -> DataProfile:
    path = resolve_path(params.get("path", ""), base)
    fmt = str(params.get("format") or guess_format(path))
    if not path.exists():
        return DataProfile(type_id, "table", source=str(path),
                           error=f"file not found: {path}",
                           findings=[Finding("error", f"data file {path} was not found",
                                             "check the path in the dataset block")])
    try:
        frame, truncated = _load_frame(path, fmt)
    except Exception as exc:  # noqa: BLE001 — reported to the user
        return DataProfile(type_id, "table", source=str(path),
                           error=f"could not read {path.name}: {exc}",
                           findings=[Finding("error", f"could not read {path.name}: {exc}")])
    if type_id == "data.csv":
        profile = profile_frame(frame, target=str(params.get("target_column") or "") or None,
                                features=_csv_list(params.get("feature_columns")) or None,
                                task=str(params.get("_task") or ""))
    elif type_id == "data.text_csv":
        text_col = str(params.get("text_column") or "text")
        label = str(params.get("label_column") or "label")
        roles = {c: "ignored" for c in map(str, frame.columns) if c not in (text_col, label)}
        roles[text_col] = "text"
        profile = profile_frame(frame, target=label, extra_roles=roles,
                                task="classification")
        if text_col in frame.columns:
            texts = frame[text_col].fillna("").astype(str)
            words = texts.str.split().str.len()
            profile.details.append(
                f"text length (words): median {int(words.median())}, 95th percentile "
                f"{int(words.quantile(0.95))}, max {int(words.max())}")
            empty = int((texts.str.strip() == "").sum())
            if empty:
                profile.findings.append(Finding("warning", f"{empty:,} empty text(s)",
                                                "remove them from the file"))
            dup_texts = int(texts.duplicated().sum())
            if dup_texts:
                profile.findings.append(Finding(
                    "warning" if dup_texts / max(len(texts), 1) > 0.01 else "info",
                    f"{dup_texts:,} duplicate text(s)", "duplicates inflate test scores"))
        else:
            profile.findings.append(Finding("error", f"text column '{text_col}' is not in "
                                                     "the table"))
    else:  # time series
        targets = _csv_list(params.get("target_columns"))
        feats = _csv_list(params.get("feature_columns"))
        roles = {t: "target" for t in targets}
        profile = profile_frame(frame, extra_roles=roles, features=None)
        profile.task = "regression"
        missing = [t for t in targets + feats if t not in frame.columns]
        if missing:
            profile.findings.append(Finding("error", "column(s) not in the table: "
                                            + ", ".join(missing)))
        window, horizon, stride = (int(params.get(k) or d) for k, d in
                                   (("window", 32), ("horizon", 1), ("stride", 1)))
        windows = max(0, (len(frame) - window - horizon) // max(stride, 1) + 1)
        profile.details.append(f"{windows:,} training windows of {window} steps "
                               f"(horizon {horizon}, stride {stride})")
        if windows < 50:
            profile.findings.append(Finding(
                "warning", f"only {windows} window(s) fit in {len(frame):,} rows",
                "use a shorter window / stride, or more data"))
    profile.type_id, profile.source, profile.truncated = type_id, str(path), truncated
    profile.format = fmt
    if truncated:
        profile.details.insert(0, f"profiled on the first {MAX_ROWS:,} rows")
    return profile


# ===================================================================== folders

def _image_stats(files: list[Path]) -> tuple[list[str], list[Finding]]:
    try:
        from PIL import Image
    except ImportError:
        return [], []
    sizes: dict[tuple[int, int], int] = {}
    modes: dict[str, int] = {}
    broken: list[str] = []
    for f in files:
        try:
            with Image.open(f) as im:
                sizes[im.size] = sizes.get(im.size, 0) + 1
                modes[im.mode] = modes.get(im.mode, 0) + 1
        except Exception:  # noqa: BLE001 — unreadable image
            broken.append(f.name)
    details, findings = [], []
    if sizes:
        common = sorted(sizes.items(), key=lambda kv: -kv[1])
        widths = [w for (w, _h) in sizes for _ in range(sizes[(w, _h)])]
        heights = [h for (_w, h) in sizes for _ in range(sizes[(_w, h)])]
        details.append(f"image sizes ({len(files)} sampled): {len(sizes)} distinct, most common "
                       + ", ".join(f"{w}×{h} ({n})" for (w, h), n in common[:3])
                       + f"; width {min(widths)}–{max(widths)}, height {min(heights)}–"
                         f"{max(heights)}")
        details.append("colour modes: " + ", ".join(f"{m} ({n})" for m, n in
                                                    sorted(modes.items(), key=lambda kv: -kv[1])))
        if min(widths) < 16 or min(heights) < 16:
            findings.append(Finding("info", f"some images are tiny ({min(widths)}×"
                                            f"{min(heights)})", "check they are not thumbnails"))
    if broken:
        findings.append(Finding("warning", f"{len(broken)} image(s) cannot be opened "
                                           f"(e.g. {broken[0]})", "remove or re-save them"))
    return details, findings


def _text_stats(files: list[Path]) -> tuple[list[str], list[Finding]]:
    lengths, empty = [], 0
    for f in files:
        try:
            words = len(f.read_text(errors="replace").split())
        except OSError:
            continue
        lengths.append(words)
        empty += words == 0
    if not lengths:
        return [], []
    lengths.sort()
    details = [f"text length (words, {len(lengths)} sampled): median "
               f"{lengths[len(lengths) // 2]}, max {lengths[-1]}"]
    findings = [Finding("warning", f"{empty} empty text file(s)", "remove them")] if empty else []
    return details, findings


def _audio_stats(files: list[Path], params: dict) -> tuple[list[str], list[Finding]]:
    import wave

    rates: dict[int, int] = {}
    durations: list[float] = []
    for f in files:
        try:
            with wave.open(str(f), "rb") as fh:
                rate = fh.getframerate()
                rates[rate] = rates.get(rate, 0) + 1
                durations.append(fh.getnframes() / max(rate, 1))
        except (OSError, EOFError, wave.Error):
            continue
    if not durations:
        return [], []
    durations.sort()
    details = [f"clips ({len(durations)} sampled): {durations[0]:.2f}–{durations[-1]:.2f} s, "
               f"median {durations[len(durations) // 2]:.2f} s; sample rates "
               + ", ".join(f"{r} Hz ({n})" for r, n in sorted(rates.items()))]
    findings = []
    target = float(params.get("duration") or 1.0)
    longer = sum(d > target * 1.5 for d in durations)
    if longer > len(durations) / 4:
        findings.append(Finding("info", f"{longer} of {len(durations)} sampled clips are much "
                                        f"longer than the {target:g} s window and get trimmed",
                                "increase duration if the sound happens later in the clip"))
    return details, findings


def profile_folder(type_id: str, root: str | Path, params: dict | None = None,
                   base=None) -> DataProfile:
    """Class-per-subfolder dataset: counts, health findings, media statistics."""
    params = params or {}
    suffixes = FOLDER_SUFFIXES.get(type_id, IMAGE_SUFFIXES)
    path = resolve_path(str(root), base)
    report = scan_class_folder(path, suffixes)
    profile = DataProfile(type_id, "folder", source=str(path), rows=report.total,
                          classes=report.classes, task="classification",
                          findings=[f for f in report.findings if f.severity == "warning"])
    if not path.exists():
        profile.error = f"folder not found: {path}"
        return profile
    profile.summary = f"{report.total:,} samples in {len(report.classes)} class(es)"
    files: list[Path] = []
    per_class = max(1, IMAGE_SAMPLE // max(len(report.classes), 1))
    for c in report.classes:
        members = sorted(p for p in (path / c.name).iterdir()
                         if p.is_file() and p.suffix.lower() in suffixes)
        profile.samples[c.name] = [str(p) for p in members[:24]]
        files += members[:per_class]
    if type_id == "data.image_folder":
        details, findings = _image_stats(files)
    elif type_id == "data.text_folder":
        details, findings = _text_stats(files)
    else:
        details, findings = _audio_stats(files, params)
    profile.details += details
    profile.findings += findings
    return profile


# ===================================================================== arrays

def _profile_numpy(params: dict, base) -> DataProfile:
    import numpy as np

    path = resolve_path(params.get("path", ""), base)
    profile = DataProfile("data.numpy", "arrays", source=str(path))
    if not path.exists():
        profile.error = f"file not found: {path}"
        profile.findings.append(Finding("error", f"data file {path} was not found"))
        return profile
    x_key, y_key = str(params.get("x_key", "x")), str(params.get("y_key", "y"))
    try:
        with np.load(path, allow_pickle=False) as archive:
            for key in archive.files:
                arr = archive[key]
                profile.details.append(f"{key}: shape {list(arr.shape)}, dtype {arr.dtype}")
            missing = [k for k in (x_key, y_key) if k not in archive.files]
            if missing:
                profile.findings.append(Finding(
                    "error", "array(s) not in the archive: " + ", ".join(missing),
                    "keys: " + ", ".join(archive.files)))
                return profile
            x, y = archive[x_key], archive[y_key]
    except Exception as exc:  # noqa: BLE001
        profile.error = f"could not read {path.name}: {exc}"
        return profile
    profile.rows = int(len(x))
    profile.summary = f"{len(x):,} samples · features {list(x.shape[1:])}"
    if len(x) != len(y):
        profile.findings.append(Finding("error", f"{x_key} has {len(x):,} samples but {y_key} "
                                                 f"has {len(y):,}"))
    if np.issubdtype(x.dtype, np.floating):
        bad = int((~np.isfinite(x)).sum())
        if bad:
            profile.findings.append(Finding("warning", f"{bad:,} NaN / infinite value(s) in "
                                                       f"{x_key}", "clean or impute them"))
        if x.size:
            profile.details.append(f"{x_key} range [{float(np.nanmin(x)):.4g}, "
                                   f"{float(np.nanmax(x)):.4g}], mean "
                                   f"{float(np.nanmean(x)):.4g}")
    if y.ndim == 1 and (not np.issubdtype(y.dtype, np.floating) or len(np.unique(y)) <= 20):
        import pandas as pd

        profile.task = "classification"
        profile.classes = _class_counts(pd.Series(y))
        imb = _imbalance(profile.classes)
        if imb:
            profile.findings.append(imb)
    else:
        profile.task = "regression"
    return profile


def _profile_json(params: dict, base) -> DataProfile:
    path = resolve_path(params.get("path", ""), base)
    profile = DataProfile("data.json", "records", source=str(path))
    if not path.exists():
        profile.error = f"file not found: {path}"
        profile.findings.append(Finding("error", f"data file {path} was not found"))
        return profile
    try:
        text = path.read_text().strip()
        records = (json.loads(text) if text.startswith("[")
                   else [json.loads(line) for line in text.splitlines() if line.strip()])
    except Exception as exc:  # noqa: BLE001
        profile.error = f"could not read {path.name}: {exc}"
        return profile
    x_field, y_field = str(params.get("x_field", "x")), str(params.get("y_field", "y"))
    profile.rows = len(records)
    profile.summary = f"{len(records):,} records"
    lacking = sum(1 for r in records if not isinstance(r, dict)
                  or x_field not in r or y_field not in r)
    if lacking:
        profile.findings.append(Finding("error", f"{lacking:,} record(s) lack '{x_field}' or "
                                                 f"'{y_field}'"))
    lengths = {len(r[x_field]) if isinstance(r.get(x_field), list) else 1
               for r in records if isinstance(r, dict) and x_field in r}
    if len(lengths) > 1:
        profile.findings.append(Finding("error", f"'{x_field}' has different lengths across "
                                                 f"records: {sorted(lengths)[:5]}"))
    profile.details += [json.dumps(r)[:300] for r in records[:5]]
    ys = [r[y_field] for r in records if isinstance(r, dict) and y_field in r]
    if ys and all(isinstance(v, (str, int, bool)) for v in ys):
        import pandas as pd

        profile.task = "classification"
        profile.classes = _class_counts(pd.Series(ys))
    return profile


# ===================================================================== dispatch

def profile_dataset(type_id: str, params: dict, base: str | Path | None = None) -> DataProfile:
    """Profile the data behind a dataset block (``params`` are its resolved params)."""
    if type_id in TABLE_BLOCKS:
        return _profile_table(type_id, params, base)
    if type_id in FOLDER_SUFFIXES:
        return profile_folder(type_id, params.get("root", ""), params, base)
    if type_id == "data.numpy":
        return _profile_numpy(params, base)
    if type_id == "data.json":
        return _profile_json(params, base)
    if type_id in ("data.torchvision", "data.sklearn"):
        name = str(params.get("dataset", ""))
        return DataProfile(type_id, "benchmark", source=name,
                           summary=BENCHMARKS.get(name, name),
                           details=["downloaded automatically on first use"])
    if type_id == "data.huggingface":
        return DataProfile(type_id, "remote", source=str(params.get("repo_id", "")),
                           summary=f"Hugging Face dataset '{params.get('repo_id')}' (split "
                                   f"'{params.get('split')}'), downloaded on first use")
    if type_id == "data.synthetic":
        kind = params.get("kind")
        return DataProfile(type_id, "synthetic", source=str(kind), rows=int(params.get(
            "n_samples") or 0), task="regression" if kind == "regression" else "classification",
            summary=f"{kind} data · {params.get('n_samples')} samples · "
                    f"{params.get('n_features')} features"
                    + (f" · {params.get('n_classes')} classes" if kind != "regression" else ""))
    return DataProfile(type_id, "unknown", summary="\n".join(f"{k}: {v}"
                                                             for k, v in params.items()))


def profile_path(path: str | Path, *, target: str | None = None, fmt: str | None = None,
                 kind: str | None = None) -> DataProfile:
    """Profile a file or folder without a dataset block (CLI, “Open file…”)."""
    p = Path(path).expanduser()
    if p.is_dir():
        sub = [f for d in p.iterdir() if d.is_dir() for f in d.iterdir() if f.is_file()][:200]
        suffixes = {f.suffix.lower() for f in sub}
        type_id = kind or ("data.text_folder" if suffixes & TEXT_SUFFIXES and not
                           suffixes & IMAGE_SUFFIXES else
                           "data.audio_folder" if suffixes & AUDIO_SUFFIXES and not
                           suffixes & IMAGE_SUFFIXES else "data.image_folder")
        return profile_folder(type_id, p)
    if p.suffix.lower() == ".npz":
        return _profile_numpy({"path": str(p)}, None)
    params = {"path": str(p), "format": fmt or guess_format(p), "target_column": target or ""}
    return _profile_table(kind or "data.csv", params, None)


def describe(type_id: str, params: dict, base=None) -> str:
    """Short text description of a dataset block (tooltips / previews)."""
    return profile_dataset(type_id, params, base).text()
