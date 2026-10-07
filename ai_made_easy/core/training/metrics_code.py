"""Emit ``compute_metrics(outputs, targets) -> dict`` for the selected metrics.

Pure numpy (no scikit-learn dependency) so the same code serves PyTorch and
Keras scripts. Binary tasks report positive-class precision/recall/F1.
"""
from __future__ import annotations

_HELPERS = {
    "confusion": '''
def confusion_matrix(y: np.ndarray, pred: np.ndarray, k: int) -> np.ndarray:
    return np.bincount(y * k + pred, minlength=k * k).reshape(k, k)
''',
    "prf": '''
def precision_recall_f1(cm: np.ndarray, average: str, binary: bool = False):
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(axis=0) - tp, cm.sum(axis=1) - tp
    prec = tp / np.maximum(tp + fp, 1e-12)
    rec = tp / np.maximum(tp + fn, 1e-12)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    if binary:
        return float(prec[1]), float(rec[1]), float(f1[1])
    if average == "micro":
        value = float(tp.sum() / max(cm.sum(), 1))
        return value, value, value
    if average == "weighted":
        w = cm.sum(axis=1) / max(cm.sum(), 1)
        return float((prec * w).sum()), float((rec * w).sum()), float((f1 * w).sum())
    return float(prec.mean()), float(rec.mean()), float(f1.mean())
''',
    "auc": '''
def binary_auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Rank-based ROC-AUC (Mann-Whitney U); NaN when one class is absent."""
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return float("nan")
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
''',
    "ap": '''
def average_precision(scores: np.ndarray, positive: np.ndarray) -> float:
    if not positive.any():
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    hits = positive[order]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(precision[hits].mean())
''',
}


def _needs(keys: set[str]) -> list[str]:
    helpers = []
    if keys & {"precision", "recall", "f1", "mcc", "kappa", "balanced_accuracy", "confusion"}:
        helpers.append("confusion")
    if keys & {"precision", "recall", "f1"}:
        helpers.append("prf")
    if "roc_auc" in keys:
        helpers.append("auc")
    if "avg_precision" in keys:
        helpers.append("ap")
    return helpers


def metrics_code(task: str, metrics: list[tuple[str, dict, str]]) -> str:
    """metrics: (type_id, params, key) triples, already filtered for the task."""
    keys = {k for _, _, k in metrics}
    params = {k: p for _, p, k in metrics}
    out = [_HELPERS[h] for h in _needs(keys)]
    body = ["", "", "def compute_metrics(outputs: np.ndarray, targets: np.ndarray) -> dict:",
            '    """Validation / test metrics from raw model outputs."""', "    m = {}"]
    add = body.append
    if task == "multiclass":
        add("    scores = to_scores(outputs)")
        add("    y = (targets.argmax(axis=1) if targets.ndim > 1 else targets).astype(int)")
        add("    pred = scores.argmax(axis=1)")
        add("    k = scores.shape[1]")
        _classification(add, keys, params, binary=False)
    elif task == "binary":
        add("    p = to_scores(outputs)[:, 0]")
        add("    y = targets.reshape(-1).astype(int)")
        add("    pred = (p >= 0.5).astype(int)")
        add("    scores = np.stack([1 - p, p], axis=1)")
        add("    k = 2")
        _classification(add, keys, params, binary=True)
    elif task == "multilabel":
        add("    scores = to_scores(outputs)")
        add("    y = targets.reshape(scores.shape).astype(int)")
        add("    pred = (scores >= 0.5).astype(int)")
        if "accuracy" in keys:
            add("    m[\"accuracy\"] = float((pred == y).all(axis=1).mean())")
        if "roc_auc" in keys:
            add("    aucs = [binary_auc(scores[:, j], y[:, j] == 1) for j in range(y.shape[1])]")
            add("    m[\"roc_auc\"] = float(np.nanmean(aucs)) if not np.all(np.isnan(aucs)) "
                "else float(\"nan\")")
        if "avg_precision" in keys:
            add("    aps = [average_precision(scores[:, j], y[:, j] == 1) for j in range(y.shape[1])]")
            add("    m[\"avg_precision\"] = float(np.nanmean(aps)) if not np.all(np.isnan(aps)) "
                "else float(\"nan\")")
    elif task == "regression":
        add("    pred = outputs.reshape(targets.shape) if outputs.size == targets.size else outputs")
        add("    err = pred - targets")
        if "mse" in keys or "rmse" in keys:
            add("    mse = float(np.mean(err ** 2))")
        if "mse" in keys:
            add("    m[\"mse\"] = mse")
        if "rmse" in keys:
            add("    m[\"rmse\"] = math.sqrt(mse)")
        if "mae" in keys:
            add("    m[\"mae\"] = float(np.mean(np.abs(err)))")
        if "median_ae" in keys:
            add("    m[\"median_ae\"] = float(np.median(np.abs(err)))")
        if "mape" in keys:
            add("    nonzero = np.abs(targets) > 1e-12")
            add("    m[\"mape\"] = float(np.mean(np.abs(err[nonzero] / targets[nonzero]))) "
                "if nonzero.any() else float(\"nan\")")
        if "r2" in keys:
            add("    ss_tot = float(np.sum((targets - targets.mean(axis=0)) ** 2))")
            add("    m[\"r2\"] = 1.0 - float(np.sum(err ** 2)) / ss_tot if ss_tot > 0 else float(\"nan\")")
        if "explained_variance" in keys:
            add("    var_y = float(np.var(targets))")
            add("    m[\"explained_variance\"] = 1.0 - float(np.var(err)) / var_y if var_y > 0 "
                "else float(\"nan\")")
    add("    return m")
    return "".join(out) + "\n".join(body)


def _classification(add, keys: set[str], params: dict, binary: bool) -> None:
    if keys & {"precision", "recall", "f1", "mcc", "kappa", "balanced_accuracy", "confusion"}:
        add("    cm = confusion_matrix(y, pred, k)")
    if "accuracy" in keys:
        add("    m[\"accuracy\"] = float((pred == y).mean())")
    if "balanced_accuracy" in keys:
        add("    support = cm.sum(axis=1)")
        add("    m[\"balanced_accuracy\"] = float(np.mean(np.diag(cm)[support > 0] "
            "/ support[support > 0]))")
    if "top_k_accuracy" in keys:
        k_top = int(params["top_k_accuracy"].get("k", 5))
        add(f"    top = np.argsort(-scores, axis=1)[:, :{k_top}]")
        add("    m[\"top_k_accuracy\"] = float((top == y[:, None]).any(axis=1).mean())")
    for key in ("precision", "recall", "f1"):
        if key in keys:
            avg = params[key].get("average", "macro")
            idx = {"precision": 0, "recall": 1, "f1": 2}[key]
            add(f"    m[\"{key}\"] = precision_recall_f1(cm, \"{avg}\", binary={binary})[{idx}]")
    if "mcc" in keys:
        add("    s, c = cm.sum(), np.trace(cm)")
        add("    pk, tk = cm.sum(axis=0), cm.sum(axis=1)")
        add("    denom = math.sqrt(max((s * s - (pk * pk).sum()) * (s * s - (tk * tk).sum()), 0))")
        add("    m[\"mcc\"] = float((c * s - (pk * tk).sum()) / denom) if denom else 0.0")
    if "kappa" in keys:
        add("    total = cm.sum()")
        add("    po = np.trace(cm) / max(total, 1)")
        add("    pe = float((cm.sum(axis=0) * cm.sum(axis=1)).sum()) / max(total * total, 1)")
        add("    m[\"kappa\"] = float((po - pe) / (1 - pe)) if pe < 1 else 0.0")
    if "roc_auc" in keys:
        if binary:
            add("    m[\"roc_auc\"] = binary_auc(p, y == 1)")
        else:
            add("    aucs = [binary_auc(scores[:, c], y == c) for c in range(k) "
                "if 0 < (y == c).sum() < len(y)]")
            add("    m[\"roc_auc\"] = float(np.mean(aucs)) if aucs else float(\"nan\")")
    if "avg_precision" in keys:
        if binary:
            add("    m[\"avg_precision\"] = average_precision(p, y == 1)")
        else:
            add("    aps = [average_precision(scores[:, c], y == c) for c in range(k) if (y == c).any()]")
            add("    m[\"avg_precision\"] = float(np.mean(aps)) if aps else float(\"nan\")")
    if "log_loss" in keys:
        add("    m[\"log_loss\"] = float(-np.mean(np.log(np.clip(scores[np.arange(len(y)), y], "
            "1e-12, 1.0))))")
    if "confusion" in keys:
        add("    m[\"confusion\"] = cm.tolist()")
