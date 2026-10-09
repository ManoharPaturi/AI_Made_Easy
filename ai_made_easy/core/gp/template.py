"""Scripts for Gaussian processes: GPyTorch (exact or sparse variational) or scikit-learn.

Both standardise the inputs (and, for real-valued targets, the target), fit the kernel
hyperparameters, report held-out accuracy and calibration against a constant baseline,
plot the fit and save what ``load_predictor`` / ``infer`` need to predict new rows.
"""
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, StrictUndefined

from ai_made_easy.core.codegen import CodegenError, sanitize_identifier
from ai_made_easy.core.gp.kernels import (
    KernelError,
    _names,
    gpytorch_expr,
    kernel_root,
    model_node,
    sklearn_expr,
)
from ai_made_easy.core.graph import Graph

_env = Environment(trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True,
                   undefined=StrictUndefined)
_env.filters["repr"] = repr

COMMON = r'''
DATASET = {{ dataset | repr }}
TARGET = {{ target | repr }}
FEATURE_SPEC = {{ features | repr }}
GP = {{ gp | repr }}
SAMPLES_DIR = Path("eval_samples")
STATE: dict = {}
FEATURES: list = []
CLASSIFY = GP["likelihood"] == "bernoulli"
COUNTS = GP["likelihood"] == "poisson"


# ======================================================================== data
def synthetic_function(kind: str, n: int, noise: float, seed: int):
    rng = np.random.default_rng(seed)
    if kind == "classes_2d":
        x = rng.uniform(-3, 3, (n, 2))
        p = 1 / (1 + np.exp(-3 * (np.sin(x[:, 0]) - 0.5 * x[:, 1])))
        return pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "y": (rng.random(n) < p).astype(int)})
    x = np.sort(rng.uniform(0, 10, n))
    f = {"smooth": np.sin(x) + 0.3 * x,
         "periodic": np.sin(2 * np.pi * x / 2.5),
         "trend_periodic": 0.4 * x + np.sin(2 * np.pi * x / 2.0),
         "step": np.where(x > 5, 1.5, -0.5)}[kind]
    return pd.DataFrame({"x": x, "y": f + noise * rng.standard_normal(n)})


def load_frame():
    if DATASET is None:
        raise SystemExit("a Gaussian process learns from data: add a table or Synthetic "
                         "Function")
    if DATASET["block"] == "data.synthetic_function":
        return synthetic_function(DATASET["kind"], int(DATASET["n_points"]),
                                  float(DATASET["noise"]), int(DATASET["seed"]))
    path = Path(DATASET["path"]).expanduser()
    fmt = DATASET.get("format", "csv")
    if fmt == "parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t" if fmt == "tsv" else ",")


def prepare(frame):
    if TARGET not in frame.columns:
        raise SystemExit(f"target column {TARGET!r} not found; columns: {list(frame.columns)}")
    features = FEATURE_SPEC or [c for c in frame.columns if c != TARGET
                                and pd.api.types.is_numeric_dtype(frame[c])]
    missing = [c for c in features if c not in frame.columns]
    if missing or not features:
        raise SystemExit(f"feature columns {missing or 'none'} not found / numeric")
    FEATURES[:] = features
    frame = frame[[*features, TARGET]].dropna()
    return frame


def fit_scalers(train) -> None:
    X = train[FEATURES].to_numpy(float)
    y = train[TARGET].to_numpy(float)
    STATE.update(features=list(FEATURES), x_mean=X.mean(0).tolist(),
                 x_std=(X.std(0) + 1e-12).tolist(),
                 y_mean=0.0 if CLASSIFY or COUNTS else float(y.mean()),
                 y_std=1.0 if CLASSIFY or COUNTS else float(y.std() + 1e-12))


def scale_x(frame) -> np.ndarray:
    return (frame[STATE["features"]].to_numpy(float) - np.array(STATE["x_mean"])) / \
        np.array(STATE["x_std"])


def dims(names: list) -> tuple:
    """Input dimensions of the named feature columns (kernels restricted to some inputs)."""
    missing = [n for n in names if n not in FEATURES]
    if missing:
        raise SystemExit(f"kernel columns {missing} are not features {FEATURES}")
    return tuple(FEATURES.index(n) for n in names)


# ======================================================================== metrics
def scores(y: np.ndarray, mean: np.ndarray, std: np.ndarray, base: float) -> dict:
    """Held-out accuracy and calibration, with a constant-prediction baseline."""
    if CLASSIFY:
        p = np.clip(mean, 1e-6, 1 - 1e-6)
        return {"accuracy": float(np.mean((p > 0.5) == (y > 0.5))),
                "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
                "brier": float(np.mean((p - y) ** 2)),
                "baseline_accuracy": float(max(np.mean(y), 1 - np.mean(y)))}
    out = {"rmse": float(np.sqrt(np.mean((mean - y) ** 2))),
           "mae": float(np.mean(np.abs(mean - y))),
           "baseline_rmse": float(np.sqrt(np.mean((base - y) ** 2)))}
    var = np.maximum(std ** 2, 1e-12)
    out["nlpd"] = float(np.mean(0.5 * np.log(2 * np.pi * var) + (y - mean) ** 2 / (2 * var)))
    out["coverage_95"] = float(np.mean(np.abs(y - mean) <= 1.96 * std))
    out["r2"] = float(1 - np.sum((y - mean) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12))
    return out


# ======================================================================== plots
def plot_fit(train, test, predict_fn, title: str) -> "Image.Image":
    w, h = 760, 360
    img = Image.new("RGB", (w, h), (24, 26, 30))
    draw = ImageDraw.Draw(img)
    if len(FEATURES) == 1 and not CLASSIFY:
        col = FEATURES[0]
        lo, hi = float(train[col].min()), float(train[col].max())
        span = hi - lo or 1
        grid = pd.DataFrame({col: np.linspace(lo - 0.15 * span, hi + 0.15 * span, 300)})
        mean, std = predict_fn(grid)
        ys = np.concatenate([train[TARGET], mean - 2 * std, mean + 2 * std])
        y0, y1 = float(ys.min()), float(ys.max())
        y1 = y1 if y1 > y0 else y0 + 1
        xs = grid[col].to_numpy()

        def xy(a, b):
            return (20 + (a - xs[0]) / (xs[-1] - xs[0]) * (w - 40),
                    h - 20 - (b - y0) / (y1 - y0) * (h - 50))

        band = [xy(a, b) for a, b in zip(xs, mean + 1.96 * std)] + \
            [xy(a, b) for a, b in zip(xs[::-1], (mean - 1.96 * std)[::-1])]
        draw.polygon(band, fill=(60, 70, 100))
        draw.line([xy(a, b) for a, b in zip(xs, mean)], fill=(240, 120, 80), width=2)
        for part, color in ((train, (200, 200, 200)), (test, (110, 210, 120))):
            if part is None:
                continue
            for a, b in zip(part[col], part[TARGET]):
                x, y = xy(a, b)
                draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=color)
        draw.text((10, 6), f"{title}: mean and 95% band (grey: train, green: test)",
                  fill=(225, 225, 225))
        return img
    part = test if test is not None else train
    mean, _std = predict_fn(part)
    y = part[TARGET].to_numpy(float)
    lo, hi = float(min(y.min(), mean.min())), float(max(y.max(), mean.max()))
    hi = hi if hi > lo else lo + 1
    for a, b in zip(y, mean):
        x0 = 30 + (a - lo) / (hi - lo) * (w - 60)
        y0 = h - 30 - (b - lo) / (hi - lo) * (h - 60)
        draw.ellipse([x0 - 2, y0 - 2, x0 + 2, y0 + 2], fill=(240, 120, 80))
    draw.text((10, 6), f"{title}: predicted vs actual (held-out rows)", fill=(225, 225, 225))
    return img


def split(frame):
    n_test = int(round(len(frame) * GP["test_fraction"])) if len(frame) >= 20 else 0
    order = np.random.default_rng(GP["seed"]).permutation(len(frame))
    test = frame.iloc[order[:n_test]].reset_index(drop=True) if n_test else None
    return frame.iloc[order[n_test:]].reset_index(drop=True), test


def finish(train, test, predict_fn, hyper: list) -> None:
    metrics = {}
    for name, part in (("train", train), ("test", test)):
        if part is None:
            continue
        mean, std = predict_fn(part)
        base = float(train[TARGET].mean())
        for k, v in scores(part[TARGET].to_numpy(float), mean, std, base).items():
            metrics[f"{name}_{k}" if name == "train" else k] = v
    print("test: " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()), flush=True)
    Path("metrics.json").write_text(json.dumps(metrics, indent=1))
    SAMPLES_DIR.mkdir(exist_ok=True)
    plot_fit(train, test, predict_fn, "Gaussian process").save(SAMPLES_DIR / "fit.png")
    (SAMPLES_DIR / "report.txt").write_text("Learned hyperparameters\n" + "\n".join(hyper) + "\n")


def infer(raw: list) -> list[dict]:
    """Rows of feature values -> predictive mean, std and a 95% interval (probability for
    classes)."""
    if _PREDICTOR is None:
        raise RuntimeError("call load_predictor(folder) first")
    frame = pd.DataFrame([dict(r) for r in raw])
    missing = [c for c in STATE["features"] if c not in frame.columns]
    if missing:
        raise ValueError(f"missing feature(s) {missing}")
    mean, std = predict(frame)
    if CLASSIFY:
        return [{"probability": round(float(m), 4)} for m in mean]
    return [{"mean": round(float(m), 6), "std": round(float(s), 6),
             "lower_95": round(float(m - 1.96 * s), 6), "upper_95": round(float(m + 1.96 * s), 6)}
            for m, s in zip(mean, std)]
'''

GPYTORCH = r'''"""{{ title }} — Gaussian process (GPyTorch) generated by AI Made Easy.

Data:   {{ data_comment }}
Usage:  python {{ filename }}          fit the kernel, score held-out rows, save the model
Needs:  torch, gpytorch, numpy, pandas, pillow
"""
from __future__ import annotations

import json
from pathlib import Path

import gpytorch
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
''' + COMMON + r'''
MODEL_FILE = "{{ name }}_gp.pt"
STATE_FILE = "inference_state.json"
_PREDICTOR = None


def init(kernel, **values):
    """Set a kernel's starting hyperparameters (in standardised input units)."""
    kernel.initialize(**values)
    return kernel


def make_kernel():
    return {{ kernel }}


def make_likelihood():
    kind = GP["likelihood"]
    if kind == "bernoulli":
        return gpytorch.likelihoods.BernoulliLikelihood()
    if kind == "poisson":
        return gpytorch.likelihoods.PoissonLikelihood()
    if kind == "student_t":
        return gpytorch.likelihoods.StudentTLikelihood()
    return gpytorch.likelihoods.GaussianLikelihood()


class ExactModel(gpytorch.models.ExactGP):
    def __init__(self, X, y, likelihood):
        super().__init__(X, y, likelihood)
        self.mean = gpytorch.means.ConstantMean()
        self.kernel = make_kernel()

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean(x), self.kernel(x))


class SparseModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing):
        dist = gpytorch.variational.CholeskyVariationalDistribution(inducing.shape[0])
        strategy = gpytorch.variational.VariationalStrategy(self, inducing, dist,
                                                            learn_inducing_locations=True)
        super().__init__(strategy)
        self.mean = gpytorch.means.ConstantMean()
        self.kernel = make_kernel()

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean(x), self.kernel(x))


def build(X: torch.Tensor, y: torch.Tensor):
    likelihood = make_likelihood()
    if GP["kind"] == "exact" and GP["likelihood"] == "gaussian":
        model = ExactModel(X, y, likelihood)
    else:
        m = min(GP["inducing_points"], len(X))
        idx = torch.randperm(len(X), generator=torch.Generator().manual_seed(GP["seed"]))[:m]
        model = SparseModel(X[idx].clone())
    for module in model.modules():
        if isinstance(module, gpytorch.kernels.SpectralMixtureKernel):
            module.initialize_from_data(X, y)
    return model, likelihood


def tensors(frame, with_target: bool = True):
    X = torch.tensor(scale_x(frame), dtype=torch.float32)
    if not with_target:
        return X, None
    y = (frame[TARGET].to_numpy(float) - STATE["y_mean"]) / STATE["y_std"]
    return X, torch.tensor(y, dtype=torch.float32)


def train(model, likelihood, X, y) -> None:
    model.train()
    likelihood.train()
    params = list(model.parameters()) + [p for p in likelihood.parameters()
                                         if not any(p is q for q in model.parameters())]
    optimizer = torch.optim.Adam(params, lr=GP["lr"])
    if isinstance(model, gpytorch.models.ExactGP):
        objective = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    else:
        objective = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(y))
    every = max(1, GP["iterations"] // 50)
    batch = len(X) if isinstance(model, gpytorch.models.ExactGP) else min(1024, len(X))
    for step in range(1, GP["iterations"] + 1):
        idx = torch.randperm(len(X))[:batch] if batch < len(X) else slice(None)
        optimizer.zero_grad()
        loss = -objective(model(X[idx]), y[idx])
        loss.backward()
        optimizer.step()
        if step % every == 0 or step == GP["iterations"]:
            print(f"epoch {step}/{GP['iterations']} train_loss={loss.item():.4f}", flush=True)


@torch.no_grad()
def predict(frame):
    model, likelihood = _PREDICTOR
    model.eval()
    likelihood.eval()
    X, _ = tensors(frame, with_target=False)
    with gpytorch.settings.fast_pred_var():
        out = likelihood(model(X))
    if CLASSIFY:
        return out.mean.numpy(), np.zeros(len(X))
    mean = out.mean.numpy() * STATE["y_std"] + STATE["y_mean"]
    std = out.variance.clamp_min(1e-12).sqrt().numpy() * STATE["y_std"]
    return mean, std


def hyperparameters(model, likelihood) -> list:
    lines = []
    for name, module in model.named_modules():
        for attr in ("outputscale", "lengthscale", "period_length", "variance", "constant"):
            value = getattr(module, attr, None)
            if isinstance(value, torch.Tensor) and value.numel() <= 16:
                lines.append(f"  {name or 'kernel'}.{attr}: "
                             f"{np.round(value.detach().numpy().ravel(), 4).tolist()}")
    noise = getattr(likelihood, "noise", None)
    if isinstance(noise, torch.Tensor):
        lines.append(f"  likelihood.noise (standardised): {round(float(noise.mean()), 5)}")
    return lines


def load_predictor(folder: str | Path = ".", device: str = "cpu"):
    global _PREDICTOR
    STATE.update(json.loads((Path(folder) / STATE_FILE).read_text()))
    FEATURES[:] = STATE["features"]
    saved = torch.load(Path(folder) / MODEL_FILE, map_location="cpu", weights_only=False)
    model, likelihood = build(saved["X"], saved["y"])
    model.load_state_dict(saved["model"])
    likelihood.load_state_dict(saved["likelihood"])
    _PREDICTOR = (model, likelihood)
    return _PREDICTOR


def main() -> None:
    global _PREDICTOR
    torch.manual_seed(GP["seed"])
    frame = prepare(load_frame())
    train_rows, test_rows = split(frame)
    fit_scalers(train_rows)
    X, y = tensors(train_rows)
    print(f"rows: train={len(train_rows)} test={0 if test_rows is None else len(test_rows)} "
          f"features={FEATURES}", flush=True)
    model, likelihood = build(X, y)
    train(model, likelihood, X, y)
    _PREDICTOR = (model, likelihood)
    finish(train_rows, test_rows, predict, hyperparameters(model, likelihood))
    torch.save({"model": model.state_dict(), "likelihood": likelihood.state_dict(), "X": X,
                "y": y}, MODEL_FILE)
    Path(STATE_FILE).write_text(json.dumps(STATE))
    print(f"saved the model to {MODEL_FILE}")


if __name__ == "__main__":
    main()
'''

SKLEARN = r'''"""{{ title }} — Gaussian process (scikit-learn) generated by AI Made Easy.

Data:   {{ data_comment }}
Usage:  python {{ filename }}          fit the kernel, score held-out rows, save the model
Needs:  scikit-learn, numpy, pandas, pillow, joblib
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (  # noqa: F401
    RBF,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    Matern,
    RationalQuadratic,
    WhiteKernel,
)
''' + COMMON + r'''
MODEL_FILE = "{{ name }}_model.joblib"
STATE_FILE = "inference.json"
_PREDICTOR = None


def make_kernel():
    return {{ kernel }}


def predict(frame):
    X = scale_x(frame)
    if CLASSIFY:
        return _PREDICTOR.predict_proba(X)[:, 1], np.zeros(len(X))
    mean, std = _PREDICTOR.predict(X, return_std=True)
    return mean * STATE["y_std"] + STATE["y_mean"], std * STATE["y_std"]


def load_predictor(folder: str | Path = ".", device: str = "cpu"):
    global _PREDICTOR
    STATE.update(json.loads((Path(folder) / STATE_FILE).read_text()))
    FEATURES[:] = STATE["features"]
    _PREDICTOR = joblib.load(Path(folder) / MODEL_FILE)
    return _PREDICTOR


def main() -> None:
    global _PREDICTOR
    frame = prepare(load_frame())
    train_rows, test_rows = split(frame)
    fit_scalers(train_rows)
    X = scale_x(train_rows)
    y = train_rows[TARGET].to_numpy(float)
    print(f"rows: train={len(train_rows)} test={0 if test_rows is None else len(test_rows)} "
          f"features={FEATURES}", flush=True)
    if CLASSIFY:
        model = GaussianProcessClassifier(make_kernel(), n_restarts_optimizer=GP["restarts"],
                                          random_state=GP["seed"])
        model.fit(X, y.astype(int))
    else:
        model = GaussianProcessRegressor(make_kernel(), normalize_y=False, alpha=1e-6,
                                         n_restarts_optimizer=GP["restarts"],
                                         random_state=GP["seed"])
        model.fit(X, (y - STATE["y_mean"]) / STATE["y_std"])
    _PREDICTOR = model
    kernel = getattr(model, "kernel_", None)
    finish(train_rows, test_rows, predict, [f"  {kernel}"])
    joblib.dump(model, MODEL_FILE)
    Path(STATE_FILE).write_text(json.dumps(STATE))
    print(f"saved the model to {MODEL_FILE}")


if __name__ == "__main__":
    main()
'''


def _dataset(graph: Graph) -> dict | None:
    node = next((n for n in graph.nodes.values()
                 if n.type_id in ("data.csv", "data.synthetic_function")), None)
    if node is None:
        return None
    out = {"block": node.type_id, **dict(node.resolved_params())}
    raw = out.get("path")
    if raw and not str(raw).startswith("~") and not Path(str(raw)).is_absolute():
        candidate = Path.cwd() / str(raw)
        if candidate.exists():
            out["path"] = str(candidate)
    return out


def render(graph: Graph, library: str) -> str:
    from ai_made_easy.core.gp.rules import gp_issues

    errors = [i for i in gp_issues(graph) if i.severity == "error"]
    if errors:
        raise CodegenError("the Gaussian process design has errors:\n"
                           + "\n".join(f"  - {e.message}" for e in errors))
    model = model_node(graph)
    p = dict(model.resolved_params())
    root = kernel_root(graph)
    try:
        if library == "sklearn":
            kernel = sklearn_expr(graph, root)
            if p["likelihood"] != "bernoulli" and "WhiteKernel" not in kernel:
                kernel = f"{kernel} + WhiteKernel(noise_level=0.1)"   # learn the noise
        else:
            kernel = gpytorch_expr(graph, root) or "gpytorch.kernels.ScaleKernel(" \
                "gpytorch.kernels.RBFKernel())"
    except KernelError as exc:
        raise CodegenError(str(exc)) from None
    name = sanitize_identifier(graph.name)
    dataset = _dataset(graph)
    gp = {k: p[k] for k in ("kind", "likelihood", "inducing_points", "iterations", "lr",
                            "restarts", "test_fraction", "seed")}
    return _env.from_string(SKLEARN if library == "sklearn" else GPYTORCH).render(
        title=graph.name, filename=f"{name}_train_{library}.py", name=name,
        data_comment=_comment(dataset), dataset=dataset, target=str(p["target_column"]),
        features=_names(p["feature_columns"]), gp=gp, kernel=kernel)


def _comment(dataset: dict | None) -> str:
    if dataset is None:
        return "none"
    if dataset["block"] == "data.synthetic_function":
        return f"{dataset['n_points']} noisy points of a {dataset['kind']} function"
    return str(dataset.get("path"))
