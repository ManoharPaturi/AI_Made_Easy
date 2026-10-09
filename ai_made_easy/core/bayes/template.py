"""The probabilistic section of supervised PyTorch training scripts.

Rendered only when a design has Bayesian layers, a Laplace block or calibration blocks.
It provides the hooks the supervised template calls — KL penalty, mixture-density loss,
Monte Carlo prediction — plus ``probabilistic_report`` (calibration, temperature
scaling, conformal prediction, last-layer Laplace) and ``serve`` (predictions with
uncertainty for ``infer``).
"""
from __future__ import annotations

PROBABILISTIC = r'''

# ============================================================= probabilistic predictions
MC_SAMPLES = {{ prob.mc_samples }}
LOSS_INPUT = "{{ loss_input or 'logits' }}"
CLASS_TASK = {{ "True" if spec.task in ("multiclass", "binary") else "False" }}
MDN = {{ prob.mdn | repr }}                # None or {"k": components, "d": targets}
LAPLACE = {{ prob.laplace | repr }}
ECE_BINS = {{ prob.ece_bins }}
TEMPERATURE = {{ "True" if prob.temperature else "False" }}
CONFORMAL_ALPHA = {{ prob.conformal | repr }}


def kl_penalty(model: nn.Module) -> torch.Tensor:
    """Sum of the Bayes-by-Backprop layers' KL(q || prior)."""
    return sum(m.kl_divergence() for m in model.modules() if hasattr(m, "kl_divergence"))


def mdn_split(out: torch.Tensor):
    k, d = MDN["k"], MDN["d"]
    logits = out[:, :k]
    mu = out[:, k:k + k * d].reshape(-1, k, d)
    log_sigma = out[:, k + k * d:].reshape(-1, k, d).clamp(-7.0, 7.0)
    return logits, mu, log_sigma


def mdn_nll(out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood of the targets under the predicted Gaussian mixture."""
    logits, mu, log_sigma = mdn_split(out.float())
    y = y.float().reshape(len(y), 1, -1)
    comp = (-0.5 * ((y - mu) / log_sigma.exp()) ** 2 - log_sigma
            - 0.5 * math.log(2 * math.pi)).sum(-1)
    return -torch.logsumexp(logits.log_softmax(-1) + comp, dim=-1).mean()


def mdn_moments(out: np.ndarray):
    logits, mu, log_sigma = mdn_split(torch.from_numpy(np.asarray(out)).float())
    w = logits.softmax(-1)[..., None]
    mean = (w * mu).sum(1)
    var = (w * (log_sigma.exp() ** 2 + mu ** 2)).sum(1) - mean ** 2
    return mean.numpy(), var.clamp_min(1e-12).sqrt().numpy()


def mdn_point(out: np.ndarray) -> np.ndarray:
    mean, _std = mdn_moments(out)
    return mean[:, 0] if mean.shape[1] == 1 else mean


def probabilities(outputs: np.ndarray) -> np.ndarray:
    """Model outputs -> class probabilities [N, C] (binary: two columns)."""
    p = np.asarray(to_scores(np.asarray(outputs, dtype=np.float32)), dtype=np.float64)
    if p.ndim == 1 or p.shape[1] == 1:
        p = p.reshape(-1)
        p = np.stack([1 - p, p], axis=1)
    return np.clip(p, 1e-7, 1.0)


def encode(probs: np.ndarray) -> np.ndarray:
    """Class probabilities back into the model's output space (so describe() / metrics
    work unchanged)."""
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
{% if spec.task == "binary" %}
    p = probs[:, 1]
    return (p if LOSS_INPUT == "probs" else np.log(p) - np.log1p(-p)).reshape(-1, 1).astype(np.float32)
{% else %}
    return (probs if LOSS_INPUT == "probs" else np.log(probs)).astype(np.float32)
{% endif %}


@torch.no_grad()
def mc_forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """The predictive mean over MC_SAMPLES stochastic passes (probabilities are averaged,
    not logits)."""
    if MC_SAMPLES <= 1 or MDN:
        return model(x)
    outs = [model(x).float().cpu().numpy() for _ in range(MC_SAMPLES)]
    if CLASS_TASK:
        mean = np.mean([probabilities(o) for o in outs], axis=0)
        return torch.from_numpy(encode(mean)).to(x.device)
    return torch.from_numpy(np.mean(outs, axis=0)).to(x.device)


# ------------------------------------------------------------- calibration helpers
def nll(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None))))


def ece(probs: np.ndarray, y: np.ndarray, bins: int) -> tuple[float, list]:
    """Expected calibration error and the reliability-diagram bins (confidence vs accuracy)."""
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    edges = np.linspace(0, 1, bins + 1)
    total, table = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
            table.append((float(conf[mask].mean()), float(correct[mask].mean()), int(mask.sum())))
    return float(total), table


def scale_temperature(probs: np.ndarray, t: float) -> np.ndarray:
    z = np.log(np.clip(probs, 1e-12, None)) / t
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def conformal_sets(probs: np.ndarray, qhat: float) -> np.ndarray:
    """Classes whose score 1 - p is within the conformal quantile (never empty: the top class
    is always kept)."""
    sets = probs >= 1 - qhat - 1e-9
    sets[np.arange(len(probs)), probs.argmax(1)] = True
    return sets


def fit_temperature(probs: np.ndarray, y: np.ndarray) -> float:
    grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 400))
    return float(min(grid, key=lambda t: nll(scale_temperature(probs, t), y)))


def reliability_png(table: list, path: Path, title: str) -> None:
    from PIL import Image, ImageDraw

    w = h = 320
    img = Image.new("RGB", (w, h), (24, 26, 30))
    draw = ImageDraw.Draw(img)

    def xy(a, b):
        return 30 + a * (w - 50), h - 30 - b * (h - 60)

    draw.line([xy(0, 0), xy(1, 1)], fill=(90, 90, 90))
    for conf, acc, count in table:
        x, y = xy(conf, acc)
        r = 2 + min(8, count ** 0.5 / 3)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(240, 120, 80))
    draw.text((8, 6), title, fill=(225, 225, 225))
    draw.text((w - 110, h - 18), "confidence ->", fill=(150, 150, 150))
    draw.text((6, 22), "accuracy ^", fill=(150, 150, 150))
    path.parent.mkdir(exist_ok=True)
    img.save(path)


# ------------------------------------------------------------- last-layer Laplace
def last_linear(model: nn.Module) -> nn.Linear:
    layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not layers:
        raise SystemExit("the Laplace approximation needs a final Dense layer")
    return layers[-1]


@torch.no_grad()
def features(model: nn.Module, loader, device) -> tuple:
    """Inputs of the last Dense layer (A), outputs and targets, deterministic pass."""
    layer, captured = last_linear(model), []
    handle = layer.register_forward_hook(lambda _m, inp, _o: captured.append(inp[0].detach()))
    model.eval()
    outs, ys = [], []
    try:
        for xb, yb in loader:
            outs.append(model(xb.to(device)).float().cpu())
            ys.append(yb)
    finally:
        handle.remove()
    return (torch.cat(captured).float().cpu().reshape(-1, layer.in_features).numpy(),
            torch.cat(outs).numpy(), torch.cat(ys).numpy())


def laplace_fit(model: nn.Module, train_loader, device) -> dict:
    layer = last_linear(model)
    A, out, y = features(model, train_loader, device)
    A1 = np.concatenate([A, np.ones((len(A), 1))], axis=1)
    if CLASS_TASK:
        p = probabilities(out)
        curvature = (p * (1 - p)) if p.shape[1] == layer.out_features else \
            (p[:, 1:] * (1 - p[:, 1:]))
        h = curvature.T @ (A1 ** 2)                                 # [outputs, F + 1]
        noise = None
    else:
        point = out.reshape(len(out), -1)
        noise = float(np.mean((point - y.reshape(len(y), -1)) ** 2)) + 1e-9
        h = np.tile((A1 ** 2).sum(0) / noise, (layer.out_features, 1))
    W = np.concatenate([layer.weight.detach().cpu().numpy(),
                        layer.bias.detach().cpu().numpy()[:, None]
                        if layer.bias is not None else np.zeros((layer.out_features, 1))], 1)
    return {"h": h, "W": W, "noise": noise, "prior": float(LAPLACE["prior_precision"])}


def laplace_predict(post: dict, A: np.ndarray, prior: float, samples: int, seed: int = 0):
    """Class probabilities or (mean, std) from weights sampled around the trained ones."""
    rng = np.random.default_rng(seed)
    A1 = np.concatenate([A, np.ones((len(A), 1))], axis=1)
    sd = 1.0 / np.sqrt(post["h"] + prior)
    draws = [A1 @ (post["W"] + sd * rng.standard_normal(sd.shape)).T for _ in range(samples)]
    if CLASS_TASK:
        return np.mean([probabilities(encode_raw(d)) for d in draws], axis=0)
    stack = np.stack(draws)
    mean = stack.mean(0)
    std = np.sqrt(stack.var(0) + post["noise"])
    return mean, std


def encode_raw(logits: np.ndarray) -> np.ndarray:
    """Last-layer outputs as the model's own outputs (the Dense layer ends the model)."""
    return logits.astype(np.float32)


def laplace_tune(post: dict, model, val_loader, device) -> float:
    if post["prior"] > 0:
        return post["prior"]
    A, _out, y = features(model, val_loader, device)
    best, best_score = 1.0, math.inf
    for prior in 10.0 ** np.arange(-3, 4, 0.5):
        pred = laplace_predict(post, A, prior, 20)
        if CLASS_TASK:
            score = nll(pred, y.astype(int).reshape(-1))
        else:
            mean, std = pred
            yy = y.reshape(len(y), -1)
            score = float(np.mean(0.5 * np.log(2 * np.pi * std ** 2) + (yy - mean) ** 2
                                  / (2 * std ** 2)))
        if score < best_score:
            best, best_score = float(prior), score
    return best


# ------------------------------------------------------------- report
@torch.no_grad()
def predictive(model: nn.Module, loader, device) -> tuple:
    """(outputs, targets, epistemic std or None) with MC averaging when stochastic."""
    outs, ys, stds = [], [], []
    model.eval()
    for xb, yb in loader:
        x = xb.to(device)
        if not CLASS_TASK and not MDN and MC_SAMPLES > 1:
            draws = torch.stack([model(x).float() for _ in range(MC_SAMPLES)])
            outs.append(draws.mean(0).cpu())
            stds.append(draws.std(0).cpu())
        else:
            outs.append(mc_forward(model, x).float().cpu())
        ys.append(yb)
    return (torch.cat(outs).numpy(), torch.cat(ys).numpy(),
            torch.cat(stds).numpy() if stds else None)


def regression_moments(outputs, eps_std, noise: float | None):
    if MDN:
        mean, std = mdn_moments(outputs)
        return mean.reshape(len(mean), -1), std.reshape(len(std), -1)
    mean = outputs.reshape(len(outputs), -1)
    if eps_std is None:
        return mean, None if noise is None else np.full_like(mean, math.sqrt(noise))
    var = eps_std.reshape(len(eps_std), -1) ** 2 + (noise or 0.0)
    return mean, np.sqrt(var)


def probabilistic_report(model: nn.Module, train_loader, val_loader, test_loader,
                         device) -> dict:
    """Calibration, temperature scaling, conformal prediction and Laplace on held-out data."""
    out = {}
    if not len(test_loader.dataset):
        return out
    val_out, val_y, val_std = predictive(model, val_loader, device)
    test_out, test_y, test_std = predictive(model, test_loader, device)
    post = None
    if LAPLACE is not None:
        post = laplace_fit(model, train_loader, device)
        post["prior"] = laplace_tune(post, model, val_loader, device)
        INFERENCE_STATE["laplace"] = post
        out["laplace_prior_precision"] = post["prior"]
    if CLASS_TASK:
        yv, yt = val_y.astype(int).reshape(-1), test_y.astype(int).reshape(-1)
        pv, pt = probabilities(val_out), probabilities(test_out)
        out["nll"] = nll(pt, yt)
        out["ece"], table = ece(pt, yt, ECE_BINS)
        if post is not None:
            Av, _o, _y = features(model, val_loader, device)
            At, _o, _y = features(model, test_loader, device)
            pv = laplace_predict(post, Av, post["prior"], LAPLACE["samples"])
            pt = laplace_predict(post, At, post["prior"], LAPLACE["samples"])
            out["laplace_nll"] = nll(pt, yt)
            out["laplace_ece"], table = ece(pt, yt, ECE_BINS)
        if TEMPERATURE and len(yv):
            t = fit_temperature(pv, yv)
            INFERENCE_STATE["temperature"] = t
            pv, pt = scale_temperature(pv, t), scale_temperature(pt, t)
            out["temperature"] = t
            out["calibrated_nll"] = nll(pt, yt)
            out["calibrated_ece"], table = ece(pt, yt, ECE_BINS)
        reliability_png(table, Path("eval_samples") / "reliability.png",
                        f"reliability (ECE {ece(pt, yt, ECE_BINS)[0]:.3f})")
        if CONFORMAL_ALPHA and len(yv):
            scores = 1 - pv[np.arange(len(yv)), yv]
            level = min(1.0, math.ceil((len(scores) + 1) * (1 - CONFORMAL_ALPHA)) / len(scores))
            qhat = float(np.quantile(scores, level, method="higher"))
            INFERENCE_STATE["conformal_qhat"] = qhat
            sets = conformal_sets(pt, qhat)
            out["conformal_coverage"] = float(sets[np.arange(len(yt)), yt].mean())
            out["conformal_set_size"] = float(sets.sum(1).mean())
        return out
    noise = None
    if not MDN and val_std is not None:            # aleatoric noise left over by MC averaging
        resid = (val_out.reshape(len(val_out), -1) - val_y.reshape(len(val_y), -1)) ** 2
        noise = max(float(resid.mean() - (val_std ** 2).mean()), 1e-12)
        INFERENCE_STATE["mc_noise"] = noise
    yv, yt = val_y.reshape(len(val_y), -1), test_y.reshape(len(test_y), -1)
    if post is not None:
        Av, _o, _y = features(model, val_loader, device)
        At, _o, _y = features(model, test_loader, device)
        mv, sv = laplace_predict(post, Av, post["prior"], LAPLACE["samples"])
        mt, st = laplace_predict(post, At, post["prior"], LAPLACE["samples"])
    else:
        mv, sv = regression_moments(val_out, val_std, noise)
        mt, st = regression_moments(test_out, test_std, noise)
    if st is not None:
        out["nll_gaussian"] = float(np.mean(0.5 * np.log(2 * np.pi * st ** 2)
                                            + (yt - mt) ** 2 / (2 * st ** 2)))
        out["coverage_90"] = float(np.mean(np.abs(yt - mt) <= 1.645 * st))
        out["mean_std"] = float(st.mean())
    if CONFORMAL_ALPHA and len(yv):
        scale_v = sv if sv is not None else np.ones_like(mv)
        scale_t = st if st is not None else np.ones_like(mt)
        scores = (np.abs(yv - mv) / scale_v).max(1)
        level = min(1.0, math.ceil((len(scores) + 1) * (1 - CONFORMAL_ALPHA)) / len(scores))
        q = float(np.quantile(scores, level, method="higher"))
        INFERENCE_STATE["conformal_q"] = q
        inside = (np.abs(yt - mt) <= q * scale_t).all(1)
        out["conformal_coverage"] = float(inside.mean())
        out["conformal_width"] = float((2 * q * scale_t).mean())
    return out


# ------------------------------------------------------------- serving
@torch.no_grad()
def serve(model: nn.Module, x: torch.Tensor) -> list[dict]:
    """infer(): predictions with uncertainty, calibrated and conformal when configured."""
    state = INFERENCE_STATE
    post = state.get("laplace")
    if CLASS_TASK:
        if post is not None:
            layer, captured = last_linear(model), []
            handle = layer.register_forward_hook(
                lambda _m, inp, _o: captured.append(inp[0].detach()))
            try:
                model(x)
            finally:
                handle.remove()
            probs = laplace_predict(post, captured[0].float().cpu().numpy(), post["prior"],
                                    LAPLACE["samples"])
        else:
            probs = probabilities(mc_forward(model, x).float().cpu().numpy())
        if "temperature" in state:
            probs = scale_temperature(probs, state["temperature"])
        results = describe(encode(probs))
        for r, p in zip(results, probs):
            r["entropy"] = round(float(-(p * np.log(np.clip(p, 1e-12, None))).sum()), 6)
            if "conformal_qhat" in state:
                r["prediction_set"] = [class_name(i) for i in np.flatnonzero(
                    conformal_sets(p[None], state["conformal_qhat"])[0])]
        return results
    eps_std = None
    if post is not None:
        layer, captured = last_linear(model), []
        handle = layer.register_forward_hook(lambda _m, inp, _o: captured.append(inp[0].detach()))
        try:
            model(x)
        finally:
            handle.remove()
        mean, std = laplace_predict(post, captured[0].float().cpu().numpy(), post["prior"],
                                    LAPLACE["samples"])
    else:
        if not MDN and MC_SAMPLES > 1:
            draws = torch.stack([model(x).float() for _ in range(MC_SAMPLES)]).cpu().numpy()
            outputs, eps_std = draws.mean(0), draws.std(0)
        else:
            outputs = model(x).float().cpu().numpy()
        mean, std = regression_moments(outputs, eps_std, state.get("mc_noise"))
    results = []
    for i in range(len(mean)):
        row = {"prediction": float(mean[i, 0]) if mean.shape[1] == 1
               else [float(v) for v in mean[i]]}
        if std is not None:
            row["std"] = float(std[i, 0]) if std.shape[1] == 1 else [float(v) for v in std[i]]
        if "conformal_q" in state:
            width = state["conformal_q"] * (std[i] if std is not None else np.ones(mean.shape[1]))
            row["interval"] = [[float(m - w), float(m + w)] for m, w in zip(mean[i], width)]
            if mean.shape[1] == 1:
                row["interval"] = row["interval"][0]
        results.append(row)
    return results
'''
