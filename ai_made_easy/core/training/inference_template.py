"""Inference section appended to every generated training script.

After training, the script saves its fitted preprocessing (``INFERENCE_STATE``)
next to the model. ``load_predictor(folder)`` + ``infer(raw)`` then turn raw
inputs into predictions with exactly the training-time preprocessing — the
deployment package's server imports the training script for this.
"""

INFERENCE_COMMON = r'''

# ============================================================= inference
INFERENCE_FILE = "inference_state.pkl"


def save_inference_state() -> None:
    """Persist the fitted preprocessing and class names for serving."""
    import pickle

    INFERENCE_STATE["classes"] = CLASS_NAMES
    with open(INFERENCE_FILE, "wb") as fh:
        pickle.dump(INFERENCE_STATE, fh)


def load_inference_state(folder: str | Path = ".") -> None:
    """Restore the state saved by a training run (trusted files only: pickle)."""
    import pickle

    global CLASS_NAMES
    with open(Path(folder) / INFERENCE_FILE, "rb") as fh:
        INFERENCE_STATE.update(pickle.load(fh))
    CLASS_NAMES = INFERENCE_STATE.get("classes")
{% if image_inputs %}


def open_image(item):
    """Path, raw bytes, base64 / data-URL string or PIL image -> PIL image."""
    import base64
    import io

    from PIL import Image

    if isinstance(item, Image.Image):
        return item
    if isinstance(item, (bytes, bytearray)):
        return Image.open(io.BytesIO(item))
    if isinstance(item, str):
        if item.startswith("data:"):
            item = item.split(",", 1)[1]
        elif len(item) < 1024 and _is_file(item):
            return Image.open(item)
        return Image.open(io.BytesIO(base64.b64decode(item)))
    arr = np.asarray(item)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.max() <= 1 else arr.astype(np.uint8)
    return Image.fromarray(arr)


def _is_file(text: str) -> bool:
    try:
        return Path(text).is_file()
    except (OSError, ValueError):  # e.g. base64 strings longer than a file name may be
        return False
{% endif %}


def class_name(index: int) -> str:
    return str(CLASS_NAMES[index]) if CLASS_NAMES and index < len(CLASS_NAMES) else str(index)


def describe(outputs: np.ndarray) -> list[dict]:
    """Model outputs -> JSON-friendly predictions."""
    scores = to_scores(np.asarray(outputs, dtype=np.float32))
    results = []
{% if spec.task == "multiclass" %}
    for row in scores:
        best = int(np.argmax(row))
        results.append({"label": class_name(best), "class_index": best,
                        "confidence": round(float(row[best]), 6),
                        "probabilities": {class_name(i): round(float(p), 6)
                                          for i, p in enumerate(row)}})
{% elif spec.task == "binary" %}
    for p in scores.reshape(len(scores), -1)[:, 0]:
        best = int(p >= 0.5)
        results.append({"label": class_name(best), "class_index": best,
                        "confidence": round(float(p if best else 1 - p), 6),
                        "probabilities": {class_name(0): round(float(1 - p), 6),
                                          class_name(1): round(float(p), 6)}})
{% elif spec.task == "multilabel" %}
    for row in scores:
        results.append({"labels": [class_name(i) for i, p in enumerate(row) if p >= 0.5],
                        "probabilities": {class_name(i): round(float(p), 6)
                                          for i, p in enumerate(row)}})
{% else %}
    values = scores.reshape(len(scores), -1)
{% if timeseries %}
    values = inverse_targets(values)
{% endif %}
    for row in values:
        results.append({"prediction": float(row[0]) if len(row) == 1
                        else [float(v) for v in row]})
{% endif %}
    return results
'''

INFERENCE_TORCH = r'''

_PREDICTOR = None


def load_predictor(folder: str | Path = ".", device: str = "cpu") -> nn.Module:
    """Load the trained weights and preprocessing state from a run folder."""
    global _PREDICTOR
    load_inference_state(folder)
    model = {{ spec.class_name }}()
    model.load_state_dict(torch.load(Path(folder) / CHECKPOINT, map_location=device))
    model.eval()
    if QUANTIZE == "dynamic_int8":       # set by a pipeline's Quantize stage (CPU serving)
        engines = torch.backends.quantized.supported_engines
        if torch.backends.quantized.engine == "none" or torch.backends.quantized.engine \
                not in engines:
            torch.backends.quantized.engine = next(e for e in ("fbgemm", "x86", "qnnpack")
                                                   if e in engines)
        model = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    _PREDICTOR = model.to(device)
    return _PREDICTOR


@torch.no_grad()
def infer(raw: list) -> list[dict]:
    """Predictions for a list of raw inputs (see ``prepare_inputs``)."""
    if _PREDICTOR is None:
        raise RuntimeError("call load_predictor(folder) first")
    device = next(_PREDICTOR.parameters()).device
    x = torch.from_numpy(np.ascontiguousarray(prepare_inputs(raw))).to(device)
{% if prob %}
    return serve(_PREDICTOR, x)
{% else %}
    return describe(_PREDICTOR(x).float().cpu().numpy())
{% endif %}
'''

INFERENCE_KERAS = r'''

_PREDICTOR = None


def load_predictor(folder: str | Path = ".", device: str = "cpu"):
    """Load the trained model and preprocessing state from a run folder."""
    global _PREDICTOR
    load_inference_state(folder)
    _PREDICTOR = keras.models.load_model(Path(folder) / MODEL_FILE)
    return _PREDICTOR


def infer(raw: list) -> list[dict]:
    """Predictions for a list of raw inputs (see ``prepare_inputs``)."""
    if _PREDICTOR is None:
        raise RuntimeError("call load_predictor(folder) first")
    x = to_model_layout(prepare_inputs(raw))
    return describe(np.asarray(_PREDICTOR.predict(x, verbose=0)))
'''

# Torch image datasets use torchvision transforms instead of DATA_TEMPLATE's
# prepare_inputs; this is their equivalent.
INFERENCE_TORCH_IMAGES = r'''


def prepare_inputs(raw: list) -> np.ndarray:
    """Images (paths, bytes, base64 strings or PIL images) -> a normalised batch."""
    mean, std = INFERENCE_STATE.get("norm") or (None, None)
    transform = build_transforms(train=False, mean=mean, std=std)
    mode = "L" if INPUT_SHAPE[0] == 1 else "RGB"
    batch = []
    for item in raw:
        img = open_image(item).convert(mode)
{% if not image_explicit_size %}
        img = img.resize((INPUT_SHAPE[2], INPUT_SHAPE[1]))
{% endif %}
        batch.append(transform(img))
    x = torch.stack(batch).numpy()
    if tuple(x.shape[1:]) != tuple(INPUT_SHAPE):
        raise ValueError(f"images become {tuple(x.shape[1:])}; the model expects "
                         f"{tuple(INPUT_SHAPE)}")
    return x
'''
