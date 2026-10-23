"""PyTorch training-script template: image data and array loaders (part of
``torch_template``)."""

TORCH_IMAGE_DATA = r'''

# ================================================================== data
INFERENCE_STATE: dict = {}  # fitted preprocessing, saved with the model for serving


def build_transforms(train: bool, mean=None, std=None) -> v2.Compose:
    steps = [v2.ToImage()]
{% if image_channels_fix %}
    steps.append({{ image_channels_fix }})
{% endif %}
{% for t in image_always %}
    steps.extend([{{ t }}])
{% endfor %}
{% if image_resize_fix %}
    steps.append({{ image_resize_fix }})
{% endif %}
    if train:
{% for t in image_train %}
        steps.extend([{{ t }}])
{% endfor %}
{% if not image_train %}
        pass
{% endif %}
    steps.append(v2.ToDtype(torch.float32, scale=True))
{% for t in image_after_tensor %}
    if train:
        steps.extend([{{ t }}])
{% endfor %}
    if mean is not None:
        steps.append(v2.Normalize(mean, std))
    return v2.Compose(steps)


def image_datasets():
    """Raw (untransformed) datasets: (train pool, official test set or None)."""
{% if d.block == "data.torchvision" %}
    root = str(Path({{ d.data_dir | repr }}).expanduser())
{% if d.download %}
    import socket

    socket.setdefaulttimeout(60)   # a dead network fails the run instead of hanging it
{% endif %}
{% if tv_split_kw == "split" %}
    return (datasets.{{ tv_class }}(root, split="train", download={{ d.download }}),
            datasets.{{ tv_class }}(root, split="test", download={{ d.download }}))
{% else %}
    return (datasets.{{ tv_class }}(root, train=True, download={{ d.download }}),
            datasets.{{ tv_class }}(root, train=False, download={{ d.download }}))
{% endif %}
{% else %}
    return datasets.ImageFolder({{ d.root | repr }}), None
{% endif %}


{% if two_views %}
class TwoViews:
    """Two independent augmentations of one image (SimCLR)."""

    def __init__(self, transform) -> None:
        self.transform = transform

    def __call__(self, image):
        return torch.stack([self.transform(image), self.transform(image)])


def views(xb: torch.Tensor, yb: torch.Tensor):
    """A batch of view pairs as one batch (first views, then their twins), else as is."""
    if xb.dim() == len(INPUT_SHAPE) + 2:
        return torch.cat([xb[:, 0], xb[:, 1]]), torch.cat([yb, yb]), True
    return xb, yb, False


{% endif %}
class TransformedSubset(Dataset):
    """A view of ``base`` restricted to ``indices`` with its own transform."""

    def __init__(self, base, indices, transform) -> None:
        self.base, self.indices, self.transform = base, list(indices), transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        image, label = self.base[self.indices[i]]
        return self.transform(image), label

    def source_file(self, i: int) -> str | None:
        samples = getattr(self.base, "samples", None)
        return samples[self.indices[i]][0] if samples else None
{% if norm_fit %}


def channel_stats(dataset, limit: int = 2000) -> tuple[list[float], list[float]]:
    """Per-channel mean/std estimated from training images (fit mode)."""
    probe = TransformedSubset(dataset.base, dataset.indices[:limit], build_transforms(train=False))
    total = squares = 0.0
    count = 0
    for xb, _ in DataLoader(probe, batch_size=256):
        total = total + xb.sum(dim=(0, 2, 3))
        squares = squares + (xb ** 2).sum(dim=(0, 2, 3))
        count += xb.numel() // xb.shape[1]
    mean = total / count
    return mean.tolist(), (squares / count - mean ** 2).clamp_min(1e-12).sqrt().tolist()
{% endif %}


def make_datasets():
    global CLASS_NAMES
    pool, official_test = image_datasets()
    CLASS_NAMES = list(getattr(pool, "classes", None) or []) or None
    idx = np.random.default_rng(SPLIT_SEED).permutation(len(pool))
    n_val = int(len(pool) * VAL_FRACTION)
{% if d.block == "data.torchvision" %}
    train_idx, val_idx, test_idx = idx[n_val:], idx[:n_val], None
{% else %}
    n_test = int(len(pool) * TEST_FRACTION)
    train_idx, val_idx, test_idx = idx[n_val + n_test:], idx[:n_val], idx[n_val:n_val + n_test]
{% endif %}
{% if norm_fixed %}
    mean, std = NORM_MEAN, NORM_STD
{% elif norm_fit %}
    mean, std = channel_stats(TransformedSubset(pool, train_idx, None))
{% else %}
    mean = std = None
{% endif %}
    INFERENCE_STATE["norm"] = (list(mean), list(std)) if mean is not None else None
    eval_tf = build_transforms(train=False, mean=mean, std=std)
{% if two_views %}
    views_tf = TwoViews(build_transforms(train=True, mean=mean, std=std))
    train = TransformedSubset(pool, train_idx, views_tf)
    val = TransformedSubset(pool, val_idx, views_tf)   # the contrastive loss needs pairs
{% else %}
    train = TransformedSubset(pool, train_idx, build_transforms(train=True, mean=mean, std=std))
    val = TransformedSubset(pool, val_idx, eval_tf)
{% endif %}
    test = (TransformedSubset(official_test, range(len(official_test)), eval_tf)
            if official_test is not None else TransformedSubset(pool, test_idx, eval_tf))
    return train, val, test


def dataset_targets(ds: TransformedSubset) -> np.ndarray:
    targets = getattr(ds.base, "targets", None)
    if targets is None:
        targets = getattr(ds.base, "labels", None)
    if targets is None:
        targets = [ds.base[i][1] for i in range(len(ds.base))]
    return np.asarray(targets)[ds.indices]
'''

TORCH_ARRAY_LOADERS = r'''

class ArrayDataset(Dataset):
    """In-memory samples with an optional per-sample training transform."""

    def __init__(self, x: np.ndarray, y: np.ndarray, transform=None) -> None:
        self.x, self.y, self.transform = torch.from_numpy(x), torch.from_numpy(y), transform

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        x = self.x[i]
        return (self.transform(x) if self.transform else x), self.y[i]


def make_datasets():
    arrays = make_arrays()
{% if image_train or spec.steps.get("prep.spec_augment") %}
    train_tf = train_transform
{% else %}
    train_tf = None
{% endif %}
    return (ArrayDataset(*arrays["train"], transform=train_tf),
            ArrayDataset(*arrays["val"]), ArrayDataset(*arrays["test"]))


def dataset_targets(ds) -> np.ndarray:
    return ds.y.numpy()
{% if image_train %}


def train_transform(x: torch.Tensor) -> torch.Tensor:
    """Training augmentation applied per sample."""
    return TRAIN_AUGMENT(x)


TRAIN_AUGMENT = v2.Compose([{{ image_train | join(", ") }}])
{% elif spec.steps.get("prep.spec_augment") %}
{% set sa = spec.steps["prep.spec_augment"] %}


def train_transform(x: torch.Tensor) -> torch.Tensor:
    """SpecAugment: random frequency and time masks (training only)."""
    if torch.rand(()) > {{ sa.p }}:
        return x
    x = x.clone()
    _, n_freq, n_time = x.shape
    f = int(torch.randint(0, {{ sa.freq_mask }} + 1, ()))
    f0 = int(torch.randint(0, max(n_freq - f, 1), ()))
    x[:, f0:f0 + f, :] = 0
    t = int(torch.randint(0, {{ sa.time_mask }} + 1, ()))
    t0 = int(torch.randint(0, max(n_time - t, 1), ()))
    x[:, :, t0:t0 + t] = 0
    return x
{% endif %}
'''
