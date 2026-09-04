"""Data-source blocks (config: they parameterize the training script that
Phase 2 generates; they carry no tensor ports and never sit in model flow)."""
from __future__ import annotations

from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import BlockDefinition, ParamSpec
from ai_made_easy.core.blocks._palette import family_color

reg = get_registry()

_DATA_COLOR = family_color("data")


def _cfg(**kwargs) -> None:
    kwargs.setdefault("category", "Data")
    kwargs.setdefault("color", _DATA_COLOR)
    reg.register(BlockDefinition(**kwargs))


_cfg(
    type_id="data.torchvision",
    display_name="Torchvision Dataset",
    params=(
        ParamSpec(name="dataset", type="enum", default="mnist",
                  options=("mnist", "fashion_mnist", "cifar10", "cifar100", "stl10")),
        ParamSpec(name="download", type="bool", default=True),
        ParamSpec(name="data_dir", type="str", default="~/.aime/data"),
    ),
)

_cfg(
    type_id="data.csv",
    display_name="CSV File",
    params=(
        ParamSpec(name="path", type="str", default="data.csv"),
        ParamSpec(name="target_column", type="str", default="label"),
    ),
)

_cfg(
    type_id="data.synthetic",
    display_name="Synthetic Generator",
    params=(
        ParamSpec(name="kind", type="enum", default="classification",
                  options=("classification", "moons", "circles", "regression")),
        ParamSpec(name="n_samples", type="int", default=1000, minimum=1),
        ParamSpec(name="n_features", type="int", default=20, minimum=1),
        ParamSpec(name="n_classes", type="int", default=2, minimum=2),
        ParamSpec(name="noise", type="float", default=0.1, minimum=0.0),
        ParamSpec(name="seed", type="int", default=42, minimum=0),
    ),
)

_cfg(
    type_id="data.numpy",
    display_name="NumPy Archive (.npz)",
    params=(
        ParamSpec(name="path", type="str", default="data.npz"),
        ParamSpec(name="x_key", type="str", default="x",
                  help="Array name for features inside the archive"),
        ParamSpec(name="y_key", type="str", default="y",
                  help="Array name for labels inside the archive"),
    ),
)

_cfg(
    type_id="data.json",
    display_name="JSON / JSONL Records",
    params=(
        ParamSpec(name="path", type="str", default="data.jsonl"),
        ParamSpec(name="x_field", type="str", default="x"),
        ParamSpec(name="y_field", type="str", default="y"),
    ),
)

_cfg(
    type_id="data.image_folder",
    display_name="Image Folder",
    params=(
        ParamSpec(name="root", type="str", default="images/",
                  help="Root folder with one subfolder per class"),
        ParamSpec(name="grayscale", type="bool", default=False),
    ),
)

_cfg(
    type_id="data.huggingface",
    display_name="HuggingFace Dataset",
    params=(
        ParamSpec(name="repo_id", type="str", default="mnist",
                  help="dataset id on the HuggingFace Hub"),
        ParamSpec(name="split", type="str", default="train"),
        ParamSpec(name="x_field", type="str", default="image"),
        ParamSpec(name="y_field", type="str", default="label"),
    ),
)
