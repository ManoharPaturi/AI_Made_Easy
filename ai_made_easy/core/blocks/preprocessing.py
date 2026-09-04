"""Preprocessing blocks (config; consumed by the Phase 2 training script)."""
from __future__ import annotations

from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import BlockDefinition, ParamSpec
from ai_made_easy.core.blocks._palette import family_color

reg = get_registry()

_PREP_COLOR = family_color("preprocess")


def _cfg(**kwargs) -> None:
    kwargs.setdefault("category", "Preprocessing")
    kwargs.setdefault("color", _PREP_COLOR)
    reg.register(BlockDefinition(**kwargs))


_cfg(
    type_id="prep.normalize",
    display_name="Normalize (z-score)",
    params=(ParamSpec(name="mean", type="str", default="0.1307",
                      help="Per-feature mean, comma-separated"),
            ParamSpec(name="std", type="str", default="0.3081",
                      help="Per-feature std, comma-separated")),
)

_cfg(
    type_id="prep.minmax",
    display_name="MinMax Scale",
    params=(
        ParamSpec(name="range_min", type="float", default=0.0, minimum=-1000.0),
        ParamSpec(name="range_max", type="float", default=1.0, minimum=-1000.0),
    ),
)

_cfg(
    type_id="prep.to_tensor",
    display_name="To Tensor",
    params=(),
)

_cfg(
    type_id="prep.split",
    display_name="Train/Val/Test Split",
    params=(
        ParamSpec(name="val_fraction", type="float", default=0.1,
                  minimum=0.0, maximum=0.9),
        ParamSpec(name="test_fraction", type="float", default=0.1,
                  minimum=0.0, maximum=0.9),
        ParamSpec(name="shuffle", type="bool", default=True),
        ParamSpec(name="seed", type="int", default=42, minimum=0),
    ),
)

_cfg(
    type_id="prep.shuffle",
    display_name="Shuffle",
    params=(ParamSpec(name="seed", type="int", default=42, minimum=0),),
)

_cfg(
    type_id="prep.dataloader",
    display_name="DataLoader",
    params=(
        ParamSpec(name="batch_size", type="int", default=32, minimum=1),
        ParamSpec(name="num_workers", type="int", default=0, minimum=0),
        ParamSpec(name="pin_memory", type="bool", default=True),
    ),
)

# --- image augmentations (applied in canonical order to image datasets) ---

_cfg(
    type_id="prep.resize",
    display_name="Resize",
    params=(
        ParamSpec(name="height", type="int", default=32, minimum=1),
        ParamSpec(name="width", type="int", default=32, minimum=1),
    ),
)

_cfg(
    type_id="prep.center_crop",
    display_name="Center Crop",
    params=(ParamSpec(name="size", type="int", default=28, minimum=1),),
)

_cfg(
    type_id="prep.random_flip",
    display_name="Random Flip",
    params=(ParamSpec(name="mode", type="enum", default="both",
                      options=("horizontal", "vertical", "both")),),
)

_cfg(
    type_id="prep.random_rotation",
    display_name="Random Rotation",
    params=(ParamSpec(name="degrees", type="float", default=10.0, minimum=-360.0, maximum=360.0),),
)

_cfg(
    type_id="prep.color_jitter",
    display_name="Color Jitter",
    params=(
        ParamSpec(name="brightness", type="float", default=0.2, minimum=0.0),
        ParamSpec(name="contrast", type="float", default=0.2, minimum=0.0),
        ParamSpec(name="saturation", type="float", default=0.2, minimum=0.0),
        ParamSpec(name="hue", type="float", default=0.05,
                  minimum=0.0, maximum=0.5),
    ),
)

# audit addition (Orange3 Impute): NaN is the #1 silent killer of a first run
reg.register(BlockDefinition(
    type_id="prep.impute",
    display_name="Fill Missing Values",
    category="Preprocessing",
    color=_PREP_COLOR,
    params=(
        ParamSpec(name="strategy", type="enum", default="mean",
                  options=("mean", "median", "mode", "constant", "drop rows"),
                  help="How to fill holes in your data"),
        ParamSpec(name="constant", type="float", default=0.0, minimum=-1000.0,
                  help="Value used when strategy = constant"),
    ),
))
