"""Regenerate the pretrained-model catalogs in ai_made_easy/core/zoo/ (offline, no weights).

Each torchvision backbone is instantiated without weights, its classifier is removed
exactly like the generated ``PretrainedBackbone`` helper does, and a forward pass
measures the feature width. Parameter counts, ImageNet accuracy and minimum input
size come from the torchvision weight metadata.

Usage: python scripts/build_zoo.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torchvision
from torch import nn

OUT = Path(__file__).resolve().parent.parent / "ai_made_easy" / "core" / "zoo"

# torchvision name -> (keras.applications class or None, fixed input side or None)
IMAGE = {
    "resnet18": (None, None), "resnet34": (None, None), "resnet50": ("ResNet50", None),
    "resnet101": ("ResNet101", None), "resnet152": ("ResNet152", None),
    "resnext50_32x4d": (None, None), "resnext101_32x8d": (None, None),
    "wide_resnet50_2": (None, None),
    "mobilenet_v2": ("MobileNetV2", None), "mobilenet_v3_small": ("MobileNetV3Small", None),
    "mobilenet_v3_large": ("MobileNetV3Large", None),
    "efficientnet_b0": ("EfficientNetB0", None), "efficientnet_b1": ("EfficientNetB1", None),
    "efficientnet_b2": ("EfficientNetB2", None), "efficientnet_b3": ("EfficientNetB3", None),
    "efficientnet_b4": ("EfficientNetB4", None),
    "efficientnet_v2_s": ("EfficientNetV2S", None), "efficientnet_v2_m": ("EfficientNetV2M", None),
    "convnext_tiny": ("ConvNeXtTiny", None), "convnext_small": ("ConvNeXtSmall", None),
    "convnext_base": ("ConvNeXtBase", None),
    "densenet121": ("DenseNet121", None), "densenet169": ("DenseNet169", None),
    "densenet201": ("DenseNet201", None),
    "vgg16": ("VGG16", None), "vgg19": ("VGG19", None),
    "regnet_y_400mf": (None, None), "regnet_y_800mf": (None, None),
    "regnet_y_1_6gf": (None, None), "regnet_y_3_2gf": (None, None),  # not in Keras 3
    "shufflenet_v2_x1_0": (None, None), "mnasnet1_0": (None, None),
    "vit_b_16": (None, 224), "vit_b_32": (None, 224), "vit_l_16": (None, 224),
    "swin_t": (None, None), "swin_s": (None, None), "swin_v2_t": (None, None),
    "maxvit_t": (None, 224),
}


def strip_head(name: str, model: nn.Module) -> nn.Module:
    """Mirror of PretrainedBackbone in core/codegen/helpers.py."""
    if name.startswith("vgg"):
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.classifier = nn.Flatten()
    elif hasattr(model, "fc"):
        model.fc = nn.Identity()
    elif hasattr(model, "heads"):
        model.heads = nn.Identity()
    elif hasattr(model, "head"):
        model.head = nn.Identity()
    elif name.startswith(("convnext", "maxvit")):
        model.classifier[-1] = nn.Identity()
    else:
        model.classifier = nn.Identity()
    return model


def image_entry(name: str, keras: str | None, fixed: int | None) -> dict:
    weights = torchvision.models.get_model_weights(name).DEFAULT
    meta = weights.meta
    model = strip_head(name, torchvision.models.get_model(name, weights=None)).eval()
    side = fixed or 224
    with torch.no_grad():
        width = int(model(torch.zeros(1, 3, side, side)).shape[-1])
    min_side = fixed or max(32, int(meta.get("min_size", (32, 32))[0]))
    acc = meta.get("_metrics", {}).get("ImageNet-1K", {}).get("acc@1")
    return {"name": name, "feature_dim": width, "params_m": round(meta["num_params"] / 1e6, 1),
            "imagenet_top1": acc, "gmacs": meta.get("_ops"),
            "gmacs_side": int(weights.transforms().crop_size[0]), "min_side": min_side, "fixed_side": fixed, "keras": keras,
            "license": "BSD-3-Clause (torchvision)", "source": "torchvision"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (keras, fixed) in IMAGE.items():
        rows.append(image_entry(name, keras, fixed))
        print(f"{name:<22} {rows[-1]['feature_dim']:>5}  {rows[-1]['params_m']:>6} M")
    (OUT / "torchvision_image.json").write_text(json.dumps(rows, indent=1) + "\n")
    print(f"wrote {len(rows)} entries to {OUT / 'torchvision_image.json'}")


if __name__ == "__main__":
    main()
