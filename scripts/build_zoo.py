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


# torchvision detection / segmentation models: name -> (task, side their GMACs refer to)
DETECTION = {
    "fasterrcnn_resnet50_fpn": ("detection", 800),
    "fasterrcnn_resnet50_fpn_v2": ("detection", 800),
    "fasterrcnn_mobilenet_v3_large_fpn": ("detection", 800),
    "fasterrcnn_mobilenet_v3_large_320_fpn": ("detection", 320),
    "retinanet_resnet50_fpn": ("detection", 800),
    "retinanet_resnet50_fpn_v2": ("detection", 800),
    "fcos_resnet50_fpn": ("detection", 800),
    "ssd300_vgg16": ("detection", 300),
    "ssdlite320_mobilenet_v3_large": ("detection", 320),
    "maskrcnn_resnet50_fpn": ("instance_segmentation", 800),
    "maskrcnn_resnet50_fpn_v2": ("instance_segmentation", 800),
    "keypointrcnn_resnet50_fpn": ("keypoints", 800),
}
SEGMENTATION = {
    "deeplabv3_resnet50": 520, "deeplabv3_resnet101": 520,
    "deeplabv3_mobilenet_v3_large": 520, "fcn_resnet50": 520, "fcn_resnet101": 520,
    "lraspp_mobilenet_v3_large": 520,
}


# curated timm image models (needs `pip install timm`; no weights are downloaded)
TIMM = """resnet18d resnet26d resnet50d resnet101d resnet152d resnext50d_32x4d seresnext50_32x4d
ecaresnet50d regnety_008 regnety_016 regnety_032 regnety_040 regnety_080 regnetz_d8
convnext_atto convnext_femto convnext_pico convnext_nano convnext_tiny convnext_small
convnext_base convnextv2_atto convnextv2_nano convnextv2_tiny convnextv2_base efficientnet_b0
efficientnet_b2 efficientnet_b3 efficientnetv2_rw_t efficientnetv2_rw_s tf_efficientnetv2_s
tf_efficientnetv2_m tf_efficientnet_b4 mobilenetv3_large_100 mobilenetv3_small_100
mobilenetv4_conv_small mobilenetv4_conv_medium mobilenetv4_conv_large mobilenetv4_hybrid_medium
ghostnetv2_100 repvit_m1 repvit_m2 fastvit_t8 fastvit_sa12 efficientformerv2_s0
efficientformerv2_s1 mobilevitv2_100 edgenext_small levit_128 tinynet_a vit_tiny_patch16_224
vit_small_patch16_224 vit_base_patch16_224 vit_base_patch32_224 vit_large_patch16_224
vit_small_patch14_dinov2 vit_base_patch14_dinov2 vit_base_patch16_clip_224
vit_base_patch16_siglip_224 deit_tiny_patch16_224 deit_small_patch16_224 deit_base_patch16_224
deit3_small_patch16_224 deit3_base_patch16_224 swin_tiny_patch4_window7_224
swin_small_patch4_window7_224 swin_base_patch4_window7_224 swinv2_tiny_window8_256
swinv2_small_window8_256 maxvit_tiny_tf_224 maxvit_small_tf_224 maxvit_nano_rw_256
coatnet_0_rw_224 coatnet_nano_rw_224 eva02_tiny_patch14_224 eva02_small_patch14_224
eva02_base_patch14_224 caformer_s18 convformer_s18 poolformerv2_s12 nextvit_small davit_tiny
efficientvit_b1 efficientvit_m5 hgnetv2_b0 hgnetv2_b2 inception_next_tiny""".split()
# input side is fixed for models with absolute position embeddings / fixed windows
_TIMM_FIXED = ("vit_", "deit", "swin", "maxvit", "coatnet", "eva02", "levit", "efficientvit_m",
               "nextvit", "davit", "efficientformerv2")


def timm_entry(name: str) -> dict:
    import timm

    model = timm.create_model(name, pretrained=False, num_classes=0).eval()
    cfg = model.pretrained_cfg
    side = int(cfg.get("input_size", (3, 224, 224))[-1])
    with torch.no_grad():
        width = int(model(torch.zeros(1, 3, side, side)).shape[-1])
    params = sum(p.numel() for p in model.parameters())
    return {"name": name, "feature_dim": width, "params_m": round(params / 1e6, 1),
            "input_side": side, "fixed_side": side if name.startswith(_TIMM_FIXED) else None,
            "min_side": 32, "license": str(cfg.get("license") or "see model card"),
            "hf_hub_id": cfg.get("hf_hub_id", ""), "source": "timm"}


def head_entry(name: str, task: str, side: int) -> dict:
    meta = torchvision.models.get_model_weights(name).DEFAULT.meta
    metrics = next(iter(meta.get("_metrics", {}).values()), {})
    return {"name": name, "task": task, "params_m": round(meta["num_params"] / 1e6, 1),
            "metrics": metrics, "gmacs": meta.get("_ops"), "gmacs_side": side,
            "pretrained_classes": len(meta.get("categories") or []),
            "license": "BSD-3-Clause (torchvision)", "source": "torchvision"}


def _write(name: str, rows: list[dict]) -> None:
    (OUT / f"{name}.json").write_text(json.dumps(rows, indent=1) + "\n")
    print(f"wrote {len(rows)} entries to {OUT / (name + '.json')}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (keras, fixed) in IMAGE.items():
        rows.append(image_entry(name, keras, fixed))
        print(f"{name:<22} {rows[-1]['feature_dim']:>5}  {rows[-1]['params_m']:>6} M")
    _write("torchvision_image", rows)
    try:
        _write("timm_image", [timm_entry(n) for n in TIMM])
    except ImportError:
        print("timm is not installed: skipping timm_image.json")
    _write("torchvision_detection", [head_entry(n, t, s) for n, (t, s) in DETECTION.items()])
    _write("torchvision_segmentation",
           [head_entry(n, "semantic_segmentation", s) for n, s in SEGMENTATION.items()])


if __name__ == "__main__":
    main()
