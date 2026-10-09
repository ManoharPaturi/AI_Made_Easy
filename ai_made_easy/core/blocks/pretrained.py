"""Pretrained feature extractors: ImageNet CNN/ViT backbones and Hugging Face
text encoders, usable frozen (feature extraction) or fine-tuned."""
from __future__ import annotations

from ai_made_easy.core.blocks._dsl import P, nn_block
from ai_made_easy.core.codegen import KerasUnsupported
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.spec import ShapeError
from ai_made_easy.core.zoo import catalog

reg = get_registry()

# name: (feature width, keras.applications class or None, min side, fixed side or None),
# from the measured torchvision catalog (scripts/build_zoo.py)
BACKBONES = {row["name"]: (row["feature_dim"], row["keras"], row["min_side"], row["fixed_side"])
             for row in catalog("torchvision_image")}

# model id: hidden size
TEXT_ENCODERS = {
    "bert-base-uncased": 768, "distilbert-base-uncased": 768, "roberta-base": 768,
    "sentence-transformers/all-MiniLM-L6-v2": 384, "albert-base-v2": 768,
    "google/electra-small-discriminator": 256, "microsoft/deberta-v3-small": 768,
    "bert-large-uncased": 1024,
}


def _backbone_shape(in_shapes, params):
    s = in_shapes[0]
    arch = params["architecture"]
    width, _keras, min_side, fixed = BACKBONES[arch]
    if len(s) != 3:
        raise ShapeError(f"{arch} expects an image [3, H, W], got {s}")
    if s[0] != 3:
        raise ShapeError(f"{arch} was pretrained on RGB images: input needs 3 channels, got {s[0]} "
                         "(add a Grayscale(3) transform or change the Input)")
    if fixed and (s[1] != fixed or s[2] != fixed):
        raise ShapeError(f"{arch} needs {fixed}×{fixed} inputs, got {s[1]}×{s[2]}")
    if min(s[1], s[2]) < min_side:
        raise ShapeError(f"{arch} needs images of at least {min_side}×{min_side}, got "
                         f"{s[1]}×{s[2]}")
    return [width]


def _keras_backbone(c):
    arch = c["architecture"]
    cls = BACKBONES[arch][1]
    if cls is None:
        raise KerasUnsupported(f"{arch} is not in keras.applications")
    weights = '"imagenet"' if c["weights"] == "imagenet" else "None"
    return (f"pretrained_backbone({c['i0']}, \"{cls}\", weights={weights}, "
            f"trainable={not c['freeze']})")


reg.register(nn_block(
    "core.pretrained_backbone", "Pretrained Image Backbone", "Pretrained Models",
    family="model",
    params=(P("architecture", "enum", "resnet18", options=tuple(BACKBONES)),
            P("weights", "enum", "imagenet", options=("imagenet", "none"),
              help="ImageNet weights expect ImageNet-normalized RGB input"),
            P("freeze", "bool", True, help="Freeze backbone weights (feature extraction)")),
    shape=_backbone_shape, layout="cl",
    torch=lambda c: (f"PretrainedBackbone(\"{c['architecture']}\", "
                     f"pretrained={c['weights'] == 'imagenet'}, freeze={bool(c['freeze'])})"),
    torch_helpers=("PretrainedBackbone",),
    keras_expr=_keras_backbone, keras_helpers=("pretrained_backbone",),
    desc="ImageNet-pretrained CNN / ViT feature extractor → pooled feature vector [F].",
))


def _text_shape(in_shapes, params):
    s = in_shapes[0]
    if len(s) != 1:
        raise ShapeError(f"text encoder expects token ids [L], got {s}")
    if s[0] > 512:
        raise ShapeError(f"sequence length {s[0]} exceeds the 512-token limit of the encoder")
    hidden = TEXT_ENCODERS[params["model_id"]]
    return [hidden] if params["pooling"] != "none" else [s[0], hidden]


reg.register(nn_block(
    "core.hf_text_encoder", "Pretrained Text Encoder", "Pretrained Models", family="model",
    params=(P("model_id", "enum", "distilbert-base-uncased", options=tuple(TEXT_ENCODERS)),
            P("pooling", "enum", "cls", options=("cls", "mean", "none"),
              help="cls/mean → [hidden]; none → per-token [L, hidden]"),
            P("freeze", "bool", True)),
    shape=_text_shape, layout="ir", input_dtype="int",
    torch=lambda c: (f"HFTextEncoder(\"{c['model_id']}\", pooling=\"{c['pooling']}\", "
                     f"freeze={bool(c['freeze'])})"),
    torch_helpers=("HFTextEncoder",),
    desc="Hugging Face transformer encoder over token ids (pair with a Hugging Face Tokenize).",
))
