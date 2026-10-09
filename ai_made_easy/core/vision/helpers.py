"""PyTorch helpers emitted into generated code for vision-task blocks.

Every task model takes images scaled to [0, 1] and normalizes internally, so
the data pipeline is the same for every architecture. Detectors expose
``losses(images, targets)`` for training and ``detect(images)`` for per-image
results; their ``forward`` returns padded detections ``[B, D, 6]``
(x1, y1, x2, y2, score, label) so the designed output shape holds. Class
indices are 0-based object classes everywhere: wrappers add / remove the
background class models like Faster R-CNN reserve at index 0.

``unet_cost`` mirrors the ``UNet`` helper to count parameters and MACs
without importing torch (summary and budget estimates).
"""
from __future__ import annotations

_NORMALIZE = '''\
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    """[0, 1] RGB -> ImageNet-normalized; grayscale is repeated to 3 channels."""
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std
'''

TV_DETECTOR = '''\
class TVDetector(nn.Module):
    """torchvision detector (Faster / Mask / Keypoint R-CNN, RetinaNet, FCOS, SSD).

    forward(images [B, C, H, W] in [0, 1]) -> padded detections [B, D, 6];
    losses(images, targets) -> dict of losses; detect(images) -> list of dicts.
    Labels are 0-based object classes (background is handled here).
    """

    def __init__(self, arch: str, num_classes: int, weights: str = "none",
                 min_size: int = 800, max_size: int = 1333, max_detections: int = 100,
                 score_threshold: float = 0.05, trainable_layers: int = 3,
                 num_keypoints: int = 17) -> None:
        super().__init__()
        from torchvision.models import detection

        self.arch, self.max_detections = arch, max_detections
        self.num_keypoints = num_keypoints
        builder = getattr(detection, arch)
        rcnn = arch.startswith(("fasterrcnn", "maskrcnn", "keypointrcnn"))
        kwargs = {}
        if rcnn:
            kwargs.update(min_size=min_size, max_size=max_size,
                          box_detections_per_img=max_detections,
                          box_score_thresh=score_threshold)
        elif arch.startswith(("retinanet", "fcos")):
            kwargs.update(min_size=min_size, max_size=max_size,
                          detections_per_img=max_detections, score_thresh=score_threshold)
        else:  # SSD family: fixed input size
            kwargs.update(detections_per_img=max_detections, score_thresh=score_threshold)
        classes = num_classes + 1  # + background
        if arch.startswith("keypointrcnn"):
            kwargs["num_keypoints"] = num_keypoints
        if weights != "none" and skip_pretrained():
            # loading trained weights: same architecture, no download
            self.model = builder(weights=None, weights_backbone=None, num_classes=classes,
                                 **kwargs)
            if not arch.endswith("_v2") and not arch.startswith("ssd"):
                freeze_batchnorm(self.model.backbone)
        elif weights == "coco":
            kwargs.pop("num_keypoints", None)
            self.model = builder(weights="DEFAULT", trainable_backbone_layers=trainable_layers,
                                 **kwargs)
            self._replace_heads(classes)
        elif weights == "imagenet":
            self.model = builder(weights=None, weights_backbone="DEFAULT", num_classes=classes,
                                 trainable_backbone_layers=trainable_layers, **kwargs)
        else:
            self.model = builder(weights=None, weights_backbone=None, num_classes=classes,
                                 **kwargs)

    def _replace_heads(self, classes: int) -> None:
        from torchvision.models.detection import faster_rcnn, fcos, keypoint_rcnn, mask_rcnn
        from torchvision.models.detection import retinanet

        m = self.model
        if self.arch.startswith(("fasterrcnn", "maskrcnn", "keypointrcnn")):
            box = m.roi_heads.box_predictor
            if box.cls_score.out_features != classes:
                m.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(
                    box.cls_score.in_features, classes)
            if self.arch.startswith("maskrcnn"):
                head = m.roi_heads.mask_predictor
                m.roi_heads.mask_predictor = mask_rcnn.MaskRCNNPredictor(
                    head.conv5_mask.in_channels, head.conv5_mask.out_channels, classes)
            if self.arch.startswith("keypointrcnn") and self.num_keypoints != 17:
                head = m.roi_heads.keypoint_predictor
                m.roi_heads.keypoint_predictor = keypoint_rcnn.KeypointRCNNPredictor(
                    head.kps_score_lowres.in_channels, self.num_keypoints)
        elif self.arch.startswith("retinanet"):
            head = m.head.classification_head
            m.head.classification_head = retinanet.RetinaNetClassificationHead(
                m.backbone.out_channels, head.num_anchors, classes)
        elif self.arch.startswith("fcos"):
            head = m.head.classification_head
            m.head.classification_head = fcos.FCOSClassificationHead(
                m.backbone.out_channels, head.num_anchors, classes)
        elif classes != 91:
            raise ValueError(f"{self.arch} COCO weights keep the 80 COCO classes: use "
                             "weights='imagenet' to train other classes")

    @staticmethod
    def _shift(targets: list[dict], by: int) -> list[dict]:
        return [{**t, "labels": t["labels"] + by} for t in targets]

    def losses(self, images: torch.Tensor, targets: list[dict]) -> dict:
        return self.model(list(images), self._shift(targets, 1))

    def detect(self, images: torch.Tensor) -> list[dict]:
        return self._shift(self.model(list(images)), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return pad_detections(self.detect(x), self.max_detections, x)
'''

SKIP_PRETRAINED = '''\
def skip_pretrained() -> bool:
    """True while loading trained weights (AIME_SKIP_PRETRAINED=1): build without downloads."""
    import os

    return os.environ.get("AIME_SKIP_PRETRAINED") == "1"


def freeze_batchnorm(module: nn.Module) -> None:
    """BatchNorm2d -> FrozenBatchNorm2d, as torchvision uses for pretrained backbones."""
    from torchvision.ops.misc import FrozenBatchNorm2d

    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, FrozenBatchNorm2d(child.num_features, eps=child.eps))
        else:
            freeze_batchnorm(child)
'''

PAD_DETECTIONS = '''\
def pad_detections(results: list[dict], size: int, like: torch.Tensor) -> torch.Tensor:
    """Per-image detections -> [B, size, 6] (x1, y1, x2, y2, score, label), zero padded."""
    out = like.new_zeros(len(results), size, 6)
    for i, r in enumerate(results):
        n = min(len(r["boxes"]), size)
        out[i, :n, :4] = r["boxes"][:n]
        out[i, :n, 4] = r["scores"][:n]
        out[i, :n, 5] = r["labels"][:n].to(out.dtype)
    return out
'''

HF_DETR = '''\
class HFDetr(nn.Module):
    """Hugging Face DETR-family detector (DETR, Conditional DETR, RT-DETR).

    Same interface as TVDetector: losses / detect / padded forward.
    """

    PRESETS = {"detr": ("facebook/detr-resnet-50", "Detr"),
               "conditional_detr": ("microsoft/conditional-detr-resnet-50", "ConditionalDetr"),
               "rt_detr": ("PekingU/rtdetr_r18vd", "RTDetr")}

    def __init__(self, arch: str, num_classes: int, weights: str = "none",
                 max_detections: int = 100, score_threshold: float = 0.05) -> None:
        super().__init__()
        import transformers

        repo, prefix = self.PRESETS[arch]
        model_cls = getattr(transformers, f"{prefix}ForObjectDetection")
        self.max_detections, self.score_threshold = max_detections, score_threshold
        if weights == "coco":
            self.model = model_cls.from_pretrained(repo, num_labels=num_classes,
                                                   ignore_mismatched_sizes=True)
        else:
            config_cls = getattr(transformers, f"{prefix}Config")
            extra = {} if arch == "rt_detr" else {"use_pretrained_backbone": False}
            queries = "num_queries" if arch == "rt_detr" else "num_queries"
            config = config_cls(num_labels=num_classes, **extra,
                                **{queries: max(max_detections, 100)})
            self.model = model_cls(config)
        self.sigmoid_scores = arch != "detr"

    def losses(self, images: torch.Tensor, targets: list[dict]) -> dict:
        h, w = images.shape[-2:]
        scale = images.new_tensor([w, h, w, h])
        labels = []
        for t in targets:
            b = t["boxes"] / scale
            cxcywh = torch.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2,
                                  b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], dim=-1)
            labels.append({"class_labels": t["labels"], "boxes": cxcywh})
        out = self.model(pixel_values=imagenet_normalize(images), labels=labels)
        return {"loss": out.loss}

    def detect(self, images: torch.Tensor) -> list[dict]:
        h, w = images.shape[-2:]
        out = self.model(pixel_values=imagenet_normalize(images))
        logits = out.logits
        probs = logits.sigmoid() if self.sigmoid_scores else logits.softmax(-1)[..., :-1]
        scores, labels = probs.max(-1)
        results = []
        for s, lab, b in zip(scores, labels, out.pred_boxes):
            keep = s >= self.score_threshold
            order = s[keep].argsort(descending=True)[: self.max_detections]
            cx, cy, bw, bh = b[keep][order].unbind(-1)
            boxes = torch.stack([(cx - bw / 2) * w, (cy - bh / 2) * h,
                                 (cx + bw / 2) * w, (cy + bh / 2) * h], dim=-1)
            results.append({"boxes": boxes, "scores": s[keep][order],
                            "labels": lab[keep][order]})
        return results

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return pad_detections(self.detect(x), self.max_detections, x)
'''

TV_SEGMENTER = '''\
class TVSegmenter(nn.Module):
    """torchvision semantic segmentation model (DeepLabV3, FCN, LR-ASPP).

    forward(images [B, C, H, W] in [0, 1]) -> class logits [B, num_classes, H, W].
    """

    def __init__(self, arch: str, num_classes: int, weights: str = "none") -> None:
        super().__init__()
        from torchvision.models import segmentation

        builder = getattr(segmentation, arch)
        if weights != "none" and skip_pretrained():
            self.model = builder(weights=None, weights_backbone=None, num_classes=num_classes,
                                 aux_loss=False)
        elif weights == "coco":
            self.model = builder(weights="DEFAULT", aux_loss=None)
            self.model.aux_classifier = None
            self._replace_head(num_classes)
        else:
            backbone = "DEFAULT" if weights == "imagenet" else None
            self.model = builder(weights=None, weights_backbone=backbone,
                                 num_classes=num_classes, aux_loss=False)

    def _replace_head(self, num_classes: int) -> None:
        head = self.model.classifier
        if hasattr(head, "low_classifier"):  # LR-ASPP
            head.low_classifier = nn.Conv2d(head.low_classifier.in_channels, num_classes, 1)
            head.high_classifier = nn.Conv2d(head.high_classifier.in_channels, num_classes, 1)
        else:
            last = head[-1]
            head[-1] = nn.Conv2d(last.in_channels, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(imagenet_normalize(x))["out"]
'''

HF_SEGFORMER = '''\
class HFSegformer(nn.Module):
    """Hugging Face SegFormer (MiT-B0..B2): logits upsampled to the input size."""

    PRESETS = {  # hidden sizes, depths, decoder width
        "segformer_b0": ([32, 64, 160, 256], [2, 2, 2, 2], 256),
        "segformer_b1": ([64, 128, 320, 512], [2, 2, 2, 2], 256),
        "segformer_b2": ([64, 128, 320, 512], [3, 4, 6, 3], 768),
    }

    def __init__(self, arch: str, num_classes: int, weights: str = "none") -> None:
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        size = arch.rsplit("_", 1)[-1]
        if weights == "ade":
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                f"nvidia/segformer-{size}-finetuned-ade-512-512", num_labels=num_classes,
                ignore_mismatched_sizes=True)
        elif weights == "imagenet":
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                f"nvidia/mit-{size}", num_labels=num_classes)
        else:
            hidden, depths, decoder = self.PRESETS[arch]
            self.model = SegformerForSemanticSegmentation(SegformerConfig(
                num_labels=num_classes, hidden_sizes=hidden, depths=depths,
                decoder_hidden_size=decoder))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(pixel_values=imagenet_normalize(x)).logits
        return nn.functional.interpolate(logits, size=x.shape[-2:], mode="bilinear",
                                         align_corners=False)
'''

UNET = '''\
class UNetBlock(nn.Module):
    """Two 3x3 conv -> norm -> ReLU; residual adds a (projected) skip."""

    def __init__(self, cin: int, cout: int, norm: str = "batch", residual: bool = False) -> None:
        super().__init__()

        def norm_layer(c: int) -> nn.Module:
            return nn.BatchNorm2d(c) if norm == "batch" else nn.GroupNorm(min(8, c), c)

        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), norm_layer(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), norm_layer(cout))
        self.skip = None
        if residual:
            self.skip = (nn.Identity() if cin == cout else nn.Sequential(
                nn.Conv2d(cin, cout, 1, bias=False), norm_layer(cout)))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.body(x)
        if self.skip is not None:
            y = y + self.skip(x)
        return self.act(y)


class UNetAttentionGate(nn.Module):
    """Additive attention gate on a skip connection (Oktay et al., 2018)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        inter = max(channels // 2, 1)
        self.wg = nn.Conv2d(channels, inter, 1)
        self.wx = nn.Conv2d(channels, inter, 1)
        self.psi = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(inter, 1, 1), nn.Sigmoid())

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return skip * self.psi(self.wg(gate) + self.wx(skip))


class UNetUp(nn.Module):
    def __init__(self, cin: int, cout: int, mode: str = "transpose") -> None:
        super().__init__()
        self.up = (nn.ConvTranspose2d(cin, cout, 2, stride=2) if mode == "transpose" else
                   nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear",
                                             align_corners=False),
                                 nn.Conv2d(cin, cout, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class UNet(nn.Module):
    """U-Net family for semantic segmentation: unet, unet_plus_plus, attention_unet,
    resunet. Input [B, C, H, W] (H, W multiples of 2**depth) -> logits
    [B, num_classes, H, W]."""

    def __init__(self, in_channels: int, num_classes: int, depth: int = 4, base: int = 32,
                 variant: str = "unet", norm: str = "batch", upsample: str = "transpose") -> None:
        super().__init__()
        self.variant, self.depth = variant, depth
        widths = [base * 2 ** i for i in range(depth + 1)]
        residual = variant == "resunet"
        self.encoders = nn.ModuleList()
        cin = in_channels
        for w in widths:
            self.encoders.append(UNetBlock(cin, w, norm, residual))
            cin = w
        self.pool = nn.MaxPool2d(2)
        if variant == "unet_plus_plus":
            # nodes[i][j] for j >= 1: conv(cat(x[i][0..j-1], up(x[i+1][j-1])))
            self.nested = nn.ModuleDict()
            self.nested_up = nn.ModuleDict()
            for j in range(1, depth + 1):
                for i in range(depth + 1 - j):
                    self.nested_up[f"{i}_{j}"] = UNetUp(widths[i + 1], widths[i], upsample)
                    self.nested[f"{i}_{j}"] = UNetBlock(widths[i] * (j + 1), widths[i], norm)
        else:
            self.ups = nn.ModuleList(UNetUp(widths[i + 1], widths[i], upsample)
                                     for i in reversed(range(depth)))
            self.decoders = nn.ModuleList(UNetBlock(2 * widths[i], widths[i], norm, residual)
                                          for i in reversed(range(depth)))
            self.gates = (nn.ModuleList(UNetAttentionGate(widths[i])
                                        for i in reversed(range(depth)))
                          if variant == "attention_unet" else None)
        self.head = nn.Conv2d(widths[0], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = []
        for i, enc in enumerate(self.encoders):
            x = enc(x if i == 0 else self.pool(x))
            feats.append(x)
        if self.variant == "unet_plus_plus":
            grid = {(i, 0): f for i, f in enumerate(feats)}
            for j in range(1, self.depth + 1):
                for i in range(self.depth + 1 - j):
                    up = self.nested_up[f"{i}_{j}"](grid[(i + 1, j - 1)])
                    parts = [grid[(i, k)] for k in range(j)] + [up]
                    grid[(i, j)] = self.nested[f"{i}_{j}"](torch.cat(parts, dim=1))
            return self.head(grid[(0, self.depth)])
        x = feats[-1]
        for k, (up, dec) in enumerate(zip(self.ups, self.decoders)):
            skip = feats[self.depth - 1 - k]
            x = up(x)
            if self.gates is not None:
                skip = self.gates[k](x, skip)
            x = dec(torch.cat([x, skip], dim=1))
        return self.head(x)
'''

TIMM_BACKBONE = '''\
class TimmBackbone(nn.Module):
    """timm image model without its classifier -> pooled features [B, F]."""

    def __init__(self, name: str, pretrained: bool = False, freeze: bool = True) -> None:
        super().__init__()
        import timm

        self.model = timm.create_model(name, pretrained=pretrained and not skip_pretrained(),
                                       num_classes=0)
        if freeze and pretrained:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(imagenet_normalize(x))
'''

HF_IMAGE_ENCODER = '''\
class HFImageEncoder(nn.Module):
    """Hugging Face vision transformer (DINOv2, CLIP, SigLIP) -> pooled features [B, F]."""

    PRESETS = {  # repo: (model class, config class, config kwargs)
        "facebook/dinov2-small": ("Dinov2Model", "Dinov2Config",
                                  {"hidden_size": 384, "num_hidden_layers": 12,
                                   "num_attention_heads": 6}),
        "facebook/dinov2-base": ("Dinov2Model", "Dinov2Config", {}),
        "openai/clip-vit-base-patch32": ("CLIPVisionModel", "CLIPVisionConfig",
                                         {"patch_size": 32}),
        "google/siglip-base-patch16-224": ("SiglipVisionModel", "SiglipVisionConfig", {}),
    }

    def __init__(self, model_id: str, pretrained: bool = False, freeze: bool = True) -> None:
        super().__init__()
        import transformers

        model_cls, config_cls, kwargs = self.PRESETS[model_id]
        model_cls = getattr(transformers, model_cls)
        if pretrained:
            self.model = model_cls.from_pretrained(model_id)
            if freeze:
                for p in self.model.parameters():
                    p.requires_grad = False
        else:
            self.model = model_cls(getattr(transformers, config_cls)(**kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=imagenet_normalize(x))
        pooled = getattr(out, "pooler_output", None)
        return pooled if pooled is not None else out.last_hidden_state[:, 0]
'''

VISION_HELPERS: dict[str, str] = {
    "skip_pretrained": SKIP_PRETRAINED,
    "imagenet_normalize": _NORMALIZE,
    "pad_detections": PAD_DETECTIONS,
    "TVDetector": TV_DETECTOR,
    "HFDetr": HF_DETR,
    "TVSegmenter": TV_SEGMENTER,
    "HFSegformer": HF_SEGFORMER,
    "UNet": UNET,
    "TimmBackbone": TIMM_BACKBONE,
    "HFImageEncoder": HF_IMAGE_ENCODER,
}


def register() -> None:
    from ai_made_easy.core.codegen.helpers import PYTORCH_HELPERS

    PYTORCH_HELPERS.update(VISION_HELPERS)


# ------------------------------------------------------------- analytic U-Net cost

def unet_cost(in_channels: int, num_classes: int, depth: int, base: int, variant: str,
              norm: str, upsample: str, height: int, width: int) -> tuple[int, int]:
    """(parameters, multiply-accumulates) of the ``UNet`` helper for one image."""
    widths = [base * 2 ** i for i in range(depth + 1)]
    residual = variant == "resunet"
    params = macs = 0

    def area(level: int) -> int:
        return (height >> level) * (width >> level)

    def block(cin: int, cout: int, level: int, res: bool) -> None:
        nonlocal params, macs
        conv = cin * cout * 9 + cout * cout * 9
        params += conv + 4 * cout  # two norms (2 params each)
        macs += conv * area(level)
        if res and cin != cout:
            params += cin * cout + 2 * cout
            macs += cin * cout * area(level)

    def up(cin: int, cout: int, level: int) -> None:
        nonlocal params, macs
        if upsample == "transpose":
            params += cin * cout * 4 + cout
            macs += cin * cout * 4 * area(level + 1)
        else:
            params += cin * cout + cout
            macs += cin * cout * area(level)

    cin = in_channels
    for level, w in enumerate(widths):
        block(cin, w, level, residual)
        cin = w
    if variant == "unet_plus_plus":
        for j in range(1, depth + 1):
            for i in range(depth + 1 - j):
                up(widths[i + 1], widths[i], i)
                block(widths[i] * (j + 1), widths[i], i, False)
    else:
        for i in reversed(range(depth)):
            up(widths[i + 1], widths[i], i)
            block(2 * widths[i], widths[i], i, residual)
            if variant == "attention_unet":
                inter = max(widths[i] // 2, 1)
                gate = 2 * (widths[i] * inter + inter) + inter + 1
                params += gate
                macs += (2 * widths[i] * inter + inter) * area(i)
    params += widths[0] * num_classes + num_classes
    macs += widths[0] * num_classes * area(0)
    return params, macs
