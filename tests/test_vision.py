"""V3 Phase 2: vision tasks — detection, instance / semantic segmentation, keypoints.

Everything runs offline with random weights and generated or tiny on-disk datasets.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ai_made_easy.core import api, budget
from ai_made_easy.core.codegen import CodegenError
from ai_made_easy.core.fixes import fix_for_issue
from ai_made_easy.core.graph import Graph
from ai_made_easy.core.registry import get_registry
from ai_made_easy.core.tasks import get_task, task_of

HAS_TORCHVISION = importlib.util.find_spec("torchvision") is not None
needs_torchvision = pytest.mark.skipif(not HAS_TORCHVISION, reason="needs torchvision")


@pytest.fixture(scope="module")
def rt():
    from ai_made_easy.core.vision.runtime import namespace

    return namespace()


def design(head: dict, *, shape: str = "3, 64, 64", data: dict | None = None,
           extra: tuple = (), epochs: int = 1, name: str = "vision_case") -> dict:
    data = data or {"type": "data.synthetic_shapes",
                    "params": {"n_images": 12, "image_size": 64, "max_objects": 2}}
    return {"name": name, "nodes": [
        {"id": "in", "type": "core.input", "params": {"shape": shape}, "position": [0, 0]},
        {"id": "h", **head, "position": [200, 0]},
        {"id": "out", "type": "core.output", "params": {}, "position": [400, 0]},
        {"id": "data", **data, "position": [0, 200]},
        {"id": "tr", "type": "train.trainer",
         "params": {"epochs": epochs, "batch_size": 4, "device": "cpu"}, "position": [0, 300]},
        *extra],
        "edges": [{"from": "in/out", "to": "h/in"}, {"from": "h/out", "to": "out/in"}]}


UNET = {"type": "vision.unet", "params": {"num_classes": 4, "depth": 2, "base_channels": 8}}
DETECTOR = {"type": "vision.detector",
            "params": {"arch": "fasterrcnn_mobilenet_v3_large_320_fpn", "num_classes": 3,
                       "weights": "none", "max_detections": 20}}


def messages(data: dict) -> list[str]:
    return [i.message for i in Graph.from_dict(data).validate()]


# ================================================================ registry / tasks

def test_vision_blocks_and_tasks_are_registered():
    reg = get_registry()
    for type_id in ("vision.detector", "vision.instance_segmenter", "vision.keypoint_detector",
                    "vision.hf_detector", "vision.segmenter", "vision.hf_segmenter",
                    "vision.unet", "vision.timm_backbone", "vision.hf_image_encoder",
                    "data.coco", "data.yolo", "data.voc", "data.mask_folder",
                    "data.synthetic_shapes", "vision.loss_dice", "eval.map", "eval.miou"):
        assert reg.has(type_id), type_id
    assert len(reg.all()) >= 370
    for task in ("detection", "instance_segmentation", "keypoints", "semantic_segmentation"):
        t = get_task(task)
        assert t.trainer_kind in ("detection", "segmentation") and t.metrics()
    assert get_task("detection").default_metrics == ("eval.map", "eval.map50")
    hf = reg.get("vision.hf_detector")
    assert hf.requires == ("transformers",) and hf.extra == "vision-tasks"


@pytest.mark.parametrize("sample, task", [("shapes_detection.json", "detection"),
                                          ("shapes_segmentation.json", "semantic_segmentation")])
def test_samples_validate_and_resolve(sample, task):
    data = api.read_sample(sample)
    result = api.validate(data)
    assert result["valid"] and result["issues"] == [], result["issues"]
    assert api.describe_design(data)["task"]["id"] == task


def test_generic_segmentation_net_is_recognised():
    data = {"name": "seg", "nodes": [
        {"id": "in", "type": "core.input", "params": {"shape": "3, 32, 32"}, "position": [0, 0]},
        {"id": "c", "type": "core.conv2d",
         "params": {"out_channels": 4, "kernel_size": 3, "padding": 1}, "position": [0, 0]},
        {"id": "out", "type": "core.output", "params": {}, "position": [0, 0]},
        {"id": "d", "type": "data.synthetic_shapes", "params": {"image_size": 32},
         "position": [0, 0]}],
        "edges": [{"from": "in/out", "to": "c/in"}, {"from": "c/out", "to": "out/in"}]}
    assert task_of(Graph.from_dict(data)).id == "semantic_segmentation"
    data["nodes"][3]["type"] = "data.csv"  # not a vision dataset: no vision task
    data["nodes"][3]["params"] = {}
    assert task_of(Graph.from_dict(data)) is None or \
        task_of(Graph.from_dict(data)).id != "semantic_segmentation"


# ================================================================ helpers

@needs_torchvision
@pytest.mark.parametrize("variant", ["unet", "unet_plus_plus", "attention_unet", "resunet"])
@pytest.mark.parametrize("upsample", ["transpose", "bilinear"])
def test_unet_cost_matches_torch(variant, upsample):
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    from ai_made_easy.core.vision.helpers import VISION_HELPERS, unet_cost

    ns = {"torch": torch, "nn": torch.nn}
    exec(VISION_HELPERS["UNet"], ns)  # noqa: S102
    model = ns["UNet"](3, 5, 3, 8, variant, "batch", upsample).eval()
    with FlopCounterMode(display=False) as counter, torch.no_grad():
        out = model(torch.zeros(1, 3, 32, 48))
    assert tuple(out.shape) == (1, 5, 32, 48)
    params, macs = unet_cost(3, 5, 3, 8, variant, "batch", upsample, 32, 48)
    assert params == sum(p.numel() for p in model.parameters())
    assert 2 * macs == counter.get_total_flops()


def test_unet_shape_rule():
    data = design({"type": "vision.unet", "params": {"depth": 4}}, shape="3, 100, 100")
    assert any("multiples of 16" in m and "resize to 96" in m for m in messages(data))


@needs_torchvision
def test_skip_pretrained_builds_the_trained_architecture(monkeypatch):
    """Loading a run never downloads weights, and the module tree matches training's."""
    import torch
    import torchvision.models._api as tv_api

    from ai_made_easy.core.vision.helpers import VISION_HELPERS

    ns = {"torch": torch, "nn": torch.nn}
    for key in ("skip_pretrained", "pad_detections", "TVDetector"):
        exec(VISION_HELPERS[key], ns)  # noqa: S102
    monkeypatch.setattr(tv_api.WeightsEnum, "get_state_dict", lambda *a, **k: {})
    monkeypatch.setattr(torch.nn.Module, "load_state_dict", lambda *a, **k: None)
    trained = ns["TVDetector"]("fasterrcnn_mobilenet_v3_large_320_fpn", 3, "imagenet")
    monkeypatch.undo()
    monkeypatch.setenv("AIME_SKIP_PRETRAINED", "1")
    rebuilt = ns["TVDetector"]("fasterrcnn_mobilenet_v3_large_320_fpn", 3, "imagenet")
    assert trained.state_dict().keys() == rebuilt.state_dict().keys()
    kinds = [type(m).__name__ for m in trained.modules()]
    assert kinds == [type(m).__name__ for m in rebuilt.modules()]
    assert "FrozenBatchNorm2d" in kinds


# ================================================================ data readers

def _image(path: Path, size=(40, 30)) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (90, 30, 30)).save(path)


def test_read_coco(rt, tmp_path):
    _image(tmp_path / "images/a.jpg")
    _image(tmp_path / "images/b.jpg")
    coco = {"images": [{"id": 1, "file_name": "a.jpg", "width": 40, "height": 30},
                       {"id": 2, "file_name": "b.jpg", "width": 40, "height": 30}],
            "categories": [{"id": 7, "name": "cat"}, {"id": 3, "name": "dog"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 7, "bbox": [2, 3, 10, 5],
                 "segmentation": [[2, 3, 12, 3, 12, 8, 2, 8]], "iscrowd": 0},
                {"id": 2, "image_id": 1, "category_id": 3, "bbox": [0, 0, 4, 4],
                 "iscrowd": 1}]}
    (tmp_path / "ann.json").write_text(json.dumps(coco))
    records, names = rt["read_coco"](tmp_path / "images", tmp_path / "ann.json")
    assert names == ["dog", "cat"]  # sorted by category id -> contiguous classes
    assert records[0]["boxes"] == [[2.0, 3.0, 12.0, 8.0]] and records[0]["labels"] == [1]
    assert records[1]["boxes"] == []  # crowd annotations are skipped
    masks = rt["instance_masks"](records[0], 30, 40)
    assert masks.shape == (1, 30, 40) and 40 <= masks.sum() <= 70
    sem = rt["semantic_mask"](records[0], 30, 40)
    assert set(np.unique(sem)) == {0, 2}


def test_read_yolo_boxes_polygons_and_classes(rt, tmp_path):
    _image(tmp_path / "images/x.png", (100, 50))
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "x.txt").write_text("1 0.5 0.5 0.2 0.4\n0 0.1 0.1 0.3 0.1 0.3 0.3\n")
    (tmp_path / "data.yaml").write_text("path: .\nnames:\n  0: red\n  1: blue\n")
    records, names = rt["read_yolo"](tmp_path / "images", labels, str(tmp_path / "data.yaml"))
    assert names == ["red", "blue"]
    np.testing.assert_allclose(records[0]["boxes"][0], [40, 15, 60, 35])
    np.testing.assert_allclose(records[0]["boxes"][1], [10, 5, 30, 15])
    assert records[0]["labels"] == [1, 0] and records[0]["polygons"][1]
    _r, inferred = rt["read_yolo"](tmp_path / "images", labels, "")
    assert inferred == ["class_0", "class_1"]


def test_read_voc_and_mask_folder(rt, tmp_path):
    root = tmp_path / "VOC"
    _image(root / "JPEGImages/p.jpg")
    (root / "Annotations").mkdir()
    (root / "Annotations/p.xml").write_text(
        "<annotation><filename>p.jpg</filename><object><name>car</name><bndbox>"
        "<xmin>5</xmin><ymin>6</ymin><xmax>20</xmax><ymax>16</ymax></bndbox></object>"
        "</annotation>")
    records, names = rt["read_voc"](root)
    assert names == ["car"] and records[0]["boxes"] == [[4.0, 5.0, 20.0, 16.0]]
    from PIL import Image

    _image(tmp_path / "imgs/q.png")
    (tmp_path / "masks").mkdir()
    mask = np.zeros((30, 40), np.uint8)
    mask[5:10, 5:10] = 2
    Image.fromarray(mask).save(tmp_path / "masks/q.png")
    records, names = rt["read_mask_folder"](tmp_path / "imgs", tmp_path / "masks", 3)
    assert names == ["background", "class_1", "class_2"]
    assert rt["semantic_mask"](records[0], 30, 40).sum() == 50


def test_synthetic_shapes_are_consistent(rt):
    records, names = rt["synthetic_shapes"](10, 64, 3, 1)
    assert names == ["square", "circle", "triangle"] and len(records) == 10
    for r in records:
        assert len(r["boxes"]) == len(r["labels"]) == len(r["masks"]) == len(r["keypoints"])
        for box, m in zip(r["boxes"], r["masks"], strict=True):
            ys, xs = np.nonzero(m)
            assert box == [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
        assert set(np.unique(r["semantic"])) <= {0, 1, 2, 3}
    again, _ = rt["synthetic_shapes"](10, 64, 3, 1)
    assert np.array_equal(again[3]["image"], records[3]["image"])


def test_split_records(rt):
    train, val, test = rt["split_records"](list(range(20)), 0.1, 0.2, 0, True)
    assert (len(train), len(val), len(test)) == (14, 2, 4)
    assert sorted(train + val + test) == list(range(20))


# ================================================================ metrics

def test_average_precision_hand_cases(rt):
    t = {"boxes": np.array([[0, 0, 10, 10], [20, 20, 30, 30]]), "labels": np.array([0, 0])}
    perfect = {"boxes": t["boxes"], "labels": t["labels"], "scores": np.array([0.9, 0.8])}
    assert rt["average_precision"]([perfect], [t])["map"] == pytest.approx(1.0)
    # a confident false positive ranked first: precision 1/2 then 2/3
    fp = {"boxes": np.array([[50, 50, 60, 60], [0, 0, 10, 10], [20, 20, 30, 30]]),
          "labels": np.array([0, 0, 0]), "scores": np.array([0.99, 0.9, 0.8])}
    result = rt["average_precision"]([fp], [t])
    assert result["map50"] == pytest.approx((51 * 2 / 3 + 50 * 2 / 3) / 101, abs=1e-6)
    assert rt["average_precision"]([{"boxes": np.zeros((0, 4)), "labels": np.zeros(0),
                                     "scores": np.zeros(0)}], [t])["map"] == 0.0


def test_average_precision_matches_pycocotools(rt):
    pytest.importorskip("pycocotools")
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    records, names = rt["synthetic_shapes"](30, 96, 4, 3)
    rng = np.random.default_rng(0)
    preds, targets, gt, dt = [], [], {"images": [], "annotations": [], "categories": [
        {"id": i + 1, "name": n} for i, n in enumerate(names)]}, []
    for i, r in enumerate(records):
        t = {"boxes": np.asarray(r["boxes"], float), "labels": np.asarray(r["labels"])}
        b = t["boxes"] + rng.normal(0, 3, t["boxes"].shape)
        b[:, 2:] = np.maximum(b[:, 2:], b[:, :2] + 1)
        extra = rng.uniform(0, 70, (2, 2))
        pb = np.concatenate([b, np.concatenate([extra, extra + 15], 1)])
        p = {"boxes": pb, "labels": np.concatenate([t["labels"], rng.integers(0, 3, 2)]),
             "scores": rng.uniform(0.05, 1, len(pb))}
        preds.append(p)
        targets.append(t)
        gt["images"].append({"id": i + 1, "width": 96, "height": 96})
        for j, (box, lab) in enumerate(zip(t["boxes"], t["labels"], strict=True)):
            gt["annotations"].append({"id": i * 100 + j + 1, "image_id": i + 1,
                                      "category_id": int(lab) + 1, "iscrowd": 0,
                                      "bbox": [*box[:2], *(box[2:] - box[:2])],
                                      "area": float(np.prod(box[2:] - box[:2]))})
        for box, lab, s in zip(p["boxes"], p["labels"], p["scores"], strict=True):
            dt.append({"image_id": i + 1, "category_id": int(lab) + 1, "score": float(s),
                       "bbox": [float(v) for v in (*box[:2], *(box[2:] - box[:2]))]})
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = gt
        coco.createIndex()
        ev = COCOeval(coco, coco.loadRes(dt), "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    ours = rt["average_precision"](preds, targets)
    assert ours["map"] == pytest.approx(ev.stats[0], abs=1e-6)
    assert ours["map50"] == pytest.approx(ev.stats[1], abs=1e-6)
    assert ours["map75"] == pytest.approx(ev.stats[2], abs=1e-6)


def test_segmentation_scores_and_pck(rt):
    true = np.array([[0, 0, 1, 1], [0, 2, 2, 255]])
    pred = np.array([[0, 1, 1, 1], [0, 2, 0, 0]])
    s = rt["segmentation_scores"](rt["confusion"](pred, true, 3))
    assert s["pixel_accuracy"] == pytest.approx(5 / 7)
    assert s["per_class_iou"] == pytest.approx([2 / 4, 2 / 3, 1 / 2])
    t = {"boxes": np.array([[0, 0, 10, 10]]), "keypoints": np.array([[[1, 1, 2], [5, 5, 2],
                                                                     [9, 9, 0]]])}
    p = {"boxes": np.array([[0, 0, 10, 10]]), "keypoints": np.array([[[1, 1, 1], [9, 9, 1],
                                                                     [0, 0, 1]]])}
    assert rt["keypoint_pck"]([p], [t], 0.2) == pytest.approx(0.5)


def test_rle_and_image_decoding(rt):
    import base64

    mask = np.zeros((7, 5), bool)
    mask[2:5, 1:3] = True
    rle = rt["mask_to_rle"](mask)
    assert rle["size"] == [7, 5] and sum(rle["counts"]) == 35
    assert np.array_equal(rt["_decode_rle"](rle, 7, 5), mask)
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (6, 4), (1, 2, 3)).save(buf, format="PNG")
    for item in (buf.getvalue(), base64.b64encode(buf.getvalue()).decode(),
                 "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
                 np.zeros((4, 6, 3)).tolist()):
        assert rt["decode_image"](item).size == (6, 4)


# ================================================================ rules / fixes

def test_detector_must_sit_between_input_and_output():
    data = design(DETECTOR)
    data["nodes"].append({"id": "r", "type": "core.relu", "params": {}, "position": [0, 0]})
    data["edges"] = [{"from": "in/out", "to": "r/in"}, {"from": "r/out", "to": "h/in"},
                     {"from": "h/out", "to": "out/in"}]
    assert any("takes the image straight from the Input" in m for m in messages(data))


def test_dataset_must_provide_the_task_annotations():
    data = design({"type": "vision.keypoint_detector", "params": {"weights": "none"}},
                  data={"type": "data.voc", "params": {}})
    assert any("has no keypoint annotations" in m for m in messages(data))
    data = design(DETECTOR, data={"type": "data.mask_folder", "params": {}})
    assert any("has no boxes" in m for m in messages(data))


def test_class_count_rules_and_quick_fixes():
    data = design({**DETECTOR, "params": {**DETECTOR["params"], "num_classes": 5}})
    graph = Graph.from_dict(data)
    issue = next(i for i in graph.validate() if "set num_classes to 3" in i.message)
    label, _desc, fixed = fix_for_issue(graph, issue)
    assert label == "Set num_classes = 3"
    assert fixed.nodes["h"].params["num_classes"] == 3
    assert not any("num_classes" in i.message for i in fixed.validate())
    seg = design({**UNET, "params": {**UNET["params"], "num_classes": 3}})
    assert any("4 pixel classes (3 classes + background)" in m for m in messages(seg))
    folder = design(UNET, data={"type": "data.mask_folder", "params": {"num_classes": 4}})
    assert not any("pixel classes" in m for m in messages(folder))
    kp = design({"type": "vision.keypoint_detector",
                 "params": {"num_classes": 3, "weights": "none", "num_keypoints": 17}})
    graph = Graph.from_dict(kp)
    issue = next(i for i in graph.validate() if "set num_keypoints to 3" in i.message)
    assert fix_for_issue(graph, issue)[2].nodes["h"].params["num_keypoints"] == 3


def test_detector_size_and_weight_rules():
    small = design(DETECTOR)
    assert any("default anchors span 32–512 px" in m for m in messages(small))
    big = design({**DETECTOR, "params": {**DETECTOR["params"], "min_size": 800}},
                 shape="3, 256, 256")
    graph = Graph.from_dict(big)
    issue = next(i for i in graph.validate() if "enlarges your 256 px images" in i.message)
    assert fix_for_issue(graph, issue)[2].nodes["h"].params["min_size"] == 0
    ok = design(DETECTOR, shape="3, 256, 256",
                data={"type": "data.synthetic_shapes", "params": {"image_size": 256}})
    assert not any("anchors" in m or "enlarges" in m for m in messages(ok))
    ssd = design({"type": "vision.detector",
                  "params": {"arch": "ssd300_vgg16", "num_classes": 3, "weights": "coco"}})
    assert any("cannot change the number of classes" in m for m in messages(ssd))


def test_pipeline_rules_for_vision():
    extra = ({"id": "n", "type": "prep.normalize", "params": {}, "position": [0, 0]},
             {"id": "a", "type": "prep.rand_augment", "params": {}, "position": [0, 0]},
             {"id": "l", "type": "train.loss_cross_entropy", "params": {}, "position": [0, 0]},
             {"id": "m", "type": "eval.miou", "params": {}, "position": [0, 0]},
             {"id": "acc", "type": "eval.accuracy", "params": {}, "position": [0, 0]})
    found = messages(design(DETECTOR, extra=extra))
    assert any("normalize images themselves" in m for m in found)
    assert any("cannot move boxes" in m for m in found)
    assert any("detectors compute their own losses" in m for m in found)
    assert any("Mean IoU does not apply to detection" in m for m in found)
    assert any("classification / regression metric" in m for m in found)
    # classification pipeline lints stay quiet for vision designs
    assert not any("CrossEntropyLoss" in m or "classification output" in m for m in found)


def test_timm_license_warning():
    data = design({"type": "vision.timm_backbone",
                   "params": {"model": "convnextv2_atto", "weights": "imagenet"}})
    assert any("non-commercial use only" in m for m in messages(data))


# ================================================================ budget / profile

def test_budget_costs_for_vision_heads():
    det = Graph.from_dict(api.read_sample("shapes_detection.json"))
    est = budget.estimate(det, "t4")
    # 4.5 GMACs at 800 px scaled to the 256 px input
    assert 15e6 < est.params < 25e6 and est.flops == pytest.approx(2 * 4.494e9 * (256 / 800) ** 2, rel=0.01)
    seg = Graph.from_dict(api.read_sample("shapes_segmentation.json"))
    summary = api.summarize(seg.to_dict())
    unet = next(layer for layer in summary["layers"] if layer["type"] == "vision.unet")
    from ai_made_easy.core.vision.helpers import unet_cost

    params, macs = unet_cost(3, 4, 4, 16, "unet", "batch", "transpose", 128, 128)
    assert unet["params"] == params and unet["flops"] == 2 * macs


def test_profile_vision_datasets(tmp_path):
    from ai_made_easy.core.data.profile import profile_dataset

    p = profile_dataset("data.synthetic_shapes",
                        {"n_images": 20, "image_size": 64, "max_objects": 3, "seed": 0})
    assert p.kind == "annotations" and p.rows == 20 and len(p.classes) == 3
    assert "boxes" in p.summary and any("COCO sizes" in d for d in p.details)
    _image(tmp_path / "images/a.png", (50, 50))
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels/a.txt").write_text("0 30 30 10 10\n")  # pixels, not 0-1
    p = profile_dataset("data.yolo", {"images_dir": "images", "labels_dir": "labels",
                                      "classes": "thing"}, tmp_path)
    assert any("outside their image" in f.message for f in p.findings)
    from PIL import Image

    _image(tmp_path / "imgs/b.png")
    (tmp_path / "masks").mkdir()
    Image.fromarray(np.full((30, 40), 5, np.uint8)).save(tmp_path / "masks/b.png")
    p = profile_dataset("data.mask_folder", {"images_dir": "imgs", "masks_dir": "masks",
                                             "num_classes": 3, "class_names": "",
                                             "ignore_index": 255}, tmp_path)
    assert any("class value 5 but num_classes is 3" in f.message for f in p.findings)
    missing = profile_dataset("data.coco", {"images_dir": "nope", "annotations": "x.json"},
                              tmp_path)
    assert "does not exist" in missing.error


# ================================================================ training scripts

def test_vision_trains_with_pytorch_only():
    from ai_made_easy.core.training.generate import generate_training

    graph = Graph.from_dict(design(UNET))
    with pytest.raises(CodegenError, match="train with PyTorch"):
        generate_training(graph, "keras")
    code = generate_training(graph, "pytorch")
    compile(code, "<vision>", "exec")
    for needle in ("def load_predictor", "def infer", "def lovasz_loss", "eval_samples",
                   "resources: peak_memory_mb"):
        assert needle in code


def _train(data: dict, tmp_path: Path) -> Path:
    from ai_made_easy.core.training.generate import generate_training

    script = tmp_path / "train.py"
    script.write_text(generate_training(Graph.from_dict(data), "pytorch"))
    proc = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True,
                          text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-3000:]
    assert "resources: peak_memory_mb" in proc.stdout
    return tmp_path


def _predict(folder: Path, image: Path) -> list[dict]:
    code = ("import sys, json; sys.path.insert(0, '.'); import train; "
            "train.load_predictor('.'); "
            f"print('OUT ' + json.dumps(train.infer([open({str(image)!r}, 'rb').read()])))")
    env = {**os.environ, "PYTHONWARNINGS": "ignore"}
    proc = subprocess.run([sys.executable, "-c", code], cwd=folder, capture_output=True,
                          text=True, timeout=300, env=env)
    assert "OUT " in proc.stdout, proc.stderr[-3000:]
    return json.loads(proc.stdout.split("OUT ", 1)[1])


@needs_torchvision
def test_semantic_segmentation_trains_and_serves(tmp_path):
    extra = ({"id": "dice", "type": "vision.loss_dice", "params": {}, "position": [0, 0]},
             {"id": "lov", "type": "vision.loss_lovasz", "params": {"weight": 0.5},
              "position": [0, 0]},
             {"id": "flip", "type": "prep.random_flip", "params": {}, "position": [0, 0]})
    run = _train(design(UNET, extra=extra, epochs=2), tmp_path)
    metrics = json.loads((run / "metrics.json").read_text())
    assert {"miou", "dice", "pixel_accuracy", "per_class_iou"} <= set(metrics)
    assert json.loads((run / "classes.json").read_text())[0] == "background"
    sample = next((run / "eval_samples").glob("*.png"))
    out = _predict(run, sample)[0]
    assert out["class_fractions"] and sum(out["class_fractions"].values()) == pytest.approx(
        1.0, abs=1e-3)


@needs_torchvision
@pytest.mark.parametrize("head", [
    DETECTOR,
    {"type": "vision.instance_segmenter", "params": {"num_classes": 3, "weights": "none",
                                                     "max_detections": 10}},
    {"type": "vision.keypoint_detector", "params": {"num_classes": 3, "num_keypoints": 3,
                                                    "weights": "none", "max_detections": 10}},
], ids=["detection", "instance", "keypoints"])
def test_box_tasks_train_and_serve(head, tmp_path):
    extra = ({"id": "flip", "type": "prep.random_flip", "params": {}, "position": [0, 0]},)
    run = _train(design(head, extra=extra), tmp_path)
    metrics = json.loads((run / "metrics.json").read_text())
    assert {"map", "map50", "map75", "per_class_ap"} <= set(metrics)
    task = get_registry().get(head["type"]).meta["task"]
    if task == "instance_segmentation":
        assert "mask_map" in metrics
    if task == "keypoints":
        assert "pck" in metrics
    out = _predict(run, next((run / "eval_samples").glob("*.png")))[0]
    assert set(out) >= {"boxes", "scores", "labels", "label_names"}
    if task == "instance_segmentation" and out["boxes"]:
        assert out["masks"][0]["size"] == [64, 64]


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None,
                    reason="needs transformers (vision-tasks extra)")
def test_transformer_detector_trains(tmp_path):
    head = {"type": "vision.hf_detector",
            "params": {"arch": "conditional_detr", "num_classes": 3, "max_detections": 10}}
    run = _train(design(head), tmp_path)
    assert "map" in json.loads((run / "metrics.json").read_text())


# ================================================================ runs, deploy, server

@needs_torchvision
def test_run_deploy_and_samples_endpoint(tmp_path, isolated_home):
    pytest.importorskip("fastapi")
    from ai_made_easy.core.deploy.package import build_package
    from ai_made_easy.core.runner.manager import RunManager
    from ai_made_easy.core.runs.history import RunHistory

    mgr = RunManager(RunHistory(tmp_path / "runs"))
    api.set_manager(mgr)
    try:
        run_id = mgr.start(Graph.from_dict(design(UNET, name="seg_run")))
        status = mgr.wait(run_id, 600)
        assert status["state"] == "finished", mgr.history.get(run_id).error[-2000:]
        record = mgr.history.get(run_id)
        assert "miou" in record.final_metrics and record.resources.get("step_ms")
        result = build_package(mgr.history.path(run_id), tmp_path / "pkg")
        assert result.verified
        meta = json.loads((tmp_path / "pkg/metadata.json").read_text())
        assert meta["task"] == "semantic_segmentation" and meta["input_kind"] == "images"
        assert "/predict/image" in (tmp_path / "pkg/README.md").read_text()
        samples = api.run_samples(run_id)
        assert samples["samples"][0]["data_url"].startswith("data:image/png;base64,")
        assert samples["per_class_metric"] == "IoU"
        from fastapi.testclient import TestClient

        from ai_made_easy.server.app import create_app

        with TestClient(create_app(token="", projects_dir=tmp_path)) as client:
            body = client.get(f"/api/runs/{run_id}/samples").json()
            assert body["classes"][0] == "background" and body["samples"]
            tasks = {t["id"] for t in client.get("/api/tasks").json()["tasks"]}
            assert {"detection", "semantic_segmentation", "keypoints"} <= tasks
    finally:
        api.set_manager(None)


@needs_torchvision
def test_augmentation_preview_for_vision(tmp_path):
    data = api.read_sample("shapes_detection.json")
    data["nodes"] += [{"id": "rot", "type": "prep.random_rotation", "params": {},
                       "position": [0, 0]},
                      {"id": "aa", "type": "prep.rand_augment", "params": {},
                       "position": [0, 0]}]
    result = api.augmentation_preview(data, str(tmp_path), images=1, variants=2)
    item = result["images"][0]
    assert item["eval_shape"] == [3, 256, 256] and len(item["train"]) == 2
    assert all(Path(p).exists() for p in (item["original"], item["eval"], *item["train"]))
    assert "v2.RandomRotation(15.0)" in result["train_transforms"]
    assert any("RandAugment is skipped" in note for note in result["skipped"])


def test_overlay_draws_keypoints_with_or_without_visibility(rt):
    img = np.zeros((20, 20, 3), np.uint8)
    for pts in ([[[5, 5]]], [[[5, 5, 2]]]):
        out = np.asarray(rt["overlay"](img, None, {"boxes": np.zeros((0, 4)), "labels": [],
                                                   "keypoints": np.asarray(pts, float)}))
        assert out[5, 5].any() and not out[15, 15].any()
    hidden = np.asarray(rt["overlay"](img, None, {"boxes": np.zeros((0, 4)), "labels": [],
                                                  "keypoints": np.asarray([[[5, 5, 0]]])}))
    assert not hidden.any()


def test_profile_reports_missing_packages(monkeypatch):
    from ai_made_easy.core.data import profile as prof

    def broken(params, base):
        raise ModuleNotFoundError("No module named 'PIL'", name="PIL")

    monkeypatch.setitem(prof.PROFILERS, "data.synthetic_shapes", broken)
    result = prof.profile_dataset("data.synthetic_shapes", {})
    assert result.error == "profiling this dataset needs pillow: pip install pillow"
