# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Model factory + ONNX export wrappers for the NF vision trainer.

Only permissive dependencies: torchvision (BSD-3) and timm (Apache-2.0).
Architecture names are capability names, not product brands; the catalog below
is what the UI offers.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Catalog (single source for the `catalog` entrypoint and the UI)
# ---------------------------------------------------------------------------

CLASSIFICATION_ARCHS = [
    {"id": "resnet18", "label": "ResNet-18", "size": 224, "params_m": 11.7},
    {"id": "resnet34", "label": "ResNet-34", "size": 224, "params_m": 21.8},
    {"id": "resnet50", "label": "ResNet-50", "size": 224, "params_m": 25.6},
    {"id": "resnet101", "label": "ResNet-101", "size": 224, "params_m": 44.5},
    {"id": "efficientnet_b0", "label": "EfficientNet-B0", "size": 224, "params_m": 5.3},
    {"id": "efficientnet_b2", "label": "EfficientNet-B2", "size": 260, "params_m": 9.1},
    {"id": "mobilenet_v3_large", "label": "MobileNetV3-Large", "size": 224, "params_m": 5.5},
    {"id": "convnext_tiny", "label": "ConvNeXt-Tiny", "size": 224, "params_m": 28.6},
    {"id": "swin_t", "label": "Swin-T", "size": 224, "params_m": 28.3},
    {"id": "vit_b_16", "label": "ViT-B/16", "size": 224, "params_m": 86.6},
]

DETECTION_ARCHS = [
    {"id": "fasterrcnn_mobilenet_v3_large_fpn", "label": "Faster R-CNN (MobileNetV3)", "size": 640, "params_m": 19.3},
    {"id": "fasterrcnn_resnet50_fpn_v2", "label": "Faster R-CNN (ResNet-50 v2)", "size": 640, "params_m": 43.4},
    {"id": "retinanet_resnet50_fpn_v2", "label": "RetinaNet (ResNet-50 v2)", "size": 640, "params_m": 38.1},
    {"id": "fcos_resnet50_fpn", "label": "FCOS (ResNet-50)", "size": 640, "params_m": 32.2},
]

INSTANCE_SEG_ARCHS = [
    {"id": "maskrcnn_resnet50_fpn_v2", "label": "Mask R-CNN (ResNet-50 v2)", "size": 640, "params_m": 46.4},
]

SEMANTIC_SEG_ARCHS = [
    {"id": "smp:Unet", "label": "U-Net (ResNet-34 encoder)", "size": 512, "params_m": 24.4,
     "encoder": "resnet34"},
    {"id": "smp:DeepLabV3Plus", "label": "DeepLabV3+ (ResNet-34 encoder)", "size": 512, "params_m": 26.7,
     "encoder": "resnet34"},
    {"id": "deeplabv3_mobilenet_v3_large", "label": "DeepLabV3 (MobileNetV3)", "size": 512, "params_m": 11.0},
    {"id": "deeplabv3_resnet50", "label": "DeepLabV3 (ResNet-50)", "size": 512, "params_m": 39.6},
    {"id": "fcn_resnet50", "label": "FCN (ResNet-50)", "size": 512, "params_m": 35.3},
    {"id": "lraspp_mobilenet_v3_large", "label": "LR-ASPP (MobileNetV3)", "size": 512, "params_m": 3.2},
]

POSE_ARCHS = [
    {"id": "keypointrcnn_resnet50_fpn", "label": "Keypoint R-CNN (ResNet-50)", "size": 640, "params_m": 59.0},
]

TASK_ARCHS = {
    "classification": CLASSIFICATION_ARCHS,
    "multilabel": CLASSIFICATION_ARCHS,
    "detection": DETECTION_ARCHS,
    "instance_seg": INSTANCE_SEG_ARCHS,
    "semantic_seg": SEMANTIC_SEG_ARCHS,
    "pose": POSE_ARCHS,
}

DEFAULT_ARCH = {
    "classification": "resnet50",
    "multilabel": "resnet50",
    "detection": "fasterrcnn_mobilenet_v3_large_fpn",
    "instance_seg": "maskrcnn_resnet50_fpn_v2",
    "semantic_seg": "deeplabv3_mobilenet_v3_large",
    "pose": "keypointrcnn_resnet50_fpn",
}


def catalog() -> dict[str, Any]:
    """Machine-readable capability catalog (consumed by the backend/UI)."""
    return {
        "framework": "torch",
        "tasks": [
            {
                "id": task,
                "architectures": archs,
                "default": DEFAULT_ARCH[task],
            }
            for task, archs in TASK_ARCHS.items()
        ],
        "entrypoints": ["train", "evaluate", "export", "catalog"],
        "contract": {
            "env": ["INPUT_URI", "OUTPUT_DIR", "CONFIG_JSON", "ARTIFACT_ID"],
            "artifacts": ["<id>.onnx", "<id>.metadata.json", "<id>.metrics.json", "<id>.pt"],
        },
        "licenses": {"torchvision": "BSD-3-Clause", "timm": "Apache-2.0"},
    }


# ---------------------------------------------------------------------------
# Head replacement (torchvision classifiers)
# ---------------------------------------------------------------------------

def _replace_classifier(model: Any, num_classes: int) -> None:
    import torch.nn as nn

    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier"):
        clf = model.classifier
        if isinstance(clf, nn.Linear):
            model.classifier = nn.Linear(clf.in_features, num_classes)
        elif isinstance(clf, nn.Sequential) and isinstance(clf[-1], nn.Linear):
            clf[-1] = nn.Linear(clf[-1].in_features, num_classes)
    elif hasattr(model, "heads") and hasattr(model.heads, "head"):
        head = model.heads.head
        model.heads.head = nn.Linear(head.in_features, num_classes)
    else:  # pragma: no cover - torchvision always exposes one of the above
        raise ValueError(f"cannot replace the classifier head of {type(model).__name__}")


def _load_weights_file(model: Any, path: str) -> None:
    """Load a base/checkpoint state dict, keeping the model's own shape rules.

    Head tensors whose shape differs (a different class count) are skipped so
    incremental learning keeps the backbone and re-initialises the head
    (ADR-014 D-07); `strict=False` alone would raise on those size mismatches.
    """
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    if not isinstance(state, dict):
        raise ValueError(f"unsupported checkpoint format in {path}")
    own = model.state_dict()
    usable = {
        key: value
        for key, value in state.items()
        if key in own and tuple(own[key].shape) == tuple(value.shape)
    }
    skipped = sorted(set(state) - set(usable))
    model.load_state_dict(usable, strict=False)
    logger.info(
        "loaded %d/%d base-weight tensors from %s (%d skipped, mostly head/optimizer)",
        len(usable), len(state), path, len(skipped),
    )
    if not usable:
        raise ValueError(f"{path} shares no tensor names with the {type(model).__name__} model")


def _fix_detection_transform(model: Any, size: int) -> None:
    """Make the model consume already-normalised, already-letterboxed input."""
    tr = getattr(model, "transform", None)
    if tr is None:
        return
    tr.min_size = (int(size),)
    tr.max_size = int(size)
    tr.image_mean = [0.0, 0.0, 0.0]
    tr.image_std = [1.0, 1.0, 1.0]


def build_model(
    task: str,
    arch: str,
    num_classes: int,
    size: int,
    pretrained: bool = True,
    weights_path: str = "",
    num_keypoints: int = 17,
    encoder_name: str = "resnet34",
) -> Any:
    """Build a model for *task* and optionally load base weights into it.

    With `weights_path` the framework's own pretrained download is skipped —
    the checkpoint is the weight source (ADR-014 D5: no weights baked in).
    """
    model = _construct_model(
        task, arch, num_classes, size,
        pretrained=bool(pretrained) and not weights_path,
        num_keypoints=num_keypoints,
        encoder_name=encoder_name,
    )
    if weights_path:
        _load_weights_file(model, weights_path)
    return model


def _replace_detection_heads(model: Any, arch: str, num_classes: int,
                             num_keypoints: int = 17) -> None:
    """Replace the COCO-91 prediction heads of a pretrained detector.

    The torchvision detection constructors do NOT accept ``num_classes``
    together with ``weights=DEFAULT`` (they raise "expected value 91"), so the
    model must be built with its stock 91-class heads loaded, then the heads
    swapped for ``num_classes`` (background included, e.g. 81 for COCO-80).
    """
    from torchvision.models.detection import (
        faster_rcnn, fcos, keypoint_rcnn, mask_rcnn, retinanet,
    )

    if arch.startswith("fasterrcnn") or arch.startswith("maskrcnn") or arch.startswith("keypointrcnn"):
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
        if arch.startswith("maskrcnn"):
            model.roi_heads.mask_predictor = mask_rcnn.MaskRCNNPredictor(
                model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, num_classes
            )
        if arch.startswith("keypointrcnn"):
            model.roi_heads.keypoint_predictor = keypoint_rcnn.KeypointRCNNPredictor(
                model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels,
                num_keypoints,
            )
    elif arch.startswith("retinanet"):
        # RetinaNet bundles classification + regression heads; replace the
        # whole head (official fine-tuning recipe).
        model.head = retinanet.RetinaNetHead(
            in_channels=model.backbone.out_channels,
            num_anchors=model.head.classification_head.num_anchors,
            num_classes=num_classes,
        )
    elif arch.startswith("fcos"):
        model.head.classification_head = fcos.FCOSClassificationHead(
            in_channels=model.backbone.out_channels,
            num_anchors=model.head.classification_head.num_anchors,
            num_classes=num_classes,
        )


def _construct_model(
    task: str,
    arch: str,
    num_classes: int,
    size: int,
    pretrained: bool = True,
    num_keypoints: int = 17,
    encoder_name: str = "resnet34",
) -> Any:
    """Build a torchvision (or timm/smp) model for *task* with a fresh head."""
    from torchvision import models

    task = task.lower()
    pretrained = bool(pretrained)

    if task in ("classification", "multilabel"):
        ctor = getattr(models, arch, None)
        if ctor is not None and arch in {a["id"] for a in CLASSIFICATION_ARCHS}:
            weights = "DEFAULT" if pretrained else None
            try:
                model = ctor(weights=weights)
            except Exception as e:  # noqa: BLE001 - offline: fall back to scratch
                logger.warning("pretrained %s unavailable (%s); training from scratch", arch, e)
                model = ctor(weights=None)
            _replace_classifier(model, num_classes)
            return model
        # timm fallback for architectures outside the torchvision set.
        try:
            import timm
        except ImportError as e:  # pragma: no cover
            raise ValueError(
                f"unknown architecture '{arch}' (not in torchvision catalog and timm missing)"
            ) from e
        return timm.create_model(arch, pretrained=pretrained, num_classes=num_classes)

    if task == "detection":
        ctor = getattr(models.detection, arch, None)
        if ctor is None:
            raise ValueError(f"unknown detection architecture '{arch}'")
        # Build with stock heads so pretrained weights load (the constructors
        # reject num_classes != 91 together with weights), then swap heads.
        try:
            model = ctor(weights="DEFAULT" if pretrained else None)
        except Exception as e:  # noqa: BLE001
            logger.warning("pretrained %s unavailable (%s); training from scratch", arch, e)
            model = ctor(weights=None)
        _replace_detection_heads(model, arch, num_classes)
        _fix_detection_transform(model, size)
        return model

    if task == "instance_seg":
        ctor = getattr(models.detection, arch, None)
        if ctor is None:
            raise ValueError(f"unknown instance-seg architecture '{arch}'")
        try:
            model = ctor(weights="DEFAULT" if pretrained else None)
        except Exception as e:  # noqa: BLE001
            logger.warning("pretrained %s unavailable (%s); training from scratch", arch, e)
            model = ctor(weights=None)
        _replace_detection_heads(model, arch, num_classes)
        _fix_detection_transform(model, size)
        return model

    if task == "pose":
        ctor = getattr(models.detection, arch, None)
        if ctor is None:
            raise ValueError(f"unknown pose architecture '{arch}'")
        try:
            model = ctor(weights="DEFAULT" if pretrained else None)
        except Exception as e:  # noqa: BLE001
            logger.warning("pretrained %s unavailable (%s); training from scratch", arch, e)
            model = ctor(weights=None)
        _replace_detection_heads(model, arch, num_classes, num_keypoints)
        _fix_detection_transform(model, size)
        return model

    if task == "semantic_seg" and arch.startswith("smp:"):
        import segmentation_models_pytorch as smp

        return smp.create_model(
            arch[4:],
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            classes=num_classes,
            in_channels=3,
        )

    if task == "semantic_seg":
        ctor = getattr(models.segmentation, arch, None)
        if ctor is None:
            raise ValueError(f"unknown semantic-seg architecture '{arch}'")
        weights = "DEFAULT" if pretrained else None
        try:
            model = ctor(weights=weights, num_classes=num_classes)
        except Exception as e:  # noqa: BLE001
            logger.warning("pretrained %s unavailable (%s); training from scratch", arch, e)
            model = ctor(weights=None, num_classes=num_classes)
        return model

    raise ValueError(f"unsupported task '{task}'")


def freeze_backbone(model: Any) -> None:
    """Freeze every parameter except the task head."""
    head_markers = ("fc", "classifier", "heads", "box_predictor", "mask_predictor", "keypoint_predictor")
    for name, param in model.named_parameters():
        param.requires_grad = any(marker in name for marker in head_markers)


# ---------------------------------------------------------------------------
# ONNX export wrappers
# ---------------------------------------------------------------------------

def _pad_rpn_proposals(model: Any, k: int = 1000) -> None:
    """Force the RPN to emit exactly *k* proposals per image (export only).

    The torchvision box coder/ROI heads reshape tensors with the proposal
    count obtained via ``Tensor.size(0)``, which the TorchScript ONNX tracer
    bakes as a constant from the dummy input. At run time a real image usually
    has a different count (NMS removes duplicates; an all-zeros dummy yields
    almost none), so ONNX Runtime fails deep in roi_heads with
    ``Reshape ... input {N, 4*C-1} requested {-1,4}`` (or silently computes
    garbage when the numbers happen to divide). Padding/truncating to the RPN
    ``post_nms_top_n`` ceiling on both trace and run time makes the count a
    fixed constant. Duplicate dummy proposals collapse in per-class NMS and
    zero-area ones are removed by min_size/score filters.
    """
    import torch

    rpn = getattr(model, "rpn", None)
    if rpn is None:  # single-stage detectors (retinanet/fcos) need no patch
        return
    orig_forward = rpn.forward

    def forward(images, features, targets=None):  # type: ignore[no-untyped-def]
        proposals, losses = orig_forward(images, features, targets)
        fixed = []
        for p in proposals:
            q = p[:k]
            # Pad with zero boxes up to k using *symbolic* shapes only: no
            # python branch on the count (the tracer would bake its one
            # path) and no advanced-indexing gather (exports to a broken
            # Gather graph). Zero-area proposals are dropped later by
            # remove_small_boxes/score threshold, so they add no detections.
            pad = k - q.shape[0]
            filler = q.new_zeros((pad, 4))
            fixed.append(torch.cat([q, filler], dim=0))
        return fixed, losses

    rpn.forward = forward


class DetectionExport(__import__("torch").nn.Module):
    """Export wrapper: one tensor in, flat top-k tensors out.

    torchvision detectors return a list of variable-length dicts, which ONNX
    cannot represent. The wrapper bakes NMS + top-k (the same output shape the
    inference adapter expects: boxes/scores/labels[/masks|keypoints]).
    """

    def __init__(self, model: Any, size: int, topk: int = 100, task: str = "detection"):
        super().__init__()
        # Two-stage models (faster/mask/keypoint-rcnn): freeze the proposal
        # count before tracing so the exported graph validates on every image.
        _pad_rpn_proposals(model, k=1000)
        self.model = model
        self.size = int(size)
        self.topk = int(topk)
        self.task = task

    def forward(self, images):
        import torch
        import torch.nn.functional as F

        out = self.model(images)[0]
        boxes = out["boxes"][: self.topk]
        scores = out["scores"][: self.topk]
        labels = out["labels"][: self.topk]
        n = boxes.shape[0]
        pad = self.topk - n

        def pad_rows(t):
            if pad <= 0:
                return t
            return torch.cat([t, t.new_zeros((pad, *t.shape[1:]))], dim=0)

        boxes = pad_rows(boxes)
        scores = pad_rows(scores)
        # torchvision reserves 0 for background; emit 0-based class ids so the
        # ONNX contract matches classification (`labels` indexes the manifest).
        labels = (labels - 1).clamp(min=0)
        labels = pad_rows(labels)

        if self.task == "instance_seg":
            masks = out.get("masks")
            if masks is None:
                masks = images.new_zeros((0, 1, self.size, self.size))
            else:
                masks = masks[: self.topk]
                if masks.shape[-1] != images.shape[-1]:
                    masks = F.interpolate(masks, size=images.shape[-2:], mode="bilinear", align_corners=False)
                masks = pad_rows(masks)
            return boxes, scores, labels, masks

        if self.task == "pose":
            kps = out.get("keypoints")
            if kps is None:
                kps = images.new_zeros((0, 17, 3))
            else:
                kps = kps[: self.topk]
                kps = pad_rows(kps)
            return boxes, scores, labels, kps

        return boxes, scores, labels


class ClassifierExport(__import__("torch").nn.Module):
    """Export wrapper: logits only (features are not needed downstream)."""

    def __init__(self, model: Any):
        super().__init__()
        self.model = model

    def forward(self, images):
        out = self.model(images)
        # timm models can return a dict in feature mode; normalise to logits.
        if isinstance(out, dict):
            out = out.get("logits", next(iter(out.values())))
        return out


class SegmentationExport(__import__("torch").nn.Module):
    """Export wrapper: logits tensor only (dict output -> tensor)."""

    def __init__(self, model: Any):
        super().__init__()
        self.model = model

    def forward(self, images):
        out = self.model(images)
        return out["out"] if isinstance(out, dict) else out


def export_onnx(
    model: Any,
    task: str,
    size: int,
    path: str,
    opset: int = 17,
    topk: int = 100,
) -> None:
    """Export *model* to ONNX with the shape contract of each task family."""
    import torch

    # ONNX is always exported on CPU (a portable artifact); a model trained on
    # cuda/xpu must be moved over or the dummy CPU tensor hits XPU/CUDA weights.
    device = next(model.parameters(), torch.nn.Parameter(torch.zeros(1))).device
    model.eval()
    model.cpu()
    dummy = torch.zeros(1, 3, size, size)

    if task in ("detection", "instance_seg", "pose"):
        wrapper = DetectionExport(model, size=size, topk=topk, task=task)
        names = ["boxes", "scores", "labels"]
        if task == "instance_seg":
            names.append("masks")
        elif task == "pose":
            names.append("keypoints")
        torch.onnx.export(
            wrapper,
            (dummy,),
            path,
            input_names=["images"],
            output_names=names,
            opset_version=opset,
            do_constant_folding=True,
            # TorchScript exporter: deterministic output names and no
            # onnxscript dependency in the runtime image.
            dynamo=False,
            dynamic_axes={"images": {0: "batch", 2: "height", 3: "width"},
                          "boxes": {0: "batch"}, "scores": {0: "batch"}, "labels": {0: "batch"}},
        )
    else:
        wrapper = ClassifierExport(model) if task in ("classification", "multilabel") else SegmentationExport(model)
        torch.onnx.export(
            wrapper,
            (dummy,),
            path,
            input_names=["images"],
            output_names=["logits"],
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
            dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        )
    logger.info("exported %s -> %s", task, path)
    model.to(device)
