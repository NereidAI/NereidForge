# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Dataset readers for the NF vision trainer (COCO / YOLO-txt / folders).

One reader covers the canonical formats summarized in
``app/builders/README.md``:

  classification / multilabel : <split>/<class>/*.jpg or labels.json
  detection / instance_seg    : annotations.json (COCO) or data.yaml + YOLO txt
  semantic_seg                : images/ + masks/ (PNG, class indices)
  pose                        : COCO keypoints (annotations.json)

Nothing here downloads weights or touches the network: the caller passes a
local dataset root prepared by common.download_dataset().
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Optional

from common import resolve_augmentation

# ImageNet statistics: the single normalisation contract used by training,
# export and the inference adapters (onnx_predict.py).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def _photometric_jitter(tensor: Any, aug: dict[str, Any]) -> Any:
    """HSV/contrast jitter on a float CHW tensor in [0, 1] (pre-normalisation)."""
    from torchvision.transforms import functional as TF

    strength = float(aug.get("hsv_s", 0.0) or 0.0)
    value = float(aug.get("hsv_v", 0.0) or 0.0)
    hue = float(aug.get("hsv_h", 0.0) or 0.0)
    if hue > 0:
        tensor = TF.adjust_hue(tensor, random.uniform(-hue, hue))
    if strength > 0:
        tensor = TF.adjust_saturation(tensor, 1.0 + random.uniform(-strength, strength))
    if value > 0:
        tensor = TF.adjust_brightness(tensor, 1.0 + random.uniform(-value, value))
        tensor = TF.adjust_contrast(tensor, 1.0 + random.uniform(-value, value))
    return tensor


def _images_in(path: str) -> list[str]:
    out: list[str] = []
    for root, _dirs, files in os.walk(path):
        for name in files:
            if name.lower().endswith(IMG_EXTS):
                out.append(os.path.join(root, name))
    return sorted(out)


def _load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _load_yaml(path: str) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data if isinstance(data, dict) else {}


def _resolve_labels(names: Any) -> list[str]:
    """Normalise data.yaml `names` (list or index-keyed dict) to a list."""
    if isinstance(names, list):
        return [str(n) for n in names]
    if isinstance(names, dict):
        items = sorted(names.items(), key=lambda kv: int(kv[0]))
        return [str(v) for _k, v in items]
    return []


def _is_video_frame(name: str) -> str:
    """"clip_000123.jpg" -> "clip" so frames of one clip never split."""
    stem = os.path.splitext(os.path.basename(name))[0]
    head, sep, tail = stem.rpartition("_")
    return head if sep and tail.isdigit() else f"__file__{stem}"


def _split(items: list[Any], ratio: float, seed: int, key=lambda x: x) -> tuple[list, list]:
    """Deterministic grouped split (groups by key; order shuffled)."""
    groups: dict[str, list[Any]] = {}
    for item in items:
        groups.setdefault(str(key(item)), []).append(item)
    names = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(names)
    n_train = max(1, int(round(len(items) * ratio)))
    train: list[Any] = []
    val: list[Any] = []
    for name in names:
        (train if len(train) < n_train else val).extend(groups[name])
    if not val and len(train) > 1:
        val = [train.pop()]
    return train, val


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

class ClassificationDataset:
    """ImageFolder-style classifier dataset (single or multi label)."""

    def __init__(self, samples: list[tuple[str, Any]], labels: list[str],
                 size: int, train: bool, multi_label: bool = False,
                 aug: Optional[dict[str, Any]] = None):
        self.samples = samples
        self.labels = labels
        self.size = size
        self.train = train
        self.multi_label = multi_label
        self.aug = aug or resolve_augmentation(None)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        import torch
        from PIL import Image
        from torchvision import transforms

        path, target = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.train:
            aug = self.aug
            scale_jitter = float(aug.get("scale", 0.0) or 0.0)
            degrees = float(aug.get("degrees", 0.0) or 0.0)
            shear = float(aug.get("shear", 0.0) or 0.0)
            translate = float(aug.get("translate", 0.0) or 0.0)
            s_jitter = float(aug.get("hsv_s", 0.0) or 0.0)
            v_jitter = float(aug.get("hsv_v", 0.0) or 0.0)
            h_jitter = float(aug.get("hsv_h", 0.0) or 0.0)
            ops = [
                transforms.RandomResizedCrop(
                    self.size, scale=(max(0.3, 1.0 - scale_jitter), 1.0)
                )
            ]
            if float(aug.get("fliplr", 0.0) or 0.0) > 0:
                ops.append(transforms.RandomHorizontalFlip(float(aug["fliplr"])))
            if float(aug.get("flipud", 0.0) or 0.0) > 0:
                ops.append(transforms.RandomVerticalFlip(float(aug["flipud"])))
            if degrees or shear or translate:
                ops.append(transforms.RandomAffine(
                    degrees=degrees,
                    translate=(translate, translate),
                    shear=shear,
                    scale=(max(0.3, 1.0 - scale_jitter * 0.5), 1.0 + scale_jitter * 0.5),
                ))
            if s_jitter or v_jitter or h_jitter:
                ops.append(transforms.ColorJitter(
                    brightness=v_jitter, contrast=v_jitter,
                    saturation=s_jitter, hue=h_jitter,
                ))
            ops += [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
            tf = transforms.Compose(ops)
        else:
            tf = transforms.Compose([
                transforms.Resize(int(self.size * 1.14)),
                transforms.CenterCrop(self.size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        tensor = tf(img)
        if self.multi_label:
            vec = torch.zeros(len(self.labels), dtype=torch.float32)
            for lbl in target:
                vec[lbl] = 1.0
            return tensor, vec
        return tensor, torch.tensor(int(target), dtype=torch.long)


def _folder_samples(root: str) -> tuple[list[tuple[str, int]], list[str]]:
    classes = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")
    )
    if not classes:
        return [], []
    samples: list[tuple[str, int]] = []
    for idx, cls in enumerate(classes):
        for path in _images_in(os.path.join(root, cls)):
            samples.append((path, idx))
    return samples, classes


def build_classification(root: str, cfg: dict[str, Any], train: bool):
    size = int(cfg.get("image_size") or cfg.get("imgsz") or 224)
    multi = str(cfg.get("task", "")).lower() in ("multilabel", "multi_label") or bool(
        cfg.get("multi_label")
    )
    split_name = "train" if train else "val"
    for split_dir in (os.path.join(root, split_name), os.path.join(root, "train" if train else "validation")):
        if os.path.isdir(split_dir):
            samples, labels = _folder_samples(split_dir)
            if samples:
                return ClassificationDataset(
                    samples, labels, size, train, multi,
                    aug=resolve_augmentation(cfg),
                ), labels

    # Flat layout: labels.json / labels.csv + one directory of images.
    labels_file = next(
        (p for p in (os.path.join(root, "labels.json"), os.path.join(root, "classes.txt"))
         if os.path.isfile(p)), None,
    )
    labels: list[str] = []
    mapping: dict[str, Any] = {}
    if labels_file and labels_file.endswith(".json"):
        raw = _load_json(labels_file)
        if isinstance(raw, dict):
            mapping = raw
            labels = sorted({str(v) for v in raw.values()}) if not multi else sorted({
                str(x) for v in raw.values() for x in (v if isinstance(v, list) else [v])
            })
        elif isinstance(raw, list):
            labels = [str(x) for x in raw]
    elif labels_file:
        labels = [ln.strip() for ln in open(labels_file, encoding="utf-8") if ln.strip()]

    index = {name: i for i, name in enumerate(labels)}
    samples = []
    for path in _images_in(root):
        rel = os.path.relpath(path, root)
        raw = mapping.get(rel) or mapping.get(os.path.basename(path))
        if raw is None:
            # <root>/<class>/<file> fallback
            parent = os.path.basename(os.path.dirname(path))
            if parent in index:
                raw = parent
            else:
                continue
        if multi:
            vals = raw if isinstance(raw, list) else [raw]
            samples.append((path, [index[str(v)] for v in vals if str(v) in index]))
        else:
            if str(raw) not in index:
                continue
            samples.append((path, index[str(raw)]))

    if not samples:
        raise RuntimeError(f"no classification samples found under {root} (expected <split>/<class>/*.jpg)")
    train_samples, val_samples = _split(
        samples, float(cfg.get("split_ratio", 0.8)), int(cfg.get("seed", 42)),
        key=lambda s: _is_video_frame(s[0]),
    )
    chosen = train_samples if train else val_samples
    return ClassificationDataset(
        chosen, labels, size, train, multi, aug=resolve_augmentation(cfg)
    ), labels


# ---------------------------------------------------------------------------
# Detection / instance segmentation / pose (COCO or YOLO txt)
# ---------------------------------------------------------------------------

class DetectionDataset:
    """Detection-style dataset: returns (normalized image tensor, target)."""

    def __init__(self, records: list[dict[str, Any]], size: int, train: bool,
                 task: str = "detection", num_keypoints: int = 0,
                 mask_dir: str = "", hflip_prob: Optional[float] = None,
                 aug: Optional[dict[str, Any]] = None):
        self.records = records
        self.size = size
        self.train = train
        self.task = task
        self.num_keypoints = num_keypoints
        self.mask_dir = mask_dir
        self.aug = aug or resolve_augmentation(None)
        # Backwards-compatible knob: an explicit hflip_prob overrides the policy.
        self.hflip_prob = (
            float(hflip_prob) if hflip_prob is not None
            else float(self.aug.get("fliplr", 0.0) or 0.0)
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        import torch
        from PIL import Image
        from torchvision.transforms import functional as TF

        rec = self.records[index]
        img = Image.open(rec["path"]).convert("RGB")
        w0, h0 = img.size
        boxes = torch.tensor(rec["boxes"], dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(rec["labels"], dtype=torch.int64)
        area = (
            (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
            * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
            if len(boxes) else torch.zeros(0)
        )
        target: dict[str, Any] = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index]),
            "area": area,
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }
        keypoints = rec.get("keypoints")
        if self.task == "pose" and keypoints is not None:
            kp = torch.tensor(keypoints, dtype=torch.float32).reshape(-1, self.num_keypoints, 3)
            target["keypoints"] = kp
        masks = None
        if self.task == "instance_seg" and rec.get("masks"):
            loaded = []
            for entry in rec["masks"]:
                loaded.append(Image.open(entry) if isinstance(entry, str) else Image.fromarray(entry))
            masks = loaded

        # Letterbox to size x size: identical geometry to onnx_predict.py at
        # inference time, so exported boxes need no second resize convention.
        scale = min(self.size / h0, self.size / w0)
        nh, nw = int(round(h0 * scale)), int(round(w0 * scale))
        pad_x, pad_y = (self.size - nw) // 2, (self.size - nh) // 2
        img = TF.resize(img, [nh, nw])
        canvas = Image.new("RGB", (self.size, self.size), (114, 114, 114))
        canvas.paste(img, (pad_x, pad_y))
        tensor = TF.to_tensor(canvas)
        if self.train:
            tensor = _photometric_jitter(tensor, self.aug)
        tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)

        if len(boxes):
            boxes = boxes * scale
            boxes[:, 0::2] += pad_x
            boxes[:, 1::2] += pad_y
            target["boxes"] = boxes
            target["area"] = area * scale * scale
        if masks is not None:
            resized = []
            for m in masks:
                m = TF.resize(m, [nh, nw], interpolation=TF.InterpolationMode.NEAREST)
                import numpy as np

                arr = (np.asarray(m) > 0).astype("uint8")
                resized.append(torch.from_numpy(arr))
            if resized:
                target["masks"] = torch.stack(resized).unsqueeze(1)
        if "keypoints" in target:
            target["keypoints"][..., :2] *= scale

        if self.train and self.hflip_prob > 0 and random.random() < self.hflip_prob:
            tensor = TF.hflip(tensor)
            if len(target["boxes"]):
                flipped = target["boxes"].clone()
                flipped[:, 0] = self.size - target["boxes"][:, 2]
                flipped[:, 2] = self.size - target["boxes"][:, 0]
                target["boxes"] = flipped
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                kp = target["keypoints"].clone()
                kp[..., 0] = self.size - kp[..., 0]
                target["keypoints"] = kp
        vflip = float(self.aug.get("flipud", 0.0) or 0.0)
        if self.train and vflip > 0 and random.random() < vflip:
            tensor = TF.vflip(tensor)
            if len(target["boxes"]):
                flipped = target["boxes"].clone()
                flipped[:, 1] = self.size - target["boxes"][:, 3]
                flipped[:, 3] = self.size - target["boxes"][:, 1]
                target["boxes"] = flipped
            if "masks" in target:
                target["masks"] = target["masks"].flip(-2)
            if "keypoints" in target:
                kp = target["keypoints"].clone()
                kp[..., 1] = self.size - kp[..., 1]
                target["keypoints"] = kp
        return tensor, target


def _rasterize_coco_segmentation(seg: Any, width: int, height: int):
    """COCO polygon list -> binary mask (uint8 HxW); None for RLE input."""
    import numpy as np
    from PIL import Image, ImageDraw

    if not width or not height or not isinstance(seg, list) or not seg:
        return None
    canvas = Image.new("L", (int(width), int(height)), 0)
    draw = ImageDraw.Draw(canvas)
    painted = False
    for poly in seg:
        if not isinstance(poly, (list, tuple)) or len(poly) < 6:
            continue
        draw.polygon(list(zip(poly[0::2], poly[1::2])), fill=1)
        painted = True
    if not painted:
        return None
    return np.asarray(canvas, dtype=np.uint8)


def _coco_records(coco: dict[str, Any], root: str, task: str,
                  mask_dir: str = "") -> list[dict[str, Any]]:
    images = {img["id"]: img for img in coco.get("images", [])}
    by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in coco.get("annotations", []):
        by_image.setdefault(ann["image_id"], []).append(ann)
    cat_ids = sorted({ann["category_id"] for ann in coco.get("annotations", [])})
    cat_to_label = {cid: i + 1 for i, cid in enumerate(cat_ids)}

    records = []
    for img_id, info in images.items():
        path = info["file_name"]
        if not os.path.isabs(path):
            candidate = os.path.join(root, path)
            path = candidate if os.path.exists(candidate) else os.path.join(root, "images", os.path.basename(path))
        if not os.path.exists(path):
            continue
        width = int(info.get("width") or 0)
        height = int(info.get("height") or 0)
        anns = by_image.get(img_id, [])
        boxes, labels, kps, masks = [], [], [], []
        for ann in anns:
            mask = None
            if task == "instance_seg":
                # Per-instance masks: COCO polygons first, PNG masks second.
                mask = _rasterize_coco_segmentation(
                    ann.get("segmentation"), width, height
                )
                if mask is None and mask_dir:
                    from PIL import Image as _Image

                    stem = os.path.splitext(os.path.basename(path))[0]
                    idx = len(boxes)
                    for ext in IMG_EXTS:
                        cand = os.path.join(mask_dir, f"{stem}_{idx}{ext}")
                        if os.path.isfile(cand):
                            mask = (_Image.open(cand).convert("L"))
                            break
                if mask is None:
                    continue  # Mask R-CNN needs one mask per box
            x, y, bw, bh = ann.get("bbox", [0, 0, 0, 0])
            boxes.append([x, y, x + bw, y + bh])
            labels.append(cat_to_label.get(ann["category_id"], 0))
            if mask is not None:
                masks.append(mask)
            if task == "pose" and ann.get("keypoints"):
                kps.append(ann["keypoints"])
        rec: dict[str, Any] = {
            "path": path,
            "boxes": boxes,
            "labels": [label for label in labels if label > 0],
            "width": width,
            "height": height,
        }
        if task == "pose":
            rec["keypoints"] = kps
        if task == "instance_seg":
            rec["masks"] = masks
        records.append(rec)
    return records


def _yolo_records(root: str, split: str) -> tuple[list[dict[str, Any]], list[str]]:
    data_yaml = next((p for p in (os.path.join(root, "data.yaml"), os.path.join(root, "dataset.yaml"))
                      if os.path.isfile(p)), "")
    names: list[str] = []
    if data_yaml:
        names = _resolve_labels(_load_yaml(data_yaml).get("names"))
    split_dirs = [os.path.join(root, "images", split), os.path.join(root, split, "images"), os.path.join(root, split)]
    img_dir = next((d for d in split_dirs if os.path.isdir(d)), "")
    if not img_dir:
        return [], names
    label_dirs = [
        img_dir.replace(f"{os.sep}images", f"{os.sep}labels"),
        os.path.join(root, "labels", split),
    ]
    lbl_dir = next((d for d in label_dirs if os.path.isdir(d)), "")
    records = []
    for path in _images_in(img_dir):
        lbl = os.path.join(lbl_dir, os.path.splitext(os.path.basename(path))[0] + ".txt") if lbl_dir else ""
        boxes, labels = [], []
        if lbl and os.path.isfile(lbl):
            from PIL import Image

            with Image.open(path) as im:
                w, h = im.size
            for line in open(lbl, encoding="utf-8"):
                parts = line.split()
                if len(parts) < 5:
                    continue
                c = int(float(parts[0]))
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
                boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                              (cx + bw / 2) * w, (cy + bh / 2) * h])
                labels.append(c + 1)  # torchvision reserves 0 for background
        records.append({"path": path, "boxes": boxes, "labels": labels})
    return records, names


def build_detection(root: str, cfg: dict[str, Any], train: bool, task: str = "detection"):
    size = int(cfg.get("image_size") or cfg.get("imgsz") or 640)
    split = "train" if train else "val"
    ratio = float(cfg.get("split_ratio", 0.8))
    seed = int(cfg.get("seed", 42))

    # 1) explicit split annotation files (produced by the platform exporter)
    split_json = os.path.join(root, f"{split}_annotations.json")
    if not os.path.isfile(split_json):
        split_json = os.path.join(root, "annotations.json") if os.path.isfile(
            os.path.join(root, "annotations.json")) else ""

    labels: list[str] = []
    if split_json:
        coco = _load_json(split_json)
        cats = sorted(coco.get("categories", []), key=lambda c: c.get("id", 0))
        labels = [str(c.get("name", c.get("id"))) for c in cats]
        mask_dir = os.path.join(root, "masks") if task == "instance_seg" else ""
        records = _coco_records(coco, root, task, mask_dir if os.path.isdir(mask_dir) else "")
        if not train and split_json.endswith("annotations.json") and not os.path.isfile(
            os.path.join(root, "val_annotations.json")
        ):
            records = _split(records, ratio, seed, key=lambda r: _is_video_frame(r["path"]))[1]
        elif train and split_json.endswith("annotations.json") and not os.path.isfile(
            os.path.join(root, "train_annotations.json")
        ):
            records = _split(records, ratio, seed, key=lambda r: _is_video_frame(r["path"]))[0]
        if not labels:
            labels = [str(i + 1) for i in range(max((max(r["labels"]) if r["labels"] else 0) for r in records) or 1)]
        if task == "instance_seg" and not any(r.get("masks") for r in records):
            raise RuntimeError(
                "instance segmentation needs per-instance masks (masks/ PNG or "
                "COCO segmentation polygons); none found in the dataset export"
            )
        return DetectionDataset(records, size, train, task,
                                num_keypoints=int(cfg.get("num_keypoints", 17)) if task == "pose" else 0), labels

    # 2) YOLO txt layout
    if split in ("train", "val") and any(
        os.path.isdir(os.path.join(root, d)) for d in
        (os.path.join("images", split), os.path.join(split, "images"), split)
    ):
        records, labels = _yolo_records(root, split)
        if records:
            return DetectionDataset(
                records, size, train, task, aug=resolve_augmentation(cfg)
            ), labels

    # 3) flat layout -> deterministic split
    records, labels = _yolo_records(root, "train")
    if not records:
        raise RuntimeError(
            f"no {task} annotations found under {root} (expected annotations.json "
            "or data.yaml + images/labels)"
        )
    train_recs, val_recs = _split(records, ratio, seed, key=lambda r: _is_video_frame(r["path"]))
    return DetectionDataset(
        train_recs if train else val_recs, size, train, task,
        aug=resolve_augmentation(cfg),
    ), labels


# ---------------------------------------------------------------------------
# Semantic segmentation
# ---------------------------------------------------------------------------

class SemanticDataset:
    def __init__(self, pairs: list[tuple[str, str]], size: int, train: bool,
                 value_map: dict[int, int], ignore_index: int = 255,
                 aug: Optional[dict[str, Any]] = None):
        self.pairs = pairs
        self.size = size
        self.train = train
        self.value_map = value_map
        self.ignore_index = ignore_index
        self.aug = aug or resolve_augmentation(None)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        import numpy as np
        import torch
        from PIL import Image
        from torchvision.transforms import functional as TF

        img_path, mask_path = self.pairs[index]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        img = TF.resize(img, [self.size, self.size])
        mask = TF.resize(mask, [self.size, self.size], interpolation=TF.InterpolationMode.NEAREST)
        tensor = TF.to_tensor(img)
        if self.train:
            tensor = _photometric_jitter(tensor, self.aug)
        tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
        arr = np.asarray(mask)
        mapped = np.full(arr.shape, self.ignore_index, dtype=np.uint8)
        for raw, idx in self.value_map.items():
            mapped[arr == raw] = idx
        target = torch.from_numpy(mapped.astype("int64"))
        if self.train:
            hflip = float(self.aug.get("fliplr", 0.0) or 0.0)
            if hflip > 0 and random.random() < hflip:
                tensor = TF.hflip(tensor)
                target = torch.flip(target, dims=[-1])
            vflip = float(self.aug.get("flipud", 0.0) or 0.0)
            if vflip > 0 and random.random() < vflip:
                tensor = TF.vflip(tensor)
                target = torch.flip(target, dims=[-2])
        return tensor, target


def build_semantic(root: str, cfg: dict[str, Any], train: bool):
    size = int(cfg.get("image_size") or cfg.get("imgsz") or 512)
    split = "train" if train else "val"
    img_dir = next((os.path.join(root, d) for d in (os.path.join("images", split), split)
                    if os.path.isdir(os.path.join(root, d))), "")
    if not img_dir:
        img_dir = os.path.join(root, "images")
    mask_dirs = [img_dir.replace(f"{os.sep}images", f"{os.sep}masks"),
                 os.path.join(root, "masks", split), os.path.join(root, "masks")]
    mask_dir = next((d for d in mask_dirs if os.path.isdir(d)), "")

    pairs: list[tuple[str, str]] = []
    for img in _images_in(img_dir):
        stem = os.path.splitext(os.path.basename(img))[0]
        for ext in IMG_EXTS:
            candidate = os.path.join(mask_dir, stem + ext) if mask_dir else ""
            if candidate and os.path.isfile(candidate):
                pairs.append((img, candidate))
                break

    if not pairs:
        raise RuntimeError(f"no image/mask pairs under {root} (expected images/ + masks/)")
    train_pairs, val_pairs = _split(pairs, float(cfg.get("split_ratio", 0.8)),
                                    int(cfg.get("seed", 42)), key=lambda p: _is_video_frame(p[0]))

    # Collect raw mask values from the training split only.
    import numpy as np
    from PIL import Image

    values: set[int] = set()
    for _img, mask in (train_pairs or pairs)[:50]:
        arr = np.asarray(Image.open(mask))
        values.update(int(v) for v in np.unique(arr))
    values.discard(255)
    ordered = sorted(values)
    value_map = {raw: idx for idx, raw in enumerate(ordered)}
    labels = [ln.strip() for ln in open(os.path.join(root, "classes.txt"), encoding="utf-8")
              if ln.strip()] if os.path.isfile(os.path.join(root, "classes.txt")) else [str(v) for v in ordered]

    chosen = train_pairs if train else (val_pairs or pairs)
    dataset = SemanticDataset(chosen, size, train, value_map,
                              aug=resolve_augmentation(cfg))
    dataset.labels = labels[: len(value_map)]  # type: ignore[attr-defined]
    dataset.value_map = value_map  # type: ignore[attr-defined]
    return dataset, labels[: len(value_map)]


def dataset_labels(dataset: Any) -> list[str]:
    return list(getattr(dataset, "labels", []) or [])


def manifest_dataset_info(root: str, splits: dict[str, int], cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "root": os.path.basename(root.rstrip("/")) or "dataset",
        "splits": splits,
        "image_size": int(cfg.get("image_size") or cfg.get("imgsz") or 0),
        "seed": int(cfg.get("seed", 42)),
        "split_ratio": float(cfg.get("split_ratio", 0.8)),
    }
