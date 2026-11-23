# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Shared dataset preparation logic for PP-YOLOE (PaddleDetection) training.

Handles: S3 download, COCO annotation handling, and train/val split.
Used by PP-YOLOE (PaddleDetection) training builders.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from .s3_utils import make_s3_client, parse_s3_uri, list_s3_objects


def download_dataset(input_uri: str, local_root: str = "/data") -> None:
    """Download all objects under *input_uri* to *local_root*.

    Exits with code 1 if no objects are found.
    """
    s3 = make_s3_client()
    bucket, prefix = parse_s3_uri(input_uri)
    print(f"[data] dataset: s3://{bucket}/{prefix}", flush=True)

    os.makedirs(local_root, exist_ok=True)
    keys = list_s3_objects(s3, bucket, prefix)
    if not keys:
        print(f"[data] no objects found under {prefix}, aborting", flush=True)
        sys.exit(1)

    downloaded = 0
    for k in keys:
        rel = k[len(prefix):].lstrip("/")
        if not rel:
            continue
        dst = os.path.join(local_root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            s3.download_file(bucket, k, dst)
            downloaded += 1
        except Exception as e:
            print(f"[data] skip {rel}: {e}", flush=True)
    print(f"[data] downloaded {downloaded} files to {local_root}", flush=True)


def convert_yolo_to_coco() -> str:
    """Convert YOLO format labels to COCO format for PaddleDetection.
    
    Returns path to COCO annotations.json.
    """
    labels_dir = "/data/yolo-train/labels" if os.path.exists("/data/yolo-train/labels") else "/data/labels"
    if os.path.exists(labels_dir):
        return

    # Find data.yaml
    data_yaml = None
    for cand in ["/data/data.yaml", "/data/yolo-train/data.yaml"]:
        if os.path.exists(cand):
            break
    if not data_yaml:
        print("[data] data.yaml not found, skipping YOLO to COCO conversion", flush=True)
        return

    import yaml
    with open(data_yaml) as f:
        yaml_data = yaml.safe_load(f)
    
    names = yaml_data.get("names", {})
    nc = yaml_data.get("nc", len(names))
    
    # Parse YOLO labels to COCO
    images_dir = "/data/yolo-train/images" if os.path.exists("/data/yolo-train/images") else "/data/images"
    labels_dir = "/data/yolo-train/labels" if os.path.exists("/data/yolo-train/labels") else "/data/labels"
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print("[data] images or labels directory not found, skipping conversion", flush=True)
        return

    import cv2
    import json
    
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in names.items()]
    }
    
    img_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    ann_id = 0
    img_id = 0
    
    import os
    for img_file in sorted(os.listdir(images_dir)):
        if not img_file.endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(images_dir, img_file)
        label_file = os.path.join(labels_dir, os.path.splitext(img_file)[0] + ".txt")
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        img_info = {
            "id": img_id,
            "file_name": img_file,
            "width": w,
            "height": h
        }
        coco_data["images"].append(img_info)
        
        label_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + ".txt")
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    x = (float(parts[1]) - float(parts[3])/2) * w
                    y = (float(parts[2]) - float(parts[4])/2) * h
                    w_box = float(parts[3]) * w
                    h_box = float(parts[4]) * h
                    ann = {
                        "id": len(coco_data["annotations"]),
                        "image_id": img_id,
                        "category_id": cls,
                        "bbox": [x, y, w_box, h_box],
                        "area": w_box * h_box,
                        "iscrowd": 0,
                        "bbox_mode": 1  # XYWH_ABS
                    }
                    coco_data["annotations"].append(ann)
        
        img_id += 1
    
    # Save COCO annotations
    coco_json_path = "/data/annotations.json"
    with open(coco_json_path, "w") as f:
        json.dump(coco_data, f)
    
    print(f"[data] Converted YOLO to COCO: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")
    return "/data/annotations.json"


def split_train_val(json_path: str, cfg: dict[str, Any], data_root: str) -> str:
    """Split COCO dataset into train/val subsets based on split_ratio.
    
    Returns path to updated COCO annotations.json.
    """
    split_ratio = float(cfg.get("split_ratio", 0.8))
    split_ratio = min(max(split_ratio, 0.0), 1.0)
    if split_ratio <= 0 or split_ratio >= 1:
        return json_path

    import random
    import json
    
    with open(json_path) as f:
        coco_data = json.load(f)
    
    # Group images by video source
    import re
    random.seed(42)
    
    def group_key(img_info: dict) -> str:
        stem = os.path.splitext(img_info["file_name"])[0]
        m = re.match(r"^(.+)_\d{6}$", stem)
        return m.group(1) if m else f"__file__{img_info['id']}"

    groups: dict[str, list[dict]] = {}
    for img in coco_data["images"]:
        k = group_key(img)
        groups.setdefault(k, []).append(img)
    
    keys = sorted(groups)
    import random
    random.shuffle(keys)
    n_target = max(1, int(len(coco_data["images"]) * split_ratio))
    train_imgs: list[dict] = []
    val_imgs: list[dict] = []
    for k in keys:
        members = groups[k]
        if len(train_imgs) < int(len(coco_data["images"]) * split_ratio):
            train_imgs.extend(members)
        else:
            val_imgs.extend(members)
    
    if not val_imgs and len(keys) > 1:
        val_imgs = train_imgs[-1:] if len(train_imgs) > 1 else []
    
    # Build train/val COCO files
    train_coco = {
        "images": train_imgs,
        "annotations": [a for a in coco_data["annotations"] if a["image_id"] in [img["id"] for img in train_imgs]],
        "categories": coco_data["categories"]
    }
    val_coco = {
        "images": val_imgs,
        "annotations": [a for a in coco_data["annotations"] if a["image_id"] in [img["id"] for img in val_imgs]],
        "categories": coco_data["categories"]
    }
    
    train_json = os.path.join("/data", "train_annotations.json")
    val_json = os.path.join("/data", "val_annotations.json")
    
    with open(train_json, "w") as f:
        json.dump(train_coco, f)
    with open(val_json, "w") as f:
        json.dump(val_coco, f)
    
    print(f"[data] split {len(train_imgs)} train / {len(val_imgs)} val (ratio={split_ratio})", flush=True)
    return train_json


def prepare_coco_dataset(input_uri: str, cfg: dict[str, Any]) -> str:
    """Full dataset preparation pipeline: download → COCO format → split.

    Returns the path to the COCO train annotations.json ready for PaddleDetection training.
    """
    download_dataset(input_uri)
    
    # Try to find COCO annotations
    coco_json = None
    for cand in ["/data/annotations.json", "/data/coco/annotations.json", "/data/results.json"]:
        if os.path.exists(cand):
            coco_json = cand
            break
    
    # If YOLO format, convert to COCO
    if not coco_json:
        coco_json = convert_yolo_to_coco()
    
    if not coco_json:
        print("[data] No COCO annotations found, aborting", flush=True)
        sys.exit(1)
    
    # Split train/val
    data_root = os.path.dirname(coco_json)
    train_json = split_train_val(coco_json, {}, "")
    
    fixed_json = "/app/data-fixed.json"
    with open(fixed_json, "w") as f:
        import json
        with open(train_json) as f_in:
            json.dump(json.load(f_in), f)
    
    print(f"[data] using train annotations: {fixed_json}", flush=True)
    return fixed_json
