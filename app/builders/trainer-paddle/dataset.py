# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""COCO detection dataset handling for the NF Paddle trainer.

The platform's canonical detection layout is COCO, and PaddleX's
`DetConfig.update_dataset()` expects exactly:

    <root>/images/<file>.jpg
    <root>/annotations/instance_train.json
    <root>/annotations/instance_val.json

Jobs whose dataset already uses that layout are passed through untouched; a flat
NF export (`images/` + `train.json`/`val.json`, or a single `annotations.json`)
is materialised into the same layout under /data/coco so PaddleX never sees a
half-migrated dataset.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any

COCO_LAYOUT = "coco-detection"

IMAGE_DIR = "images"
ANNOTATION_DIR = "annotations"
TRAIN_ANNOTATION = "instance_train.json"
VAL_ANNOTATION = "instance_val.json"

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
# Flat export names the backend may produce, in preference order.
_TRAIN_CANDIDATES = ("instance_train.json", "train.json", "annotations.json")
_VAL_CANDIDATES = ("instance_val.json", "val.json", "annotations.json")


def _find_dir(root: str, name: str) -> str:
    """Case-insensitive, depth-2 search for a directory called *name*."""
    direct = os.path.join(root, name)
    if os.path.isdir(direct):
        return direct
    for current, dirs, _files in os.walk(root):
        if len(os.path.relpath(current, root).split(os.sep)) > 2:
            dirs[:] = []
            continue
        for candidate in dirs:
            if candidate.lower() == name.lower():
                return os.path.join(current, candidate)
    return ""


def _find_file(root: str, names: tuple[str, ...]) -> str:
    for name in names:
        for current, _dirs, files in os.walk(root):
            if name in files:
                return os.path.join(current, name)
    return ""


def _image_count(path: str) -> int:
    if not os.path.isdir(path):
        return 0
    return sum(1 for _r, _d, files in os.walk(path)
               for f in files if f.lower().endswith(_IMAGE_SUFFIXES))


def is_coco_layout(root: str) -> bool:
    """True when *root* already looks like the PaddleX COCO layout."""
    images = os.path.join(root, IMAGE_DIR)
    annotations = os.path.join(root, ANNOTATION_DIR)
    return (
        _image_count(images) > 0
        and os.path.isfile(os.path.join(annotations, TRAIN_ANNOTATION))
        and os.path.isfile(os.path.join(annotations, VAL_ANNOTATION))
    )


def prepare_coco_dataset(data_root: str, target: str = "/data/coco") -> str:
    """Return a directory PaddleX can consume, materialising one if needed."""
    if is_coco_layout(data_root):
        print(f"[data] COCO layout detected at {data_root}", flush=True)
        return data_root

    images_src = _find_dir(data_root, IMAGE_DIR)
    train_src = _find_file(data_root, _TRAIN_CANDIDATES)
    val_src = _find_file(data_root, _VAL_CANDIDATES)
    if not images_src or not train_src:
        raise RuntimeError(
            f"no COCO dataset found under {data_root}: expected an '{IMAGE_DIR}/' "
            f"directory and one of {_TRAIN_CANDIDATES}"
        )
    if not val_src:
        print("[data] no validation annotations; reusing the train split for evaluation", flush=True)
        val_src = train_src

    os.makedirs(os.path.join(target, ANNOTATION_DIR), exist_ok=True)
    _link_tree(images_src, os.path.join(target, IMAGE_DIR))
    shutil.copyfile(train_src, os.path.join(target, ANNOTATION_DIR, TRAIN_ANNOTATION))
    shutil.copyfile(val_src, os.path.join(target, ANNOTATION_DIR, VAL_ANNOTATION))
    if train_src != val_src:
        _migrate_val_images(data_root, val_src, target)
    print(f"[data] COCO layout materialised at {target}", flush=True)
    return target


def _migrate_val_images(data_root: str, val_annotations: str, target: str) -> None:
    """Copy validation images that live in their own split directory."""
    try:
        with open(val_annotations, encoding="utf-8") as fh:
            coco = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return
    images = os.path.join(target, IMAGE_DIR)
    for entry in coco.get("images") or []:
        name = os.path.basename(str(entry.get("file_name") or ""))
        if not name or os.path.isfile(os.path.join(images, name)):
            continue
        for current, _dirs, files in os.walk(data_root):
            if name in files:
                shutil.copyfile(os.path.join(current, name), os.path.join(images, name))
                break


def _link_tree(src: str, dst: str) -> None:
    """Hard-link (fallback: copy) every file of *src* into *dst*."""
    os.makedirs(dst, exist_ok=True)
    for current, _dirs, files in os.walk(src):
        rel = os.path.relpath(current, src)
        out_dir = dst if rel == "." else os.path.join(dst, rel)
        os.makedirs(out_dir, exist_ok=True)
        for name in files:
            source = os.path.join(current, name)
            target = os.path.join(out_dir, name)
            if os.path.exists(target):
                continue
            try:
                os.link(source, target)
            except OSError:
                shutil.copyfile(source, target)


def load_coco_labels(dataset_dir: str) -> list[str]:
    """Category names from the train annotations, ordered by category id.

    This order is the model's label order and goes into the artifact manifest,
    which is what the inference side trusts (ADR-014 D-04).

    Side effect: the densified corpus is written back to `annotations/dense/`
    with 0-based category ids and copied into the active split, because
    PaddleDetection's COCO loader assumes contiguous ids starting at 0 while
    NF datasets may keep 1-based ids (the common COCO export). Keeping the
    original files makes the step idempotent and inspectable.
    """
    path = os.path.join(dataset_dir, ANNOTATION_DIR, TRAIN_ANNOTATION)
    if not os.path.isfile(path):
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            coco = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[data] cannot read {path}: {exc}", flush=True)
        return []

    categories = [c for c in (coco.get("categories") or []) if isinstance(c, dict)]
    if not categories:
        return []
    categories.sort(key=lambda c: int(c.get("id") or 0))
    labels = [str(c.get("name") or f"class_{c.get('id')}") for c in categories]
    print(f"[data] {len(labels)} categories: {labels}", flush=True)

    if [int(c.get("id") or 0) for c in categories] != list(range(len(categories))):
        dense_dir = os.path.join(dataset_dir, ANNOTATION_DIR, "dense")
        os.makedirs(dense_dir, exist_ok=True)
        for name in (TRAIN_ANNOTATION, VAL_ANNOTATION):
            source = os.path.join(dataset_dir, ANNOTATION_DIR, name)
            if not os.path.isfile(source):
                continue
            try:
                with open(source, encoding="utf-8") as fh:
                    payload = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            apply_label_map(payload)
            with open(os.path.join(dense_dir, name), "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            shutil.copyfile(os.path.join(dense_dir, name), source)
        print(f"[data] category ids densified to 0-based in {ANNOTATION_DIR}/", flush=True)
    return labels


def apply_label_map(annotations: dict[str, Any]) -> dict[str, Any]:
    """Rewrite category ids to dense 0-based indices (helper for callers/tests)."""
    categories = sorted(
        (c for c in (annotations.get("categories") or []) if isinstance(c, dict)),
        key=lambda c: int(c.get("id") or 0),
    )
    remap = {int(c["id"]): index for index, c in enumerate(categories)}
    for category, index in zip(categories, range(len(categories))):
        category["id"] = index
    annotations["categories"] = categories
    for annotation in annotations.get("annotations") or []:
        if isinstance(annotation, dict) and int(annotation.get("category_id") or -1) in remap:
            annotation["category_id"] = remap[int(annotation["category_id"])]
    return annotations
