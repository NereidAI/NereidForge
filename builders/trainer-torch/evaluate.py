# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a trained NF vision checkpoint against a dataset split.

Contract (``builders/sdk/README.md``):
  INPUT_URI       dataset root (same export format as training)
  OUTPUT_DIR      where <ARTIFACT_ID>.metrics.json is written
  CONFIG_JSON     {"task", "arch", "labels", "checkpoint_uri"|"model_uri", ...}
  ARTIFACT_ID     output id

Exit code 0 on success; metrics are the same schema the training run writes,
so the evaluation gate and the UI share one parser.
"""

from __future__ import annotations

import json
import os
import sys

import torch

from common import download_dataset, publish_file, write_json
from dataset import build_classification, build_detection, build_semantic
from models import DEFAULT_ARCH, build_model
from train import (
    _eval_classifier,
    _eval_semantic,
    _eval_torchvision_task,
    load_config,
    pick_device,
)


def _download_uri(uri: str) -> str:
    from common import parse_uri, s3_client

    bucket, key = parse_uri(uri)
    local = os.path.join("/tmp", os.path.basename(key) or "checkpoint.pt")
    s3_client().download_file(bucket, key, local)
    return local


def main() -> int:
    input_uri = os.environ.get("INPUT_URI", "")
    output_uri = os.environ.get("OUTPUT_DIR", "")
    artifact_id = os.environ.get("ARTIFACT_ID", "") or "eval"
    cfg = load_config()

    task = str(cfg.get("task") or "detection").lower()
    arch = str(cfg.get("arch") or cfg.get("model") or DEFAULT_ARCH.get(task, "resnet50"))
    size = int(cfg.get("image_size") or 0) or {"classification": 224, "multilabel": 224,
                                              "detection": 640, "instance_seg": 640,
                                              "pose": 640, "semantic_seg": 512}.get(task, 640)
    device = pick_device(cfg)

    ckpt_uri = str(cfg.get("checkpoint_uri") or cfg.get("model_uri") or "")
    if not ckpt_uri:
        print("[evaluate] checkpoint_uri/model_uri is required", flush=True)
        return 2
    local_ckpt = _download_uri(ckpt_uri) if ckpt_uri.startswith("s3://") else ckpt_uri
    checkpoint = torch.load(local_ckpt, map_location="cpu", weights_only=False)
    labels = list(checkpoint.get("labels") or cfg.get("labels") or [])
    task = str(checkpoint.get("task") or task)
    arch = str(checkpoint.get("arch") or arch)

    if not input_uri:
        print("[evaluate] INPUT_URI is required", flush=True)
        return 2
    root = download_dataset(input_uri, "/data") if input_uri.startswith("s3://") else input_uri

    if task in ("classification", "multilabel"):
        val, val_labels = build_classification(root, cfg, train=False)
        labels = labels or val_labels
        model = build_model(task, arch, len(labels), size, pretrained=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        from torch.utils.data import DataLoader

        loader = DataLoader(val, batch_size=int(cfg.get("batch") or 16), shuffle=False, num_workers=0)
        metrics = _eval_classifier(model, loader, labels, device, task == "multilabel")
    elif task == "semantic_seg":
        val, val_labels = build_semantic(root, cfg, train=False)
        labels = labels or val_labels
        model = build_model(task, arch, len(labels), size, pretrained=False,
                            encoder_name=str(cfg.get("encoder") or "resnet34"))
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        from torch.utils.data import DataLoader

        loader = DataLoader(val, batch_size=int(cfg.get("batch") or 4), shuffle=False, num_workers=0)
        metrics = _eval_semantic(model, loader, labels, device)
    else:
        val, val_labels = build_detection(root, cfg, train=False, task=task)
        labels = labels or val_labels
        model = build_model(task, arch, len(labels) + 1, size, pretrained=False,
                            num_keypoints=int(cfg.get("num_keypoints") or 17))
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        from torch.utils.data import DataLoader

        def collate(items):
            return torch.stack([i[0] for i in items]), [i[1] for i in items]

        loader = DataLoader(val, batch_size=int(cfg.get("batch") or 1), shuffle=False,
                            num_workers=0, collate_fn=collate)
        metrics = _eval_torchvision_task(model, loader, labels, device, task)

    metrics = {
        "task": task,
        "architecture": arch,
        "framework": "torch",
        "checkpoint_uri": ckpt_uri,
        "source": "evaluate",
        **metrics,
    }
    local = f"/tmp/{artifact_id}.metrics.json"
    write_json(local, metrics)
    if output_uri:
        publish_file(local, output_uri, f"{artifact_id}.metrics.json")
    print(f"[evaluate] {json.dumps({k: v for k, v in metrics.items() if not isinstance(v, (dict, list))})}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
