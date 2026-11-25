# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Re-export a trained checkpoint to ONNX + NF manifest (no training).

Useful when the serving runtime changes (opset, top-k), when a model was
imported from a user checkpoint, or when only the manifest needs a refresh.

  CONFIG_JSON     {"task", "arch", "labels", "checkpoint_uri", "export_topk"}
  OUTPUT_DIR      where <ARTIFACT_ID>.onnx / .metadata.json land
  ARTIFACT_ID     output id
"""

from __future__ import annotations

import os
import sys

import torch

from common import build_manifest, publish_file, write_json
from models import DEFAULT_ARCH, build_model, export_onnx
from train import load_config


def main() -> int:
    output_uri = os.environ.get("OUTPUT_DIR", "")
    artifact_id = os.environ.get("ARTIFACT_ID", "") or "export"
    cfg = load_config()

    from common import parse_uri, s3_client

    ckpt_uri = str(cfg.get("checkpoint_uri") or cfg.get("model_uri") or "")
    if not ckpt_uri:
        print("[export] checkpoint_uri/model_uri is required", flush=True)
        return 2
    if ckpt_uri.startswith("s3://"):
        bucket, key = parse_uri(ckpt_uri)
        local_ckpt = os.path.join("/tmp", os.path.basename(key) or "checkpoint.pt")
        s3_client().download_file(bucket, key, local_ckpt)
    else:
        local_ckpt = ckpt_uri

    checkpoint = torch.load(local_ckpt, map_location="cpu", weights_only=False)
    task = str(checkpoint.get("task") or cfg.get("task") or "detection").lower()
    arch = str(checkpoint.get("arch") or cfg.get("arch") or DEFAULT_ARCH.get(task, "resnet50"))
    labels = list(checkpoint.get("labels") or cfg.get("labels") or [])
    size = int(cfg.get("image_size") or 0) or {"classification": 224, "multilabel": 224,
                                              "detection": 640, "instance_seg": 640,
                                              "pose": 640, "semantic_seg": 512}.get(task, 640)

    det_offset = 1 if task in ("detection", "instance_seg", "pose") else 0
    model = build_model(task, arch, len(labels) + det_offset, size, pretrained=False,
                        num_keypoints=int(cfg.get("num_keypoints") or 17),
                        encoder_name=str(cfg.get("encoder") or "resnet34"))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    local_onnx = f"/tmp/{artifact_id}.onnx"
    export_onnx(model, task, size, local_onnx, topk=int(cfg.get("export_topk") or 100))

    manifest = build_manifest(
        task=task,
        architecture=arch,
        framework="torch",
        input_size=size,
        labels=labels,
        metrics=checkpoint.get("metrics") or {},
        dataset={"source": "re-export"},
    )
    local_manifest = f"/tmp/{artifact_id}.metadata.json"
    write_json(local_manifest, manifest)

    if output_uri:
        publish_file(local_onnx, output_uri, f"{artifact_id}.onnx")
        publish_file(local_manifest, output_uri, f"{artifact_id}.metadata.json")
    print(f"[export] task={task} arch={arch} labels={len(labels)} -> {artifact_id}.onnx", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
