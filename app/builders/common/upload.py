# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Shared upload utilities for builder entrypoints.

Handles uploading weights, ONNX exports, TensorBoard logs, and metrics to S3.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .s3_utils import make_s3_client, parse_s3_uri


def upload_training_artifacts(
    output_uri: str,
    artifact_id: str,
    metrics: dict[str, Any],
    onnx_path: str | None = None,
    weight_candidates: list[str] | None = None,
    tensorboard_dirs: list[str] | None = None,
) -> dict[str, Any]:
    """Upload weights, ONNX, TensorBoard, and metrics to S3.

    Returns the updated metrics dict with uploaded URIs.
    """
    if not output_uri:
        return metrics

    s3 = make_s3_client()
    ob, okey = parse_s3_uri(output_uri)

    # Upload weights
    if weight_candidates is None:
        weight_candidates = [
            "runs/detect/train/weights/best.pt",
            "runs/detect/train/weights/last.pt",
            "/app/runs/detect/train/weights/best.pt",
            "/app/runs/detect/train/weights/last.pt",
        ]
    uploaded = False
    for w in weight_candidates:
        if os.path.exists(w):
            wkey = okey + "/" + artifact_id + ".pt"
            s3.upload_file(w, ob, wkey)
            print(f"[upload] weights uploaded s3://{ob}/{wkey}", flush=True)
            metrics["weights_uri"] = f"s3://{ob}/{wkey}"
            uploaded = True
            break
    if not uploaded:
        print("[upload] WARN: no weights file found", flush=True)

    # Upload ONNX
    if onnx_path and os.path.exists(onnx_path):
        onnx_key = okey + "/" + artifact_id + ".onnx"
        s3.upload_file(onnx_path, ob, onnx_key)
        print(f"[upload] onnx uploaded s3://{ob}/{onnx_key}", flush=True)
        metrics["onnx_uri"] = f"s3://{ob}/{onnx_key}"
    else:
        print("[upload] WARN: no onnx file to upload", flush=True)

    # Upload TensorBoard events
    if tensorboard_dirs is None:
        tensorboard_dirs = ["runs/detect/train", "/app/runs/detect/train"]
    tb_uploaded = False
    for tb_dir in tensorboard_dirs:
        if not os.path.isdir(tb_dir):
            continue
        for fn in os.listdir(tb_dir):
            if fn.startswith("events.out.tfevents") or fn == "results.csv":
                tb_key = okey + "/tensorboard/" + fn
                s3.upload_file(os.path.join(tb_dir, fn), ob, tb_key)
                print(f"[upload] tensorboard {fn} uploaded s3://{ob}/{tb_key}", flush=True)
                tb_uploaded = True
    if tb_uploaded:
        metrics["tensorboard_uri"] = f"s3://{ob}/{okey}/tensorboard/"

    # Upload metrics
    with open("/app/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    mkey = okey + "/" + artifact_id + ".metrics.json"
    s3.upload_file("/app/metrics.json", ob, mkey)
    print(f"[upload] metrics uploaded s3://{ob}/{mkey}", flush=True)
    print(f"[upload] metrics: {json.dumps(metrics)}", flush=True)

    return metrics
