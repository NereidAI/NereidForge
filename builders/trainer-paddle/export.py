# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Export entrypoint for the NF Paddle trainer (`SCRIPT=export`).

Turns a trained snapshot into the deployable artifact pair the platform serves:
`<id>.onnx` plus `<id>.metadata.json`. Useful for re-exporting a registered
model (e.g. with a different input size) without retraining.

Env: `INPUT_URI` (dataset, for the label order), `OUTPUT_DIR`, `CONFIG_JSON`
(`model`, `weights_uri`/`weights_path`, `image_size`), `ARTIFACT_ID`.
"""

from __future__ import annotations

import os
import sys

from dataset import load_coco_labels, prepare_coco_dataset
from models import input_size, resolve_model_name
from train import _export_onnx

from common import (
    build_manifest,
    download_dataset,
    parse_uri,
    publish_file,
    read_config,
    s3_client,
    write_json,
)


def _weights_local(cfg: dict) -> str:
    """Local snapshot path: explicit path, or download `weights_uri`."""
    path = str(cfg.get("weights_path") or "").strip()
    if path and os.path.isfile(path):
        return path
    uri = str(cfg.get("weights_uri") or "").strip()
    if not uri.startswith("s3://"):
        return ""
    bucket, prefix = parse_uri(uri)
    local = "/tmp/nf-weights.pdparams"
    s3_client().download_file(bucket, prefix, local)
    return local


def main() -> int:
    cfg = read_config()
    input_uri = os.environ.get("INPUT_URI", "")
    output_uri = os.environ.get("OUTPUT_DIR", "")
    artifact_id = os.environ.get("ARTIFACT_ID", "model")

    weights = _weights_local(cfg)
    if not weights:
        print("[export] CONFIG_JSON needs weights_path or weights_uri (s3://...)", flush=True)
        return 2

    arch = str(cfg.get("model") or "")
    image_size = int(cfg.get("image_size") or input_size(arch))
    model_name = resolve_model_name(arch, int(cfg.get("num_classes") or 0))

    labels: list[str] = []
    if input_uri:
        dataset_dir = prepare_coco_dataset(download_dataset(input_uri, "/data"))
        labels = load_coco_labels(dataset_dir)

    from paddlex.repo_apis.base import PaddleModel

    device = "gpu:0" if str(os.environ.get("NF_DEVICE", "cpu")).lower() in ("gpu", "cuda", "cuda:0") else "cpu"
    model = PaddleModel(model_name=model_name)
    onnx_local = _export_onnx(model, weights, "/app/output", artifact_id, image_size, device)
    if not onnx_local:
        return 1

    publish_file(onnx_local, output_uri, f"{artifact_id}.onnx")
    manifest = build_manifest(
        task="detection",
        architecture=model_name,
        framework="paddle",
        input_size=image_size,
        labels=labels,
        metrics={},
        dataset={"uri": input_uri, "layout": "coco-detection"},
        extra={"license": {"code": "Apache-2.0", "weights": "NF-trained"}},
    )
    local = f"/tmp/{artifact_id}.metadata.json"
    write_json(local, manifest)
    publish_file(local, output_uri, f"{artifact_id}.metadata.json")
    print(f"[export] SUCCESS onnx={onnx_local} labels={len(labels)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
