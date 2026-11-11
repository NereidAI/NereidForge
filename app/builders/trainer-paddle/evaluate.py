# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Evaluation entrypoint for the NF Paddle trainer (`SCRIPT=evaluate`).

Re-runs PaddleX's COCO evaluation on a trained snapshot and publishes
`<id>.metrics.json`, so a model can be re-scored without retraining (and so the
platform can evaluate registered artifacts).

Env: `INPUT_URI` (COCO dataset), `OUTPUT_DIR` (metrics destination),
`CONFIG_JSON` (`model`, `weights_uri` or `weights_path`, `batch`), `ARTIFACT_ID`.
"""

from __future__ import annotations

import json
import os
import sys

from dataset import load_coco_labels, prepare_coco_dataset
from models import resolve_model_name
from trainer_metrics import parse_metrics

from common import download_dataset, parse_uri, publish_file, read_config, s3_client, write_json


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
    if not input_uri:
        print("[eval] INPUT_URI is required", flush=True)
        return 2

    weights = _weights_local(cfg)
    if not weights:
        print("[eval] CONFIG_JSON needs weights_path or weights_uri (s3://...)", flush=True)
        return 2

    device = "gpu:0" if str(cfg.get("device") or os.environ.get("NF_DEVICE", "cpu")).lower().startswith(("gpu", "cuda")) else "cpu"
    model_name = resolve_model_name(str(cfg.get("model") or ""), int(cfg.get("num_classes") or 0))

    dataset_dir = prepare_coco_dataset(download_dataset(input_uri, "/data"))
    labels = load_coco_labels(dataset_dir)

    from paddlex.repo_apis.base import PaddleModel

    model = PaddleModel(model_name=model_name)
    model.config.update_dataset(dataset_dir)
    # Build the 2-class head before loading the custom-class snapshot, or the
    # default 80-class model silently skips its weights ("unmatched ... head").
    model.config.update_num_class(len(labels))
    result = model.evaluate(weight_path=weights, batch_size=int(cfg.get("batch") or 8), device=device)

    stdout = "\n".join(str(p) for p in (getattr(result, "stdout", "") or "", getattr(result, "stderr", "") or "") if p)
    code = int(getattr(result, "returncode", 0) or 0)
    if code != 0:
        print(stdout[-4000:], flush=True)
        print(f"[eval] paddle evaluate failed with exit code {code}", flush=True)
        return code

    # DetRunner.evaluate attaches a structured metrics dict ({'AP': x}); the
    # stdout table is the fallback for other/older PaddleX revisions.
    repo_metrics = getattr(result, "metrics", None) or {}
    metrics = {
        **parse_metrics(stdout),
        **({"AP": repo_metrics["AP"]} if isinstance(repo_metrics.get("AP"), (int, float)) else {}),
        "task": "detection",
        "architecture": model_name,
        "num_classes": len(labels),
        "device": device,
        "weights_uri": str(cfg.get("weights_uri") or ""),
    }
    local = f"/tmp/{artifact_id}.metrics.json"
    write_json(local, metrics)
    published = publish_file(local, output_uri, f"{artifact_id}.metrics.json")
    print(f"[eval] SUCCESS {json.dumps(metrics, sort_keys=True)}", flush=True)
    print(f"[eval] metrics published to {published}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
