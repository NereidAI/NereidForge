# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""NereidForge Paddle vision trainer — object detection via PaddleX (Apache-2.0).

Second supported framework of the default matrix (ADR-014 D-01). One entrypoint
per role, selected by `SCRIPT`:

    SCRIPT=train      python /app/train.py     (default)
    SCRIPT=evaluate   python /app/evaluate.py
    SCRIPT=export     python /app/export.py
    SCRIPT=profile    python /app/nf_profile.py

PaddleX has no train/evaluate/export CLI, so everything goes through the
`paddlex.repo_apis` Python API: `PaddleModel(model_name=...)` (the repo API,
NOT `paddlex.create_model`, which builds an inference predictor) ->
`model.config.update_dataset` -> `model.train/evaluate/export`, then
`paddlex --paddle2onnx` for the ONNX step. Training requires the PaddleDetection
repo baked into the image by the Dockerfile (the wheel ships register.py and
the repo API code but neither the training YAML configs nor the engine).

Dataset: COCO detection layout (canonical formats are summarized in
``builders/README.md``):

    images/*.jpg
    annotations/instance_train.json
    annotations/instance_val.json

Artifacts written to OUTPUT_DIR:
    <id>.onnx              inference graph (manifest-driven adapters consume it)
    <id>.metadata.json     NF model manifest (task/labels/input/normalisation)
    <id>.metrics.json      final validation metrics + URIs (flat keys for the UI)
    <id>.pdparams          trained snapshot (re-export / fine-tune source)
    progress.json          live epoch progress (polled by the NF worker)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from typing import Any

from dataset import COCO_LAYOUT, load_coco_labels, prepare_coco_dataset
from trainer_metrics import metrics_summary, parse_metrics
from models import DEFAULT_ARCH, input_size, resolve_model_name

from common import (
    Progress,
    build_manifest,
    build_training_metrics,
    download_dataset,
    publish_file,
    read_config,
    write_json,
)

TASK = "detection"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config() -> dict[str, Any]:
    """Job config; `CONFIG_JSON` is inline JSON or a path (SDK helper)."""
    try:
        cfg = read_config()
    except Exception as exc:  # noqa: BLE001
        print(f"[train] CONFIG_JSON is not valid JSON: {exc}", flush=True)
        cfg = {}
    return cfg or {}


def _device(cfg: dict[str, Any]) -> str:
    """PaddleX device string. The image's device is set by NF_DEVICE."""
    requested = str(cfg.get("device") or os.environ.get("NF_DEVICE") or "cpu").strip().lower()
    if requested.startswith("gpu") or requested in ("cuda", "cuda:0"):
        return "gpu:0"
    return "cpu"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _run_result(result: Any) -> tuple[int, str]:
    """Normalise a PaddleX CompletedProcess into (returncode, stdout+stderr)."""
    code = int(getattr(result, "returncode", 0) or 0)
    text = "\n".join(
        str(part) for part in (getattr(result, "stdout", "") or "", getattr(result, "stderr", "") or "")
        if part
    )
    return code, text


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def main() -> int:
    cfg = load_config()
    input_uri = os.environ.get("INPUT_URI", "")
    output_uri = os.environ.get("OUTPUT_DIR", "")
    artifact_id = os.environ.get("ARTIFACT_ID", "model")

    if not input_uri:
        print("[train] INPUT_URI is required", flush=True)
        return 2

    epochs = int(cfg.get("epochs") or 50)
    batch = int(cfg.get("batch") or 8)
    learning_rate = float(cfg.get("learning_rate") or 0.001)
    num_classes = int(cfg.get("num_classes") or 0)
    arch = str(cfg.get("model") or DEFAULT_ARCH).strip()
    device = _device(cfg)

    data_root = download_dataset(input_uri, "/data")
    dataset_dir = prepare_coco_dataset(data_root)
    labels = load_coco_labels(dataset_dir)
    if not num_classes:
        num_classes = len(labels)

    model_name = resolve_model_name(arch, num_classes)
    print(
        f"[train] arch={arch} model={model_name} classes={num_classes} "
        f"epochs={epochs} batch={batch} lr={learning_rate} device={device}",
        flush=True,
    )
    print(f"[train] dataset={dataset_dir} layout={COCO_LAYOUT} labels={len(labels)}", flush=True)

    from paddlex.repo_apis.base import PaddleModel

    # Repo API, NOT paddlex.create_model: the latter builds an inference
    # predictor (and pulls in PDF/reader deps), while training/evaluate/export
    # live on the PaddleDetection repo model registered at `import paddlex`.
    model = PaddleModel(model_name=model_name)
    model.config.update_dataset(dataset_dir)
    if num_classes:
        model.config.update_num_class(int(num_classes))

    save_dir = "/app/output"
    os.makedirs(save_dir, exist_ok=True)

    progress = Progress(output_uri, epochs)
    training_started = time.monotonic()
    result = model.train(
        epochs_iters=epochs,
        batch_size=batch,
        learning_rate=learning_rate,
        device=device,
        save_dir=save_dir,
        use_vdl=False,
        # Paddle 3.x: the legacy-IR dy2static export crashes on PicoDet
        # (InvalidType in lcnet.py). The official PaddleX module trainer
        # defaults to the new PIR serializer (FLAGS_json_format_model); pass
        # it explicitly on the repo-API call.
        export_with_pir=True,
    )
    train_time = time.monotonic() - training_started
    code, log = _run_result(result)
    if code != 0:
        print(log[-4000:], flush=True)
        print(f"[train] paddle train failed with exit code {code}", flush=True)
        return code

    metrics = parse_metrics(log)
    print(f"[train] {metrics_summary(metrics)}", flush=True)
    metrics.update({
        "task": TASK,
        "architecture": model_name,
        "epochs": epochs,
        "num_classes": num_classes,
        "device": device,
    })
    metrics = build_training_metrics(metrics, train_time_s=train_time)
    progress.log(epochs, {k: v for k, v in metrics.items() if isinstance(v, (int, float))})

    weights = _find_weights(save_dir)
    if weights:
        publish_file(weights, output_uri, f"{artifact_id}.pdparams")
        metrics["weights_uri"] = _uri(output_uri, f"{artifact_id}.pdparams")

    onnx_local = _export_onnx(
        model, weights, save_dir, artifact_id,
        input_size=int(cfg.get("image_size") or input_size(arch)),
        device=device,
    )
    if onnx_local:
        publish_file(onnx_local, output_uri, f"{artifact_id}.onnx")
        metrics["onnx_uri"] = _uri(output_uri, f"{artifact_id}.onnx")
        manifest = build_manifest(
            task=TASK,
            architecture=model_name,
            framework="paddle",
            input_size=int(cfg.get("image_size") or input_size(arch)),
            labels=labels,
            metrics=metrics,
            dataset={"uri": input_uri, "layout": COCO_LAYOUT},
            extra={"license": {"code": "Apache-2.0", "weights": "NF-trained"}},
        )
        manifest_local = f"/tmp/{artifact_id}.metadata.json"
        write_json(manifest_local, manifest)
        publish_file(manifest_local, output_uri, f"{artifact_id}.metadata.json")
        metrics["manifest_uri"] = _uri(output_uri, f"{artifact_id}.metadata.json")
    else:
        print("[train] WARNING: no ONNX artifact produced; deployment will not be possible", flush=True)

    metrics_local = f"/tmp/{artifact_id}.metrics.json"
    write_json(metrics_local, metrics)
    publish_file(metrics_local, output_uri, f"{artifact_id}.metrics.json")

    print(f"[train] SUCCESS metrics={json.dumps(metrics, sort_keys=True)}", flush=True)
    return 0


def _uri(output_uri: str, name: str) -> str:
    if not output_uri:
        return ""
    if output_uri.startswith("s3://"):
        return output_uri.rstrip("/") + "/" + name
    return os.path.join(output_uri, name)


def _find_weights(save_dir: str) -> str:
    """Locate the best snapshot PaddleDetection wrote (`best_model/model.pdparams`)."""
    candidates = [
        os.path.join(save_dir, "best_model", "model.pdparams"),
        os.path.join(save_dir, "best_model.pdparams"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    for root, _dirs, files in os.walk(save_dir):
        for name in files:
            if name.endswith(".pdparams"):
                return os.path.join(root, name)
    return ""


def _export_onnx(model: Any, weights: str, save_dir: str, artifact_id: str,
                 input_size: int = 640, device: str = "cpu") -> str:
    """PaddleX inference export, then `paddlex --paddle2onnx` -> <id>.onnx.

    `paddle2onnx` is a plugin PaddleX normally installs on demand; the image
    installs it at build time so this works with no network access.
    """
    if not weights:
        print("[export] no weights to export", flush=True)
        return ""
    inference_dir = os.path.join(save_dir, "inference")
    code, log = _run_result(model.export(
        weight_path=weights,
        save_dir=inference_dir,
        input_shape=[3, input_size, input_size],
        export_with_pir=True,
        # The export subprocess otherwise inherits use_gpu=true from the
        # default config and aborts on a CPU-only paddle build.
        device=device,
    ))
    if code != 0:
        print(log[-2000:], flush=True)
        print(f"[export] paddle export failed with exit code {code}", flush=True)
        return ""

    onnx_dir = os.path.join(save_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)
    try:
        proc = subprocess.run(
            [
                sys.executable, "-m", "paddlex", "--paddle2onnx",
                "--paddle_model_dir", inference_dir,
                "--onnx_model_dir", onnx_dir,
                "--opset_version", "17",
            ],
            capture_output=True, text=True, timeout=1800, check=False,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[export] paddle2onnx failed to start: {exc}", flush=True)
        return ""
    if proc.returncode != 0:
        print((proc.stdout or "")[-2000:], flush=True)
        print((proc.stderr or "")[-2000:], flush=True)
        print(f"[export] paddle2onnx failed with exit code {proc.returncode}", flush=True)
        return ""

    for root, _dirs, files in os.walk(onnx_dir):
        for name in files:
            if name.endswith(".onnx"):
                dst = os.path.join(save_dir, f"{artifact_id}.onnx")
                shutil.copyfile(os.path.join(root, name), dst)
                print(f"[export] onnx -> {dst}", flush=True)
                return dst
    print("[export] paddle2onnx produced no .onnx file", flush=True)
    return ""


if __name__ == "__main__":
    sys.exit(main())
