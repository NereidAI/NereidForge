# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""NereidForge vision trainer entrypoint (torchvision + timm).

One entrypoint for every supported vision task; the task is a config field, not
a separate image:

    classification | multilabel | detection | instance_seg | semantic_seg | pose

Env contract (unchanged from the platform contract):
    INPUT_URI   s3:// dataset root
    OUTPUT_DIR  s3:// artifact destination
    CONFIG_JSON flat training config
    ARTIFACT_ID unique id (artifacts are <id>.<ext>)

Artifacts written to OUTPUT_DIR:
    <id>.onnx              inference graph (manifest-driven adapters consume it)
    <id>.metadata.json     NF model manifest (task/labels/input/normalisation)
    <id>.metrics.json      final validation metrics + URIs (flat keys for the UI)
    <id>.pt                checkpoint (resume / re-export)
    progress.json          live epoch progress (polled by the NF worker)

No third-party weights are baked into the image: either `base_weights_uri`
points at an artifact in S3, or `pretrained=true` downloads the permissive
upstream backbone weights at runtime (network deployments).
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import torch
from torch.utils.data import DataLoader

from common import (
    Progress,
    augmentation_summary,
    build_manifest,
    build_training_metrics,
    download_dataset,
    publish_file,
    resolve_augmentation,
    write_json,
)
from dataset import build_classification, build_detection, build_semantic
from metrics import (
    classification_metrics,
    detection_map,
    multilabel_metrics,
    pose_ap,
    segmentation_metrics,
)
from models import DEFAULT_ARCH, build_model, export_onnx, freeze_backbone

IGNORE_INDEX = 255
DET_TASKS = ("detection", "instance_seg", "pose")


# ---------------------------------------------------------------------------
# Config / device
# ---------------------------------------------------------------------------

def load_config() -> dict[str, Any]:
    raw = os.environ.get("CONFIG_JSON", "{}")
    cfg: dict[str, Any] = {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            cfg = parsed
    except json.JSONDecodeError:
        print(f"[train] CONFIG_JSON is not valid JSON: {raw[:200]!r}", flush=True)
    # Legacy aliases: trainer v1 configs use `model`, `imgsz`.
    cfg.setdefault("model", cfg.get("arch") or "")
    cfg.setdefault("arch", cfg.get("model") or "")
    if "image_size" not in cfg:
        cfg["image_size"] = cfg.get("imgsz") or 0
    return cfg


def pick_device(cfg: dict[str, Any]) -> torch.device:
    want = str(cfg.get("device") or "auto").strip().lower()
    if want in ("", "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu:0")
        return torch.device("cpu")
    if want.startswith("xpu"):
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            print("[train] XPU requested but torch.xpu is unavailable; using CPU", flush=True)
            return torch.device("cpu")
        return torch.device(want if ":" in want else "xpu:0")
    if want.isdigit():
        if not torch.cuda.is_available():
            print("[train] CUDA requested but unavailable; using CPU", flush=True)
            return torch.device("cpu")
        return torch.device(f"cuda:{want}")
    if want in ("cuda", "gpu"):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def _materialize_base_weights(cfg: dict[str, Any]) -> str:
    """Download `base_weights_uri` to a local path (empty when not set)."""
    uri = str(cfg.get("base_weights_uri") or "").strip()
    if not uri:
        return ""
    if not uri.startswith("s3://"):
        return uri if os.path.isfile(uri) else ""
    from common import parse_uri, s3_client

    bucket, key = parse_uri(uri)
    local = os.path.join("/tmp", os.path.basename(key) or "base_weights.pt")
    s3_client().download_file(bucket, key, local)
    print(f"[train] base weights -> {local}", flush=True)
    return local


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _checkpoint_labels(path: str) -> list[str]:
    """Read the label space of a previous NF checkpoint (empty when unknown)."""
    if not path or not os.path.isfile(path):
        return []
    try:
        import torch

        obj = torch.load(path, map_location="cpu", weights_only=False)
        labels = obj.get("labels") if isinstance(obj, dict) else None
        return [str(name) for name in labels] if isinstance(labels, list) else []
    except Exception as exc:  # noqa: BLE001 - a foreign checkpoint may not load
        print(f"[train] cannot read labels from {path}: {exc}", flush=True)
        return []


def _safe_workers(requested: int) -> int:
    """Clamp DataLoader workers to what /dev/shm can serve.

    Each worker feeds tensors back over a file-descriptor backed by /dev/shm;
    a container without an explicit emptyDir mount gets Docker's 64 MB tmpfs
    and crashes with "unable to allocate shared memory ... No space left on
    device". Run single-process loading on such pods instead of failing.
    """
    if requested <= 0:
        return 0
    try:
        stat = os.statvfs("/dev/shm")
        free_mb = stat.f_bavail * stat.f_frsize / 1024 / 1024
    except OSError:
        return requested
    if free_mb < 1024:
        print(
            f"[train] /dev/shm only {free_mb:.0f} MB: forcing num_workers=0 "
            "(mount a memory-backed /dev/shm to use worker processes)",
            flush=True,
        )
        return 0
    return requested


def build_loaders(task: str, root: str, cfg: dict[str, Any]):
    batch = max(1, int(cfg.get("batch") or cfg.get("batch_size") or 16))
    # A missing key retains the historical default; an explicit zero disables
    # worker processes and must not be replaced by a truthiness fallback.
    requested_workers = int(cfg["workers"]) if cfg.get("workers") is not None else 2
    workers = _safe_workers(max(0, requested_workers))
    train: Any
    val: Any
    labels: list[str]
    if task in ("classification", "multilabel"):
        train, labels = build_classification(root, cfg, train=True)
        val, val_labels = build_classification(root, cfg, train=False)
        labels = labels or val_labels
        train_loader = DataLoader(train, batch_size=batch, shuffle=True,
                                  num_workers=workers, drop_last=False, pin_memory=True)
        val_loader = DataLoader(val, batch_size=batch, shuffle=False,
                                num_workers=workers, pin_memory=True)
        return train_loader, val_loader, labels, len(train), len(val)

    if task == "semantic_seg":
        train, labels = build_semantic(root, cfg, train=True)
        val, val_labels = build_semantic(root, cfg, train=False)
        labels = labels or val_labels
        train_loader = DataLoader(train, batch_size=batch, shuffle=True,
                                  num_workers=workers, pin_memory=True)
        val_loader = DataLoader(val, batch_size=batch, shuffle=False,
                                num_workers=workers, pin_memory=True)
        return train_loader, val_loader, labels, len(train), len(val)

    train, labels = build_detection(root, cfg, train=True, task=task)
    val, val_labels = build_detection(root, cfg, train=False, task=task)
    labels = labels or val_labels

    def collate(batch_items):
        images = torch.stack([item[0] for item in batch_items])
        targets = [item[1] for item in batch_items]
        return images, targets

    train_loader = DataLoader(train, batch_size=batch, shuffle=True,
                              num_workers=workers, collate_fn=collate, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch, shuffle=False,
                            num_workers=workers, collate_fn=collate, pin_memory=True)
    return train_loader, val_loader, labels, len(train), len(val)


# ---------------------------------------------------------------------------
# Training steps
# ---------------------------------------------------------------------------

def _train_classifier(model, loader, val_loader, labels, cfg, device, progress):
    import torch.nn as nn

    epochs = max(1, int(cfg.get("epochs") or 10))
    lr = float(cfg.get("lr") or cfg.get("learning_rate") or 1e-3)
    wd = float(cfg.get("weight_decay") or 1e-4)
    multi = str(cfg.get("task", "")).lower() in ("multilabel", "multi_label") or bool(cfg.get("multi_label"))
    criterion = nn.BCEWithLogitsLoss() if multi else nn.CrossEntropyLoss()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    best = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                out = model(images)
                if isinstance(out, dict):
                    out = out.get("logits", next(iter(out.values())))
                loss = criterion(out, targets.float() if multi else targets)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += float(loss.detach()) * len(images)
            seen += len(images)
        sched.step()

        metrics = _eval_classifier(model, val_loader, labels, device, multi)
        metrics["train_loss"] = round(running / max(seen, 1), 4)
        metrics["lr"] = round(opt.param_groups[0]["lr"], 6)
        score = metrics.get("mAP") if multi else metrics.get("accuracy")
        if score is not None and score > best:
            best = float(score)
        progress.log(epoch, {**metrics, "best_score": round(best, 4)})
        print(f"[train] epoch {epoch}/{epochs} {metrics}", flush=True)
    return best


def _eval_classifier(model, loader, labels, device, multi: bool):
    import torch.nn.functional as F

    model.eval()
    probs, truths = [], []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            out = model(images)
            if isinstance(out, dict):
                out = out.get("logits", next(iter(out.values())))
            if multi:
                probs.append(torch.sigmoid(out).cpu().numpy())
            else:
                probs.append(F.softmax(out, dim=1).cpu().numpy())
            truths.append(targets.cpu().numpy())
    import numpy as np

    if not probs:
        return {"accuracy": 0.0, "samples": 0}
    if multi:
        return multilabel_metrics(np.concatenate(truths), np.concatenate(probs), labels)
    return classification_metrics(np.concatenate(truths), np.concatenate(probs), labels)


def _train_torchvision_task(model, train_loader, val_loader, labels, cfg, device, progress, task):
    epochs = max(1, int(cfg.get("epochs") or 10))
    lr = float(cfg.get("lr") or cfg.get("learning_rate") or 1e-4)
    wd = float(cfg.get("weight_decay") or 1e-4)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 3), gamma=0.3)
    best = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        running, steps = 0.0, 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            opt.zero_grad(set_to_none=True)
            losses = model(images, targets)
            loss = sum(v for v in losses.values())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
            opt.step()
            running += float(loss.detach())
            steps += 1
        sched.step()

        metrics = _eval_torchvision_task(model, val_loader, labels, device, task)
        metrics["train_loss"] = round(running / max(steps, 1), 4)
        metrics["lr"] = round(opt.param_groups[0]["lr"], 6)
        score = float(metrics.get("mAP50") or metrics.get("OKS_AP") or 0.0)
        best = max(best, score)
        progress.log(epoch, {**metrics, "best_score": round(best, 4)})
        print(f"[train] epoch {epoch}/{epochs} {metrics}", flush=True)
    return best


def _eval_torchvision_task(model, loader, labels, device, task):
    model.eval()
    preds: list[dict[str, Any]] = []
    gts: list[dict[str, Any]] = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            outputs = model(images)
            for out, tgt in zip(outputs, targets):
                pred = {
                    "boxes": out["boxes"].detach().cpu().numpy(),
                    "scores": out["scores"].detach().cpu().numpy(),
                    "labels": out["labels"].detach().cpu().numpy(),
                }
                gt = {
                    "boxes": tgt["boxes"].numpy(),
                    "labels": tgt["labels"].numpy(),
                }
                if task == "instance_seg":
                    pred["masks"] = [
                        (m[0].detach().cpu().numpy() > 0.5) for m in out.get("masks", [])
                    ]
                    gt["masks"] = [m[0].numpy() > 0 for m in tgt.get("masks", [])]
                if task == "pose":
                    pred["keypoints"] = out["keypoints"].detach().cpu().numpy()
                    gt["keypoints"] = tgt["keypoints"].numpy()
                preds.append(pred)
                gts.append(gt)

    if task == "pose":
        return pose_ap(preds, gts)
    if task == "instance_seg":
        thresholds = (0.5, 0.75)
        return detection_map(preds, gts, thresholds, use_masks=True)
    thresholds = (float(x) for x in range(50, 100, 5))
    return detection_map(preds, gts, [t / 100 for t in thresholds])


def _train_semantic(model, train_loader, val_loader, labels, cfg, device, progress):
    import torch.nn.functional as F

    epochs = max(1, int(cfg.get("epochs") or 10))
    lr = float(cfg.get("lr") or cfg.get("learning_rate") or 1e-3)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=float(cfg.get("weight_decay") or 1e-4))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        running, steps = 0.0, 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(images)
            logits = out["out"] if isinstance(out, dict) else out
            loss = F.cross_entropy(logits, masks, ignore_index=IGNORE_INDEX)
            loss.backward()
            opt.step()
            running += float(loss.detach())
            steps += 1
        sched.step()
        metrics = _eval_semantic(model, val_loader, labels, device)
        metrics["train_loss"] = round(running / max(steps, 1), 4)
        best = max(best, float(metrics.get("mean_iou") or 0.0))
        progress.log(epoch, {**metrics, "best_score": round(best, 4)})
        print(f"[train] epoch {epoch}/{epochs} {metrics}", flush=True)
    return best


def _eval_semantic(model, loader, labels, device):
    import numpy as np

    n = len(labels)
    confusion = np.zeros((n, n), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            out = model(images)
            logits = out["out"] if isinstance(out, dict) else out
            pred = logits.argmax(1).cpu().numpy()
            truth = masks.numpy()
            for p, t in zip(pred, truth):
                valid = (t != IGNORE_INDEX) & (t >= 0) & (t < n)
                idx = t[valid].astype(np.int64) * n + p[valid].astype(np.int64)
                confusion += np.bincount(idx, minlength=n * n).reshape(n, n)
    return segmentation_metrics(confusion, labels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    input_uri = os.environ.get("INPUT_URI", "")
    output_uri = os.environ.get("OUTPUT_DIR", "")
    artifact_id = os.environ.get("ARTIFACT_ID", "") or "artifact"
    cfg = load_config()

    task = str(cfg.get("task") or cfg.get("task_type") or "detection").lower()
    if task in ("cls", "classification", "classify"):
        task = "classification"
    elif task in ("det", "detection", "object_detection"):
        task = "detection"
    elif task in ("seg", "instance_seg", "instance_segmentation"):
        task = "instance_seg"
    elif task in ("semantic_seg", "semantic_segmentation"):
        task = "semantic_seg"
    elif task in ("keypoint", "keypoints", "pose"):
        task = "pose"
    elif task in ("multilabel", "multi_label"):
        task = "multilabel"
    else:
        print(f"[train] unknown task '{task}', defaulting to detection", flush=True)
        task = "detection"

    arch = str(cfg.get("arch") or cfg.get("model") or DEFAULT_ARCH[task])
    if arch.endswith(".pt") or arch.endswith(".onnx"):
        arch = DEFAULT_ARCH[task]
    epochs = max(1, int(cfg.get("epochs") or 10))
    seed = int(cfg.get("seed") or 42)
    torch.manual_seed(seed)
    device = pick_device(cfg)

    print(f"[train] task={task} arch={arch} device={device} epochs={epochs} artifact={artifact_id}", flush=True)
    aug = resolve_augmentation(cfg)
    print(f"[train] augmentation: {augmentation_summary(aug)}", flush=True)
    skipped = [k for k in ("mosaic", "mixup", "copy_paste") if float(
        (cfg.get("augmentation") or {}).get(k, 0) or 0
    ) > 0]
    if skipped:
        print(f"[train] note: {', '.join(skipped)} need multi-image collation; "
              "accepted but not applied", flush=True)
    for key in ("INPUT_URI", "OUTPUT_DIR"):
        if not os.environ.get(key):
            print(f"[train] {key} not set; running without S3 I/O", flush=True)

    if not input_uri:
        print("[train] INPUT_URI is required", flush=True)
        return 2
    if input_uri.startswith("s3://"):
        root = download_dataset(input_uri, "/data")
    else:
        root = input_uri  # local dataset (dev runs and the trainer self-test)
        if not os.path.isdir(root):
            print(f"[train] INPUT_URI is not a directory: {root}", flush=True)
            return 2

    progress = Progress(output_uri, epochs)
    progress.write(0, {"stage": "preparing"})

    train_loader, val_loader, labels, n_train, n_val = build_loaders(task, root, cfg)

    weights_path = _materialize_base_weights(cfg)
    # Incremental learning (ADR-014 D-07): a continuation keeps the previous
    # label space, so a dataset containing only the new classes still produces
    # a model whose head covers old + new. Dataset indices stay valid because
    # the previous labels are appended after the dataset's own order; the head
    # is re-initialised (tensor shapes differ) while the backbone transfers.
    base_labels = _checkpoint_labels(weights_path)
    added: list[str] = []
    for name in base_labels + [str(c) for c in (cfg.get("new_classes") or [])]:
        if name and name not in labels:
            labels.append(name)
            added.append(name)

    # torchvision detectors reserve class 0 for background: the model is built
    # with N+1 logits while the dataset uses labels 1..N. The ONNX wrapper maps
    # them back to 0-based ids, which is the adapter contract.
    num_classes = len(labels) + 1 if task in DET_TASKS else max(1, len(labels))
    size = int(cfg.get("image_size") or 0) or {
        "classification": 224, "multilabel": 224, "detection": 640,
        "instance_seg": 640, "pose": 640, "semantic_seg": 512,
    }[task]
    print(f"[train] {n_train} train / {n_val} val, {num_classes} classes, {len(labels)} labels", flush=True)

    model = build_model(
        task, arch, num_classes, size,
        pretrained=bool(cfg.get("pretrained", True)),
        weights_path=weights_path,
        num_keypoints=int(cfg.get("num_keypoints") or 17),
        encoder_name=str(cfg.get("encoder") or "resnet34"),
    )
    if bool(cfg.get("freeze_backbone")):
        freeze_backbone(model)
    model.to(device)

    training_started = time.monotonic()
    if task in ("classification", "multilabel"):
        best = _train_classifier(model, train_loader, val_loader, labels, cfg, device, progress)
    elif task == "semantic_seg":
        best = _train_semantic(model, train_loader, val_loader, labels, cfg, device, progress)
    else:
        best = _train_torchvision_task(model, train_loader, val_loader, labels, cfg, device, progress, task)
    train_time = time.monotonic() - training_started

    # Final re-validation and export are artifact work, not training time.
    if task in ("classification", "multilabel"):
        final = _eval_classifier(model, val_loader, labels, device, task == "multilabel")
    elif task == "semantic_seg":
        final = _eval_semantic(model, val_loader, labels, device)
    else:
        final = _eval_torchvision_task(model, val_loader, labels, device, task)

    # ---- artifacts -------------------------------------------------------
    local_onnx = f"/tmp/{artifact_id}.onnx"
    export_onnx(model, task, size, local_onnx, topk=int(cfg.get("export_topk") or 100))

    local_ckpt = f"/tmp/{artifact_id}.pt"
    torch.save(
        {
            "task": task,
            "arch": arch,
            "labels": labels,
            "state_dict": model.state_dict(),
            "config": cfg,
            "framework": "torch",
        },
        local_ckpt,
    )

    metrics: dict[str, Any] = {
        "task": task,
        "architecture": arch,
        "framework": "torch",
        "epochs": epochs,
        "final_epoch": epochs,
        "num_classes": num_classes,
        "train_samples": n_train,
        "val_samples": n_val,
        "best_score": round(float(best), 4),
        "device": str(device),
        **final,
    }
    metrics = build_training_metrics(metrics, train_time_s=train_time)
    local_metrics = f"/tmp/{artifact_id}.metrics.json"
    write_json(local_metrics, metrics)

    manifest = build_manifest(
        task=task,
        architecture=arch,
        framework="torch",
        input_size=size,
        labels=labels,
        metrics=final,
        dataset={"splits": {"train": n_train, "val": n_val}, "source": os.path.basename(input_uri)},
        extra={
            "checkpoint": f"{artifact_id}.pt",
            "incremental": {
                "base_weights_uri": str(cfg.get("base_weights_uri") or "") or None,
                "previous_labels": base_labels,
                # Classes the label space gained from the previous checkpoint or
                # the `new_classes` config (ADR-014 D-07).
                "added_to_label_space": added,
                "head": "reinitialised" if base_labels else "fresh",
            },
            "export": {"opset": 17, "outputs": "see outputs"},
            "base_weights": {
                "source": "s3" if weights_path else ("upstream" if cfg.get("pretrained", True) else "scratch"),
                # ADR-014 D-09: weights carry their own license, independent of
                # the code license. Model-pack weights declare it in the pack.
                "license": {
                    "spdx": str(cfg.get("base_weights_license") or "unknown"),
                    "commercial_use": bool(cfg.get("base_weights_commercial", False)),
                    "source": str(weights_path or cfg.get("base_weights_uri") or "upstream"),
                },
            },
            "license": {"code": "Apache-2.0", "framework": "torchvision BSD-3-Clause / timm Apache-2.0"},
        },
    )
    local_manifest = f"/tmp/{artifact_id}.metadata-src.json"
    write_json(local_manifest, manifest)

    uris: dict[str, str] = {}
    if output_uri:
        for local, name, key in (
            (local_onnx, f"{artifact_id}.onnx", "onnx"),
            (local_ckpt, f"{artifact_id}.pt", "pt"),
            (local_metrics, f"{artifact_id}.metrics.json", "metrics"),
            (local_manifest, f"{artifact_id}.metadata.json", "metadata"),
        ):
            try:
                uris[key] = publish_file(local, output_uri, name)
            except Exception as e:  # noqa: BLE001
                print(f"[train] upload {name} failed: {e}", flush=True)
        metrics["onnx_uri"] = uris.get("onnx", "")
        metrics["weights_uri"] = uris.get("pt", "")
        metrics["manifest_uri"] = uris.get("metadata", "")
        write_json(local_metrics, metrics)
        try:
            publish_file(local_metrics, output_uri, f"{artifact_id}.metrics.json")
        except Exception as e:  # noqa: BLE001
            print(f"[train] metrics re-upload failed: {e}", flush=True)

    progress.write(epochs, {"stage": "done", "best_score": round(float(best), 4)})
    print(f"[train] SUCCESS {json.dumps({k: v for k, v in metrics.items() if not isinstance(v, dict)})}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
