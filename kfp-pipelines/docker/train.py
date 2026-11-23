# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""NereidForge trainer entrypoint - PP-YOLOE (PaddleDetection) - Apache-2.0.

Reads training config, downloads dataset from seaweedfs S3,
runs PP-YOLOE training (CPU or GPU), uploads weights + metrics to seaweedfs.

Usage (env vars):
  INPUT_URI   s3://nereidforge/yolo-train/   (dataset uri with data.yaml/images/labels)
  OUTPUT_DIR  s3://nereidforge/models/...     (weights + metrics destination)
  CONFIG_JSON JSON training config (epochs, imgsz, batch, model base, etc)
  ARTIFACT_ID unique id
"""

import json
import os
import sys
import time
import subprocess

import boto3

S3_ENDPOINT = os.environ.get(
    "S3_ENDPOINT",
    "http://seaweedfs-s3.nereidforge.svc.cluster.local:8333",
)


def s3_client():
    """s3_client creates the standalone job's client with explicit credentials."""
    access_key = os.environ.get("S3_ACCESS_KEY", "")
    secret_key = os.environ.get("S3_SECRET_KEY", "")
    if not access_key.strip() or not secret_key.strip():
        raise RuntimeError(
            "S3_ACCESS_KEY and S3_SECRET_KEY must contain non-empty "
            "operator-supplied credentials before S3 access."
        )
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="us-east-1",
        config=boto3.session.Config(signature_version="s3v4"),
    )


def parse_uri(uri):
    rest = uri[len("s3://"):]
    parts = rest.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key.rstrip("/")


def upload_progress(s3, bucket, okey, epoch, total, loss):
    """Upload a small progress.json so the API/UI can show live training progress."""
    try:
        data = {
            "current_epoch": epoch,
            "total_epochs": total,
            "progress_pct": int(round(epoch * 100 / total)),
            "loss": loss,
            "updated_at": time.time(),
        }
        s3.put_object(Bucket=bucket, Key=okey.rstrip("/") + "/progress.json", Body=json.dumps(data))
    except Exception as e:
        print(f"[train] progress upload failed: {e}", flush=True)


def list_s3_objects(s3, bucket, prefix):
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def list_paddlex_objects(s3, bucket, prefix):
    """List objects for PaddleDetection dataset structure."""
    return list_s3_objects(s3, bucket, prefix)


def parse_yolo_to_coco(input_uri, local_root):
    """Convert YOLO format dataset to COCO format for PaddleDetection."""
    import yaml
    import json
    import os
    import shutil
    
    s3 = s3_client()
    bucket, prefix = parse_uri(uri)
    keys = list_s3_objects(s3, bucket, prefix)
    
    # Download dataset
    local_root = "/data"
    os.makedirs(local_root, exist_ok=True)
    
    for k in keys:
        rel = k[len(prefix):].lstrip("/")
        if not rel:
            continue
        dst = os.path.join(local_root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            s3.download_file(bucket, k, dst)
        except Exception as e:
            print(f"[train] skip {rel}: {e}", flush=True)
    
    # Find data.yaml
    data_yaml = None
    for cand in ["/data/data.yaml", "/data/yolo-train/data.yaml"]:
        if os.path.exists(cand):
            data_yaml = cand
            break
    
    if data_yaml is None:
        print("[train] data.yaml not found in dataset, aborting", flush=True)
        sys.exit(1)
    
    with open(data_yaml) as f:
        yaml_data = yaml.safe_load(f)
    
    # Parse YOLO format to COCO for PaddleDetection
    names = yaml_data.get("names", {})
    nc = yaml_data.get("nc", len(names))
    
    # Convert YOLO labels to COCO format for PaddleDetection
    images_dir = "/data/yolo-train/images" if os.path.exists("/data/yolo-train/images") else "/data/images"
    labels_dir = "/data/yolo-train/labels" if os.path.exists("/data/yolo-train/labels") else "/data/labels"
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print("[train] images or labels directory not found", flush=True)
        sys.exit(1)
    
    # Create COCO format for PaddleDetection
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in names.items()]
    }
    
    img_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    ann_id = 0
    img_id = 0
    
    import cv2
    for img_file in img_files:
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
                    x = (cx - bw/2) * w
                    y = (cy - bh/2) * h
                    w_box = bw * w
                    h_box = bh * h
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
    
    print(f"[train] Converted YOLO to COCO format: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")
    return "/data"


def main():
    input_uri = os.environ.get("INPUT_URI", "")
    output_uri = os.environ.get("OUTPUT_DIR", "")
    config_json = os.environ.get("CONFIG_JSON", "{}")
    artifact_id = os.environ.get("ARTIFACT_ID", "")
    cfg = json.loads(config_json)

    s3 = s3_client()
    bucket, prefix = parse_uri(input_uri)
    print(f"[train] dataset: s3://{bucket}/{prefix}", flush=True)

    # Download the dataset to local storage
    local_root = "/data"
    os.makedirs(local_root, exist_ok=True)
    keys = list_s3_objects(s3, bucket, prefix)
    if not keys:
        print(f"[train] no objects found under {prefix}, aborting", flush=True)
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
            print(f"[train] skip {rel}: {e}", flush=True)
    print(f"[train] downloaded {downloaded} files to {local_root}", flush=True)

    # Convert YOLO format to COCO for PaddleDetection
    data_root = parse_yolo_to_coco(input_uri, "/data")
    
    # Training config
    epochs = int(cfg.get("epochs", 2))
    imgsz = int(cfg.get("imgsz", 640))
    batch = int(cfg.get("batch", 4))
    base = cfg.get("model", "PP-YOLOE-L")
    device = cfg.get("device", "cpu")
    
    # Training config for PaddleDetection
    epochs = int(cfg.get("epochs", 2))
    imgsz = int(cfg.get("imgsz", 640))
    batch = int(cfg.get("batch", 4))
    base = cfg.get("model", "PP-YOLOE-L")
    device = cfg.get("device", "cpu")
    
    # Build training command for PaddleDetection
    # Using PaddleX high-level API
    from paddlex import create_model
    import paddle
    
    # Set device
    if device == "gpu" or (device != "cpu" and paddle.device.is_compiled_with_cuda()):
        paddle.set_device("gpu:0")
    else:
        paddle.set_device("cpu")
    
    # Create PP-YOLOE model
    model = create_model(model_name="PP-YOLOE-L")
    
    # Training config
    train_config = {
        "epochs": epochs,
        "batch_size": batch,
        "learning_rate": 0.001,
        "save_interval": 1,
        "eval_interval": 1,
        "save_dir": "/app/output",
        "device": "gpu" if paddle.device.is_compiled_with_cuda() else "cpu",
        "pretrain_weights": "PP-YOLOE-L",
    }
    
    print(f"[train] PP-YOLOE training epochs={epochs} imgsz={imgsz} batch={batch} device={device}", flush=True)
    print(f"[train] dataset root: {data_root}", flush=True)
    
    t0 = time.time()
    
    # Train using PaddleX
    try:
        model.train(
            train_data=os.path.join(data_root, "train"),
            val_data=os.path.join(data_root, "val"),
            **train_config
        )
        elapsed = time.time() - t0
        print(f"[train] training done in {elapsed:.1f}s", flush=True)
    except Exception as e:
        print(f"[train] training failed: {e}", flush=True)
        sys.exit(1)
    
    # Export model to ONNX
    onnx_path = None
    try:
        model.export(output_dir="/app/output", format="onnx")
        onnx_files = [f for f in os.listdir("/app/output") if f.endswith(".onnx")]
        if onnx_files:
            onnx_path = os.path.join("/app/output", onnx_files[0])
            print(f"[train] ONNX exported: {onnx_path}", flush=True)
    except Exception as e:
        print(f"[train] ONNX export failed: {e}", flush=True)
    
    # Metrics - extract from training logs
    metrics = {
        "mAP50-95": 0.0,
        "mAP50": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "final_epoch": epochs,
        "train_time_s": round(time.time() - t0, 1),
    }
    
    # Try to extract metrics from training logs
    try:
        log_file = os.path.join("/app/output", "train.log")
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                content = f.read()
                # Try to extract mAP from logs
                import re
                map50_match = re.search(r"mAP.*?50.*?(\d+\.?\d*)", content)
                if map50_match:
                    metrics["mAP50"] = float(map50_match.group(1))
                map_match = re.search(r"mAP.*?(\d+\.\d+)", content)
                if map_match:
                    metrics["mAP50-95"] = float(map_match.group(1))
    except Exception as e:
        print(f"[train] metrics extraction failed: {e}", flush=True)
    
    # Upload weights, ONNX artifacts, and metrics to SeaweedFS
    if output_uri:
        ob, okey = parse_uri(output_uri)
        s3 = s3_client()
        
        # Upload best model
        best_model_path = "/app/output/best_model/model.pdparams"
        if os.path.exists(best_model_path):
            wkey = okey + "/" + artifact_id + ".pdparams"
            s3.upload_file(best_model_path, ob, wkey)
            print(f"[train] weights uploaded s3://{ob}/{wkey}", flush=True)
            metrics["weights_uri"] = f"s3://{ob}/{wkey}"
        
        # Upload ONNX model
        onnx_files = [f for f in os.listdir("/app/output") if f.endswith(".onnx")]
        if onnx_files:
            onnx_path = os.path.join("/app/output", onnx_files[0])
            onnx_key = okey + "/" + artifact_id + ".onnx"
            s3.upload_file(onnx_path, ob, onnx_key)
            print(f"[train] onnx uploaded s3://{ob}/{onnx_key}", flush=True)
            metrics["onnx_uri"] = f"s3://{ob}/{onnx_key}"
        
        # Upload metrics
        with open("/app/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        mkey = okey + "/" + artifact_id + ".metrics.json"
        s3.upload_file("/app/metrics.json", ob, mkey)
        print(f"[train] metrics uploaded s3://{ob}/{mkey}", flush=True)
        print(f"[train] metrics: {json.dumps(metrics)}", flush=True)

    print("[train] SUCCESS", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
