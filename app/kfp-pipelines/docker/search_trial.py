# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Katib search trial for model optimization.

Runs one optimization+validation pass with a parameter set injected by Katib
and writes the Katib metric file at /output/metrics (or $OUTPUT_DIR).

Metrics emitted:
  latency_ms      mean inference latency of the optimized model (minimize)
  mAP50           validation mAP@50 of the optimized model (maximize)
  map_drop_pct    mAP drop vs the source model (minimize)
  size_reduction_pct  model size reduction (maximize)

Env contract (injected by Katib trialParameters plus NF job context):
  INPUT_URI       s3:// source onnx
  DATASET_URI     s3:// evaluation dataset (YOLO or COCO layout)
  OUTPUT_URI      s3:// artifact dir for this trial
  ARTIFACT_ID     trial id
  QUANT_METHOD    graph|fp16|int8-dynamic|int8-static
  CALIB_N         calibration images for int8-static
  IMGSZ           inference size
  CONF_THR        detection confidence threshold
"""

import json
import os
import sys

import boto3

S3_ENDPOINT = os.environ.get(
    "S3_ENDPOINT", "http://seaweedfs-s3.nereidforge.svc.cluster.local:8333"
)


def s3_client():
    """s3_client creates the standalone trial's client with explicit credentials."""
    access_key = os.environ.get("S3_ACCESS_KEY", "")
    secret_key = os.environ.get("S3_SECRET_KEY", "")
    if not access_key.strip() or not secret_key.strip():
        raise RuntimeError(
            "S3_ACCESS_KEY and S3_SECRET_KEY must contain non-empty "
            "operator-supplied credentials before S3 access."
        )
    return boto3.client(
        "s3", endpoint_url=S3_ENDPOINT,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="us-east-1",
    )


def parse_uri(uri):
    rest = uri[len("s3://"):]
    parts = rest.split("/", 1)
    return parts[0], (parts[1] if len(parts) > 1 else "").rstrip("/")


def download(s3, uri, local):
    b, k = parse_uri(uri)
    print(f"[trial] download parse: bucket={b!r} key={k!r} from {uri!r}", file=sys.stderr, flush=True)
    s3.download_file(b, k, local)


def main():
    # Katib's StdOut collector parses ONLY stdout lines as 'key: value'; keep
    # stdout clean by routing all diagnostic output to stderr.
    _stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        _main()
    finally:
        sys.stdout = _stdout


def _main():
    input_uri = os.environ.get("INPUT_URI", "")
    dataset_uri = os.environ.get("DATASET_URI", "")
    output_uri = os.environ.get("OUTPUT_URI", "")
    artifact_id = os.environ.get("ARTIFACT_ID", "")
    method = os.environ.get("QUANT_METHOD", "graph")
    calib_n = int(os.environ.get("CALIB_N", "200"))
    imgsz = int(os.environ.get("IMGSZ", "640"))
    conf_thr = float(os.environ.get("CONF_THR", "0.25"))
    target_device = os.environ.get("TARGET_DEVICE", "cpu")

    import optimize  # noqa: PLC0415  (same image; reuse quantize+benchmark)

    s3 = s3_client()
    src = "/tmp/trial-src.onnx"
    print(f"[trial] downloading {input_uri!r}", file=sys.stderr, flush=True)
    download(s3, input_uri, src)

    # --- Quantize -------------------------------------------------------------
    cfg = {
        "target_device": target_device,
        "quant_method": method,
        "dtype": "fp16" if method == "fp16" else ("int8" if method.startswith("int8") else ""),
        "calib_n": calib_n,
        "imgsz": imgsz,
        "calib_uri": dataset_uri,
    }
    try:
        qpath = optimize.apply_quantization(src, method, cfg, s3, imgsz)
    except Exception as e:  # noqa: BLE001
        print(f"[trial] quantization failed ({e}); marking trial invalid", file=sys.stderr, flush=True)
        _write_metrics({"invalid": 1.0, "mAP50": 0.0, "latency_ms": 1e6,
                        "map_drop_pct": 100.0, "size_reduction_pct": 0.0})
        sys.exit(1)

    # --- Benchmark ------------------------------------------------------------
    import onnxruntime  # noqa: PLC0415

    shape = optimize.input_shape_for(
        onnxruntime.InferenceSession(src, providers=["CPUExecutionProvider"]), imgsz
    )
    _, src_stats = optimize.benchmark_ort(src, shape, warmup=5, measure=30)
    _, opt_stats = optimize.benchmark_ort(qpath, shape, warmup=5, measure=30)
    opt_size = os.path.getsize(qpath)
    src_size = os.path.getsize(src)

    # --- Evaluate (subprocess; evaluate.py reads/writes via S3) ---------------
    import subprocess  # noqa: PLC0415

    opt_s3_uri = f"{output_uri.rstrip('/')}/{artifact_id}/{artifact_id}.optimized.onnx"
    optimize.upload(s3, qpath, opt_s3_uri)

    eval_uri = f"{output_uri.rstrip('/')}/{artifact_id}/"
    env = dict(os.environ)
    env.update({
        "INPUT_URI": opt_s3_uri,
        "DATASET_URI": dataset_uri,
        "OUTPUT_URI": eval_uri,
        "ARTIFACT_ID": artifact_id,
        "CONFIG_JSON": json.dumps({
            "task_type": "detection", "conf_threshold": conf_thr,
            "imgsz": imgsz, "max_images": 0,
        }),
    })
    import shutil  # noqa: PLC0415
    subprocess.run(["python", "/app/evaluate.py"], env=env, check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    mp = f"/tmp/{artifact_id}.metrics.json"
    opt_mp = "/tmp/opt.metrics.json"
    if os.path.exists(mp):
        shutil.copy(mp, opt_mp)
    metrics = {}
    if os.path.exists(opt_mp):
        metrics = json.load(open(opt_mp))
    map50 = float(metrics.get("mAP50") or 0.0)

    env["INPUT_URI"] = input_uri
    subprocess.run(["python", "/app/evaluate.py"], env=env, check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    src_metrics = {}
    if os.path.exists(mp):
        src_metrics = json.load(open(mp))
    src_map50 = float(src_metrics.get("mAP50") or 0.0)
    map_drop = round((src_map50 - map50) * 100, 4) if src_map50 > 0 else 0.0

    result = {
        "mAP50": round(map50, 4),
        "latency_ms": round(float(opt_stats.get("avg_ms", 0)), 4),
        "map_drop_pct": map_drop,
        "size_reduction_pct": round((1 - opt_size / src_size) * 100, 2) if src_size else 0.0,
    }
    print(f"[trial] result: {json.dumps(result)}", file=sys.stderr, flush=True)
    _write_metrics(result)
    # Persist the full result to S3 so the worker can pick the best trial by
    # objective even when Katib's StdOut collector fails to surface metrics.
    try:
        s3.put_object(
            Bucket=parse_uri(output_uri)[0],
            Key=parse_uri(output_uri)[1].rstrip("/") + f"/{artifact_id}/{artifact_id}.trial.json",
            Body=json.dumps(result),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[trial] result upload failed: {e}", file=sys.stderr, flush=True)


def _write_metrics(metrics: dict) -> None:
    out_dir = os.environ.get("OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "metrics")
    with open(path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    # Katib's StdOut collector parses ONLY stdout lines as 'key: value'; write
    # the metric lines to the real stdout (sys.__stdout__ survives our
    # diagnostic-output redirection).
    real_out = sys.__stdout__
    for k, v in metrics.items():
        print(f"{k}: {v}", file=real_out, flush=True)
    print(f"[trial] wrote {path}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
