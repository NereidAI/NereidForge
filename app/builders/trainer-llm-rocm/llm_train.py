# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""NereidForge LLM/VLM trainer entrypoint (LLaMA-Factory).

Reads the train config, downloads the base model + `llm-sft` JSONL dataset from
SeaweedFS S3, runs `llamafactory-cli train`, uploads the LoRA adapter together
with live progress.json / final metrics.json back to SeaweedFS.

Usage (env vars, consistent with the CV trainer):
  INPUT_URI   s3://nereidforge/<dataset>/v<n>/    (dataset prefix containing .jsonl)
  OUTPUT_DIR  s3://nereidforge/models/<job_id>/   (weights + metrics destination)
  CONFIG_JSON JSON training config (contains the `llm` LF argument block)
  ARTIFACT_ID unique id (job id)
  NPROC       number of local GPUs to run under torchrun (default 1)
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from nereidforge_trainer import build_training_metrics, s3_client

OUT_ROOT = "/app/output"
DATASET_ROOT = "/datasets"
BASE_ROOT = "/base"


def parse_uri(uri: str):
    rest = uri[len("s3://"):]
    parts = rest.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key.rstrip("/")


def list_s3_objects(s3, bucket: str, prefix: str) -> list[str]:
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def download_dir(s3, bucket: str, prefix: str, local_root: str) -> int:
    """Download every object under `prefix` into `local_root`, mirroring keys."""
    keys = list_s3_objects(s3, bucket, prefix)
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
            print(f"[llm-train] skip {rel}: {e}", flush=True)
    return downloaded


def upload_progress(s3, bucket: str, okey: str, epoch: float, total: float, loss) -> None:
    """Upload a small progress.json so the worker can surface live progress."""
    try:
        data = {
            "current_epoch": int(epoch),
            "total_epochs": int(total),
            "progress_pct": int(round(epoch * 100.0 / total)) if total else 99,
            "loss": loss,
            "updated_at": time.time(),
        }
        s3.put_object(Bucket=bucket, Key=okey.rstrip("/") + "/progress.json", Body=json.dumps(data))
    except Exception as e:
        print(f"[llm-train] progress upload failed: {e}", flush=True)


def wait_and_upload_progress(s3, bucket: str, okey: str, total_epochs: float, proc) -> None:
    """Poll trainer_state.json while training runs and push progress to S3."""
    state_path = Path(OUT_ROOT) / "trainer_state.json"
    last_epoch = -1.0
    while proc.poll() is None:
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text())
                epoch = float(state.get("epoch") or 0)
                history = state.get("log_history", []) or []
                loss = history[-1].get("loss") if history else None
                if epoch != last_epoch:
                    upload_progress(s3, bucket, okey, epoch, total_epochs, loss)
                    last_epoch = epoch
            except Exception:
                pass
        time.sleep(30)
    # Final flush after the process exits.
    try:
        state = json.loads(state_path.read_text())
        history = state.get("log_history", []) or []
        loss = history[-1].get("loss") if history else None
        upload_progress(s3, bucket, okey, float(state.get("epoch") or 0), total_epochs, loss)
    except Exception:
        pass


def write_dataset_info(jsonl_files: list[str]) -> None:
    """Write the LLaMA-Factory dataset_info.json mapping our JSONL to 'train'/'val'."""
    os.makedirs(DATASET_ROOT, exist_ok=True)
    info: dict = {}
    if "train.jsonl" in jsonl_files:
        info["train"] = {
            "file_name": "train.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
        }
    else:
        info["train"] = {
            "file_name": sorted(jsonl_files)[0],
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
        }
    if "val.jsonl" in jsonl_files:
        info["val"] = {
            "file_name": "val.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
        }
    with open(os.path.join(DATASET_ROOT, "dataset_info.json"), "w") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def build_lf_yaml(llm: dict) -> str:
    """Render the LLaMA-Factory configuration YAML from the NF `llm` block."""
    yaml_path = "/app/lf_config.yaml"
    cfg = dict(llm)
    # drop NF-internal keys that LLaMA-Factory does not understand
    for internal in ("base_model_uri", "num_nodes", "num_proc_per_node"):
        cfg.pop(internal, None)
    cfg["model_name_or_path"] = BASE_ROOT
    cfg["output_dir"] = OUT_ROOT
    cfg["dataset_dir"] = DATASET_ROOT
    cfg["dataset"] = ["train"]
    if "use_cpu" not in cfg:
        try:
            import torch
            cfg["use_cpu"] = not torch.cuda.is_available()
        except Exception:
            cfg["use_cpu"] = True
    # Adapter-only artifacts are the v1 default; merged export is a follow-up.
    cfg.setdefault("save_only_model", True)

    with open(yaml_path, "w") as f:
        for key, value in cfg.items():
            if isinstance(value, (list, dict)):
                f.write(f"{key}: {json.dumps(value, ensure_ascii=False)}\n")
            elif isinstance(value, bool):
                f.write(f"{key}: {'true' if value else 'false'}\n")
            elif value is None:
                f.write(f"{key}: null\n")
            else:
                f.write(f"{key}: {value}\n")
    return yaml_path


def lf_command(yaml_path: str, nproc: int) -> list[str]:
    """Build the llamafactory CLI command, using torchrun for NPROC > 1."""
    entry = shutil.which("llamafactory-cli")
    if entry:
        base = [entry, "train", yaml_path]
    else:
        base = [sys.executable, "-m", "llamafactory.cli.train", yaml_path]
    if nproc > 1:
        return [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(nproc), *base]
    return base


def upload_dir(s3, local_dir: str, bucket: str, okey: str) -> list[str]:
    """Recursively upload a local directory (LoRA adapter) to S3."""
    keys = []
    for root, _, files in os.walk(local_dir):
        for fn in files:
            fp = os.path.join(root, fn)
            rel = os.path.relpath(fp, local_dir)
            dst = okey.rstrip("/") + "/" + rel.replace(os.sep, "/")
            s3.upload_file(fp, bucket, dst)
            keys.append(dst)
    return keys


def load_config(raw: str, s3) -> dict:
    """Parse CONFIG_JSON: an inline JSON document or an s3:// path to one."""
    if raw.startswith("s3://"):
        bucket, key = parse_uri(raw)
        resp = s3.get_object(Bucket=bucket, Key=key)
        raw = resp["Body"].read().decode("utf-8")
    return json.loads(raw)


def main() -> None:
    input_uri = os.environ.get("INPUT_URI", "")
    output_uri = os.environ.get("OUTPUT_DIR", "")
    s3 = s3_client()
    cfg = load_config(os.environ.get("CONFIG_JSON", "{}") or "{}", s3)
    artifact_id = os.environ.get("ARTIFACT_ID", cfg.get("artifact_id", ""))

    llm = cfg.get("llm", {}) or {}
    dataset_uri = cfg.get("dataset_uri") or cfg.get("dataset_output_dir") or input_uri
    out_uri = cfg.get("output_dir") or output_uri
    base_uri = str(llm.get("base_model_uri") or "")
    if not dataset_uri or not out_uri or not base_uri:
        print("[llm-train] FIXME: missing dataset_uri/output_dir/base_model_uri, aborting", flush=True)
        sys.exit(1)

    dBucket, dPrefix = parse_uri(dataset_uri)
    oBucket, oKey = parse_uri(out_uri)

    # --- download base model ---
    bBucket, bPrefix = parse_uri(base_uri)
    n = download_dir(s3, bBucket, bPrefix, BASE_ROOT)
    print(f"[llm-train] base model files: {n} under {BASE_ROOT}", flush=True)
    if n == 0:
        print(f"[llm-train] base model empty at {base_uri}, aborting", flush=True)
        sys.exit(1)

    # --- download jsonl dataset ---
    jl = download_jsonl(s3, dBucket, dPrefix)
    if not jl:
        print(f"[llm-train] no .jsonl under {dataset_uri}, aborting", flush=True)
        sys.exit(1)
    write_dataset_info(jl)

    # --- render LF config and start training ---
    total_epochs = float(llm.get("num_train_epochs") or 3)
    yaml_path = build_lf_yaml(llm)
    nproc = int(os.environ.get("NPROC", "1") or cfg.get("numProcPerNode") or 1)
    cmd = lf_command(yaml_path, nproc)
    print(f"[llm-train] running: {' '.join(cmd)}", flush=True)
    training_started = time.monotonic()
    proc = subprocess.Popen(cmd)
    wait_and_upload_progress(s3, oBucket, oKey, total_epochs, proc)
    train_time = time.monotonic() - training_started
    rc = proc.returncode
    print(f"[llm-train] llamafactory exit={rc}", flush=True)
    if rc != 0:
        sys.exit(rc)

    # --- upload adapter + metrics ---
    state_path = Path(OUT_ROOT) / "trainer_state.json"
    metrics: dict = {"final_epoch": int(total_epochs)}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
            history = state.get("log_history", []) or []
            losses = [h.get("loss") for h in history if h.get("loss") is not None]
            if losses:
                metrics["final_loss"] = float(losses[-1])
            if state.get("global_step"):
                metrics["global_step"] = int(state["global_step"])
        except Exception:
            pass
    uploaded = upload_dir(s3, OUT_ROOT, oBucket, oKey + "/" + artifact_id)
    metrics["weights_uri"] = f"s3://{oBucket}/{oKey}/{artifact_id}/adapter_config.json"
    metrics["adapter_key"] = f"{oKey}/{artifact_id}"
    metrics = build_training_metrics(metrics, train_time_s=train_time)
    with open("/app/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    mkey = oKey + "/" + artifact_id + ".metrics.json"
    s3.upload_file("/app/metrics.json", oBucket, mkey)
    print(f"[llm-train] uploaded {len(uploaded)} files; metrics -> s3://{oBucket}/{mkey}", flush=True)
    upload_progress(s3, oBucket, oKey, total_epochs, total_epochs, metrics.get("final_loss"))
    print("[llm-train] SUCCESS", flush=True)
    sys.exit(0)


def download_jsonl(s3, bucket: str, prefix: str) -> list[str]:
    """Download every .jsonl under the dataset prefix into /datasets."""
    os.makedirs(DATASET_ROOT, exist_ok=True)
    names = []
    for k in list_s3_objects(s3, bucket, prefix):
        if not k.endswith(".jsonl"):
            continue
        rel = os.path.basename(k)
        dst = os.path.join(DATASET_ROOT, rel)
        try:
            s3.download_file(bucket, k, dst)
            names.append(rel)
        except Exception as e:
            print(f"[llm-train] skip dataset {rel}: {e}", flush=True)
    return names


if __name__ == "__main__":
    main()
