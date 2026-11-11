# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""NereidForge LLM/VLM trainer entrypoint for Intel XPU (native torch.xpu).

LLaMA-Factory 0.9.4 does not support native torch.xpu, so this vendor variant
uses the stack validated on Arc B580 (see app/builders/trainer-xpu/README.md):
transformers + peft + bitsandbytes XPU 4-bit QLoRA, single-card, bf16.

It keeps the exact same S3/config/artifact contract as the CUDA/ROCm
(LLaMA-Factory) entrypoint so the control plane is vendor-agnostic:
  INPUT_URI   s3://... dataset prefix containing .jsonl (sharegpt)
  OUTPUT_DIR  s3://... models/<job_id>/
  CONFIG_JSON inline JSON or s3:// path; the `llm` block uses LF-style keys
  ARTIFACT_ID job id
"""

import json
import os
import sys
import time

import torch
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


def list_s3_objects(s3, bucket: str, prefix: str):
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def download_dir(s3, bucket: str, prefix: str, local_root: str) -> int:
    n = 0
    for k in list_s3_objects(s3, bucket, prefix):
        rel = k[len(prefix):].lstrip("/")
        if not rel:
            continue
        dst = os.path.join(local_root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            s3.download_file(bucket, k, dst)
            n += 1
        except Exception as e:
            print(f"[llm-xpu] skip {rel}: {e}", flush=True)
    return n


def download_jsonl(s3, bucket: str, prefix: str):
    os.makedirs(DATASET_ROOT, exist_ok=True)
    paths = []
    for k in list_s3_objects(s3, bucket, prefix):
        if not k.endswith(".jsonl"):
            continue
        dst = os.path.join(DATASET_ROOT, os.path.basename(k))
        try:
            s3.download_file(bucket, k, dst)
            paths.append(dst)
        except Exception as e:
            print(f"[llm-xpu] skip {k}: {e}", flush=True)
    return paths


def upload_dir(s3, local_dir: str, bucket: str, okey: str):
    keys = []
    for root, _, files in os.walk(local_dir):
        for fn in files:
            fp = os.path.join(root, fn)
            rel = os.path.relpath(fp, local_dir)
            dst = okey.rstrip("/") + "/" + rel.replace(os.sep, "/")
            s3.upload_file(fp, bucket, dst)
            keys.append(dst)
    return keys


def upload_progress(s3, bucket, okey, epoch, total, loss):
    try:
        s3.put_object(
            Bucket=bucket,
            Key=okey.rstrip("/") + "/progress.json",
            Body=json.dumps(
                {
                    "current_epoch": int(epoch),
                    "total_epochs": int(total),
                    "progress_pct": int(round(epoch * 100.0 / total)) if total else 99,
                    "loss": loss,
                    "updated_at": time.time(),
                }
            ),
        )
    except Exception as e:
        print(f"[llm-xpu] progress upload failed: {e}", flush=True)


# Live loss-history buffer (one JSON line per HF log event); the NF curves API
# renders history.jsonl while the run is still in progress.
_HISTORY: list = []
_HISTORY_LAST_FLUSH = 0.0


def upload_history(s3, bucket, okey, record=None, force=False, min_interval=5.0):
    """Append one loss event and throttled-rewrite history.jsonl on S3."""
    global _HISTORY_LAST_FLUSH
    if record is not None:
        _HISTORY.append(record)
    now = time.time()
    if not force and (not _HISTORY or now - _HISTORY_LAST_FLUSH < min_interval):
        return
    _HISTORY_LAST_FLUSH = now
    try:
        body = "".join(json.dumps(r) + "\n" for r in _HISTORY)
        s3.put_object(Bucket=bucket, Key=okey.rstrip("/") + "/history.jsonl", Body=body)
    except Exception as e:
        print(f"[llm-xpu] history upload failed: {e}", flush=True)


def load_config(raw: str, s3) -> dict:
    if raw.startswith("s3://"):
        bucket, key = parse_uri(raw)
        raw = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
    return json.loads(raw)


# sharegpt role name -> chat-template role
_ROLE_MAP = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant",
             "system": "system", "bot": "assistant"}


def _to_messages(conv):
    msgs = []
    for turn in conv:
        role = _ROLE_MAP.get(str(turn.get("from", "")).lower(), "user")
        msgs.append({"role": role, "content": turn.get("value", "")})
    return msgs


def build_dataset(jsonl_paths, tokenizer, max_len: int):
    from datasets import Dataset

    records = []
    for p in jsonl_paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    def tokenize(rec):
        msgs = _to_messages(rec.get("conversations") or rec.get("messages") or [])
        full = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        prompt_msgs = msgs[:-1] if msgs and msgs[-1]["role"] == "assistant" else msgs
        prompt = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        full_ids = tokenizer(full, truncation=True, max_length=max_len, add_special_tokens=False)["input_ids"]
        prompt_len = len(tokenizer(prompt, truncation=True, max_length=max_len, add_special_tokens=False)["input_ids"])
        labels = [-100] * min(prompt_len, len(full_ids)) + full_ids[prompt_len:]
        labels = labels[:len(full_ids)]
        return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}

    return Dataset.from_list(records).map(tokenize, remove_columns=Dataset.from_list(records).column_names)


def _as_list(v):
    if v is None:
        return None
    if isinstance(v, list):
        return v
    return [s.strip() for s in str(v).split(",") if s.strip()]


def train(cfg: dict, s3, obucket, okey) -> float:
    """train returns actual Trainer wall-clock seconds, excluding setup/export."""
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    class NFProgressCallback(TrainerCallback):
        """Mirror HF log/loss into S3 progress.json; other hooks are no-ops."""

        def on_log(self, args, state, control, logs=None, **kw):
            logs = logs or {}
            upload_progress(
                s3, obucket, okey, float(state.epoch or 0), epochs,
                logs.get("loss"),
            )
            # Train steps carry "loss"; evaluation passes carry "eval_loss".
            if "loss" in logs or "eval_loss" in logs:
                upload_history(s3, obucket, okey, {
                    "step": int(state.global_step),
                    "epoch": round(float(state.epoch or 0), 4),
                    "loss": logs.get("loss"),
                    "eval_loss": logs.get("eval_loss"),
                    "learning_rate": logs.get("learning_rate"),
                })

    if not torch.xpu.is_available():
        print("[llm-xpu] FATAL torch.xpu unavailable", flush=True)
        sys.exit(2)
    torch.xpu.init()
    device = "xpu:0"
    print(f"[llm-xpu] device={torch.xpu.get_device_name(0)}", flush=True)

    llm = cfg.get("llm", {}) or {}
    epochs = float(llm.get("num_train_epochs") or 3)
    max_len = int(llm.get("cutoff_len") or llm.get("max_seq_length") or 1024)
    bs = int(llm.get("per_device_train_batch_size") or 1)
    grad_accum = int(llm.get("gradient_accumulation_steps") or 8)
    rank = int(llm.get("lora_rank") or 16)
    alpha = int(llm.get("lora_alpha") or rank * 2)
    dropout = float(llm.get("lora_dropout") or 0.05)
    target = _as_list(llm.get("lora_target")) or [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ]

    tokenizer = AutoTokenizer.from_pretrained(BASE_ROOT, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_ROOT,
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": device},
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    jsonl = download_jsonl(s3, *parse_uri(cfg.get("dataset_uri") or os.environ["INPUT_URI"])[:2])
    if not jsonl:
        print("[llm-xpu] no .jsonl dataset, aborting", flush=True)
        sys.exit(1)
    train_ds = build_dataset(jsonl, tokenizer, max_len)
    print(f"[llm-xpu] train examples={len(train_ds)}", flush=True)

    os.makedirs(OUT_ROOT, exist_ok=True)
    args = TrainingArguments(
        output_dir=OUT_ROOT,
        num_train_epochs=epochs,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=float(llm.get("learning_rate") or 1e-4),
        warmup_ratio=float(llm.get("warmup_ratio") or 0.03),
        lr_scheduler_type=llm.get("lr_scheduler_type") or "cosine",
        logging_steps=int(llm.get("logging_steps") or 5),
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=bool(llm.get("gradient_checkpointing", True)),
        optim=llm.get("optim") or "adamw_8bit",
        report_to=[],
        dataloader_num_workers=0,
        use_cpu=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, processing_class=tokenizer)
    trainer.add_callback(NFProgressCallback())
    training_started = time.monotonic()
    trainer.train()
    train_time = time.monotonic() - training_started
    upload_history(s3, obucket, okey, force=True)
    model.save_pretrained(OUT_ROOT, safe_serialization=True)
    tokenizer.save_pretrained(OUT_ROOT)
    # save_strategy="no" means no checkpoint (and no trainer_state.json) is
    # written, but the metrics upload in main() needs log_history/global_step;
    # persist the in-memory TrainerState explicitly so final_loss is captured.
    st = trainer.state
    with open(os.path.join(OUT_ROOT, "trainer_state.json"), "w") as f:
        json.dump({
            "epoch": float(st.epoch or 0),
            "global_step": int(st.global_step),
            "log_history": list(st.log_history),
        }, f)
    return train_time


def main():
    s3 = s3_client()
    cfg = load_config(os.environ.get("CONFIG_JSON", "{}") or "{}", s3)
    artifact_id = os.environ.get("ARTIFACT_ID", cfg.get("artifact_id", ""))
    llm = cfg.get("llm", {}) or {}
    out_uri = cfg.get("output_dir") or os.environ.get("OUTPUT_DIR", "")
    base_uri = str(llm.get("base_model_uri") or "")
    dataset_uri = cfg.get("dataset_uri") or cfg.get("dataset_output_dir") or os.environ.get("INPUT_URI", "")
    if not dataset_uri or not out_uri or not base_uri:
        print("[llm-xpu] FIXME: missing dataset_uri/output_dir/base_model_uri", flush=True)
        sys.exit(1)

    b_bucket, b_prefix = parse_uri(base_uri)
    n = download_dir(s3, b_bucket, b_prefix, BASE_ROOT)
    print(f"[llm-xpu] base model files: {n}", flush=True)
    if n == 0:
        sys.exit(1)

    cfg["dataset_uri"] = dataset_uri
    o_bucket, o_key = parse_uri(out_uri)
    total_epochs = float(llm.get("num_train_epochs") or 3)

    train_time = train(cfg, s3, o_bucket, o_key)

    state_path = os.path.join(OUT_ROOT, "trainer_state.json")
    metrics = {"final_epoch": int(total_epochs)}
    final_loss = None
    if os.path.exists(state_path):
        try:
            state = json.loads(open(state_path).read())
            losses = [h.get("loss") for h in state.get("log_history", []) if h.get("loss") is not None]
            if losses:
                final_loss = float(losses[-1])
                metrics["final_loss"] = final_loss
            if state.get("global_step"):
                metrics["global_step"] = int(state["global_step"])
        except Exception:
            pass

    uploaded = upload_dir(s3, OUT_ROOT, o_bucket, o_key + "/" + artifact_id)
    metrics["weights_uri"] = f"s3://{o_bucket}/{o_key}/{artifact_id}/adapter_config.json"
    metrics["adapter_key"] = f"{o_key}/{artifact_id}"
    metrics = build_training_metrics(metrics, train_time_s=train_time)
    mkey = o_key + "/" + artifact_id + ".metrics.json"
    s3.put_object(Bucket=o_bucket, Key=mkey, Body=json.dumps(metrics, indent=2))
    upload_progress(s3, o_bucket, o_key, total_epochs, total_epochs, final_loss)
    print(f"[llm-xpu] uploaded {len(uploaded)} files; SUCCESS", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
