# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""NereidForge VLM trainer entrypoint for Intel XPU (native torch.xpu).

Minimal vision-language fine-tuning path validated with SmolVLM-500M
(Idefics3 architecture) on Arc B580: vision tower + connector are frozen, only
the Llama text backbone receives LoRA adapters, bf16, single card. The S3 /
config / artifact contract is identical to llm_train.py so the control plane
stays the same; datasets use sharegpt rows plus an `image` key whose value is a
relative object path under dataset_uri (e.g. "images/a.png"):

  {"image": "images/a.png",
   "conversations": [
     {"from": "human", "value": "What is in the image?"},
     {"from": "gpt",   "value": "A blue circle on a red background."}]}
"""

import json
import os
import sys
import time

import torch
from nereidforge_trainer import build_training_metrics, s3_client

OUT_ROOT = "/app/output"
DATA_ROOT = "/datasets"
BASE_ROOT = "/base"


def parse_uri(uri: str):
    rest = uri[len("s3://"):]
    bucket, _, key = rest.partition("/")
    return bucket, key.rstrip("/")


def list_objs(s3, bucket: str, prefix: str):
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(o["Key"] for o in page.get("Contents", []))
    return keys


def download_dir(s3, bucket: str, prefix: str, root: str) -> int:
    n = 0
    for k in list_objs(s3, bucket, prefix):
        rel = k[len(prefix):].lstrip("/")
        if not rel:
            continue
        dst = os.path.join(root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            s3.download_file(bucket, k, dst)
            n += 1
        except Exception as e:
            print(f"[vlm-xpu] skip {rel}: {e}", flush=True)
    return n


def upload_dir(s3, local_dir: str, bucket: str, okey: str):
    keys = []
    for root, _, files in os.walk(local_dir):
        for fn in files:
            fp = os.path.join(root, fn)
            dst = okey.rstrip("/") + "/" + os.path.relpath(fp, local_dir).replace(os.sep, "/")
            s3.upload_file(fp, bucket, dst)
            keys.append(dst)
    return keys


def upload_progress(s3, bucket, okey, epoch, total, loss):
    try:
        s3.put_object(
            Bucket=bucket, Key=okey.rstrip("/") + "/progress.json",
            Body=json.dumps({"current_epoch": int(epoch), "total_epochs": int(total),
                             "progress_pct": int(round(epoch * 100.0 / total)) if total else 99,
                             "loss": loss, "updated_at": time.time()}),
        )
    except Exception as e:
        print(f"[vlm-xpu] progress upload failed: {e}", flush=True)


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
        print(f"[vlm-xpu] history upload failed: {e}", flush=True)


def load_config(raw: str, s3) -> dict:
    if raw.startswith("s3://"):
        b, k = parse_uri(raw)
        raw = s3.get_object(Bucket=b, Key=k)["Body"].read().decode()
    return json.loads(raw or "{}")


_ROLE = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant",
         "system": "system"}


def to_messages(conv, with_image: bool):
    """Convert sharegpt turns to the Idefics3 chat content list.

    The single image placeholder is attached to the first user turn only
    (SmolVLM accepts one image per example).
    """
    out = []
    image_added = not with_image
    for t in conv:
        role = _ROLE.get(str(t.get("from", "")).lower(), "user")
        content = []
        if role == "user" and not image_added:
            content.append({"type": "image"})
            image_added = True
        content.append({"type": "text", "text": t.get("value", "")})
        out.append({"role": role, "content": content})
    return out


def train(cfg: dict, s3, obucket, okey) -> float:
    """train returns actual Trainer wall-clock seconds, excluding setup/export."""
    from PIL import Image
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model

    if not torch.xpu.is_available():
        print("[vlm-xpu] FATAL torch.xpu unavailable", flush=True)
        sys.exit(2)
    device = "xpu:0"
    torch.xpu.init()
    print(f"[vlm-xpu] device={torch.xpu.get_device_name(0)}", flush=True)

    llm = cfg.get("llm", {}) or {}
    epochs = float(llm.get("num_train_epochs") or 5)
    max_len = int(llm.get("cutoff_len") or llm.get("max_seq_length") or 512)
    grad_accum = int(llm.get("gradient_accumulation_steps") or 8)
    rank = int(llm.get("lora_rank") or 16)
    lr = float(llm.get("learning_rate") or 1e-4)

    processor = AutoProcessor.from_pretrained(BASE_ROOT, do_image_splitting=False)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Load records (jsonl) + their images from DATA_ROOT.
    records = []
    for root, _, files in os.walk(DATA_ROOT):
        for fn in sorted(files):
            if not fn.endswith(".jsonl"):
                continue
            for line in open(os.path.join(root, fn)):
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    if not records:
        print("[vlm-xpu] no jsonl records, aborting", flush=True)
        sys.exit(1)
    print(f"[vlm-xpu] records={len(records)}", flush=True)

    def encode(rec):
        img = Image.open(os.path.join(DATA_ROOT, rec["image"])).convert("RGB")
        messages = to_messages(rec.get("conversations") or rec.get("messages") or [], True)
        full = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt_messages = messages[:-1] if messages[-1]["role"] == "assistant" else messages
        prompt = processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True)
        feats = processor(text=full, images=[img], return_tensors="pt",
                          truncation=True, max_length=max_len)
        prompt_len = processor(text=prompt, images=[img], return_tensors="pt",
                               truncation=True, max_length=max_len)["input_ids"].shape[1]
        ids = feats["input_ids"][0]
        labels = ids.clone()
        labels[: min(prompt_len, labels.shape[0])] = -100
        feats["labels"] = labels.unsqueeze(0)
        return {k: v for k, v in feats.items()}

    encoded = [encode(r) for r in records]
    print(f"[vlm-xpu] encoded examples={len(encoded)} keys={list(encoded[0].keys())}", flush=True)

    model = AutoModelForImageTextToText.from_pretrained(BASE_ROOT, dtype=torch.bfloat16)
    for mod in (getattr(model.model, "vision_model", None), getattr(model.model, "connector", None)):
        if mod is not None:
            for p in mod.parameters():
                p.requires_grad = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=rank, lora_alpha=rank * 2, lora_dropout=0.05,
            # Restrict adapters to the Llama text backbone (the vision tower
            # also has q/k/v_proj modules, so a scoped regex is mandatory).
            target_modules=r".*text_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
            bias="none", task_type=None,
        ),
    )
    model.print_trainable_parameters()
    model.to(device)

    class SingleExampleCollator:
        """Batch size is fixed to 1; every tensor is already batched."""

        def __call__(self, features):
            return features[0]

    class NFProgressCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            logs = logs or {}
            upload_progress(s3, obucket, okey, float(state.epoch or 0), epochs,
                            logs.get("loss"))
            # Train steps carry "loss"; evaluation passes carry "eval_loss".
            if "loss" in logs or "eval_loss" in logs:
                upload_history(s3, obucket, okey, {
                    "step": int(state.global_step),
                    "epoch": round(float(state.epoch or 0), 4),
                    "loss": logs.get("loss"),
                    "eval_loss": logs.get("eval_loss"),
                    "learning_rate": logs.get("learning_rate"),
                })

    os.makedirs(OUT_ROOT, exist_ok=True)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=OUT_ROOT, num_train_epochs=epochs,
            per_device_train_batch_size=1, gradient_accumulation_steps=grad_accum,
            learning_rate=lr, warmup_ratio=0.05, lr_scheduler_type="cosine",
            logging_steps=max(1, int(llm.get("logging_steps") or 2)),
            save_strategy="no", bf16=True, gradient_checkpointing=False,
            optim="adamw_torch", report_to=[], remove_unused_columns=False,
            dataloader_num_workers=0),
        train_dataset=encoded,
        data_collator=SingleExampleCollator(),
        processing_class=processor.tokenizer,
    )
    trainer.add_callback(NFProgressCallback())
    training_started = time.monotonic()
    trainer.train()
    train_time = time.monotonic() - training_started
    upload_history(s3, obucket, okey, force=True)
    model.save_pretrained(OUT_ROOT, safe_serialization=True)
    processor.save_pretrained(OUT_ROOT)
    st = trainer.state
    with open(os.path.join(OUT_ROOT, "trainer_state.json"), "w") as f:
        json.dump({"epoch": float(st.epoch or 0), "global_step": int(st.global_step),
                   "log_history": list(st.log_history)}, f)
    return train_time


def main():
    s3 = s3_client()
    cfg = load_config(os.environ.get("CONFIG_JSON", "{}"), s3)
    artifact_id = os.environ.get("ARTIFACT_ID", cfg.get("artifact_id", ""))
    llm = cfg.get("llm", {}) or {}
    out_uri = cfg.get("output_dir") or os.environ.get("OUTPUT_DIR", "")
    base_uri = str(llm.get("base_model_uri") or "")
    dataset_uri = cfg.get("dataset_uri") or os.environ.get("INPUT_URI", "")
    if not dataset_uri or not out_uri or not base_uri:
        print("[vlm-xpu] FIXME: missing dataset_uri/output_dir/base_model_uri", flush=True)
        sys.exit(1)

    b_bucket, b_prefix = parse_uri(base_uri)
    if download_dir(s3, b_bucket, b_prefix, BASE_ROOT) == 0:
        print("[vlm-xpu] empty base model, aborting", flush=True)
        sys.exit(1)
    d_bucket, d_prefix = parse_uri(dataset_uri)
    if download_dir(s3, d_bucket, d_prefix, DATA_ROOT) == 0:
        print("[vlm-xpu] empty dataset, aborting", flush=True)
        sys.exit(1)

    o_bucket, o_key = parse_uri(out_uri)
    total_epochs = float(llm.get("num_train_epochs") or 5)
    train_time = train(cfg, s3, o_bucket, o_key)

    metrics = {"final_epoch": int(total_epochs)}
    final_loss = None
    sp = os.path.join(OUT_ROOT, "trainer_state.json")
    if os.path.exists(sp):
        state = json.loads(open(sp).read())
        losses = [h.get("loss") for h in state.get("log_history", []) if h.get("loss") is not None]
        if losses:
            final_loss = float(losses[-1])
            metrics["final_loss"] = final_loss
        if state.get("global_step"):
            metrics["global_step"] = int(state["global_step"])
    n = upload_dir(s3, OUT_ROOT, o_bucket, o_key + "/" + artifact_id)
    metrics["weights_uri"] = f"s3://{o_bucket}/{o_key}/{artifact_id}/adapter_config.json"
    metrics["adapter_key"] = f"{o_key}/{artifact_id}"
    metrics = build_training_metrics(metrics, train_time_s=train_time)
    s3.put_object(Bucket=o_bucket, Key=o_key + "/" + artifact_id + ".metrics.json",
                  Body=json.dumps(metrics, indent=2))
    upload_progress(s3, o_bucket, o_key, total_epochs, total_epochs, final_loss)
    print(f"[vlm-xpu] uploaded {n} files; SUCCESS", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
