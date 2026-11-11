# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""LoRA inference smoke test for a fine-tuned VLM adapter on Intel XPU.

Downloads an operator-selected base model, trained adapter and sharegpt+image
SFT dataset from S3, loads the adapter with PEFT on torch.xpu, asks the first
image question in the jsonl and prints VLM_LORA_INFERENCE_OK. The expected
answer comes from the selected dataset record, and matching key words produce
exit code 0.

Run as a pod using vlm_lora_infer_e2e.yaml. Every value below is required and
must be non-empty; a missing value stops the script before it accesses S3:
- S3_ENDPOINT: S3-compatible endpoint URL. The companion manifest retains the
  generic in-cluster service DNS endpoint.
- S3_BUCKET: bucket containing the model, adapter and dataset objects.
- S3_ACCESS_KEY and S3_SECRET_KEY: credentials for that bucket.
- BASE_PREFIX: base-model object prefix.
- ADAPTER_PREFIX: trained LoRA adapter object prefix.
- DATA_PREFIX: sharegpt+image SFT dataset object prefix.

Prefixes are relative to S3_BUCKET and must end in "/". The manifest sets
HF_HUB_OFFLINE=1 so model loading uses only the downloaded files.
"""
import json
import os
import sys

import boto3
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor


def _required_env(name: str) -> str:
    """Return a non-empty operator input or stop before accessing S3."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Set {name} to a non-empty value before running this E2E test")
    return value


EP = _required_env("S3_ENDPOINT")
BUCKET = _required_env("S3_BUCKET")
ACCESS_KEY = _required_env("S3_ACCESS_KEY")
SECRET_KEY = _required_env("S3_SECRET_KEY")
BASE_PREFIX = _required_env("BASE_PREFIX")
ADAPTER_PREFIX = _required_env("ADAPTER_PREFIX")
DATA_PREFIX = _required_env("DATA_PREFIX")
BASE_ROOT = "/vbase"
ADAPTER_ROOT = "/vadapter"
DATA_ROOT = "/vdata"

s3 = boto3.client(
    "s3",
    endpoint_url=EP,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name="us-east-1",
)


def download_prefix(prefix, root):
    """Download every object under an S3 prefix; return the file count."""
    count = 0
    for obj in s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix).get("Contents", []):
        key = obj["Key"]
        rel = key[len(prefix):]
        if not rel:
            continue
        dst = os.path.join(root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        s3.download_file(BUCKET, key, dst)
        count += 1
    return count


print("[vlm-lora] base files:", download_prefix(BASE_PREFIX, BASE_ROOT), flush=True)
print("[vlm-lora] adapter files:", download_prefix(ADAPTER_PREFIX, ADAPTER_ROOT), flush=True)
print("[vlm-lora] dataset files:", download_prefix(DATA_PREFIX, DATA_ROOT), flush=True)

if not torch.xpu.is_available():
    print("[vlm-lora] FATAL torch.xpu unavailable", flush=True)
    sys.exit(2)

device = "xpu:0"
print("[vlm-lora] device:", torch.xpu.get_device_name(0), flush=True)
processor = AutoProcessor.from_pretrained(BASE_ROOT)
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

model = AutoModelForImageTextToText.from_pretrained(BASE_ROOT, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, ADAPTER_ROOT)
model.to(device)
model.eval()

record = None
jsonl_path = None
for root, _, files in os.walk(DATA_ROOT):
    for name in sorted(files):
        if name.endswith(".jsonl"):
            jsonl_path = os.path.join(root, name)
            break
    if jsonl_path:
        break

with open(jsonl_path) as f:
    for line in f:
        row = json.loads(line)
        if row.get("image"):
            record = row
            break

messages = []
image_added = False
expected = ""
for turn in record["conversations"]:
    role = {"human": "user", "gpt": "assistant"}.get(turn["from"], turn["from"])
    content = []
    if role == "user" and not image_added:
        content.append({"type": "image"})
        image_added = True
    text = turn.get("value", "")
    content.append({"type": "text", "text": text})
    messages.append({"role": role, "content": content})
    if role == "assistant":
        expected = text

prompt_messages = messages[:-1] if messages and messages[-1]["role"] == "assistant" else messages
image = Image.open(os.path.join(DATA_ROOT, record["image"])).convert("RGB")
prompt = processor.apply_chat_template(prompt_messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)

with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=80, do_sample=False)

answer = processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
print("[vlm-lora] question:", prompt_messages[-1]["content"][-1]["text"], flush=True)
print("[vlm-lora] expected:", expected, flush=True)
print("[vlm-lora] answer:", answer, flush=True)
expected_words = {w.strip(".,!?;:").lower() for w in expected.split() if len(w.strip(".,!?;:")) > 3}
answer_words = {w.strip(".,!?;:").lower() for w in answer.split()}
matched = expected_words & answer_words
ok = len(matched) >= max(1, len(expected_words) // 2)
print("[vlm-lora] matched:", sorted(matched), flush=True)
print("[vlm-lora] VLM_LORA_INFERENCE_OK", ok, flush=True)
sys.exit(0 if ok else 3)
