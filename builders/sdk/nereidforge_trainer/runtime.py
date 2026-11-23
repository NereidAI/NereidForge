# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Job runtime helpers: config parsing, JSON output and live progress."""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Mapping
from typing import Any

from .s3 import publish_file


def read_config(raw: str = "") -> dict[str, Any]:
    """Read the job's flat training config.

    `CONFIG_JSON` may hold inline JSON or a path to a JSON file (the KFP step
    accepts both). Missing/empty config is an empty dict, never an error: the
    trainer applies its own defaults.
    """
    value = (raw or os.environ.get("CONFIG_JSON", "") or "").strip()
    if not value:
        return {}
    if os.path.exists(value):
        with open(value, encoding="utf-8") as fh:
            data = json.load(fh)
    else:
        data = json.loads(value)
    if not isinstance(data, dict):
        raise ValueError("CONFIG_JSON must decode to a JSON object")
    return data


def _json_safe(value: Any) -> Any:
    """_json_safe marks non-finite measurements null while preserving their rows.

    Frameworks may temporarily report NaN or Infinity during validation. Strict
    JSON consumers reject those Python-specific tokens, so unavailable numeric
    values cross the artifact boundary as null instead of invalidating a document.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_json(path: str, data: Any) -> None:
    """write_json serializes one trainer artifact as deterministic strict JSON."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(data), fh, indent=2, sort_keys=True, allow_nan=False)


#: TRAINING_METRICS_SCHEMA identifies the versioned final trainer-metrics document.
TRAINING_METRICS_SCHEMA = "nf.training-metrics/1"


def validate_training_metrics(document: Any) -> list[str]:
    """validate_training_metrics checks the portable final-metrics contract.

    Task-specific metric keys remain open. The schema and a finite, positive
    wall-clock training duration are mandatory so every trainer provides the
    same lifecycle observability regardless of its framework.
    """
    if not isinstance(document, dict):
        return ["document must be a JSON object"]
    errors: list[str] = []
    if document.get("schema") != TRAINING_METRICS_SCHEMA:
        errors.append(f"schema must be {TRAINING_METRICS_SCHEMA}")
    value = document.get("train_time_s")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append("train_time_s must be a JSON number")
    elif not math.isfinite(value) or value <= 0:
        errors.append("train_time_s must be finite and greater than zero")
    return errors


def build_training_metrics(
    metrics: Mapping[str, Any], *, train_time_s: float,
) -> dict[str, Any]:
    """build_training_metrics returns a validated final-metrics document.

    Contract-owned values override same-named custom keys, and the caller's
    mapping is never mutated. Invalid durations fail inside the trainer before
    it can accidentally publish an apparently conformant artifact.
    """
    document = {
        **dict(metrics),
        "schema": TRAINING_METRICS_SCHEMA,
        "train_time_s": train_time_s,
    }
    errors = validate_training_metrics(document)
    if errors:
        raise ValueError("invalid training metrics: " + "; ".join(errors))
    return document


class Progress:
    """Live progress reporter (`progress.json`, polled by the NF worker)."""

    def __init__(self, output_uri: str, total_epochs: int):
        self.output_uri = output_uri
        self.total = max(1, int(total_epochs))
        self.started = time.time()
        self.history: list[dict[str, Any]] = []

    def write(self, epoch: int, metrics: dict[str, Any], key: str = "progress.json") -> None:
        payload = {
            "current_epoch": int(epoch),
            "total_epochs": self.total,
            "progress_pct": int(min(99, epoch * 100 / self.total)),
            "elapsed_s": round(time.time() - self.started, 1),
            **{k: v for k, v in metrics.items() if isinstance(v, (int, float, str))},
            "history": self.history,
        }
        local = "/tmp/progress.json"
        write_json(local, payload)
        if self.output_uri:
            try:
                publish_file(local, self.output_uri, key)
            except Exception as e:  # noqa: BLE001
                print(f"[progress] upload failed (non-fatal): {e}", flush=True)

    def log(self, epoch: int, metrics: dict[str, Any]) -> None:
        self.history.append({"epoch": int(epoch), **{
            k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics.items()
        }})
        self.write(epoch, metrics)
