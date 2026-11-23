# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""COCO metric parsing for the NF Paddle trainer.

PaddleDetection prints its COCO evaluation table to stdout; PaddleX returns the
subprocess result. Two entrypoints need the same parser, so it lives in its own
module (importable without PaddleX installed, which keeps it testable).

Example line:

    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.412
"""

from __future__ import annotations

import re
from typing import Any

_NUM = r"(-?\d+(?:\.\d+)?)"

_METRIC_PATTERNS: dict[str, str] = {
    "mAP50-95": rf"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50:0\.95\s*\|\s*area=\s*all[^\]]*\]\s*=\s*{_NUM}",
    "mAP50": rf"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50\s*\|\s*area=\s*all[^\]]*\]\s*=\s*{_NUM}",
    "mAP75": rf"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.75\s*\|\s*area=\s*all[^\]]*\]\s*=\s*{_NUM}",
    "mAP_small": rf"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50:0\.95\s*\|\s*area=\s*small[^\]]*\]\s*=\s*{_NUM}",
    "mAP_medium": rf"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50:0\.95\s*\|\s*area=medium[^\]]*\]\s*=\s*{_NUM}",
    "mAP_large": rf"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50:0\.95\s*\|\s*area=\s*large[^\]]*\]\s*=\s*{_NUM}",
    "AR100": rf"Average Recall\s+\(AR\)\s+@\[\s*IoU=0\.50:0\.95\s*\|\s*area=\s*all[^\]]*maxDets=100\s*\]\s*=\s*{_NUM}",
    "per_class_AP50": r"",
}

#: UI/metrics keys the platform understands for detection (see evaluation-spec.md).
CANONICAL_METRICS = ("mAP50-95", "mAP50", "mAP75", "mAP_small", "mAP_medium",
                     "mAP_large", "AR100")


def parse_metrics(text: str) -> dict[str, float]:
    """Extract the COCO metric table from PaddleDetection output (best effort)."""
    metrics: dict[str, float] = {}
    for key in CANONICAL_METRICS:
        pattern = _METRIC_PATTERNS.get(key)
        if not pattern:
            continue
        match = re.search(pattern, text or "")
        if not match:
            continue
        try:
            metrics[key] = round(float(match.group(1)), 6)
        except ValueError:
            continue
    return metrics


def metrics_summary(metrics: dict[str, Any]) -> str:
    """One-line, log-friendly summary of the parsed metrics."""
    keys = [k for k in CANONICAL_METRICS if k in metrics]
    if not keys:
        return "no COCO metrics found in the training log"
    return " ".join(f"{k}={metrics[k]:.4f}" for k in keys)
