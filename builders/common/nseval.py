# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""NereidForge shared evaluation core (imported by evaluate.py / optimize.py).

Implements real detection metrics without heavyweight dependencies:
- detection output decoding for the three common ONNX conventions: named
  `boxes`/`scores`/`labels` outputs (NF-trained torchvision models), a
  post-NMS `[N, 6]` tensor, and an anchor-free `[1, N, 4+nc]` / `[1, 4+nc, N]`
  raw head (YOLO v8/v11, PP-YOLOE)
- Vectorized greedy NMS and pairwise IoU
- Per-class precision/recall/F1 at a fixed IoU threshold
- mAP@50 and mAP@50:95 (all-point interpolated AP, classes averaged)
- Classification metrics (top-1/top-5 accuracy, per-class P/R)
- Gate evaluation (hard: metric drop + NaN/Inf; soft: latency/throughput)

All functions are pure numpy so they run identically inside the optimizer
(calibration preprocessing) and evaluator images.
"""

from __future__ import annotations

import json
import platform
import socket
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np

# IoU thresholds for COCO-style mAP@50:95 (inclusive both ends).
MAP_THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------

def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert (N, 4) cx,cy,w,h to x1,y1,x2,y2."""
    out = np.zeros_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def pair_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N,4) and (M,4) xyxy boxes -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Greedy NMS over xyxy boxes; returns kept indices (score order)."""
    order = np.argsort(-scores)
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = pair_iou(boxes[i:i + 1], boxes[rest])[0]
        order = rest[ious <= iou_thr]
    return keep


# ---------------------------------------------------------------------------
# Detection output decoding
# ---------------------------------------------------------------------------

def _orient_rows(pred: np.ndarray, num_classes: int) -> np.ndarray | None:
    """Orient an anchor-free head to rows=anchors, cols=4+nc."""
    while pred.ndim > 2:
        pred = pred[0]
    if pred.ndim != 2:
        return None
    if pred.shape[0] < pred.shape[1] and num_classes and pred.shape[0] == 4 + num_classes:
        pred = pred.T
    elif pred.shape[1] < pred.shape[0] and num_classes and pred.shape[1] == 4 + num_classes:
        pass
    elif pred.shape[1] < 4:
        return None
    return pred


def _looks_like_indices(values: np.ndarray) -> bool:
    if values.size == 0:
        return False
    return bool(np.all(np.isfinite(values))
                and np.all(values >= 0)
                and np.all(values < 4096)
                and np.allclose(values, np.round(values)))


def _looks_like_scores(values: np.ndarray) -> bool:
    if values.size == 0:
        return False
    return bool(np.all(np.isfinite(values))
                and np.all(values >= 0.0)
                and np.all(values <= 1.0001))


def decode_post_nms(pred: np.ndarray, conf_thr: float = 0.25,
                    max_det: int = 300) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a `[N, 6]` post-NMS tensor.

    Two orderings are in the wild and neither is declared in the graph:
    ``[class, score, x1, y1, x2, y2]`` (OpenVINO NMS, many NPU exports) and
    ``[x1, y1, x2, y2, score, class]`` (EfficientNMS plugin). Column type
    inspection picks the right one.
    """
    pred = np.asarray(pred, dtype=np.float32).reshape(-1, pred.shape[-1] if pred.ndim else 6)
    if pred.size == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    class_first = (_looks_like_indices(pred[:, 0])
                   and _looks_like_scores(pred[:, 1]))
    box_first = _looks_like_scores(pred[:, 4]) and _looks_like_indices(pred[:, 5])
    if class_first or not box_first:
        cls_ids = pred[:, 0].astype(int)
        conf = pred[:, 1]
        boxes = pred[:, 2:6]
    else:
        boxes = pred[:, :4]
        conf = pred[:, 4]
        cls_ids = pred[:, 5].astype(int)
    keep = conf >= conf_thr
    order = np.argsort(-conf[keep])
    keep_idx = np.where(keep)[0][order][:max_det]
    return boxes[keep_idx], conf[keep_idx], cls_ids[keep_idx]


def decode_detections(raw: np.ndarray, num_classes: int = 0,
                      conf_thr: float = 0.25, iou_thr: float = 0.45,
                      max_det: int = 300) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a single-tensor detection output into post-NMS detections.

    Handles the three single-tensor conventions: post-NMS ``[N, 6]``,
    anchor-free ``[1, N, 4+nc]`` and its transpose ``[1, 4+nc, N]``. Boxes
    come out as xyxy in input-pixel scale, scores in score space.
    """
    pred = np.asarray(raw, dtype=np.float32)
    while pred.ndim > 2:
        pred = pred[0]
    if pred.ndim != 2:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

    # Post-NMS layout: 6 columns (or a generic head whose class count is
    # unknown and whose shape has no room for one-hot scores).
    if pred.shape[1] == 6:
        return decode_post_nms(pred, conf_thr, max_det)
    if num_classes <= 0:
        num_classes = max(pred.shape[1] - 4, 1)

    rows = _orient_rows(pred, num_classes)
    if rows is None or rows.shape[1] < 5:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    boxes = xywh_to_xyxy(rows[:, :4])
    cls_scores = rows[:, 4:4 + num_classes]
    if cls_scores.shape[1] == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    cls_ids = cls_scores.argmax(axis=1)
    conf = cls_scores.max(axis=1)
    mask = conf >= conf_thr
    boxes, conf, cls_ids = boxes[mask], conf[mask], cls_ids[mask]
    if len(conf) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    # Per-class NMS
    keep_all: list[int] = []
    for c in np.unique(cls_ids):
        idx = np.where(cls_ids == c)[0]
        kept = nms(boxes[idx], conf[idx], iou_thr)
        keep_all.extend(idx[k] for k in kept)
    keep_all = sorted(keep_all, key=lambda i: -conf[i])[:max_det]
    sel = np.array(sorted(keep_all))
    return boxes[sel], conf[sel], cls_ids[sel]


def decode_named(outputs: list, out_names: list[str], conf_thr: float = 0.25,
                 max_det: int = 300) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode named `boxes`/`scores`/`labels` outputs (NF/torchvision export).

    NMS is already baked into the exported graph, so this only filters by
    confidence and caps the detection count. A graph without a `boxes` output
    raises ValueError; the caller falls back to the single-tensor decoder.
    """
    index = {str(name): i for i, name in enumerate(out_names)}
    if "boxes" not in index:
        raise ValueError("no 'boxes' output in the graph")
    boxes = np.asarray(outputs[index["boxes"]], dtype=np.float32)
    while boxes.ndim > 2:
        boxes = boxes[0]
    boxes = boxes.reshape(-1, 4)
    n = len(boxes)
    if "scores" in index:
        scores = np.asarray(outputs[index["scores"]], dtype=np.float32).reshape(-1)[:n]
    else:
        scores = np.ones(n, dtype=np.float32)
    if "labels" in index:
        labels = np.asarray(outputs[index["labels"]]).reshape(-1)[:n].astype(int)
    else:
        labels = np.zeros(n, dtype=int)
    keep = np.where(scores >= conf_thr)[0]
    order = keep[np.argsort(-scores[keep])][:max_det]
    return boxes[order], scores[order], labels[order]


# Backwards-compatible alias for callers written against the old name.
decode_yolo = decode_detections


# ---------------------------------------------------------------------------
# Dataset structures: per image {class_id: [boxes]} collections
# ---------------------------------------------------------------------------

def parse_yolo_label(text: str, img_w: int, img_h: int, input_size: int) -> dict[int, np.ndarray]:
    """Parse a YOLO txt label into {cls: (K,4) xyxy} in ORIGINAL image pixels.

    Predictions are remapped to original pixels after letterboxing, so ground
    truth must stay in the same space (normalized coords * image dims).
    """
    per_cls: dict[int, list[list[float]]] = {}
    for line in text.strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c = int(float(parts[0]))
        w = float(parts[3]) * img_w
        h = float(parts[4]) * img_h
        cx = float(parts[1]) * img_w
        cy = float(parts[2]) * img_h
        per_cls.setdefault(c, []).append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return {c: np.asarray(v, dtype=np.float64) for c, v in per_cls.items()}


# ---------------------------------------------------------------------------
# Detection metrics
# ---------------------------------------------------------------------------

def _tp_fp_for_threshold(preds: list[dict[int, list[tuple[np.ndarray, float]]]],
                         gts: list[dict[int, np.ndarray]],
                         num_classes: int, iou_thr: float,
                         ) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """Greedy score-ordered matching at one IoU threshold.

    Returns (tp_flags, scores, gt_counts) per class over the whole dataset.
    """
    tp_by_cls: list[list[float]] = [[] for _ in range(num_classes)]
    sc_by_cls: list[list[float]] = [[] for _ in range(num_classes)]
    gt_counts = [0] * num_classes
    matched_global: list[dict[int, set[int]]] = []
    for g in gts:
        gt_counts = [gt_counts[c] + len(g.get(c, [])) for c in range(num_classes)]
    for img_preds in preds:
        matched_global.append({c: set() for c in range(num_classes)})
    # Flatten predictions with image index, sort globally per class by score.
    flat: list[tuple[int, int, np.ndarray, float]] = []
    for i, img_preds in enumerate(preds):
        for c, dets in img_preds.items():
            for box, score in dets:
                flat.append((i, c, box, score))
    flat.sort(key=lambda x: -x[3])
    for i, c, box, score in flat:
        sc_by_cls[c].append(score)
        gt_boxes = gts[i].get(c)
        if gt_boxes is None or len(gt_boxes) == 0:
            tp_by_cls[c].append(0.0)
            continue
        used = matched_global[i][c]
        if len(used) >= len(gt_boxes):
            tp_by_cls[c].append(0.0)
            continue
        ious = pair_iou(box[None, :], gt_boxes)[0]
        order = np.argsort(-ious)
        hit = False
        for j in order:
            j = int(j)
            if j in used:
                continue
            if ious[j] >= iou_thr:
                used.add(j)
                hit = True
            break  # only the best-IoU candidate counts (greedy)
        tp_by_cls[c].append(1.0 if hit else 0.0)
    return ([np.asarray(v) for v in tp_by_cls], [np.asarray(v) for v in sc_by_cls], gt_counts)


def _ap_from_tp_fp(tp: np.ndarray, scores: np.ndarray, n_gt: int) -> float:
    """COCO 101-point interpolated AP for one class."""
    if n_gt == 0 or len(tp) == 0:
        return 0.0
    order = np.argsort(-scores)
    tpc = np.cumsum(tp[order])
    fpc = np.arange(1, len(tp) + 1) - tpc
    recall = tpc / n_gt
    precision = tpc / np.maximum(tpc + fpc, 1e-9)
    # COCO-style 101-point interpolation: AP = mean of max precision at
    # 101 equally spaced recall levels (0.0 .. 1.0).
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = 0.0
    for i in idx:
        ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]
    return float(ap)


def detection_metrics(preds: list[dict[int, list[tuple[np.ndarray, float]]]],
                      gts: list[dict[int, np.ndarray]],
                      num_classes: int, class_names: list[str] | None = None,
                      ) -> dict[str, Any]:
    """Compute the full detection metric suite (mAP50, mAP50-95, P/R/F1, per-class)."""
    names = class_names or [str(i) for i in range(num_classes)]
    # Confusion-based P/R/F1 at IoU=0.5 (fixed conf threshold upstream)
    tp50, _, gt_counts = _tp_fp_for_threshold(preds, gts, num_classes, 0.5)
    n_pred = sum(len(v) for img in preds for v in img.values())
    n_tp50 = float(sum(v.sum() for v in tp50))
    n_gt = sum(gt_counts)
    precision = n_tp50 / n_pred if n_pred else 0.0
    recall = n_tp50 / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    ap_tables: dict[float, list[float]] = {t: [] for t in MAP_THRESHOLDS}
    for thr in MAP_THRESHOLDS:
        tp, sc, gcnt = _tp_fp_for_threshold(preds, gts, num_classes, thr)
        for c in range(num_classes):
            if gcnt[c] == 0:
                continue  # classes absent from GT don't drag the mean down
            ap_tables[thr].append(_ap_from_tp_fp(tp[c], sc[c], gcnt[c]))
    map50 = float(np.mean(ap_tables[0.5])) if ap_tables[0.5] else 0.0
    map_5095 = float(np.mean([np.mean(v) for v in ap_tables.values() if v])) if any(ap_tables.values()) else 0.0

    # Per-class P/R at IoU=0.5 plus AP50 (classes absent from GT excluded)
    gt_classes = [c for c in range(num_classes) if gt_counts[c] > 0]
    per_class = {}
    for c, ap in zip(gt_classes, ap_tables[0.5]):
        p_c = float(tp50[c].sum()) / max(len(tp50[c]), 1)
        r_c = float(tp50[c].sum()) / gt_counts[c]
        per_class[names[c]] = {
            "ap50": round(float(ap), 4),
            "precision": round(p_c, 4), "recall": round(r_c, 4),
            "gt": gt_counts[c], "pred": int(len(tp50[c])),
        }

    return {
        "mAP50": round(map50, 4),
        "mAP50_95": round(map_5095, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "gt_boxes": n_gt,
        "predictions": n_pred,
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def classification_metrics(probs: np.ndarray, labels: np.ndarray,
                           class_names: list[str] | None = None) -> dict[str, Any]:
    """Top-1/top-5 accuracy and per-class precision/recall from softmax probs."""
    names = class_names or [str(i) for i in range(probs.shape[1])]
    top1 = probs.argmax(axis=1)
    top5 = np.argsort(-probs, axis=1)[:, :5]
    n = len(labels)
    acc1 = float((top1 == labels).mean()) if n else 0.0
    acc5 = float(np.mean([labels[i] in top5[i] for i in range(n)])) if n else 0.0
    per_class: dict[str, Any] = {}
    nc = probs.shape[1]
    for c in range(nc):
        pred_mask = top1 == c
        gt_mask = labels == c
        tp = int((pred_mask & gt_mask).sum())
        per_class[names[c]] = {
            "precision": round(tp / max(int(pred_mask.sum()), 1), 4),
            "recall": round(tp / max(int(gt_mask.sum()), 1), 4),
            "gt": int(gt_mask.sum()), "pred": int(pred_mask.sum()),
        }
    return {"top1_accuracy": round(acc1, 4), "top5_accuracy": round(acc5, 4),
            "per_class": per_class, "samples": n}


# ---------------------------------------------------------------------------
# Gates + report assembly
# ---------------------------------------------------------------------------

def evaluate_gates(metrics: dict, baseline: dict | None, gates_cfg: dict) -> dict:
    """Evaluate hard/soft gates; returns {gates, gate_result, hard_passed}.

    Hard gates: mAP drop vs baseline and NaN/Inf freedom.
    Soft gates (latency/throughput) only warn (YELLOW), never fail.
    """
    g: dict[str, Any] = {}
    drop_thr = float(gates_cfg.get("max_map_drop_pct", 1.0))
    if baseline and baseline.get("mAP50") is not None and metrics.get("mAP50") is not None:
        base = float(baseline["mAP50"])
        cur = float(metrics["mAP50"])
        drop_pct = (base - cur) / base * 100 if base > 0 else 0.0
        g["max_map_drop_pct"] = {
            "hard": True, "threshold": drop_thr, "value": round(drop_pct, 4),
            "passed": drop_pct <= drop_thr + 1e-9,
            "detail": f"mAP50 {base:.4f} -> {cur:.4f}",
        }
        # Absolute mAP50 drop gate (edge INT8 quality contract, solution D3):
        # present only when configured, so legacy optimization batches keep
        # their historic relative-percent behaviour.
        drop_abs_cfg = gates_cfg.get("max_map_drop_abs")
        if drop_abs_cfg is not None:
            drop_abs_thr = float(drop_abs_cfg)
            drop_abs = base - cur
            g["max_map_drop_abs"] = {
                "hard": True, "threshold": drop_abs_thr,
                "value": round(drop_abs, 4),
                "passed": drop_abs <= drop_abs_thr + 1e-9,
                "detail": f"mAP50 {base:.4f} -> {cur:.4f} (absolute)",
            }
    nan_count = int(metrics.get("nan_inf_count", 0))
    g["nan_inf_free"] = {
        "hard": True, "threshold": 0, "value": nan_count,
        "passed": nan_count == 0,
    }
    bench = metrics.get("benchmark") or {}
    p99_thr = gates_cfg.get("p99_latency_ms")
    if p99_thr is not None and bench.get("p99_ms") is not None:
        g["p99_latency_ms"] = {
            "hard": False, "threshold": float(p99_thr), "value": bench["p99_ms"],
            "passed": float(bench["p99_ms"]) <= float(p99_thr),
        }
    qps_thr = gates_cfg.get("min_throughput_qps")
    if qps_thr is not None and bench.get("qps") is not None:
        g["min_throughput_qps"] = {
            "hard": False, "threshold": float(qps_thr), "value": bench["qps"],
            "passed": float(bench["qps"]) >= float(qps_thr),
        }
    hard_ok = all(x["passed"] for x in g.values() if x.get("hard"))
    soft_ok = all(x["passed"] for x in g.values() if not x.get("hard"))
    result = "PASS" if (hard_ok and soft_ok) else ("YELLOW" if hard_ok else "FAIL")
    return {"gates": g, "gate_result": result, "hard_gates_passed": hard_ok}


def build_report(job_type: str, job_id: str, task_type: str,
                 model_uri: str, dataset_uri: str, metrics: dict,
                 baseline_uri: str = "", baseline_metrics: dict | None = None,
                 gates_out: dict | None = None, extra: dict | None = None) -> dict:
    """Assemble the canonical evaluation report JSON."""
    rep: dict[str, Any] = {
        "job_type": job_type, "job_id": job_id, "task_type": task_type,
        "model_uri": model_uri, "dataset_uri": dataset_uri,
        "metrics": metrics,
        "environment": {
            "hostname": socket.gethostname(),
            "cpu": platform.processor() or platform.machine(),
            "python": platform.python_version(),
            "threads": int(__import__("os").cpu_count() or 1),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if baseline_uri:
        rep["baseline_model_uri"] = baseline_uri
        rep["baseline_metrics"] = baseline_metrics or {}
        if baseline_metrics:
            rep["delta"] = compute_delta(metrics, baseline_metrics)
    if gates_out:
        rep["gates"] = gates_out["gates"]
        rep["gate_result"] = gates_out["gate_result"]
        rep["hard_gates_passed"] = gates_out["hard_gates_passed"]
    if extra:
        rep.update(extra)
    return rep


def compute_delta(metrics: dict, baseline: dict) -> dict:
    """Signed deltas (candidate - baseline) for headline metrics."""
    out = {}
    for key in ("mAP50", "mAP50_95", "precision", "recall", "f1_score", "top1_accuracy"):
        if key in metrics and key in baseline and baseline[key] is not None and metrics[key] is not None:
            out["d_" + key] = round(float(metrics[key]) - float(baseline[key]), 4)
    bm, cm = (metrics.get("benchmark") or {}).get("avg_ms"), (baseline.get("benchmark") or {}).get("avg_ms")
    if bm is not None and cm is not None:
        out["d_avg_latency_ms"] = round(float(bm) - float(cm), 3)
        if cm > 0:
            out["speedup_vs_baseline"] = round(float(cm) / max(float(bm), 1e-9), 3)
    return out


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile (q in [0,1])."""
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return float(s[k])


def benchmark_summary(latencies_ms: list[float]) -> dict:
    """Summarize per-image latencies into the benchmark block."""
    if not latencies_ms:
        return {}
    avg = float(np.mean(latencies_ms))
    return {
        "avg_ms": round(avg, 3),
        "p50_ms": round(percentile(latencies_ms, 0.50), 3),
        "p95_ms": round(percentile(latencies_ms, 0.95), 3),
        "p99_ms": round(percentile(latencies_ms, 0.99), 3),
        "qps": round(1000.0 / avg, 2) if avg > 0 else 0.0,
        "measure_count": len(latencies_ms),
        "method": "sequential single-image, wall clock",
    }


def dump_json(obj: dict) -> str:
    """Pretty JSON dump (report/metrics files)."""
    return json.dumps(obj, indent=2, ensure_ascii=False)


class Timer:
    """Context-free wall-clock stopwatch helper."""

    def __init__(self) -> None:
        self.t0 = time.time()

    def ms(self) -> float:
        return (time.time() - self.t0) * 1000
