# Copyright (C) 2026 YIQISOFT
# SPDX-License-Identifier: Apache-2.0

"""Task-aware metrics for the NF vision trainer (no COCO toolkit needed).

Detection and instance segmentation use the standard 101-point interpolated AP
with greedy score-ordered matching, which is what the evaluation pipeline
reports. Pose uses COCO OKS. Semantic segmentation uses a confusion matrix.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

# COCO keypoint sigmas (17 keypoints, person layout).
COCO_SIGMAS = np.array([
    0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72,
    0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89,
], dtype=np.float32)


def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """101-point interpolated average precision."""
    if recall.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    grid = np.linspace(0, 1, 101)
    idx = np.searchsorted(mrec, grid, side="left")
    idx = np.clip(idx, 0, mpre.size - 1)
    return float(np.mean(mpre[idx]))


def _box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union else 0.0


def detection_map(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    iou_thresholds: Iterable[float] = (0.5,),
    use_masks: bool = False,
) -> dict[str, Any]:
    """mAP over images for detection (or mask mAP for instance seg).

    predictions[i]: {"boxes": Nx4, "scores": N, "labels": N, "masks": [N,H,W]}
    ground_truths[i]: {"boxes": Mx4, "labels": M, "masks": [M,H,W]}
    """
    thresholds = list(iou_thresholds)
    labels = sorted({int(label) for gt in ground_truths for label in gt["labels"]})
    per_threshold: dict[str, float] = {}
    per_class_ap: dict[str, dict[str, float]] = {}

    for thr in thresholds:
        aps: list[float] = []
        for label in labels:
            scores_all: list[float] = []
            matches: list[tuple[int, float, bool]] = []  # (image idx, iou, is_tp)
            n_gt = 0
            for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
                gt_mask = np.asarray(gt["labels"]) == label
                gt_boxes = np.asarray(gt["boxes"])[gt_mask] if len(gt["labels"]) else np.zeros((0, 4))
                gt_masks = [m for m, keep in zip(gt.get("masks") or [], gt_mask) if keep]
                n_gt += len(gt_boxes)

                p_mask = np.asarray(pred["labels"]) == label
                p_boxes = np.asarray(pred["boxes"])[p_mask] if len(pred["labels"]) else np.zeros((0, 4))
                p_scores = np.asarray(pred["scores"])[p_mask] if len(pred["labels"]) else np.zeros((0,))
                p_masks = [m for m, keep in zip(pred.get("masks") or [], p_mask) if keep]

                if len(p_boxes) == 0:
                    continue
                order = np.argsort(-p_scores)
                ious = _box_iou(p_boxes[order], gt_boxes)
                used = np.zeros(len(gt_boxes), dtype=bool)
                for rank, det_idx in enumerate(order):
                    scores_all.append(float(p_scores[det_idx]))
                    best_iou, best_j = 0.0, -1
                    for j in range(len(gt_boxes)):
                        if used[j]:
                            continue
                        iou = float(ious[rank, j])
                        if iou > best_iou:
                            best_iou, best_j = iou, j
                    tp = best_iou >= thr
                    if tp and use_masks and p_masks and gt_masks:
                        tp = _mask_iou(p_masks[det_idx], gt_masks[best_j]) >= thr
                    if tp:
                        used[best_j] = True
                    matches.append((img_idx, best_iou, tp))

            if not matches or n_gt == 0:
                continue
            order = np.argsort(-np.asarray([s for s in scores_all]))
            tp_flags = np.asarray([matches[i][2] for i in order], dtype=np.float32)
            fp_flags = 1.0 - tp_flags
            tp_cum = np.cumsum(tp_flags)
            fp_cum = np.cumsum(fp_flags)
            recall = tp_cum / max(n_gt, 1)
            precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
            ap = _voc_ap(recall, precision)
            aps.append(ap)
            per_class_ap.setdefault(str(label), {})[f"{thr:.2f}"] = round(ap, 4)

        per_threshold[f"mAP@{thr:.2f}"] = round(float(np.mean(aps)) if aps else 0.0, 4)

    result: dict[str, Any] = {**per_threshold, "per_class": per_class_ap}
    if 0.5 in thresholds:
        result["mAP50"] = per_threshold.get("mAP@0.50", 0.0)
    if len(thresholds) > 1:
        result["mAP50-95"] = round(float(np.mean([v for k, v in per_threshold.items() if k.startswith("mAP@")])), 4)
    return result


def _oks(pred_kp: np.ndarray, gt_kp: np.ndarray, area: float,
         sigmas: np.ndarray = COCO_SIGMAS) -> float:
    """COCO object keypoint similarity for one instance."""
    if pred_kp.shape[0] != gt_kp.shape[0]:
        return 0.0
    vars_ = (sigmas * 2) ** 2
    dx = pred_kp[:, 0] - gt_kp[:, 0]
    dy = pred_kp[:, 1] - gt_kp[:, 1]
    visible = gt_kp[:, 2] > 0
    if not visible.any():
        return 0.0
    e = (dx ** 2 + dy ** 2) / (2 * vars_ * max(area, 1.0) + 1e-9)
    return float(np.mean(np.exp(-e[visible])))


def pose_ap(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    thresholds: Iterable[float] = (0.5, 0.75, 0.95),
) -> dict[str, Any]:
    """OKS-based AP (person-style single class keypoint evaluation)."""
    results: dict[str, Any] = {}
    aps = []
    for thr in thresholds:
        tp, fp, n_gt = [], [], 0
        scores: list[float] = []
        for pred, gt in zip(predictions, ground_truths):
            gt_kps = np.asarray(gt.get("keypoints") or [])
            gt_boxes = np.asarray(gt.get("boxes") or [])
            n_gt += len(gt_boxes)
            if not len(pred.get("keypoints") or []) or not len(gt_kps):
                continue
            used = np.zeros(len(gt_kps), dtype=bool)
            order = np.argsort(-np.asarray(pred["scores"]))
            for idx in order:
                scores.append(float(pred["scores"][idx]))
                best, best_j = 0.0, -1
                for j in range(len(gt_kps)):
                    if used[j]:
                        continue
                    area = float((gt_boxes[j][2] - gt_boxes[j][0]) * (gt_boxes[j][3] - gt_boxes[j][1])) if len(gt_boxes) else 1.0
                    score = _oks(np.asarray(pred["keypoints"][idx]), gt_kps[j], area)
                    if score > best:
                        best, best_j = score, j
                ok = best >= thr
                if ok:
                    used[best_j] = True
                tp.append(1.0 if ok else 0.0)
                fp.append(0.0 if ok else 1.0)
        if not scores or n_gt == 0:
            results[f"AP@{thr:.2f}"] = 0.0
            continue
        order = np.argsort(-np.asarray(scores))
        tp_arr = np.asarray(tp)[order]
        fp_arr = np.asarray(fp)[order]
        recall = np.cumsum(tp_arr) / n_gt
        precision = np.cumsum(tp_arr) / np.maximum(np.cumsum(tp_arr) + np.cumsum(fp_arr), 1e-9)
        ap = _voc_ap(recall, precision)
        results[f"AP@{thr:.2f}"] = round(ap, 4)
        aps.append(ap)
    results["OKS_AP"] = round(float(np.mean(aps)) if aps else 0.0, 4)
    return results


def classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, labels: list[str], topk: int = 5
) -> dict[str, Any]:
    """Top-1/top-5, macro precision/recall/F1 and per-class support."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    n_classes = y_prob.shape[1]
    pred = y_prob.argmax(1)
    top1 = float((pred == y_true).mean()) if len(y_true) else 0.0
    k = min(topk, n_classes)
    topk_idx = np.argsort(-y_prob, axis=1)[:, :k]
    topk_acc = float(np.any(topk_idx == y_true[:, None], axis=1).mean()) if len(y_true) else 0.0

    per_class: dict[str, Any] = {}
    precisions, recalls, f1s = [], [], []
    for c in range(n_classes):
        tp = int(((pred == c) & (y_true == c)).sum())
        fp = int(((pred == c) & (y_true != c)).sum())
        fn = int(((pred != c) & (y_true == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        support = int((y_true == c).sum())
        if support:
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)
        name = labels[c] if c < len(labels) else str(c)
        per_class[name] = {"precision": round(p, 4), "recall": round(r, 4),
                           "f1": round(f1, 4), "support": support}

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, pred):
        cm[int(t), int(p)] += 1

    return {
        "accuracy": round(top1, 4),
        f"top{topk}_accuracy": round(topk_acc, 4),
        "macro_precision": round(float(np.mean(precisions)) if precisions else 0.0, 4),
        "macro_recall": round(float(np.mean(recalls)) if recalls else 0.0, 4),
        "macro_f1": round(float(np.mean(f1s)) if f1s else 0.0, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classes": labels,
        "samples": int(len(y_true)),
    }


def multilabel_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, labels: list[str], threshold: float = 0.5
) -> dict[str, Any]:
    """Per-class AP + macro AP for multi-label classification."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    aps: list[float] = []
    per_class: dict[str, Any] = {}
    for c in range(y_prob.shape[1]):
        scores = y_prob[:, c]
        truth = y_true[:, c]
        order = np.argsort(-scores)
        truth_sorted = truth[order]
        tp = np.cumsum(truth_sorted)
        fp = np.cumsum(1 - truth_sorted)
        recall = tp / max(int(truth.sum()), 1)
        precision = tp / np.maximum(tp + fp, 1e-9)
        ap = _voc_ap(recall, precision)
        aps.append(ap)
        pred = (scores >= threshold).astype(int)
        p = float((pred & truth).sum() / max(pred.sum(), 1))
        r = float((pred & truth).sum() / max(truth.sum(), 1))
        name = labels[c] if c < len(labels) else str(c)
        per_class[name] = {"ap": round(ap, 4), "precision": round(p, 4),
                           "recall": round(r, 4), "support": int(truth.sum())}
    return {
        "mAP": round(float(np.mean(aps)) if aps else 0.0, 4),
        "per_class": per_class,
        "classes": labels,
        "samples": int(len(y_true)),
    }


def segmentation_metrics(confusion: np.ndarray, labels: list[str],
                         ignore_index: int = 255) -> dict[str, Any]:
    """Pixel accuracy + per-class IoU from a (C, C) confusion matrix."""
    cm = np.asarray(confusion, dtype=np.float64)
    if ignore_index is not None and ignore_index < cm.shape[0]:
        pass  # ignore row/col already excluded by the accumulator
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    iou = tp / np.maximum(tp + fp + fn, 1e-9)
    present = (tp + fn) > 0
    per_class = {
        (labels[i] if i < len(labels) else str(i)): {
            "iou": round(float(iou[i]), 4),
            "pixels": int(tp[i] + fn[i]),
        }
        for i in range(len(iou)) if present[i]
    }
    return {
        "pixel_accuracy": round(float(tp.sum() / max(cm.sum(), 1e-9)), 4),
        "mean_iou": round(float(iou[present].mean()) if present.any() else 0.0, 4),
        "per_class": per_class,
        "classes": labels,
    }


def mask_to_rle(mask: np.ndarray) -> dict[str, Any]:
    """COCO-style RLE (used by the evaluation report for instance masks)."""
    flat = np.asarray(mask, dtype=np.uint8).reshape(-1)
    counts: list[int] = []
    current, run = 0, 1
    for value in flat:
        if value != current:
            counts.append(run)
            run = 1
            current = int(value)
        else:
            run += 1
    counts.append(run)
    return {"counts": counts, "size": list(mask.shape)}
