from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    boxes1 = np.asarray(boxes1, dtype=np.float32).reshape(-1, 4)
    boxes2 = np.asarray(boxes2, dtype=np.float32).reshape(-1, 4)

    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    area1 = np.maximum(0.0, boxes1[:, 2] - boxes1[:, 0]) * np.maximum(
        0.0, boxes1[:, 3] - boxes1[:, 1]
    )
    area2 = np.maximum(0.0, boxes2[:, 2] - boxes2[:, 0]) * np.maximum(
        0.0, boxes2[:, 3] - boxes2[:, 1]
    )

    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.maximum(0.0, rb - lt)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter

    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _voc_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def average_precision(
    records: List[Dict[str, np.ndarray]],
    iou_threshold: float,
) -> Tuple[float, np.ndarray, np.ndarray]:
    total_gt = int(sum(len(record["gt_boxes"]) for record in records))
    if total_gt == 0:
        return 0.0, np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    predictions = []
    gt_by_image = {}
    matched_by_image = {}

    for image_idx, record in enumerate(records):
        gt_boxes = np.asarray(record["gt_boxes"], dtype=np.float32).reshape(-1, 4)
        pred_boxes = np.asarray(record["pred_boxes"], dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(record["scores"], dtype=np.float32).reshape(-1)

        gt_by_image[image_idx] = gt_boxes
        matched_by_image[image_idx] = np.zeros(len(gt_boxes), dtype=bool)

        for pred_idx, score in enumerate(scores):
            predictions.append((float(score), image_idx, pred_boxes[pred_idx]))

    if len(predictions) == 0:
        return 0.0, np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32)

    predictions.sort(key=lambda item: item[0], reverse=True)
    tp = np.zeros(len(predictions), dtype=np.float32)
    fp = np.zeros(len(predictions), dtype=np.float32)

    for idx, (_, image_idx, pred_box) in enumerate(predictions):
        gt_boxes = gt_by_image[image_idx]
        matched = matched_by_image[image_idx]

        if len(gt_boxes) == 0:
            fp[idx] = 1.0
            continue

        ious = box_iou(pred_box[None, :], gt_boxes)[0]
        best_gt = int(np.argmax(ious))
        best_iou = float(ious[best_gt])

        if best_iou >= iou_threshold and not matched[best_gt]:
            tp[idx] = 1.0
            matched[best_gt] = True
        else:
            fp[idx] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / max(total_gt, 1)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    return _voc_ap(recalls, precisions), recalls, precisions


def precision_recall_at_threshold(
    records: List[Dict[str, np.ndarray]],
    iou_threshold: float,
    score_threshold: float,
) -> Dict[str, float]:
    tp = 0
    fp = 0
    fn = 0

    for record in records:
        gt_boxes = np.asarray(record["gt_boxes"], dtype=np.float32).reshape(-1, 4)
        pred_boxes = np.asarray(record["pred_boxes"], dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(record["scores"], dtype=np.float32).reshape(-1)

        keep = scores >= score_threshold
        pred_boxes = pred_boxes[keep]
        scores = scores[keep]
        order = np.argsort(-scores)
        pred_boxes = pred_boxes[order]

        matched = np.zeros(len(gt_boxes), dtype=bool)
        for pred_box in pred_boxes:
            if len(gt_boxes) == 0:
                fp += 1
                continue

            ious = box_iou(pred_box[None, :], gt_boxes)[0]
            best_gt = int(np.argmax(ious))
            best_iou = float(ious[best_gt])

            if best_iou >= iou_threshold and not matched[best_gt]:
                tp += 1
                matched[best_gt] = True
            else:
                fp += 1

        fn += int((~matched).sum())

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def mean_best_iou(records: List[Dict[str, np.ndarray]]) -> float:
    best_ious = []
    for record in records:
        gt_boxes = np.asarray(record["gt_boxes"], dtype=np.float32).reshape(-1, 4)
        pred_boxes = np.asarray(record["pred_boxes"], dtype=np.float32).reshape(-1, 4)
        if len(gt_boxes) == 0:
            continue
        ious = box_iou(gt_boxes, pred_boxes)
        if ious.shape[1] == 0:
            best_ious.extend([0.0] * len(gt_boxes))
        else:
            best_ious.extend(np.max(ious, axis=1).tolist())

    return float(np.mean(best_ious)) if best_ious else 0.0


def evaluate_detection_records(
    records: List[Dict[str, np.ndarray]],
    iou_thresholds: Iterable[float] = None,
    pr_iou_threshold: float = 0.5,
    score_threshold: float = 0.5,
) -> Dict[str, float]:
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)

    metrics: Dict[str, float] = {
        "num_images": float(len(records)),
        "num_gt_boxes": float(sum(len(record["gt_boxes"]) for record in records)),
        "num_pred_boxes": float(sum(len(record["pred_boxes"]) for record in records)),
        "mean_best_iou": mean_best_iou(records),
    }

    ap_values = []
    for threshold in iou_thresholds:
        threshold = round(float(threshold), 2)
        ap, _, _ = average_precision(records, threshold)
        metrics[f"ap_{int(round(threshold * 100)):02d}"] = float(ap)
        ap_values.append(ap)

    metrics["map_50"] = metrics.get("ap_50", 0.0)
    metrics["map_50_95"] = float(np.mean(ap_values)) if ap_values else 0.0
    metrics.update(
        {
            f"{key}_iou{int(pr_iou_threshold * 100)}_score{int(score_threshold * 100)}": value
            for key, value in precision_recall_at_threshold(
                records,
                iou_threshold=pr_iou_threshold,
                score_threshold=score_threshold,
            ).items()
        }
    )

    return metrics
