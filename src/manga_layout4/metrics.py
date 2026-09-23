from __future__ import annotations
from collections import defaultdict
from typing import Any

import torch
from torch.nn import functional as F

from . import CLASSES
from .decode import decode_one


def _iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-9)


def _match_predictions(preds: list[tuple[float, str, list[float]]], gts: dict[str, list[list[float]]]):
    matched = {key: set() for key in gts}
    rows = []
    duplicate = 0
    for score, page, box in sorted(preds, key=lambda x: -x[0]):
        candidates = gts.get(page, [])
        best = -1
        best_iou = 0.0
        best_any = 0.0
        for j, gt in enumerate(candidates):
            value = _iou(box, gt)
            best_any = max(best_any, value)
            if j in matched.setdefault(page, set()):
                continue
            if value > best_iou:
                best_iou = value
                best = j
        is_tp = best >= 0 and best_iou >= 0.5
        if is_tp:
            matched[page].add(best)
        else:
            # A prediction overlapping an already matched GT is a duplicate FP.
            if best_any >= 0.5:
                duplicate += 1
        rows.append((score, 1.0 if is_tp else 0.0, 0.0 if is_tp else 1.0))
    return rows, matched, duplicate


def _ap50(preds: list[tuple[float, str, list[float]]], gts: dict[str, list[list[float]]]):
    total = sum(len(v) for v in gts.values())
    if total == 0:
        return float("nan")
    rows, _, _ = _match_predictions(preds, gts)
    if not rows:
        return 0.0
    tp = torch.tensor([r[1] for r in rows]).cumsum(0)
    fp = torch.tensor([r[2] for r in rows]).cumsum(0)
    rec = tp / total
    prec = tp / (tp + fp).clamp_min(1e-9)
    vals = []
    for recall_point in torch.linspace(0, 1, 101):
        candidates = prec[rec >= recall_point]
        vals.append(float(candidates.max()) if len(candidates) else 0.0)
    return sum(vals) / 101.0


def _diagnostics(preds, gts, page_count: int):
    total = sum(len(v) for v in gts.values())
    rows, matched, duplicates = _match_predictions(preds, gts)
    true_positive = sum(len(v) for v in matched.values())
    false_positive = sum(int(r[2]) for r in rows)
    return {
        "gt": total,
        "predictions": len(preds),
        "recall50": true_positive / max(total, 1),
        "fp": false_positive,
        "fp_per_page": false_positive / max(page_count, 1),
        "duplicate_fp": duplicates,
    }


def _boundary(mask: torch.Tensor) -> torch.Tensor:
    x = mask.float().unsqueeze(0).unsqueeze(0)
    eroded = 1 - F.max_pool2d(1 - x, 3, 1, 1)
    return ((x - eroded) > 0.5)[0, 0]


def _boundary_f1(pred: torch.Tensor, gt: torch.Tensor, tol: int = 2) -> float:
    pb = _boundary(pred)
    gb = _boundary(gt)
    if not pb.any() and not gb.any():
        return 1.0
    if not pb.any() or not gb.any():
        return 0.0
    pred_near_gt = F.max_pool2d(gb.float()[None, None], 2 * tol + 1, 1, tol)[0, 0] > 0
    gt_near_pred = F.max_pool2d(pb.float()[None, None], 2 * tol + 1, 1, tol)[0, 0] > 0
    precision = float((pb & pred_near_gt).sum()) / max(1, int(pb.sum()))
    recall = float((gb & gt_near_pred).sum()) / max(1, int(gb.sum()))
    return 2 * precision * recall / max(precision + recall, 1e-9)


@torch.no_grad()
def evaluate(model, loader, cfg, device):
    model.eval()
    preds = {class_name: [] for class_name in CLASSES}
    gts = {class_name: defaultdict(list) for class_name in CLASSES}
    mask_ious = []
    boundary = []
    empty_mask_fp_fraction = []
    page_count = 0

    for images, targets in loader:
        images = images.to(device)
        out = model(images)
        predicted_masks = (
            out["balloon_mask_logits"].sigmoid()
            >= float(cfg["postprocess"]["balloon_mask_threshold"])
        ).cpu()
        for batch_index, target in enumerate(targets):
            page_count += 1
            page = str(target["image_id"])
            det = decode_one(out, batch_index, cfg)
            for box, label, score in zip(
                det["boxes"].cpu().tolist(),
                det["labels"].cpu().tolist(),
                det["scores"].cpu().tolist(),
            ):
                preds[CLASSES[label]].append((float(score), page, [float(x) for x in box]))
            for box, label in zip(target["boxes"].tolist(), target["labels"].tolist()):
                gts[CLASSES[label]][page].append([float(x) for x in box])

            gt = target["balloon_mask"].bool()
            pred = predicted_masks[batch_index]
            if gt.any():
                inter = (gt & pred).sum().item()
                union = (gt | pred).sum().item()
                mask_ious.append(inter / max(union, 1))
                boundary.append(_boundary_f1(pred[0], gt[0]))
            else:
                empty_mask_fp_fraction.append(float(pred.float().mean()))

    metrics = {f"{class_name}_ap50": _ap50(preds[class_name], gts[class_name]) for class_name in CLASSES}
    metrics["balloon_box_ap50"] = metrics.pop("balloon_ap50")
    metrics["balloon_mask_iou"] = sum(mask_ious) / max(1, len(mask_ious))
    metrics["balloon_boundary_f1"] = sum(boundary) / max(1, len(boundary))
    metrics["balloon_mask_positive_pages"] = len(mask_ious)
    metrics["balloon_empty_page_fp_fraction"] = (
        sum(empty_mask_fp_fraction) / max(1, len(empty_mask_fp_fraction))
    )
    metrics["page_count"] = page_count
    metrics["detection_diagnostics"] = {
        class_name: _diagnostics(preds[class_name], gts[class_name], page_count)
        for class_name in CLASSES
    }

    weights = cfg["selection"]["composite_weights"]
    metrics["composite"] = (
        metrics["frame_ap50"] * float(weights["frame_ap50"])
        + metrics["text_ap50"] * float(weights["text_ap50"])
        + metrics["balloon_box_ap50"] * float(weights["balloon_box_ap50"])
        + metrics["onomatopoeia_ap50"] * float(weights["onomatopoeia_ap50"])
        + metrics["balloon_mask_iou"] * float(weights["balloon_mask_iou"])
        + metrics["balloon_boundary_f1"] * float(weights["balloon_boundary_f1"])
    )
    return metrics
