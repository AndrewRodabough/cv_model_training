"""Swap / collapse / joint x-y mixing metrics for partner overlap analysis."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from metrics import compute_oks, decode_simcc, person_scale_in_crop


IOU_BUCKETS = (
    ('0.0-0.3', 0.0, 0.3),
    ('0.3-0.6', 0.3, 0.6),
    ('0.6-1.0', 0.6, 1.01),
)


def iou_bucket(max_partner_iou: float) -> str:
    for name, lo, hi in IOU_BUCKETS:
        if lo <= max_partner_iou < hi:
            return name
    return IOU_BUCKETS[-1][0]


def binomial_se(rate: float, n: int) -> float:
    if n <= 0:
        return 0.0
    p = min(max(rate, 0.0), 1.0)
    return math.sqrt(p * (1.0 - p) / n)


@dataclass
class PartnerContext:
    partner_kps: np.ndarray
    partner_bbox: tuple[float, float, float, float]
    max_partner_iou: float


def pick_partner(
    self_bbox: tuple[float, float, float, float],
    others: list[tuple[np.ndarray, tuple[float, float, float, float], str | None]],
) -> PartnerContext | None:
    """Choose the other person on the frame with highest bbox IoU."""
    from train import bbox_iou

    best = None
    best_iou = -1.0
    for kps, bbox, _role in others:
        iou = bbox_iou(self_bbox, bbox)
        if iou > best_iou:
            best_iou = iou
            best = (kps, bbox)
    if best is None:
        return None
    kps, bbox = best
    return PartnerContext(partner_kps=kps, partner_bbox=bbox, max_partner_iou=float(best_iou))


def joint_xy_mix_fraction(
    pred_x: np.ndarray,
    pred_y: np.ndarray,
    self_gt: np.ndarray,
    partner_gt: np.ndarray,
    split_ratio: float,
) -> tuple[float, int]:
    """Fraction of visible joints where x-peak and y-peak vote for different bodies."""
    k = pred_x.shape[0]
    mixed = 0
    counted = 0
    for j in range(k):
        if self_gt[j, 2] <= 0 or partner_gt[j, 2] <= 0:
            continue
        x_peak = float(np.argmax(pred_x[j]) / split_ratio)
        y_peak = float(np.argmax(pred_y[j]) / split_ratio)
        sx, sy = float(self_gt[j, 0]), float(self_gt[j, 1])
        px, py = float(partner_gt[j, 0]), float(partner_gt[j, 1])
        x_votes_self = abs(x_peak - sx) <= abs(x_peak - px)
        y_votes_self = abs(y_peak - sy) <= abs(y_peak - py)
        if x_votes_self != y_votes_self:
            mixed += 1
        counted += 1
    if counted == 0:
        return 0.0, 0
    return mixed / counted, counted


def _ownership_rate_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {
            'n': 0,
            'swap_rate': None,
            'swap_rate_se': None,
            'collapse_rate': None,
            'collapse_rate_se': None,
            'xy_mix_rate': None,
            'xy_mix_rate_se': None,
            'xy_mix_samples': 0,
            'mean_oks_self': None,
            'mean_oks_partner': None,
        }
    swaps = [r['swap'] for r in rows]
    collapses = [r['collapse'] for r in rows]
    xy_mix = [r['xy_mix_rate'] for r in rows if r['xy_mix_joints'] > 0]
    swap_rate = float(np.mean(swaps))
    collapse_rate = float(np.mean(collapses))
    xy_mix_rate = float(np.mean(xy_mix)) if xy_mix else 0.0
    xy_n = len(xy_mix)
    return {
        'n': n,
        'swap_rate': swap_rate,
        'swap_rate_se': binomial_se(swap_rate, n),
        'collapse_rate': collapse_rate,
        'collapse_rate_se': binomial_se(collapse_rate, n),
        'xy_mix_rate': xy_mix_rate,
        'xy_mix_rate_se': binomial_se(xy_mix_rate, xy_n) if xy_n else 0.0,
        'xy_mix_samples': xy_n,
        'mean_oks_self': float(np.mean([r['oks_self'] for r in rows])),
        'mean_oks_partner': float(np.mean([r['oks_partner'] for r in rows])),
    }


def summarize_ownership_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-sample records into overall + IoU-bucketed rates (no per-task)."""
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_bucket[rec['iou_bucket']].append(rec)

    partner_rows = [r for r in records if r['has_partner']]
    out: dict[str, Any] = {
        'num_samples': len(records),
        'num_with_partner': len(partner_rows),
        'overall': _ownership_rate_block(partner_rows),
        'buckets': {},
    }
    for bucket_name, _, _ in IOU_BUCKETS:
        out['buckets'][bucket_name] = _ownership_rate_block(by_bucket.get(bucket_name, []))
    return out


def write_ownership_json(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)


def _format_ownership_row(label: str, b: dict[str, Any]) -> str:
    n = b['n']
    if n == 0 or b['swap_rate'] is None:
        return f'{label:<10} {n:5d}     —        —        —       —       —'
    return (
        f"{label:<10} {n:5d} {b['swap_rate']:7.3f}±{b['swap_rate_se']:.3f} "
        f"{b['collapse_rate']:7.3f}±{b['collapse_rate_se']:.3f} "
        f"{b['xy_mix_rate']:7.3f}±{b['xy_mix_rate_se']:.3f} "
        f"{b['mean_oks_self']:8.3f} {b['mean_oks_partner']:8.3f}"
    )


def format_ownership_summary(metrics: dict[str, Any]) -> str:
    lines = [
        f"samples={metrics['num_samples']} with_partner={metrics['num_with_partner']}",
        '',
        f"{'bucket':<10} {'n':>5} {'swap':>8} {'collapse':>10} {'xy_mix':>8} {'oks_self':>9} {'oks_part':>9}",
    ]
    if 'overall' in metrics:
        lines.append(_format_ownership_row('overall', metrics['overall']))
    for bucket_name, _, _ in IOU_BUCKETS:
        lines.append(_format_ownership_row(bucket_name, metrics['buckets'][bucket_name]))
    return '\n'.join(lines)
