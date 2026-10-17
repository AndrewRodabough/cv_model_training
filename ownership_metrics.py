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


def _contra_map(left_right_pairs: list[tuple[int, int]]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for left_i, right_i in left_right_pairs:
        mapping[int(left_i)] = int(right_i)
        mapping[int(right_i)] = int(left_i)
    return mapping


def _nearest_source(value: float, sources: dict[str, float | None]) -> str | None:
    """Pick source name with min |value - coord|; ignore None sources."""
    best_name = None
    best_dist = float('inf')
    for name, coord in sources.items():
        if coord is None:
            continue
        dist = abs(value - float(coord))
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name


def classify_axis_partial_swap(
    x_peak: float,
    y_peak: float,
    self_xy: tuple[float, float],
    contra_xy: tuple[float, float] | None,
    partner_xy: tuple[float, float] | None,
    partner_contra_xy: tuple[float, float] | None = None,
) -> str:
    """Label one joint's SimCC peaks by nearest 1D sources.

    Returns one of:
      both_self, partial_lr, partial_partner, partial_partner_lr,
      both_lr, both_partner, other
    """
    sx, sy = self_xy
    x_sources: dict[str, float | None] = {'self': sx, 'contra': None, 'partner': None, 'partner_contra': None}
    y_sources: dict[str, float | None] = {'self': sy, 'contra': None, 'partner': None, 'partner_contra': None}
    if contra_xy is not None:
        x_sources['contra'] = contra_xy[0]
        y_sources['contra'] = contra_xy[1]
    if partner_xy is not None:
        x_sources['partner'] = partner_xy[0]
        y_sources['partner'] = partner_xy[1]
    if partner_contra_xy is not None:
        x_sources['partner_contra'] = partner_contra_xy[0]
        y_sources['partner_contra'] = partner_contra_xy[1]

    x_src = _nearest_source(x_peak, x_sources)
    y_src = _nearest_source(y_peak, y_sources)
    if x_src is None or y_src is None:
        return 'other'
    if x_src == 'self' and y_src == 'self':
        return 'both_self'
    if {x_src, y_src} == {'self', 'contra'}:
        return 'partial_lr'
    if {x_src, y_src} == {'self', 'partner'}:
        return 'partial_partner'
    if {x_src, y_src} == {'self', 'partner_contra'}:
        return 'partial_partner_lr'
    if x_src == 'contra' and y_src == 'contra':
        return 'both_lr'
    if x_src == 'partner' and y_src == 'partner':
        return 'both_partner'
    return 'other'


def axis_partial_swap_records(
    pred_x: np.ndarray,
    pred_y: np.ndarray,
    self_gt: np.ndarray,
    partner_gt: np.ndarray | None,
    split_ratio: float,
    left_right_pairs: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    """Per-joint axis attribution for one person crop (partner optional)."""
    contra = _contra_map(left_right_pairs)
    k = pred_x.shape[0]
    rows: list[dict[str, Any]] = []
    for j in range(k):
        if self_gt[j, 2] <= 0:
            continue
        x_peak = float(np.argmax(pred_x[j]) / split_ratio)
        y_peak = float(np.argmax(pred_y[j]) / split_ratio)
        self_xy = (float(self_gt[j, 0]), float(self_gt[j, 1]))
        contra_xy = None
        cj = contra.get(j)
        if cj is not None and self_gt[cj, 2] > 0:
            contra_xy = (float(self_gt[cj, 0]), float(self_gt[cj, 1]))
        partner_xy = None
        partner_contra_xy = None
        if partner_gt is not None and partner_gt[j, 2] > 0:
            partner_xy = (float(partner_gt[j, 0]), float(partner_gt[j, 1]))
            if cj is not None and partner_gt[cj, 2] > 0:
                partner_contra_xy = (float(partner_gt[cj, 0]), float(partner_gt[cj, 1]))
        label = classify_axis_partial_swap(
            x_peak, y_peak, self_xy, contra_xy, partner_xy, partner_contra_xy
        )
        rows.append(
            {
                'joint': j,
                'label': label,
                'has_contra': contra_xy is not None,
                'has_partner_joint': partner_xy is not None,
            }
        )
    return rows


def summarize_axis_partial_swaps(
    joint_rows: list[dict[str, Any]],
    *,
    joint_names: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Aggregate axis-partial-swap labels (global + per-joint)."""
    labels = (
        'both_self',
        'partial_lr',
        'partial_partner',
        'partial_partner_lr',
        'both_lr',
        'both_partner',
        'other',
    )
    n = len(joint_rows)
    counts = {lab: 0 for lab in labels}
    for row in joint_rows:
        counts[row['label']] = counts.get(row['label'], 0) + 1

    def rate(name: str) -> float | None:
        return (counts[name] / n) if n else None

    partial_any = counts['partial_lr'] + counts['partial_partner'] + counts['partial_partner_lr']
    out: dict[str, Any] = {
        'num_joints': n,
        'rates': {lab: rate(lab) for lab in labels},
        'counts': counts,
        'partial_any_rate': (partial_any / n) if n else None,
        'partial_lr_rate': rate('partial_lr'),
        'partial_partner_rate': rate('partial_partner'),
        'partial_partner_lr_rate': rate('partial_partner_lr'),
    }

    by_joint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joint_rows:
        key = joint_names.get(row['joint'], str(row['joint'])) if joint_names else str(row['joint'])
        by_joint[key].append(row)
    per_joint = {}
    for name, rows in by_joint.items():
        nj = len(rows)
        c = defaultdict(int)
        for r in rows:
            c[r['label']] += 1
        per_joint[name] = {
            'n': nj,
            'partial_lr': c['partial_lr'] / nj if nj else 0.0,
            'partial_partner': c['partial_partner'] / nj if nj else 0.0,
            'partial_partner_lr': c['partial_partner_lr'] / nj if nj else 0.0,
            'partial_any': (c['partial_lr'] + c['partial_partner'] + c['partial_partner_lr']) / nj
            if nj
            else 0.0,
            'both_self': c['both_self'] / nj if nj else 0.0,
        }
    out['per_joint'] = per_joint
    return out


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
