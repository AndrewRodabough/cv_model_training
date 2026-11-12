"""Torch-free OKS part / percentile helpers (safe for the Gradio dashboard)."""

from __future__ import annotations

from typing import Any

import numpy as np

BODY_PART_HISTORY_ORDER: tuple[str, ...] = ('feet', 'legs', 'arms', 'head')

BODY_PART_JOINT_NAMES: dict[str, tuple[str, ...]] = {
    'head': ('ear_l', 'ear_r', 'glabella'),
    'arms': (
        'shoulder_l',
        'shoulder_r',
        'elbow_l',
        'elbow_r',
        'wrist_l',
        'wrist_r',
    ),
    'legs': ('hip_l', 'hip_r', 'knee_l', 'knee_r', 'ankle_l', 'ankle_r'),
    'feet': (
        'toe_b_l',
        'toe_s_l',
        'heel_l',
        'heel_spike_l',
        'toe_b_r',
        'toe_s_r',
        'heel_r',
        'heel_spike_r',
    ),
}

OKS_PERCENTILE_BANDS: tuple[tuple[str, int, int], ...] = (
    ('p0_10', 0, 10),
    ('p10_20', 10, 20),
    ('p20_50', 20, 50),
    ('p50_80', 50, 80),
    ('p80_100', 80, 100),
)


def oks_percentile_band_means(oks_values: np.ndarray | list[float]) -> dict[str, float | None]:
    """Mean per-person OKS inside percentile bands (sorted ascending = worst first)."""
    oks = np.asarray(oks_values, dtype=np.float64)
    oks = oks[np.isfinite(oks)]
    out: dict[str, float | None] = {}
    if oks.size == 0:
        return {name: None for name, _, _ in OKS_PERCENTILE_BANDS}
    order = np.sort(oks)
    n = int(order.size)
    for name, lo, hi in OKS_PERCENTILE_BANDS:
        start = int(np.floor(n * lo / 100.0))
        end = int(np.ceil(n * hi / 100.0))
        if end <= start:
            out[name] = None
            continue
        out[name] = float(order[start:end].mean())
    return out


def part_mean_oks_from_metrics(metrics: dict[str, Any]) -> dict[str, float | None]:
    """Extract feet/legs/arms/head mean_OKS from a summarize_pose_metrics result."""
    parts = metrics.get('custom', {}).get('body_parts', {}).get('parts', {})
    out: dict[str, float | None] = {}
    for part in BODY_PART_HISTORY_ORDER:
        value = parts.get(part, {}).get('mean_OKS')
        out[part] = None if value is None else float(value)
    return out


def part_mean_oks_from_joint_arrays(
    joint_oks: np.ndarray,
    visibility: np.ndarray,
    part_indices: dict[str, list[int]],
) -> dict[str, float | None]:
    """Approximate part mean_OKS from per-joint OKS (dashboard backfill from val_preds)."""
    joint_oks = np.asarray(joint_oks, dtype=np.float64)
    visibility = np.asarray(visibility)
    out: dict[str, float | None] = {}
    for part in BODY_PART_HISTORY_ORDER:
        ids = [i for i in part_indices.get(part, []) if 0 <= i < joint_oks.shape[1]]
        if not ids:
            out[part] = None
            continue
        person_means: list[float] = []
        for person_i in range(joint_oks.shape[0]):
            mask = visibility[person_i, ids] > 0
            if not np.any(mask):
                continue
            vals = joint_oks[person_i, ids][mask]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                person_means.append(float(vals.mean()))
        out[part] = float(np.mean(person_means)) if person_means else None
    return out
