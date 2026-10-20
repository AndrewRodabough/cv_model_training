from __future__ import annotations
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np

from src.config.coco_wholebody_config import CocoWholeBodyConfig

# COCO body keypoint index → dance joint name (subset of 17 COCO body joints).
COCO_BODY_NAME_BY_INDEX: dict[int, str] = {
    0: 'nose',  # proxy source for glabella
    3: 'ear_l',
    4: 'ear_r',
    5: 'shoulder_l',
    6: 'shoulder_r',
    7: 'elbow_l',
    8: 'elbow_r',
    9: 'wrist_l',
    10: 'wrist_r',
    11: 'hip_l',
    12: 'hip_r',
    13: 'knee_l',
    14: 'knee_r',
    15: 'ankle_l',
    16: 'ankle_r',
}

# WholeBody foot_kpts order (6 joints × 3) → dance names.
WHOLEBODY_FOOT_NAMES: tuple[str, ...] = (
    'toe_b_l',
    'toe_s_l',
    'heel_l',
    'toe_b_r',
    'toe_s_r',
    'heel_r',
)

HEEL_TO_SPIKE: dict[str, str] = {
    'heel_l': 'heel_spike_l',
    'heel_r': 'heel_spike_r',
}

Sample = tuple[Path, np.ndarray, tuple[float, float, float, float], None]




def _local_id_by_name(keypoints_by_name: dict[str, dict]) -> dict[str, int]:
    return {name: int(meta['id']) for name, meta in keypoints_by_name.items()}


def _set_joint(
    keypoints: np.ndarray,
    local_ids: dict[str, int],
    name: str,
    x: float,
    y: float,
    visibility: float,
) -> None:
    joint_id = local_ids.get(name)
    if joint_id is None:
        return
    if visibility <= 0:
        return
    keypoints[joint_id] = [x, y, visibility]


def _count_labeled(keypoints: np.ndarray) -> int:
    return int(np.sum(keypoints[:, 2] > 0))


def annotation_to_keypoints(
    ann: dict[str, Any],
    num_joints: int,
    local_ids: dict[str, int],
    *,
    proxy_glabella_from_nose: bool,
    proxy_heel_spike_from_heel: bool,
) -> np.ndarray:
    """Map one COCO-WholeBody annotation into dance_* [K, 3] keypoints."""
    keypoints = np.zeros((num_joints, 3), dtype=np.float32)
    body = ann.get('keypoints') or []
    if len(body) >= 51:
        for coco_idx, dance_name in COCO_BODY_NAME_BY_INDEX.items():
            base = coco_idx * 3
            x, y, v = float(body[base]), float(body[base + 1]), float(body[base + 2])
            if dance_name == 'nose':
                if proxy_glabella_from_nose and 'glabella' in local_ids and v > 0:
                    # Soft proxy: occluded-style weight in SimCC loss.
                    _set_joint(keypoints, local_ids, 'glabella', x, y, 1.0)
                continue
            _set_joint(keypoints, local_ids, dance_name, x, y, v)

    foot = ann.get('foot_kpts') or []
    foot_valid = bool(ann.get('foot_valid', True))
    if foot_valid and len(foot) >= 18:
        for foot_i, dance_name in enumerate(WHOLEBODY_FOOT_NAMES):
            base = foot_i * 3
            x, y, v = float(foot[base]), float(foot[base + 1]), float(foot[base + 2])
            _set_joint(keypoints, local_ids, dance_name, x, y, v)

    if proxy_heel_spike_from_heel:
        for heel_name, spike_name in HEEL_TO_SPIKE.items():
            heel_id = local_ids.get(heel_name)
            spike_id = local_ids.get(spike_name)
            if heel_id is None or spike_id is None:
                continue
            hx, hy, hv = keypoints[heel_id]
            if hv > 0:
                # Soft prior only — never hard-label spike = heel.
                keypoints[spike_id] = [hx, hy, 1.0]

    return keypoints


def build_coco_wholebody_samples(
    cfg: CocoWholeBodyConfig,
    keypoints_by_name: dict[str, dict],
    num_joints: int,
    *,
    seed: int | None = None,
    target_count: int | None = None,
) -> list[Sample]:
    """Build single-person samples from COCO-WholeBody train annotations.

    When ``target_count`` and/or ``cfg.max_samples`` is set, uses reservoir
    sampling so the full annotation list is not materialized.
    """
    cfg.validate()
    if not cfg.enabled:
        return []

    local_ids = _local_id_by_name(keypoints_by_name)
    ann_path = Path(cfg.ann_file)
    image_root = Path(cfg.image_root)

    keep_cap: int | None = target_count
    if cfg.max_samples is not None:
        keep_cap = (
            cfg.max_samples
            if keep_cap is None
            else min(keep_cap, cfg.max_samples)
        )

    with open(ann_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    images_by_id = {img['id']: img for img in data['images']}
    samples: list[Sample] = []
    n_skipped = 0
    n_seen_valid = 0
    rng = random.Random(seed)

    for ann in data['annotations']:
        if ann.get('iscrowd', 0):
            n_skipped += 1
            continue
        image_info = images_by_id.get(ann['image_id'])
        if image_info is None:
            n_skipped += 1
            continue
        bbox = ann.get('bbox')
        if not bbox or len(bbox) != 4:
            n_skipped += 1
            continue
        x, y, w, h = (float(v) for v in bbox)
        if w <= 1.0 or h <= 1.0:
            n_skipped += 1
            continue

        kpts = annotation_to_keypoints(
            ann,
            num_joints,
            local_ids,
            proxy_glabella_from_nose=cfg.proxy_glabella_from_nose,
            proxy_heel_spike_from_heel=cfg.proxy_heel_spike_from_heel,
        )
        if _count_labeled(kpts) < cfg.min_labeled_joints:
            n_skipped += 1
            continue

        file_name = image_info['file_name']
        image_path = image_root / file_name
        bbox_xyxy = (x, y, x + w, y + h)
        sample: Sample = (image_path, kpts, bbox_xyxy, None)
        n_seen_valid += 1

        if keep_cap is None:
            samples.append(sample)
            continue
        if len(samples) < keep_cap:
            samples.append(sample)
            continue
        # Algorithm R reservoir sampling.
        j = rng.randint(0, n_seen_valid - 1)
        if j < keep_cap:
            samples[j] = sample

    if not samples:
        raise ValueError(
            f'No COCO-WholeBody samples kept from {ann_path} '
            f'(skipped={n_skipped}, min_labeled_joints={cfg.min_labeled_joints})'
        )

    print(
        f'COCO-WholeBody: kept {len(samples)} person samples '
        f'(valid_seen={n_seen_valid}, skipped={n_skipped}, '
        f'keep_cap={keep_cap}, min_labeled_joints={cfg.min_labeled_joints}, '
        f'proxy_glabella={cfg.proxy_glabella_from_nose}, '
        f'proxy_heel_spike={cfg.proxy_heel_spike_from_heel})'
    )
    return samples


def target_coco_count(n_dance_train: int, dance_fraction: float) -> int:
    """n_coco ≈ n_dance * (1 - f) / f so dance is ~dance_fraction of the mix."""
    if n_dance_train <= 0:
        raise ValueError('n_dance_train must be > 0 to mix with COCO')
    if dance_fraction >= 1.0:
        return 0
    return max(0, int(round(n_dance_train * (1.0 - dance_fraction) / dance_fraction)))


def subsample_coco_for_dance_fraction(
    coco_samples: list[Sample],
    n_dance_train: int,
    dance_fraction: float,
    seed: int,
) -> list[Sample]:
    """Pick COCO count so dance is ~dance_fraction of the concatenated train list."""
    if not coco_samples:
        return []

    target_coco = target_coco_count(n_dance_train, dance_fraction)
    if target_coco == 0:
        return []
    if target_coco >= len(coco_samples):
        if target_coco > len(coco_samples):
            print(
                f'COCO mix: requested {target_coco} samples for dance_fraction='
                f'{dance_fraction:.2f} but only {len(coco_samples)} available; using all'
            )
        return list(coco_samples)

    rng = random.Random(seed)
    return rng.sample(coco_samples, target_coco)


def mix_dance_with_coco(
    dance_samples: list[Sample],
    coco_pool: list[Sample],
    dance_fraction: float,
    seed: int,
) -> tuple[list[Sample], int]:
    """Subsample COCO to the target mix, concatenate with dance, and shuffle."""
    coco_train = subsample_coco_for_dance_fraction(
        coco_pool,
        len(dance_samples),
        dance_fraction,
        seed=seed,
    )
    mixed: list[Sample] = list(dance_samples) + list(coco_train)
    rng = random.Random(seed + 31)
    rng.shuffle(mixed)
    return mixed, len(coco_train)


def mean_labeled_joints(samples: list[Sample]) -> float:
    if not samples:
        return 0.0
    return float(np.mean([_count_labeled(k) for _, k, _, _ in samples]))
