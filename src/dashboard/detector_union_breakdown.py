"""Val breakdown: unioned vs separate GT boxes for a detector run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.dashboard.detector_overlay import (
    box_iou,
    load_pred_cache,
    read_yolo_labels,
    resolve_data_paths,
)
from src.dataset_review.export_bbox_detection import (
    apply_label_policy,
    collect_cleaned_person_boxes,
)


def _best_iou(pred_boxes: list[dict], gt_xyxy) -> float:
    if not pred_boxes:
        return 0.0
    return max(box_iou(pred['xyxy'], gt_xyxy) for pred in pred_boxes)


def _coverage_of_union(pred_boxes: list[dict], gt_union, iou_thresh: float = 0.5) -> str:
    if not pred_boxes:
        return 'miss'
    ious = [box_iou(pred['xyxy'], gt_union) for pred in pred_boxes]
    best = max(ious)
    if any(iou >= iou_thresh for iou in ious):
        return 'one_cover'
    if len(pred_boxes) >= 2:
        merged = pred_boxes[0]['xyxy']
        for pred in pred_boxes[1:]:
            merged = (
                min(merged[0], pred['xyxy'][0]),
                min(merged[1], pred['xyxy'][1]),
                max(merged[2], pred['xyxy'][2]),
                max(merged[3], pred['xyxy'][3]),
            )
        if box_iou(merged, gt_union) >= iou_thresh:
            return 'two_cover'
    if best > 0:
        return 'under_cover'
    return 'miss'


def ensure_box_kinds(data_root: Path) -> dict[str, Any]:
    """Load box_kinds.json or rebuild it from cleaned detector annotations."""
    path = data_root / 'box_kinds.json'
    if path.is_file():
        return json.loads(path.read_text(encoding='utf-8'))

    meta_path = data_root / 'export_meta.json'
    if not meta_path.is_file():
        raise FileNotFoundError(f'Missing {meta_path}')
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    version = str(meta.get('annotation_version') or '')
    always_separate = bool(meta.get('always_separate'))
    tau = meta.get('union_iou_threshold')
    seed = int(meta.get('seed') or 44)
    min_track = int(meta.get('min_track_length') or 10)
    from src.util.get_version import get_detector_version_folder

    ann = get_detector_version_folder(version) / 'cleaned_annotations.xml'
    frames, _, _ = collect_cleaned_person_boxes(ann, min_track_length=min_track, frame_stride=None)
    train_tasks = set(str(t) for t in (meta.get('train_tasks') or []))
    val_tasks = set(str(t) for t in (meta.get('val_tasks') or []))

    payload_frames: dict[str, Any] = {}
    for (task_id, relative_frame), people in frames.items():
        if task_id not in train_tasks and task_id not in val_tasks:
            continue
        boxes, _tiers, stats = apply_label_policy(
            people,
            union_iou_threshold=None if always_separate else tau,
            always_separate=always_separate,
            soft_band=0.0,
            soft_mode='hard',
            decision_key=(seed, task_id, relative_frame),
        )
        if not boxes:
            continue
        kinds = list(stats.get('box_kinds') or [])
        stem = f'task_{task_id}_frame_{relative_frame:06d}'
        payload_frames[stem] = {
            'task_id': task_id,
            'frame': relative_frame,
            'split': 'train' if task_id in train_tasks else 'val',
            'kinds': kinds,
            'n_union': kinds.count('union'),
            'n_separate': kinds.count('separate'),
            'n_unpaired': kinds.count('unpaired'),
            'boxes_xyxy': [list(b) for b in boxes],
        }
    payload = {
        'policy': meta.get('policy'),
        'union_iou_threshold': tau,
        'frames': payload_frames,
    }
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return payload


def compute_union_breakdown(
    run_dir: Path,
    checkpoint: str = 'best',
    iou_thresh: float = 0.5,
) -> dict[str, Any]:
    """Aggregate val metrics for union / separate / unpaired GT boxes."""
    import yaml

    run_dir = Path(run_dir)
    args = yaml.safe_load((run_dir / 'args.yaml').read_text(encoding='utf-8')) or {}
    data_yaml = Path(str(args.get('data') or ''))
    data_root = data_yaml.parent if data_yaml.is_file() else None
    if data_root is None or not data_root.is_dir():
        raise FileNotFoundError(f'Cannot resolve detection export from {run_dir}')

    kinds_doc = ensure_box_kinds(data_root)
    cache = load_pred_cache(run_dir, checkpoint)
    if not cache:
        raise FileNotFoundError(
            f'No pred cache for {checkpoint} under {run_dir}/val_browser — '
            f'cache preds first'
        )

    _, images_val, labels_val = resolve_data_paths(run_dir)
    from PIL import Image

    kind_stats = {
        kind: {
            'n_gt': 0,
            'tp': 0,
            'sum_iou': 0.0,
            'ious': [],
        }
        for kind in ('union', 'separate', 'unpaired')
    }
    union_cover = {'one_cover': 0, 'two_cover': 0, 'under_cover': 0, 'miss': 0}
    frame_bucket = {
        'has_union': {'n': 0, 'sum_mean_iou': 0.0, 'tp': 0, 'fn': 0, 'fp': 0},
        'separate_only': {'n': 0, 'sum_mean_iou': 0.0, 'tp': 0, 'fn': 0, 'fp': 0},
        'unpaired_only': {'n': 0, 'sum_mean_iou': 0.0, 'tp': 0, 'fn': 0, 'fp': 0},
    }

    frames_meta = kinds_doc.get('frames') or {}
    pred_frames = (cache.get('frames') or {})
    used = 0
    for stem, meta in frames_meta.items():
        if meta.get('split') != 'val':
            continue
        if stem not in pred_frames:
            continue
        entry = pred_frames[stem]
        preds = [
            {
                'cls': int(b.get('cls', 0)),
                'xyxy': tuple(b['xyxy']),
                'conf': float(b.get('conf', 0)),
            }
            for b in entry.get('preds') or []
        ]
        kinds = list(meta.get('kinds') or [])
        gt_boxes = meta.get('boxes_xyxy')
        if not gt_boxes:
            # Fall back to YOLO labels on disk.
            image_path = images_val / f'{stem}.jpg'
            if not image_path.is_file():
                image_path = images_val / f'{stem}.png'
            if not image_path.is_file():
                continue
            with Image.open(image_path) as im:
                width, height = im.size
            labeled = read_yolo_labels(labels_val / f'{stem}.txt', width, height)
            gt_boxes = [list(b['xyxy']) for b in labeled]
        if len(kinds) != len(gt_boxes):
            # Kinds must align with emitted boxes; skip inconsistent rows.
            continue

        for kind, xyxy in zip(kinds, gt_boxes):
            if kind not in kind_stats:
                continue
            iou = _best_iou(preds, tuple(xyxy))
            bucket = kind_stats[kind]
            bucket['n_gt'] += 1
            bucket['sum_iou'] += iou
            bucket['ious'].append(iou)
            if iou >= iou_thresh:
                bucket['tp'] += 1
            if kind == 'union':
                union_cover[_coverage_of_union(preds, tuple(xyxy), iou_thresh)] += 1

        n_union = int(meta.get('n_union') or kinds.count('union'))
        n_sep = int(meta.get('n_separate') or kinds.count('separate'))
        if n_union > 0:
            fb = frame_bucket['has_union']
        elif n_sep > 0:
            fb = frame_bucket['separate_only']
        else:
            fb = frame_bucket['unpaired_only']
        fb['n'] += 1
        fb['sum_mean_iou'] += float(entry.get('mean_iou') or 0.0)
        fb['tp'] += int(entry.get('tp') or 0)
        fb['fn'] += int(entry.get('fn') or 0)
        fb['fp'] += int(entry.get('fp') or 0)
        used += 1

    summary = {
        'run_dir': str(run_dir),
        'checkpoint': checkpoint,
        'policy': kinds_doc.get('policy'),
        'union_iou_threshold': kinds_doc.get('union_iou_threshold'),
        'n_val_frames_used': used,
        'iou_thresh': iou_thresh,
        'by_box_kind': {},
        'union_cover': union_cover,
        'by_frame_bucket': {},
    }
    for kind, bucket in kind_stats.items():
        n = bucket['n_gt']
        summary['by_box_kind'][kind] = {
            'n_gt': n,
            'recall': (bucket['tp'] / n) if n else None,
            'mean_iou': (bucket['sum_iou'] / n) if n else None,
            'tp': bucket['tp'],
        }
    for name, bucket in frame_bucket.items():
        n = bucket['n']
        summary['by_frame_bucket'][name] = {
            'n_frames': n,
            'mean_frame_iou': (bucket['sum_mean_iou'] / n) if n else None,
            'tp': bucket['tp'],
            'fn': bucket['fn'],
            'fp': bucket['fp'],
        }
    return summary
