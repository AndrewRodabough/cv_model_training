#!/usr/bin/env python3
"""Export cleaned bbox tracks to COCO + Ultralytics detection labels.

Label policy (default):
  - Keep only effectively cleaned bbox shapes (same rules as dataset_parser).
  - Pair partners by couple_id when present; else lead/follow + max IoU.
  - If pair IoU >= union_iou_threshold, emit one union box; else both boxes.
  - Single class: person.
"""

from __future__ import annotations

import argparse
import json
import random
import struct
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from src.dataset_review.dataset_parser import (
    IMAGES_DIR,
    attribute_is_true,
    attribute_value,
    load_project,
    shape_should_include,
    track_attribute_value,
)
from src.util.get_version import get_pose_estimator_version_folder

PERSON_CATEGORY_ID = 1
PERSON_CATEGORY = {'id': PERSON_CATEGORY_ID, 'name': 'person', 'supercategory': 'person'}


def bbox_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def union_bbox(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return (min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2))


def task_start_frames(tasks: dict[str, ET.Element]) -> dict[str, int]:
    starts = {}
    global_offset = 0
    for task_id in sorted(tasks, key=lambda value: int(value)):
        starts[task_id] = global_offset
        global_offset += int(tasks[task_id].findtext('size') or 0)
    return starts


def jpeg_size(path: Path) -> tuple[int, int]:
    """Return (width, height) from a JPEG without decoding the full image."""
    with path.open('rb') as handle:
        if handle.read(2) != b'\xff\xd8':
            raise ValueError(f'Not a JPEG: {path}')
        while True:
            marker_prefix = handle.read(1)
            if not marker_prefix:
                break
            if marker_prefix != b'\xff':
                continue
            marker = handle.read(1)
            while marker == b'\xff':
                marker = handle.read(1)
            if marker in (
                b'\xd0', b'\xd1', b'\xd2', b'\xd3', b'\xd4', b'\xd5', b'\xd6', b'\xd7',
                b'\x01', b'\x00',
            ):
                continue
            if marker == b'\xd9':  # EOI
                break
            length_bytes = handle.read(2)
            if len(length_bytes) != 2:
                break
            length = struct.unpack('>H', length_bytes)[0]
            if marker in (b'\xc0', b'\xc1', b'\xc2', b'\xc3', b'\xc5', b'\xc6', b'\xc7',
                          b'\xc9', b'\xca', b'\xcb', b'\xcd', b'\xce', b'\xcf'):
                handle.read(1)  # precision
                height, width = struct.unpack('>HH', handle.read(4))
                return int(width), int(height)
            handle.seek(length - 2, 1)
    raise ValueError(f'Could not read JPEG size: {path}')


def parse_box_xyxy(box_el) -> tuple[float, float, float, float]:
    return (
        float(box_el.get('xtl')),
        float(box_el.get('ytl')),
        float(box_el.get('xbr')),
        float(box_el.get('ybr')),
    )


def collect_cleaned_person_boxes(
    annotations_file: Path,
    *,
    min_track_length: int,
    frame_stride: int | None,
):
    """Return frames dict: (task_id, relative_frame) -> list of person dicts."""
    root, _project, tasks = load_project(annotations_file)
    starts = task_start_frames(tasks)
    task_names = {tid: tasks[tid].findtext('name', '') for tid in tasks}

    # First pass: track lengths (non-outside boxes) for junk filtering.
    track_lengths: dict[str, int] = defaultdict(int)
    for track in root.findall('track'):
        if track.get('label') != 'bbox':
            continue
        track_id = track.get('id', '')
        for shape in track:
            if shape.tag != 'box' or shape.get('outside') == '1':
                continue
            track_lengths[track_id] += 1

    frames: dict[tuple[str, int], list[dict]] = defaultdict(list)
    skipped_short = 0
    skipped_missing_image_attrs = 0

    for track in root.findall('track'):
        if track.get('label') != 'bbox':
            continue
        task_id = track.get('task_id')
        if task_id is None or task_id not in tasks:
            continue
        track_id = track.get('id', '')
        if track_lengths[track_id] < min_track_length:
            skipped_short += 1
            continue

        start_frame = starts[task_id]
        person_id = track_attribute_value(track, 'person_id')
        role_raw = track_attribute_value(track, 'role')
        role = role_raw.lower() if role_raw else None
        couple_id = track_attribute_value(track, 'couple_id')

        for shape in track:
            if shape.tag != 'box' or shape.get('frame') is None:
                continue
            if shape.get('outside') == '1':
                continue
            if not shape_should_include(shape, start_frame, track_id, task_id=task_id):
                continue

            global_frame = int(shape.get('frame'))
            relative_frame = global_frame - start_frame
            if frame_stride is not None and frame_stride > 1:
                if relative_frame % frame_stride != 0:
                    continue

            # Prefer shape-level attrs when present.
            shape_person = attribute_value(shape, 'person_id') or person_id
            shape_role = attribute_value(shape, 'role')
            if shape_role:
                shape_role = shape_role.lower()
            else:
                shape_role = role
            shape_couple = attribute_value(shape, 'couple_id') or couple_id

            try:
                box = parse_box_xyxy(shape)
            except (TypeError, ValueError):
                skipped_missing_image_attrs += 1
                continue
            if box[2] <= box[0] or box[3] <= box[1]:
                continue

            frames[(task_id, relative_frame)].append(
                {
                    'track_id': track_id,
                    'person_id': shape_person,
                    'role': shape_role,
                    'couple_id': shape_couple,
                    'bbox': box,
                }
            )

    return frames, task_names, {
        'skipped_short_tracks': skipped_short,
        'skipped_bad_boxes': skipped_missing_image_attrs,
        'tasks': len(tasks),
    }


def greedy_lead_follow_pairs(people: list[dict]) -> list[tuple[dict, dict]]:
    leads = [p for p in people if p.get('role') == 'lead']
    follows = [p for p in people if p.get('role') == 'follow']
    candidates = []
    for lead in leads:
        for follow in follows:
            candidates.append((bbox_iou(lead['bbox'], follow['bbox']), lead, follow))
    candidates.sort(key=lambda item: item[0], reverse=True)
    used = set()
    pairs = []
    for _iou, lead, follow in candidates:
        lead_key = id(lead)
        follow_key = id(follow)
        if lead_key in used or follow_key in used:
            continue
        used.add(lead_key)
        used.add(follow_key)
        pairs.append((lead, follow))
    return pairs


def pair_people(people: list[dict]) -> list[tuple[dict, dict]]:
    """Pair partners: couple_id groups first, then lead/follow IoU greedy."""
    by_couple: dict[str, list[dict]] = defaultdict(list)
    unpaired = []
    for person in people:
        cid = person.get('couple_id')
        if cid is not None and cid != '':
            by_couple[cid].append(person)
        else:
            unpaired.append(person)

    pairs = []
    leftovers = list(unpaired)
    for _cid, group in by_couple.items():
        if len(group) == 2:
            pairs.append((group[0], group[1]))
        elif len(group) > 2:
            # Prefer lead/follow within the couple group; leftover stay separate.
            group_pairs = greedy_lead_follow_pairs(group)
            pairs.extend(group_pairs)
            paired_ids = {id(a) for a, b in group_pairs} | {id(b) for a, b in group_pairs}
            leftovers.extend(p for p in group if id(p) not in paired_ids)
        else:
            leftovers.extend(group)

    if leftovers:
        more = greedy_lead_follow_pairs(leftovers)
        pairs.extend(more)
    return pairs


def apply_label_policy(
    people: list[dict],
    *,
    union_iou_threshold: float | None,
    always_separate: bool,
    soft_band: float = 0.0,
    soft_mode: str = 'hard',
    decision_key: tuple | None = None,
) -> tuple[list[tuple[float, float, float, float]], dict]:
    """Return emitted xyxy boxes and stats for one frame.

    soft_band > 0 with soft_mode='random' (train): pairs with IoU in
    [τ - soft_band, τ + soft_band] randomly get union OR separate (seeded by
    decision_key) so either accurate merge or accurate split is acceptable near
    the boundary. Val should use soft_mode='hard'.
    """
    stats = {
        'input_people': len(people),
        'pairs': 0,
        'unions': 0,
        'separate_pairs': 0,
        'soft_pairs': 0,
        'unpaired': 0,
        'emitted': 0,
    }
    if not people:
        return [], stats

    if always_separate or union_iou_threshold is None:
        boxes = [p['bbox'] for p in people]
        stats['emitted'] = len(boxes)
        stats['unpaired'] = len(people)
        return boxes, stats

    pairs = pair_people(people)
    stats['pairs'] = len(pairs)
    paired_ids = {id(a) for a, b in pairs} | {id(b) for a, b in pairs}
    boxes = []
    tau = float(union_iou_threshold)
    soft_low = tau - soft_band
    soft_high = tau + soft_band
    use_soft = soft_band > 0 and soft_mode == 'random'

    for pair_index, (person_a, person_b) in enumerate(pairs):
        iou = bbox_iou(person_a['bbox'], person_b['bbox'])
        do_union: bool
        if use_soft and soft_low <= iou <= soft_high:
            stats['soft_pairs'] += 1
            # Deterministic per-frame/pair coin flip so re-exports are stable.
            rng = random.Random(
                (
                    decision_key if decision_key is not None else ('soft',),
                    pair_index,
                    round(iou, 4),
                )
            )
            do_union = rng.random() < 0.5
        else:
            do_union = iou >= tau

        if do_union:
            boxes.append(union_bbox(person_a['bbox'], person_b['bbox']))
            stats['unions'] += 1
        else:
            boxes.append(person_a['bbox'])
            boxes.append(person_b['bbox'])
            stats['separate_pairs'] += 1
    for person in people:
        if id(person) not in paired_ids:
            boxes.append(person['bbox'])
            stats['unpaired'] += 1
    stats['emitted'] = len(boxes)
    return boxes, stats


def split_tasks(task_ids: list[str], train_ratio: float, seed: int) -> tuple[set[str], set[str]]:
    ids = sorted(task_ids, key=lambda value: int(value))
    rng = random.Random(seed)
    rng.shuffle(ids)
    if len(ids) == 1:
        return set(ids), set()
    n_train = max(1, int(round(len(ids) * train_ratio)))
    n_train = min(n_train, len(ids) - 1)
    train = set(ids[:n_train])
    val = set(ids[n_train:])
    return train, val


def xyxy_to_coco_bbox(box):
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def xyxy_to_yolo(box, width: int, height: int):
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return cx, cy, bw, bh


def policy_dirname(
    always_separate: bool,
    union_iou_threshold: float | None,
    soft_band: float = 0.0,
) -> str:
    if always_separate or union_iou_threshold is None:
        return 'separate'
    name = f'union_iou{union_iou_threshold:g}'
    if soft_band > 0:
        name += f'_soft{soft_band:g}'
    return name


def write_coco(path: Path, images, annotations, categories):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'info': {'description': 'Dance bbox detection export'},
        'licenses': [],
        'images': images,
        'annotations': annotations,
        'categories': categories,
    }
    path.write_text(json.dumps(payload), encoding='utf-8')


def export_detection_dataset(
    *,
    annotation_version: str,
    images_dir: Path,
    output_root: Path | None,
    union_iou_threshold: float,
    always_separate: bool,
    train_ratio: float,
    seed: int,
    min_track_length: int,
    frame_stride: int | None,
    soft_band: float = 0.0,
) -> Path:
    dataset_dir = get_pose_estimator_version_folder(annotation_version)
    annotation_candidates = (
        dataset_dir / 'cleaned_annotations.xml',
        dataset_dir / 'annotations.xml',
    )
    annotations_file = next(
        (candidate for candidate in annotation_candidates if candidate.is_file()),
        None,
    )
    if annotations_file is None:
        expected = ', '.join(str(path) for path in annotation_candidates)
        raise FileNotFoundError(f'Missing annotations XML; checked: {expected}')

    frames, task_names, collect_stats = collect_cleaned_person_boxes(
        annotations_file,
        min_track_length=min_track_length,
        frame_stride=frame_stride,
    )
    if not frames:
        raise RuntimeError('No cleaned bbox frames found to export.')

    policy = policy_dirname(
        always_separate,
        None if always_separate else union_iou_threshold,
        soft_band=0.0 if always_separate else soft_band,
    )
    if output_root is None:
        output_root = Path('dataset') / 'detection' / annotation_version / policy
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Split tasks first so train can use soft band and val stays hard-threshold.
    all_task_ids = sorted({task_id for task_id, _ in frames}, key=lambda v: int(v))
    train_tasks, val_tasks = split_tasks(all_task_ids, train_ratio, seed)

    emitted: dict[tuple[str, int], list[tuple[float, float, float, float]]] = {}
    aggregate = defaultdict(int)
    aggregate_train = defaultdict(int)
    aggregate_val = defaultdict(int)
    for key, people in frames.items():
        task_id, relative_frame = key
        is_train = task_id in train_tasks
        # Train: soft random band. Val: hard τ only (clean metrics).
        soft_mode = 'random' if (is_train and soft_band > 0 and not always_separate) else 'hard'
        boxes, stats = apply_label_policy(
            people,
            union_iou_threshold=None if always_separate else union_iou_threshold,
            always_separate=always_separate,
            soft_band=soft_band if soft_mode == 'random' else 0.0,
            soft_mode=soft_mode,
            decision_key=(seed, task_id, relative_frame),
        )
        if boxes:
            emitted[key] = boxes
        bucket = aggregate_train if is_train else aggregate_val
        for name, value in stats.items():
            aggregate[name] += value
            bucket[name] += value

    tasks_with_frames = sorted({task_id for task_id, _ in emitted}, key=lambda v: int(v))

    splits = {
        'train': [k for k in emitted if k[0] in train_tasks],
        'val': [k for k in emitted if k[0] in val_tasks],
    }

    size_cache: dict[str, tuple[int, int]] = {}
    export_meta = {
        'annotation_version': annotation_version,
        'policy': policy,
        'union_iou_threshold': None if always_separate else union_iou_threshold,
        'soft_band': soft_band if not always_separate else 0.0,
        'soft_mode_train': 'random' if soft_band > 0 and not always_separate else 'hard',
        'soft_mode_val': 'hard',
        'always_separate': always_separate,
        'train_ratio': train_ratio,
        'seed': seed,
        'min_track_length': min_track_length,
        'frame_stride': frame_stride,
        'train_tasks': sorted(train_tasks, key=int),
        'val_tasks': sorted(val_tasks, key=int),
        'task_names': {tid: task_names.get(tid, '') for tid in tasks_with_frames},
        'collect_stats': collect_stats,
        'label_stats': dict(aggregate),
        'label_stats_train': dict(aggregate_train),
        'label_stats_val': dict(aggregate_val),
        'frames': {},
    }

    for split_name, keys in splits.items():
        images_out = output_root / 'images' / split_name
        labels_out = output_root / 'labels' / split_name
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        coco_images = []
        coco_annotations = []
        ann_id = 1
        image_id = 1
        keys_sorted = sorted(keys, key=lambda item: (int(item[0]), item[1]))

        for task_id, relative_frame in keys_sorted:
            image_name = f'task_{task_id}_frame_{relative_frame:06d}.jpg'
            src = images_dir / image_name
            if not src.is_file():
                print(f'Warning: missing image {src}; skipping frame.')
                continue
            if image_name not in size_cache:
                size_cache[image_name] = jpeg_size(src)
            width, height = size_cache[image_name]

            dest_image = images_out / image_name
            if dest_image.exists() or dest_image.is_symlink():
                dest_image.unlink()
            dest_image.symlink_to(src.resolve())

            boxes = emitted[(task_id, relative_frame)]
            label_path = labels_out / f'{image_name.rsplit(".", 1)[0]}.txt'
            yolo_lines = []
            for box in boxes:
                coco_annotations.append(
                    {
                        'id': ann_id,
                        'image_id': image_id,
                        'category_id': PERSON_CATEGORY_ID,
                        'bbox': xyxy_to_coco_bbox(box),
                        'area': max(0.0, (box[2] - box[0]) * (box[3] - box[1])),
                        'iscrowd': 0,
                        'segmentation': [],
                    }
                )
                ann_id += 1
                cx, cy, bw, bh = xyxy_to_yolo(box, width, height)
                yolo_lines.append(f'0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')
            label_path.write_text('\n'.join(yolo_lines) + ('\n' if yolo_lines else ''), encoding='utf-8')

            coco_images.append(
                {
                    'id': image_id,
                    'file_name': image_name,
                    'width': width,
                    'height': height,
                    'task_id': task_id,
                    'frame': relative_frame,
                }
            )
            image_id += 1

        write_coco(
            output_root / 'annotations' / f'instances_{split_name}.json',
            coco_images,
            coco_annotations,
            [PERSON_CATEGORY],
        )
        export_meta['frames'][split_name] = len(coco_images)
        export_meta[f'{split_name}_annotations'] = len(coco_annotations)

    data_yaml = {
        'path': str(output_root.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'person'},
    }
    (output_root / 'data.yaml').write_text(
        # Minimal YAML without requiring PyYAML dump ordering tricks
        'path: {path}\n'
        'train: {train}\n'
        'val: {val}\n'
        'names:\n'
        '  0: person\n'.format(**data_yaml),
        encoding='utf-8',
    )
    (output_root / 'export_meta.json').write_text(
        json.dumps(export_meta, indent=2),
        encoding='utf-8',
    )
    return output_root


def main():
    parser = argparse.ArgumentParser(
        description='Export cleaned bbox tracks to COCO + Ultralytics detection labels.'
    )
    parser.add_argument(
        '--annotation-version',
        default='1.7.0',
        help="Annotation version, e.g. '1.7.0' (default: 1.7.0)",
    )
    parser.add_argument(
        '--images',
        type=Path,
        default=IMAGES_DIR,
        help='Image store directory (default: dataset/image_store)',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output root (default: dataset/detection/<version>/<policy>)',
    )
    parser.add_argument(
        '--union-iou-threshold',
        type=float,
        default=0.3,
        help='IoU threshold for unioning paired partners (default: 0.3)',
    )
    parser.add_argument(
        '--soft-band',
        type=float,
        default=0.0,
        help=(
            'Train-only soft band around the threshold: pairs with IoU in '
            '[τ-band, τ+band] randomly union OR stay separate (default: 0 = hard)'
        ),
    )
    parser.add_argument(
        '--always-separate',
        action='store_true',
        help='Disable union-when-close; emit every cleaned person box',
    )
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=44)
    parser.add_argument(
        '--min-track-length',
        type=int,
        default=10,
        help='Skip bbox tracks with fewer than N non-outside boxes (default: 10)',
    )
    parser.add_argument(
        '--frame-stride',
        type=int,
        default=None,
        help='Optional extra subsample on relative frame after cleaned filter',
    )
    args = parser.parse_args()

    output_root = export_detection_dataset(
        annotation_version=args.annotation_version,
        images_dir=args.images,
        output_root=args.output,
        union_iou_threshold=args.union_iou_threshold,
        always_separate=args.always_separate,
        train_ratio=args.train_ratio,
        seed=args.seed,
        min_track_length=args.min_track_length,
        frame_stride=args.frame_stride,
        soft_band=args.soft_band,
    )
    print(f'Wrote detection dataset to {output_root}')
    meta = json.loads((output_root / 'export_meta.json').read_text(encoding='utf-8'))
    print(
        f"frames train={meta['frames'].get('train', 0)} "
        f"val={meta['frames'].get('val', 0)} "
        f"unions={meta['label_stats'].get('unions', 0)} "
        f"separate_pairs={meta['label_stats'].get('separate_pairs', 0)} "
        f"soft_pairs={meta['label_stats'].get('soft_pairs', 0)}"
    )


if __name__ == '__main__':
    main()
