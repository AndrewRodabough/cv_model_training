#!/usr/bin/env python3
"""Evaluate a fine-tuned detector on the held-out detection split.

Reports:
  - Ultralytics box metrics (AP / precision / recall) when available
  - Limb coverage on frames that also have skeletons in cleaned_annotations.xml
  - Close-pair behavior vs GT unions (one covering box vs two vs under-coverage)
  - Worst-frame overlay dumps
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import yaml

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from src.dataset_review.export_bbox_detection import bbox_iou, union_bbox


def load_coco(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def coco_xywh_to_xyxy(bbox):
    x, y, w, h = bbox
    return (x, y, x + w, y + h)


def load_gt_by_image(coco: dict) -> dict[int, list[tuple[float, float, float, float]]]:
    by_image: dict[int, list] = defaultdict(list)
    for ann in coco['annotations']:
        by_image[ann['image_id']].append(coco_xywh_to_xyxy(ann['bbox']))
    return by_image


def load_images(coco: dict) -> dict[int, dict]:
    return {img['id']: img for img in coco['images']}


def box_contains_point(box, x: float, y: float, margin: float = 0.0) -> bool:
    x1, y1, x2, y2 = box
    return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)


def best_iou_match(pred_boxes, gt_box) -> float:
    if not pred_boxes:
        return 0.0
    return max(bbox_iou(pred, gt_box) for pred in pred_boxes)


def coverage_of_union(pred_boxes, gt_union, iou_thresh: float = 0.5) -> str:
    """Classify close-pair prediction behavior relative to GT union."""
    if not pred_boxes:
        return 'miss'
    ious = [bbox_iou(pred, gt_union) for pred in pred_boxes]
    best = max(ious)
    covering = [pred for pred, iou in zip(pred_boxes, ious) if iou >= iou_thresh]
    if covering:
        return 'one_cover'
    # Two boxes that together cover most of the union?
    if len(pred_boxes) >= 2:
        merged = pred_boxes[0]
        for pred in pred_boxes[1:]:
            merged = union_bbox(merged, pred)
        if bbox_iou(merged, gt_union) >= iou_thresh:
            return 'two_cover'
    if best > 0:
        return 'under_cover'
    return 'miss'


def load_skeleton_keypoints(cleaned_xml: Path) -> dict[tuple[str, int], list[tuple[float, float, int]]]:
    """Map (task_id, relative_frame) -> list of (x, y, visibility)."""
    if not cleaned_xml.is_file():
        return {}
    root = ET.parse(cleaned_xml).getroot()
    out: dict[tuple[str, int], list] = defaultdict(list)
    for video in root.findall('.//video'):
        task_id = video.get('id')
        for frame in video.findall('frame'):
            frame_number = int(frame.get('number'))
            for person in frame.findall('person'):
                for kp in person.findall('keypoint'):
                    vis = int(kp.get('visibility', '0'))
                    if vis <= 0:
                        continue
                    out[(task_id, frame_number)].append(
                        (float(kp.get('x')), float(kp.get('y')), vis)
                    )
    return out


def run_predictions(weights: Path, image_paths: list[Path], conf: float, iou: float, device: str | None):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            'ultralytics is required. Install with: pip install ultralytics'
        ) from exc

    model = YOLO(str(weights))
    kwargs = {'conf': conf, 'iou': iou, 'verbose': False}
    if device is not None:
        kwargs['device'] = device
    preds_by_name: dict[str, list[tuple[float, float, float, float]]] = {}
    for path in image_paths:
        result = model.predict(source=str(path), **kwargs)[0]
        boxes = []
        if result.boxes is not None and len(result.boxes):
            xyxy = result.boxes.xyxy.cpu().tolist()
            boxes = [tuple(map(float, row)) for row in xyxy]
        preds_by_name[path.name] = boxes
    return preds_by_name


def draw_overlay(image_path: Path, gt_boxes, pred_boxes, out_path: Path, title: str):
    if cv2 is None:
        return
    image = cv2.imread(str(image_path))
    if image is None:
        return
    for box in gt_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for box in pred_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 128, 255), 2)
    cv2.putText(image, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)


def evaluate(
    *,
    data_yaml: Path,
    weights: Path,
    cleaned_xml: Path | None,
    conf: float,
    iou: float,
    close_pair_iou: float,
    match_iou: float,
    overlays: int,
    out_dir: Path,
    device: str | None,
) -> dict:
    with data_yaml.open(encoding='utf-8') as handle:
        data_cfg = yaml.safe_load(handle)
    root = Path(data_cfg['path'])
    coco_path = root / 'annotations' / 'instances_val.json'
    coco = load_coco(coco_path)
    images = load_images(coco)
    gt_by_image = load_gt_by_image(coco)

    image_dir = root / 'images' / 'val'
    image_paths = [image_dir / img['file_name'] for img in coco['images']]
    image_paths = [p for p in image_paths if p.is_file()]
    preds_by_name = run_predictions(weights, image_paths, conf=conf, iou=iou, device=device)

    keypoints = {}
    if cleaned_xml is not None:
        keypoints = load_skeleton_keypoints(cleaned_xml)

    limb_total = 0
    limb_covered = 0
    close_pair_counts = defaultdict(int)
    per_image_scores = []

    for image_id, meta in images.items():
        name = meta['file_name']
        gt_boxes = gt_by_image.get(image_id, [])
        pred_boxes = preds_by_name.get(name, [])
        task_id = str(meta.get('task_id', ''))
        frame = int(meta.get('frame', -1))

        # Mean best-IoU recall proxy per GT box.
        if gt_boxes:
            recalls = [best_iou_match(pred_boxes, gt) >= match_iou for gt in gt_boxes]
            recall = sum(recalls) / len(recalls)
        else:
            recall = 1.0 if not pred_boxes else 0.0

        # Limb coverage when skeletons exist for this frame.
        kps = keypoints.get((task_id, frame), [])
        visible = [(x, y) for x, y, vis in kps if vis >= 1]
        frame_limb_total = len(visible)
        frame_limb_hit = 0
        for x, y in visible:
            limb_total += 1
            if any(box_contains_point(box, x, y) for box in pred_boxes):
                limb_covered += 1
                frame_limb_hit += 1

        # Close-pair behavior relative to GT (union exports often have one box).
        if len(gt_boxes) == 1:
            # Likely a union target: check coverage quality.
            behavior = coverage_of_union(pred_boxes, gt_boxes[0], iou_thresh=match_iou)
            if behavior == 'one_cover':
                close_pair_counts['union_gt_one_cover'] += 1
            elif behavior == 'two_cover':
                close_pair_counts['union_gt_two_cover'] += 1
            elif behavior == 'under_cover':
                close_pair_counts['union_gt_under_cover'] += 1
            else:
                close_pair_counts['union_gt_miss'] += 1
        elif len(gt_boxes) >= 2:
            # Separate GT: check any high-IoU GT pair for under-merge desire.
            for i, box_a in enumerate(gt_boxes):
                for box_b in gt_boxes[i + 1 :]:
                    pair_iou = bbox_iou(box_a, box_b)
                    if pair_iou < close_pair_iou:
                        continue
                    gt_union = union_bbox(box_a, box_b)
                    behavior = coverage_of_union(pred_boxes, gt_union, iou_thresh=match_iou)
                    close_pair_counts[f'separate_gt_{behavior}'] += 1

        limb_rate = (frame_limb_hit / frame_limb_total) if frame_limb_total else None
        score = recall
        if limb_rate is not None:
            score = 0.5 * recall + 0.5 * limb_rate
        per_image_scores.append(
            {
                'file_name': name,
                'task_id': task_id,
                'frame': frame,
                'gt_count': len(gt_boxes),
                'pred_count': len(pred_boxes),
                'recall_at_iou': recall,
                'limb_coverage': limb_rate,
                'score': score,
                'gt_boxes': gt_boxes,
                'pred_boxes': pred_boxes,
            }
        )

    # Ultralytics val metrics (optional, uses data.yaml)
    ultra_metrics = {}
    try:
        from ultralytics import YOLO

        model = YOLO(str(weights))
        val_kwargs = {'data': str(data_yaml.resolve()), 'split': 'val', 'verbose': False}
        if device is not None:
            val_kwargs['device'] = device
        metrics = model.val(**val_kwargs)
        box = getattr(metrics, 'box', None)
        if box is not None:
            ultra_metrics = {
                'precision': float(getattr(box, 'mp', float('nan'))),
                'recall': float(getattr(box, 'mr', float('nan'))),
                'mAP50': float(getattr(box, 'map50', float('nan'))),
                'mAP50_95': float(getattr(box, 'map', float('nan'))),
            }
    except Exception as exc:  # noqa: BLE001 - keep custom metrics even if val fails
        ultra_metrics = {'error': str(exc)}

    limb_coverage = (limb_covered / limb_total) if limb_total else None
    summary = {
        'weights': str(weights),
        'data_yaml': str(data_yaml),
        'num_val_images': len(images),
        'ultralytics': ultra_metrics,
        'mean_gt_recall_at_iou': (
            sum(row['recall_at_iou'] for row in per_image_scores) / max(len(per_image_scores), 1)
        ),
        'limb_coverage': limb_coverage,
        'limb_keypoints_evaluated': limb_total,
        'close_pair_counts': dict(close_pair_counts),
        'match_iou': match_iou,
        'close_pair_iou': close_pair_iou,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    worst = sorted(per_image_scores, key=lambda row: row['score'])[: max(0, overlays)]
    overlay_dir = out_dir / 'overlays'
    for rank, row in enumerate(worst, start=1):
        image_path = image_dir / row['file_name']
        title = (
            f"#{rank} recall={row['recall_at_iou']:.2f} "
            f"limb={row['limb_coverage'] if row['limb_coverage'] is not None else 'n/a'}"
        )
        draw_overlay(
            image_path,
            row['gt_boxes'],
            row['pred_boxes'],
            overlay_dir / f'{rank:02d}_{row["file_name"]}',
            title,
        )

    return summary


def main():
    parser = argparse.ArgumentParser(description='Evaluate dance bbox detector.')
    parser.add_argument('--weights', type=Path, required=True, help='Path to best.pt')
    parser.add_argument(
        '--data',
        type=Path,
        default=Path('dataset/detection/1.7.0/union_iou0.3/data.yaml'),
        help='Ultralytics data.yaml from export_bbox_detection.py',
    )
    parser.add_argument(
        '--cleaned-xml',
        type=Path,
        default=None,
        help='Optional cleaned_annotations.xml for limb-coverage (default: version from data path)',
    )
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.7, help='NMS IoU for predictions')
    parser.add_argument(
        '--match-iou',
        type=float,
        default=0.5,
        help='IoU threshold for GT match / covering classification',
    )
    parser.add_argument(
        '--close-pair-iou',
        type=float,
        default=0.3,
        help='GT pair IoU treated as close-hold for behavior stats',
    )
    parser.add_argument('--overlays', type=int, default=15)
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=None,
        help='Output directory (default: <weights_parent>/eval)',
    )
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    cleaned_xml = args.cleaned_xml
    if cleaned_xml is None:
        # Detector evaluation uses the corresponding pose version's cleaned annotations.
        parts = Path(args.data).resolve().parts
        version = None
        if 'detection' in parts:
            idx = parts.index('detection')
            if idx + 1 < len(parts):
                version = parts[idx + 1]
        if version:
            from src.util.get_version import get_pose_estimator_version_folder

            candidate = get_pose_estimator_version_folder(version) / 'cleaned_annotations.xml'
            if candidate.is_file():
                cleaned_xml = candidate

    out_dir = args.out_dir or (args.weights.parent.parent / 'eval')
    summary = evaluate(
        data_yaml=args.data,
        weights=args.weights,
        cleaned_xml=cleaned_xml,
        conf=args.conf,
        iou=args.iou,
        close_pair_iou=args.close_pair_iou,
        match_iou=args.match_iou,
        overlays=args.overlays,
        out_dir=out_dir,
        device=args.device,
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir / 'metrics.json'}")


if __name__ == '__main__':
    main()
