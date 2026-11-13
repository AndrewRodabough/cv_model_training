#!/usr/bin/env python3
"""Fine-tune an Ultralytics YOLO detector on the exported dance bbox dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.dataset_review.export_bbox_detection import export_detection_dataset
from src.training.yolo_partial_label import (
    enable_partial_label_loss,
    load_full_cls_task_ids,
)


def load_config(path: Path) -> dict:
    with path.open(encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def maybe_export(cfg: dict, images: Path | None) -> Path:
    dataset_cfg = cfg['dataset']
    data_yaml = Path(dataset_cfg['data_yaml'])
    if data_yaml.is_file():
        return data_yaml

    print(f'{data_yaml} missing; exporting detection dataset...')
    always_separate = bool(dataset_cfg.get('always_separate', False))
    output_root = export_detection_dataset(
        annotation_version=str(dataset_cfg['annotation_version']),
        images_dir=images or Path('dataset/image_store'),
        output_root=None,
        union_iou_threshold=float(dataset_cfg.get('union_iou_threshold', 0.3)),
        always_separate=always_separate,
        train_ratio=float(dataset_cfg.get('train_ratio', 0.8)),
        seed=int(cfg.get('training', {}).get('seed', 44)),
        min_track_length=int(dataset_cfg.get('min_track_length', 10)),
        frame_stride=dataset_cfg.get('frame_stride'),
        soft_band=float(dataset_cfg.get('soft_band', 0.0)),
        val_tasks=dataset_cfg.get('val_tasks') or dataset_cfg.get('test_clips'),
        pseudo_fill_unlabeled=bool(dataset_cfg.get('pseudo_fill_unlabeled', False)),
        pseudo_teacher=str(dataset_cfg.get('pseudo_teacher', 'weights/detector/yolo11m.pt')),
        pseudo_conf=float(dataset_cfg.get('pseudo_conf', 0.25)),
        pseudo_device=str(dataset_cfg.get('pseudo_device', 'cpu')),
    )
    data_yaml = output_root / 'data.yaml'
    if not data_yaml.is_file():
        raise FileNotFoundError(f'Export did not create {data_yaml}')
    return data_yaml


def train(cfg: dict, *, data_yaml: Path, device: str | None, resume: bool) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            'ultralytics is required. Install with: pip install ultralytics'
        ) from exc

    model_cfg = cfg['model']
    train_cfg = cfg['training']
    model = YOLO(model_cfg['name'])

    # Sparse keypoint-driven boxes: mask cls outside GT (crowd not = background).
    # Optional allowlist restores full cls on tasks tagged label-complete / safe-bg.
    mask_outside = bool(train_cfg.get('mask_outside_gt', False))
    pos_only = bool(train_cfg.get('positive_only_cls', False))
    full_cls_tasks: set[str] = set()
    completeness = train_cfg.get('label_completeness_yaml')
    if completeness:
        full_cls_tasks = load_full_cls_task_ids(completeness)
    for raw in train_cfg.get('full_cls_tasks') or []:
        full_cls_tasks.add(str(raw).removeprefix('task_'))
    # Always enable when side-tier weights or masking are requested.
    side_cls = float(train_cfg.get('side_cls_weight', 0.5))
    side_box = float(train_cfg.get('side_box_weight', 0.2))
    use_side_weights = bool(train_cfg.get('side_tier_weights', True))
    if mask_outside or pos_only or use_side_weights:
        enable_partial_label_loss(
            model,
            mask_outside_gt=mask_outside,
            gt_mask_dilate=float(train_cfg.get('gt_mask_dilate', 1.15)),
            positive_only_cls=pos_only,
            neg_weight=float(train_cfg.get('background_cls_weight', 0.0)),
            full_cls_task_ids=full_cls_tasks,
            side_cls_weight=side_cls if use_side_weights else 1.0,
            side_box_weight=side_box if use_side_weights else 1.0,
            side_class_id=int(train_cfg.get('side_class_id', 1)),
        )

    kwargs = {
        'data': str(data_yaml.resolve()),
        'epochs': int(train_cfg['epochs']),
        'imgsz': int(model_cfg.get('imgsz', 640)),
        'batch': int(train_cfg.get('batch_size', 16)),
        'workers': int(train_cfg.get('workers', 8)),
        'patience': int(train_cfg.get('patience', 20)),
        'seed': int(train_cfg.get('seed', 44)),
        'project': str(Path(train_cfg.get('project', 'training/detector_runs')).resolve()),
        'name': str(train_cfg.get('run_name', 'bbox')),
        'exist_ok': True,
        'iou': float(train_cfg.get('iou', 0.7)),
        'conf': float(train_cfg.get('conf', 0.25)),
        'resume': resume,
    }
    if device is not None:
        kwargs['device'] = device
    if 'lr0' in train_cfg:
        kwargs['lr0'] = float(train_cfg['lr0'])
    if 'optimizer' in train_cfg:
        kwargs['optimizer'] = str(train_cfg['optimizer'])
    if 'save_period' in train_cfg:
        kwargs['save_period'] = int(train_cfg['save_period'])

    results = model.train(**kwargs)
    save_dir = Path(results.save_dir)
    print(f'Training complete. Artifacts: {save_dir}')
    print(f'Best weights: {save_dir / "weights" / "best.pt"}')
    return save_dir


def main():
    parser = argparse.ArgumentParser(description='Train YOLO on exported dance bboxes.')
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('config_bbox_union.yaml'),
        help='Detector training config (default: config_bbox_union.yaml)',
    )
    parser.add_argument(
        '--export',
        action='store_true',
        help='Force re-export even if data.yaml already exists',
    )
    parser.add_argument(
        '--images',
        type=Path,
        default=None,
        help='Image store for export (default: dataset/image_store)',
    )
    parser.add_argument('--device', default=None, help='Ultralytics device, e.g. 0 or cpu')
    parser.add_argument('--resume', action='store_true', help='Resume last run')
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_yaml = Path(cfg['dataset']['data_yaml'])
    if args.export or not data_yaml.is_file():
        if args.export and data_yaml.is_file():
            # Re-export into the expected policy directory.
            always_separate = bool(cfg['dataset'].get('always_separate', False))
            export_detection_dataset(
                annotation_version=str(cfg['dataset']['annotation_version']),
                images_dir=args.images or Path('dataset/image_store'),
                output_root=data_yaml.parent,
                union_iou_threshold=float(cfg['dataset'].get('union_iou_threshold', 0.3)),
                always_separate=always_separate,
                train_ratio=float(cfg['dataset'].get('train_ratio', 0.8)),
                seed=int(cfg.get('training', {}).get('seed', 44)),
                min_track_length=int(cfg['dataset'].get('min_track_length', 10)),
                frame_stride=cfg['dataset'].get('frame_stride'),
                soft_band=float(cfg['dataset'].get('soft_band', 0.0)),
                val_tasks=cfg['dataset'].get('val_tasks')
                or cfg['dataset'].get('test_clips'),
                pseudo_fill_unlabeled=bool(
                    cfg['dataset'].get('pseudo_fill_unlabeled', False)
                ),
                pseudo_teacher=str(cfg['dataset'].get('pseudo_teacher', 'weights/detector/yolo11m.pt')),
                pseudo_conf=float(cfg['dataset'].get('pseudo_conf', 0.25)),
                pseudo_device=str(cfg['dataset'].get('pseudo_device', 'cpu')),
            )
        else:
            data_yaml = maybe_export(cfg, args.images)

    train(cfg, data_yaml=data_yaml, device=args.device, resume=args.resume)


if __name__ == '__main__':
    main()
