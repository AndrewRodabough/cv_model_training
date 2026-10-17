#!/usr/bin/env python3
"""Fine-tune an Ultralytics YOLO detector on the exported dance bbox dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from export_bbox_detection import export_detection_dataset


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
            )
        else:
            data_yaml = maybe_export(cfg, args.images)

    train(cfg, data_yaml=data_yaml, device=args.device, resume=args.resume)


if __name__ == '__main__':
    main()
