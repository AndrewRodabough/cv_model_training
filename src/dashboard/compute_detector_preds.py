"""Cache YOLO val predictions for the detector browser.

Run under the training env (has ultralytics)::

    PYTHONPATH=. python -m src.dashboard.compute_detector_preds \\
      --run-dir training/detector_runs/bbox_union_iou0.5_102 \\
      --checkpoint best \\
      --device cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.dashboard.detector_overlay import (
    cache_dir,
    checkpoint_weight_path,
    list_val_stems,
    match_boxes,
    read_yolo_labels,
    resolve_data_paths,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_run_dir(run_dir: Path) -> Path:
    path = Path(run_dir)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f'Missing run dir: {path}')
    return path


def cache_checkpoint_preds(
    run_dir: Path,
    checkpoint: str,
    *,
    device: str = 'cpu',
    conf: float | None = None,
    iou: float | None = None,
    imgsz: int | None = None,
    max_frames: int | None = None,
) -> Path:
    from ultralytics import YOLO
    import yaml

    run_dir = _resolve_run_dir(run_dir)
    weights = checkpoint_weight_path(run_dir, checkpoint)
    _, images_val, labels_val = resolve_data_paths(run_dir)
    args = yaml.safe_load((run_dir / 'args.yaml').read_text(encoding='utf-8')) or {}
    conf = float(args.get('conf', 0.25) if conf is None else conf)
    iou = float(args.get('iou', 0.7) if iou is None else iou)
    imgsz = int(args.get('imgsz', 640) if imgsz is None else imgsz)

    stems = list_val_stems(run_dir)
    if max_frames is not None:
        stems = stems[: max(0, int(max_frames))]

    model = YOLO(str(weights))
    frames: dict[str, dict] = {}
    for i, stem in enumerate(stems):
        image_path = images_val / f'{stem}.jpg'
        if not image_path.is_file():
            image_path = images_val / f'{stem}.png'
        if not image_path.is_file():
            continue
        from PIL import Image

        with Image.open(image_path) as im:
            width, height = im.size
        gt_boxes = read_yolo_labels(labels_val / f'{stem}.txt', width, height)
        results = model.predict(
            source=str(image_path),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        pred_boxes = []
        if results:
            boxes = results[0].boxes
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.cpu().tolist()
                confs = boxes.conf.cpu().tolist()
                clss = boxes.cls.cpu().tolist()
                for box, score, cls_id in zip(xyxy, confs, clss):
                    pred_boxes.append(
                        {
                            'cls': int(cls_id),
                            'xyxy': [float(v) for v in box],
                            'conf': float(score),
                        }
                    )
        stats = match_boxes(gt_boxes, pred_boxes, iou_thresh=0.5)
        frames[stem] = {
            'preds': pred_boxes,
            'mean_iou': stats['mean_iou'],
            'tp': stats['tp'],
            'fp': stats['fp'],
            'fn': stats['fn'],
            'n_gt': stats['n_gt'],
            'n_pred': stats['n_pred'],
            'gt_ious': stats['gt_ious'],
        }
        if (i + 1) % 50 == 0 or i + 1 == len(stems):
            print(f'cached {i + 1}/{len(stems)}', flush=True)

    out_dir = cache_dir(run_dir, checkpoint)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'run_dir': str(run_dir),
        'checkpoint': checkpoint.replace('.pt', ''),
        'weights': str(weights),
        'conf': conf,
        'iou': iou,
        'imgsz': imgsz,
        'device': device,
        'n_frames': len(frames),
        'frames': frames,
    }
    out_path = out_dir / 'preds.json'
    out_path.write_text(json.dumps(payload), encoding='utf-8')
    print(f'Wrote {out_path} ({len(frames)} frames)', flush=True)
    return out_path


def main():
    parser = argparse.ArgumentParser(description='Cache detector val preds for browser')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', default='best', help='best | last | epoch12')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--conf', type=float, default=None)
    parser.add_argument('--iou', type=float, default=None)
    parser.add_argument('--imgsz', type=int, default=None)
    parser.add_argument('--max-frames', type=int, default=None)
    args = parser.parse_args()
    cache_checkpoint_preds(
        args.run_dir,
        args.checkpoint,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        max_frames=args.max_frames,
    )


if __name__ == '__main__':
    main()
