"""Detector val-frame overlays for the dashboard (no torch required)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

DEFAULT_DISPLAY_MAX = 960


def yolo_label_to_xyxy(
    cls_id: int,
    cx: float,
    cy: float,
    bw: float,
    bh: float,
    width: int,
    height: int,
) -> tuple[int, tuple[float, float, float, float]]:
    x1 = (cx - bw / 2.0) * width
    y1 = (cy - bh / 2.0) * height
    x2 = (cx + bw / 2.0) * width
    y2 = (cy + bh / 2.0) * height
    return cls_id, (x1, y1, x2, y2)


def read_yolo_labels(path: Path, width: int, height: int) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    boxes: list[dict[str, Any]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id, cx, cy, bw, bh = (
            int(float(parts[0])),
            float(parts[1]),
            float(parts[2]),
            float(parts[3]),
            float(parts[4]),
        )
        _, xyxy = yolo_label_to_xyxy(cls_id, cx, cy, bw, bh, width, height)
        boxes.append({'cls': cls_id, 'xyxy': xyxy, 'conf': 1.0})
    return boxes


def box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def match_boxes(
    gt_boxes: list[dict[str, Any]],
    pred_boxes: list[dict[str, Any]],
    iou_thresh: float = 0.5,
) -> dict[str, Any]:
    """Greedy one-to-one matching by IoU. Returns summary + per-gt best IoU."""
    preds = sorted(pred_boxes, key=lambda b: float(b.get('conf', 0.0)), reverse=True)
    used_pred: set[int] = set()
    gt_ious: list[float] = []
    tp = 0
    for gt in gt_boxes:
        best_i = -1
        best_iou = 0.0
        for i, pred in enumerate(preds):
            if i in used_pred:
                continue
            iou = box_iou(gt['xyxy'], pred['xyxy'])
            if iou > best_iou:
                best_iou = iou
                best_i = i
        gt_ious.append(best_iou)
        if best_i >= 0 and best_iou >= iou_thresh:
            used_pred.add(best_i)
            tp += 1
    fp = max(0, len(preds) - tp)
    fn = max(0, len(gt_boxes) - tp)
    mean_iou = sum(gt_ious) / len(gt_ious) if gt_ious else (1.0 if not preds else 0.0)
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'mean_iou': mean_iou,
        'gt_ious': gt_ious,
        'n_gt': len(gt_boxes),
        'n_pred': len(preds),
    }


def resolve_data_paths(run_dir: Path) -> tuple[Path, Path, Path]:
    """Return (data_yaml, images_val, labels_val) from args.yaml."""
    import yaml

    args_path = run_dir / 'args.yaml'
    if not args_path.is_file():
        raise FileNotFoundError(f'Missing {args_path}')
    args = yaml.safe_load(args_path.read_text(encoding='utf-8')) or {}
    data_yaml = Path(str(args.get('data') or ''))
    if not data_yaml.is_file():
        raise FileNotFoundError(f'Missing data yaml: {data_yaml}')
    data = yaml.safe_load(data_yaml.read_text(encoding='utf-8')) or {}
    root = Path(str(data.get('path') or data_yaml.parent))
    val_rel = str(data.get('val') or 'images/val')
    images_val = root / val_rel
    # Ultralytics layout: labels/val mirrors images/val
    if val_rel.startswith('images/'):
        labels_val = root / val_rel.replace('images/', 'labels/', 1)
    else:
        labels_val = root / 'labels' / 'val'
    return data_yaml, images_val, labels_val


def list_val_stems(run_dir: Path) -> list[str]:
    _, images_val, _ = resolve_data_paths(run_dir)
    stems = sorted(p.stem for p in images_val.glob('*.jpg'))
    stems += sorted(p.stem for p in images_val.glob('*.png') if p.stem not in stems)
    return stems


def list_detector_checkpoints(run_dir: Path) -> list[str]:
    weights = run_dir / 'weights'
    if not weights.is_dir():
        return []
    names: list[str] = []
    for preferred in ('best.pt', 'last.pt'):
        if (weights / preferred).is_file():
            names.append(preferred.replace('.pt', ''))
    # Ultralytics save_period files: epoch0.pt, epoch5.pt, ...
    epoch_files = sorted(
        weights.glob('epoch*.pt'),
        key=lambda p: int(''.join(ch for ch in p.stem if ch.isdigit()) or '0'),
    )
    for path in epoch_files:
        stem = path.stem  # epoch12
        if stem not in names:
            names.append(stem)
    return names


def checkpoint_weight_path(run_dir: Path, checkpoint: str) -> Path:
    name = checkpoint.strip()
    if not name.endswith('.pt'):
        name = f'{name}.pt'
    path = run_dir / 'weights' / name
    if not path.is_file():
        raise FileNotFoundError(f'Missing weights: {path}')
    return path


def normalize_checkpoint_name(checkpoint: str | None) -> str:
    name = (checkpoint or 'best').strip()
    if name.endswith('.pt'):
        name = name[:-3]
    return name or 'best'


def cache_dir(run_dir: Path, checkpoint: str) -> Path:
    safe = normalize_checkpoint_name(checkpoint).replace('/', '_')
    return run_dir / 'val_browser' / safe


def load_pred_cache(run_dir: Path, checkpoint: str) -> dict[str, Any] | None:
    path = cache_dir(run_dir, checkpoint) / 'preds.json'
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def list_cached_checkpoints(run_dir: Path) -> list[str]:
    root = run_dir / 'val_browser'
    if not root.is_dir():
        return []
    found = []
    for path in sorted(root.iterdir()):
        if path.is_dir() and (path / 'preds.json').is_file():
            found.append(path.name)
    return found


def frame_sort_key(stem: str) -> tuple[int, int, str]:
    task_id, frame = 10**9, 10**9
    if '_frame_' in stem:
        head, _, frame_s = stem.partition('_frame_')
        try:
            frame = int(frame_s)
        except ValueError:
            frame = 10**9
        if head.startswith('task_'):
            try:
                task_id = int(head.split('_', 1)[1])
            except ValueError:
                task_id = 10**9
    return task_id, frame, stem


def sorted_stems(
    stems: list[str],
    cache: dict[str, Any] | None,
    sort_by: str,
) -> list[str]:
    key = (sort_by or 'task').strip().lower()
    frames = (cache or {}).get('frames') or {}
    if key in ('1-iou', 'worst', 'mean_iou') and frames:
        return sorted(
            stems,
            key=lambda s: (
                float((frames.get(s) or {}).get('mean_iou', 1.0)),
                *frame_sort_key(s),
            ),
        )
    if key in ('fn', 'misses') and frames:
        return sorted(
            stems,
            key=lambda s: (
                -int((frames.get(s) or {}).get('fn', 0)),
                float((frames.get(s) or {}).get('mean_iou', 1.0)),
                *frame_sort_key(s),
            ),
        )
    return sorted(stems, key=frame_sort_key)


def _font(size: int = 18):
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size=size)
    except Exception:
        return ImageFont.load_default()


def draw_detector_overlay(
    image: Image.Image,
    gt_boxes: list[dict[str, Any]],
    pred_boxes: list[dict[str, Any]],
    *,
    title: str = '',
    max_side: int = DEFAULT_DISPLAY_MAX,
) -> Image.Image:
    """Draw GT (green) and preds (orange) on a display-sized copy."""
    rgb = image.convert('RGB')
    w, h = rgb.size
    scale = 1.0
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        rgb = rgb.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
        w, h = rgb.size
    draw = ImageDraw.Draw(rgb)
    font = _font(18)
    small = _font(14)

    def _scaled(box):
        x1, y1, x2, y2 = box
        return [x1 * scale, y1 * scale, x2 * scale, y2 * scale]

    # GT first (thinner green). Preds on top (thicker orange) so overlaps stay visible.
    for box in gt_boxes:
        xy = _scaled(box['xyxy'])
        draw.rectangle(xy, outline=(40, 200, 90), width=2)
        label = 'GT'
        tw = draw.textlength(label, font=small) if hasattr(draw, 'textlength') else 16
        tx = xy[0] + 2
        ty = max(0, xy[1] - 16)
        draw.rectangle([tx - 1, ty - 1, tx + tw + 2, ty + 14], fill=(0, 0, 0))
        draw.text((tx, ty), label, fill=(40, 220, 100), font=small)
    for box in pred_boxes:
        xy = _scaled(box['xyxy'])
        conf = float(box.get('conf', 0.0))
        draw.rectangle(xy, outline=(255, 140, 40), width=4)
        label = f'P {conf:.2f}'
        tw = draw.textlength(label, font=small) if hasattr(draw, 'textlength') else 40
        # Park pred label on the opposite corner from GT when boxes overlap.
        tx = max(0, xy[2] - tw - 4)
        ty = max(0, xy[1] - 16)
        draw.rectangle([tx - 1, ty - 1, tx + tw + 2, ty + 14], fill=(0, 0, 0))
        draw.text((tx, ty), label, fill=(255, 170, 60), font=small)
    if title:
        draw.rectangle([0, 0, w, 28], fill=(0, 0, 0))
        draw.text((6, 5), title, fill=(240, 240, 240), font=font)
    return rgb


def render_frame(
    run_dir: Path,
    stem: str,
    checkpoint: str | None = None,
) -> tuple[Image.Image | None, str]:
    """Render one val frame with GT (+ preds if cache exists)."""
    try:
        _, images_val, labels_val = resolve_data_paths(run_dir)
    except Exception as exc:
        return None, str(exc)

    image_path = images_val / f'{stem}.jpg'
    if not image_path.is_file():
        image_path = images_val / f'{stem}.png'
    if not image_path.is_file():
        return None, f'Missing image for {stem}'

    image = Image.open(image_path)
    width, height = image.size
    gt_boxes = read_yolo_labels(labels_val / f'{stem}.txt', width, height)
    pred_boxes: list[dict[str, Any]] = []
    stats = None
    ckpt = normalize_checkpoint_name(checkpoint)
    cache = load_pred_cache(run_dir, ckpt)
    if cache and stem in (cache.get('frames') or {}):
        entry = cache['frames'][stem]
        pred_boxes = [
            {'cls': int(b.get('cls', 0)), 'xyxy': tuple(b['xyxy']), 'conf': float(b.get('conf', 0))}
            for b in entry.get('preds') or []
        ]
        stats = {
            'mean_iou': entry.get('mean_iou'),
            'tp': entry.get('tp'),
            'fp': entry.get('fp'),
            'fn': entry.get('fn'),
            'n_gt': entry.get('n_gt'),
            'n_pred': entry.get('n_pred'),
        }
    elif gt_boxes:
        stats = match_boxes(gt_boxes, [])

    title_bits = [stem, f'ckpt={ckpt}']
    if stats:
        title_bits.append(
            f"IoU={float(stats.get('mean_iou') or 0):.2f} "
            f"tp={stats.get('tp')} fp={stats.get('fp')} fn={stats.get('fn')}"
        )
    if not pred_boxes:
        title_bits.append('GT only — cache preds to see model boxes')
    overlay = draw_detector_overlay(
        image, gt_boxes, pred_boxes, title=' | '.join(title_bits)
    )
    info = (
        f'{stem}  ckpt={ckpt}  gt={len(gt_boxes)}  pred={len(pred_boxes)}\n'
        f'cache={"yes" if cache and stem in (cache.get("frames") or {}) else "no"}'
    )
    if stats:
        info += (
            f"  mean_iou={stats.get('mean_iou')}  "
            f"tp/fp/fn={stats.get('tp')}/{stats.get('fp')}/{stats.get('fn')}"
        )
    return overlay, info
