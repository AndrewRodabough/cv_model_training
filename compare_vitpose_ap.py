#!/usr/bin/env python3
"""Compare OKS-AP of a training-run best.pt vs VitPose++ on the same test split.

Uses the run folder's config.yaml for annotation version, seed, and split_mode so the
test set matches training. Reports:

  1. best.pt — all dance_20 joints
  2. best.pt — joints shared with VitPose++ (excludes heel_spike_*)
  3. best.pt — feet-only shared joints (ankles + toes + heels)
  4. VitPose++ — shared joints (exact GT bbox)
  5. VitPose++ — feet-only (same preds as 4)
  6. VitPose++ — shared joints with jittered bbox (side metric)

Example:
  python compare_vitpose_ap.py --run training/runs/2026_09_26_001
  python compare_vitpose_ap.py --run training/runs/2026_09_26_001 \\
      --jitter-frac 0.03 --jitter-scale-frac 0.03
"""

from __future__ import annotations

import argparse
import base64
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
import yaml
from PIL import Image

import train as train_mod
from metrics import (
    compute_oks,
    decode_simcc,
    format_metrics_summary,
    person_scale_in_crop,
    summarize_pose_metrics,
    write_metrics_json,
)

DEFAULT_FUNCTION_NAME = 'pth-vitpose-plus-plus-wholebody'

# dance_20 name -> COCO-WholeBody / VitPose++ index (body + feet).
# heel_spike_l/r have no VitPose counterpart and are excluded from shared eval.
VITPOSE_SHARED_BY_NAME: dict[str, int] = {
    'shoulder_l': 5,
    'shoulder_r': 6,
    'elbow_l': 7,
    'elbow_r': 8,
    'wrist_l': 9,
    'wrist_r': 10,
    'hip_l': 11,
    'hip_r': 12,
    'knee_l': 13,
    'knee_r': 14,
    'ankle_l': 15,
    'ankle_r': 16,
    'toe_b_l': 17,
    'toe_s_l': 18,
    'heel_l': 19,
    'toe_b_r': 20,
    'toe_s_r': 21,
    'heel_r': 22,
}

FEET_SHARED_NAMES = (
    'ankle_l',
    'ankle_r',
    'toe_b_l',
    'toe_s_l',
    'heel_l',
    'toe_b_r',
    'toe_s_r',
    'heel_r',
)


def resolve_function_url(explicit: str | None, function_name: str) -> str:
    if explicit:
        return explicit.rstrip('/')
    port = _docker_published_port(function_name)
    if port is None:
        raise RuntimeError(
            f'Could not find Nuclio port for {function_name!r}. '
            'Pass --function-url explicitly '
            f'(e.g. http://localhost:32768/api/{function_name}).'
        )
    return f'http://localhost:{port}/api/{function_name}'


def _docker_published_port(function_name: str) -> int | None:
    container = f'nuclio-nuclio-{function_name}'
    try:
        result = subprocess.run(
            ['docker', 'port', container, '8080/tcp'],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    for line in result.stdout.splitlines():
        if ':' in line:
            try:
                return int(line.rsplit(':', 1)[-1].strip())
            except ValueError:
                continue
    return None


def joint_subset_from_names(
    keypoints_by_name: dict,
    names: tuple[str, ...] | list[str],
    vitpose_by_name: dict[str, int],
) -> tuple[list[int], list[int], list[str]]:
    local_ids: list[int] = []
    vitpose_ids: list[int] = []
    out_names: list[str] = []
    for name in names:
        if name not in keypoints_by_name:
            raise ValueError(f'Keypoint format missing joint: {name}')
        if name not in vitpose_by_name:
            raise ValueError(f'No VitPose mapping for joint: {name}')
        local_ids.append(int(keypoints_by_name[name]['id']))
        vitpose_ids.append(vitpose_by_name[name])
        out_names.append(name)
    return local_ids, vitpose_ids, out_names


def person_scale_image(bbox: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return float(np.sqrt(max(x2 - x1, 1e-6) * max(y2 - y1, 1e-6)))


def jitter_bbox(
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    rng: np.random.Generator,
    shift_frac: float,
    scale_frac: float,
) -> tuple[float, float, float, float]:
    """Randomly shift/scale a box, then clamp to the image."""
    x1, y1, x2, y2 = bbox
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    scale = 1.0 + float(rng.uniform(-scale_frac, scale_frac))
    shift_x = float(rng.uniform(-shift_frac, shift_frac)) * width
    shift_y = float(rng.uniform(-shift_frac, shift_frac)) * height
    new_w = width * scale
    new_h = height * scale
    nx1 = cx + shift_x - 0.5 * new_w
    ny1 = cy + shift_y - 0.5 * new_h
    nx2 = nx1 + new_w
    ny2 = ny1 + new_h
    nx1 = float(np.clip(nx1, 0.0, image_width - 1.0))
    ny1 = float(np.clip(ny1, 0.0, image_height - 1.0))
    nx2 = float(np.clip(nx2, nx1 + 1.0, image_width))
    ny2 = float(np.clip(ny2, ny1 + 1.0, image_height))
    return nx1, ny1, nx2, ny2


def call_vitpose(
    function_url: str,
    image_bytes: bytes,
    regions: list[dict[str, Any]],
    timeout: float,
) -> list[dict[str, Any]]:
    payload = {
        'image': base64.b64encode(image_bytes).decode('ascii'),
        'regions': regions,
    }
    resp = requests.post(function_url, json=payload, timeout=timeout)
    if resp.status_code not in (200, 201, 202):
        raise RuntimeError(f'ViTPose request failed ({resp.status_code}): {resp.text}')
    body = resp.json()
    if isinstance(body, list):
        return body
    if isinstance(body, dict) and isinstance(body.get('result'), list):
        return body['result']
    raise RuntimeError(f'Unexpected ViTPose response shape: {type(body)}')


def skeleton_to_xy(
    prediction: dict[str, Any],
    vitpose_ids: list[int],
) -> tuple[np.ndarray, float]:
    """Map a VitPose skeleton to [K, 2] xy and a ranking score."""
    elements = prediction.get('elements') or []
    by_label: dict[int, tuple[float, float]] = {}
    for element in elements:
        try:
            label = int(element.get('label'))
        except (TypeError, ValueError):
            continue
        points = element.get('points') or []
        if len(points) < 2:
            continue
        by_label[label] = (float(points[0]), float(points[1]))

    xy = np.zeros((len(vitpose_ids), 2), dtype=np.float32)
    for row, vit_idx in enumerate(vitpose_ids):
        if vit_idx in by_label:
            xy[row] = by_label[vit_idx]
        else:
            xy[row] = np.nan

    score = prediction.get('confidence')
    if score is None:
        score = 1.0
    return xy, float(score)


def evaluate_best_pt(
    model,
    checkpoint_path: Path,
    test_samples: list,
    device: torch.device,
    batch_size: int,
    joint_indices: list[int] | None,
    load_weights: bool = True,
) -> dict[str, Any]:
    """OKS-AP for best.pt; optionally restrict to a joint subset (local ids)."""
    if load_weights:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_sigmas = train_mod.OKS_SIGMAS.detach().cpu().numpy()
    if joint_indices is None:
        use_indices = list(range(len(all_sigmas)))
    else:
        use_indices = list(joint_indices)
    sigmas = all_sigmas[use_indices]

    records: list[dict[str, Any]] = []
    for start in range(0, len(test_samples), batch_size):
        batch = test_samples[start:start + batch_size]
        pixels_batch = []
        keypoints_batch = []
        role_batch = []
        scales = []

        for image_path, keypoints, bbox, role in batch:
            image = Image.open(image_path).convert('RGB')
            crop_x1, crop_y1, crop_width, crop_height = train_mod.compute_padded_crop(
                bbox, image.width, image.height
            )
            crop = image.crop((crop_x1, crop_y1, crop_x1 + crop_width, crop_y1 + crop_height))
            crop = crop.resize(
                (train_mod.IMAGE_SIZE.x, train_mod.IMAGE_SIZE.y),
                Image.Resampling.BILINEAR,
            )
            pixels = torch.from_numpy(np.asarray(crop, dtype=np.float32)).permute(2, 0, 1) / 255.0
            pixels = (pixels - train_mod.IMAGE_MEAN) / train_mod.IMAGE_STD
            crop_keypoints = train_mod.transform_keypoints_to_crop(
                keypoints, crop_x1, crop_y1, crop_width, crop_height
            )
            pixels_batch.append(pixels)
            keypoints_batch.append(torch.from_numpy(crop_keypoints))
            role_batch.append(train_mod.encode_role(role))
            scales.append(
                person_scale_in_crop(
                    bbox,
                    crop_width,
                    crop_height,
                    train_mod.IMAGE_SIZE.x,
                    train_mod.IMAGE_SIZE.y,
                )
            )

        pixels_tensor = torch.stack(pixels_batch).to(device)
        keypoints_tensor = torch.stack(keypoints_batch)
        role_tensor = torch.stack(role_batch).to(device)
        target_x, target_y, weights = train_mod.make_targets(keypoints_tensor, device)

        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            pred_x, pred_y = model(
                pixels_tensor,
                role=role_tensor if train_mod.USE_ROLE else None,
            )
            sample_losses = train_mod.per_sample_simcc_loss(
                pred_x, pred_y, target_x, target_y, weights
            )
            coords, joint_conf = decode_simcc(pred_x, pred_y, train_mod.SIMCC_SPLIT_RATIO)

        coords_np = coords.detach().cpu().numpy()
        conf_np = joint_conf.detach().cpu().numpy()
        keypoints_np = keypoints_tensor.numpy()
        losses_np = sample_losses.detach().cpu().numpy()

        for offset, (image_path, _keypoints, _bbox, _role) in enumerate(batch):
            gt_full = keypoints_np[offset]
            pred_full = coords_np[offset]
            conf_full = conf_np[offset]
            gt = gt_full[use_indices]
            pred = pred_full[use_indices]
            conf = conf_full[use_indices]
            visible = gt[:, 2] > 0
            score = float(conf[visible].mean()) if np.any(visible) else float(conf.mean())
            oks = compute_oks(pred, gt, sigmas, scales[offset])
            records.append(
                {
                    'video_id': train_mod.clip_id_from_image_path(image_path),
                    'oks': oks,
                    'loss': float(losses_np[offset]),
                    'score': score,
                }
            )

    return summarize_pose_metrics(records)


def collect_vitpose_predictions(
    test_samples: list,
    function_url: str,
    vitpose_ids: list[int],
    timeout: float,
    label: str,
    jitter_shift_frac: float = 0.0,
    jitter_scale_frac: float = 0.0,
    jitter_seed: int = 0,
) -> list[dict[str, Any]]:
    """Call VitPose once per image; return per-sample pred dicts aligned to test_samples."""
    rng = np.random.default_rng(jitter_seed)
    use_jitter = jitter_shift_frac > 0.0 or jitter_scale_frac > 0.0

    by_image: dict[Path, list[tuple[int, np.ndarray, tuple[float, float, float, float]]]] = (
        defaultdict(list)
    )
    for index, (image_path, keypoints, bbox, _role) in enumerate(test_samples):
        by_image[Path(image_path)].append((index, keypoints, bbox))

    out: list[dict[str, Any] | None] = [None] * len(test_samples)
    image_paths = sorted(by_image.keys(), key=lambda path: str(path))
    for image_idx, image_path in enumerate(image_paths, start=1):
        entries = by_image[image_path]
        image_bytes = image_path.read_bytes()
        with Image.open(image_path) as image:
            image_width, image_height = image.size

        regions = []
        request_bboxes = []
        for _index, _keypoints, bbox in entries:
            if use_jitter:
                req_bbox = jitter_bbox(
                    bbox,
                    image_width,
                    image_height,
                    rng,
                    shift_frac=jitter_shift_frac,
                    scale_frac=jitter_scale_frac,
                )
            else:
                req_bbox = bbox
            request_bboxes.append(req_bbox)
            regions.append(
                {
                    'points': [
                        float(req_bbox[0]),
                        float(req_bbox[1]),
                        float(req_bbox[2]),
                        float(req_bbox[3]),
                    ]
                }
            )

        print(
            f'  VitPose {label} [{image_idx}/{len(image_paths)}] {image_path.name} '
            f'({len(regions)} person(s))'
        )
        predictions = call_vitpose(function_url, image_bytes, regions, timeout=timeout)
        skeletons = [pred for pred in predictions if isinstance(pred, dict) and pred.get('elements')]
        if len(skeletons) < len(regions):
            skeletons = skeletons + [{}] * (len(regions) - len(skeletons))
        skeletons = skeletons[: len(regions)]

        for (sample_index, keypoints, gt_bbox), prediction, req_bbox in zip(
            entries, skeletons, request_bboxes
        ):
            pred_xy, score = skeleton_to_xy(prediction, vitpose_ids)
            out[sample_index] = {
                'video_id': train_mod.clip_id_from_image_path(image_path),
                'keypoints': keypoints,
                'gt_bbox': gt_bbox,
                'request_bbox': req_bbox,
                'pred_xy': pred_xy,
                'score': score,
            }

    assert all(item is not None for item in out)
    return out  # type: ignore[return-value]


def score_vitpose_predictions(
    predictions: list[dict[str, Any]],
    local_ids: list[int],
    subset_rows: list[int] | None,
    sigmas_full: np.ndarray,
) -> dict[str, Any]:
    """Score cached VitPose preds on a joint subset (rows into the shared pred vector)."""
    if subset_rows is None:
        subset_rows = list(range(len(local_ids)))
    use_local = [local_ids[row] for row in subset_rows]
    sigmas = sigmas_full[use_local]

    records: list[dict[str, Any]] = []
    for item in predictions:
        pred_xy = item['pred_xy'][subset_rows]
        gt = item['keypoints'][use_local].copy()
        missing = np.isnan(pred_xy).any(axis=1)
        pred_xy = np.nan_to_num(pred_xy, nan=0.0)
        gt[missing, 2] = 0.0
        scale = person_scale_image(item['gt_bbox'])
        oks = compute_oks(pred_xy, gt, sigmas, scale)
        records.append(
            {
                'video_id': item['video_id'],
                'oks': oks,
                'loss': float('nan'),
                'score': item['score'],
            }
        )
    return summarize_pose_metrics(records)


def mean_pred_jitter(
    exact_preds: list[dict[str, Any]],
    jitter_preds: list[dict[str, Any]],
    subset_rows: list[int] | None = None,
) -> dict[str, float]:
    """Mean L2 pixel move between exact-bbox and jittered-bbox VitPose preds."""
    distances: list[float] = []
    for exact, jittered in zip(exact_preds, jitter_preds):
        a = exact['pred_xy']
        b = jittered['pred_xy']
        if subset_rows is not None:
            a = a[subset_rows]
            b = b[subset_rows]
        valid = ~(np.isnan(a).any(axis=1) | np.isnan(b).any(axis=1))
        if not np.any(valid):
            continue
        delta = a[valid] - b[valid]
        distances.extend(np.linalg.norm(delta, axis=1).tolist())
    if not distances:
        return {'mean_l2_px': float('nan'), 'median_l2_px': float('nan'), 'n_joints': 0}
    arr = np.asarray(distances, dtype=np.float64)
    return {
        'mean_l2_px': float(arr.mean()),
        'median_l2_px': float(np.median(arr)),
        'n_joints': int(arr.size),
    }


def skipped_metrics() -> dict[str, Any]:
    return {
        'AP': float('nan'),
        'AP50': float('nan'),
        'AP75': float('nan'),
        'mean_OKS': float('nan'),
        'std_OKS': float('nan'),
        'mean_loss': float('nan'),
        'std_loss': float('nan'),
        'num_samples': 0,
        'std_AP_across_videos': float('nan'),
        'per_video': {},
        'skipped': True,
    }


def print_comparison(results: dict[str, Any]) -> None:
    print('\n=== Comparison (same test split) ===')
    order = [
        ('best_full', 'best.pt (all dance_20)'),
        ('best_shared', 'best.pt (shared)'),
        ('best_feet', 'best.pt (feet only)'),
        ('vitpose_shared', 'VitPose++ (shared, exact bbox)'),
        ('vitpose_feet', 'VitPose++ (feet only, exact bbox)'),
        ('vitpose_shared_jittered', 'VitPose++ (shared, jittered bbox)'),
        ('vitpose_feet_jittered', 'VitPose++ (feet only, jittered bbox)'),
    ]
    for key, label in order:
        metrics = results.get(key)
        if not metrics or metrics.get('skipped'):
            print(f'{label}: skipped')
            continue
        print(
            f'{label}: AP={metrics["AP"]:.3f}  AP50={metrics["AP50"]:.3f}  '
            f'AP75={metrics["AP75"]:.3f}  mean_OKS={metrics["mean_OKS"]:.3f}  '
            f'n={metrics["num_samples"]}'
        )
    jitter_stats = results.get('jitter_pred_delta')
    if jitter_stats:
        print(
            f'VitPose pred move under bbox jitter: '
            f'mean={jitter_stats["mean_l2_px"]:.2f}px  '
            f'median={jitter_stats["median_l2_px"]:.2f}px  '
            f'n_joints={jitter_stats["n_joints"]}'
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Compare best.pt vs VitPose++ OKS-AP on a run\'s test split.'
    )
    parser.add_argument(
        '--run',
        required=True,
        type=Path,
        help='Training run folder containing config.yaml and best.pt',
    )
    parser.add_argument('--function-url', default=None)
    parser.add_argument('--function-name', default=DEFAULT_FUNCTION_NAME)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--device', default=None, help='cuda | cpu (default: auto)')
    parser.add_argument('--timeout', type=float, default=180.0)
    parser.add_argument('--skip-vitpose', action='store_true')
    parser.add_argument(
        '--jitter-frac',
        type=float,
        default=0.03,
        help='Bbox center shift as fraction of box size (default: 0.03)',
    )
    parser.add_argument(
        '--jitter-scale-frac',
        type=float,
        default=0.03,
        help='Bbox scale jitter fraction (default: 0.03)',
    )
    parser.add_argument(
        '--jitter-seed',
        type=int,
        default=0,
        help='RNG seed for bbox jitter (default: 0)',
    )
    parser.add_argument(
        '--no-jitter',
        action='store_true',
        help='Skip the jittered-bbox VitPose side metric',
    )
    args = parser.parse_args()

    run_dir = Path(args.run)
    config_path = run_dir / 'config.yaml'
    checkpoint_path = run_dir / 'best.pt'
    if not config_path.exists():
        raise FileNotFoundError(f'Missing run config: {config_path}')
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Missing checkpoint: {checkpoint_path}')

    with open(config_path, encoding='utf-8') as config_file:
        config = train_mod.parse_config(yaml.safe_load(config_file))
    train_mod.setup_from_config(config)
    train_mod.set_seed(config['training'].seed)

    if train_mod.OKS_SIGMAS.numel() == 0:
        raise ValueError('oks_sigmas missing from keypoint format mapping')

    local_ids, vitpose_ids, shared_names = joint_subset_from_names(
        train_mod.KEYPOINTS_BY_NAME,
        list(VITPOSE_SHARED_BY_NAME.keys()),
        VITPOSE_SHARED_BY_NAME,
    )
    feet_local_ids, _feet_vitpose_ids, feet_names = joint_subset_from_names(
        train_mod.KEYPOINTS_BY_NAME,
        FEET_SHARED_NAMES,
        VITPOSE_SHARED_BY_NAME,
    )
    feet_rows = [shared_names.index(name) for name in feet_names]
    excluded = sorted(
        name
        for name in train_mod.KEYPOINTS_BY_NAME
        if name not in VITPOSE_SHARED_BY_NAME
    )

    device = torch.device(
        args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    batch_size = args.batch_size or config['training'].batch_size
    test_samples = train_mod.load_test_samples(config)
    model = train_mod.build_model(config, device)
    sigmas_full = train_mod.OKS_SIGMAS.detach().cpu().numpy()

    training_cfg = config['training']
    dataset_cfg = config['dataset']
    print(
        f'Run {run_dir.name} | annotation={dataset_cfg.annotation_version} | '
        f'split={training_cfg.train_test_split:.2f} {training_cfg.split_mode} | '
        f'seed={training_cfg.seed} | test_samples={len(test_samples)}'
    )
    print(
        f'Shared joints ({len(shared_names)}): {", ".join(shared_names)}\n'
        f'Feet joints ({len(feet_names)}): {", ".join(feet_names)}\n'
        f'Excluded from shared eval: {", ".join(excluded) if excluded else "(none)"}'
    )

    print('\n[1/6] best.pt — all joints')
    best_full = evaluate_best_pt(
        model, checkpoint_path, test_samples, device, batch_size, joint_indices=None
    )
    print(format_metrics_summary('best_full', best_full))

    print('\n[2/6] best.pt — shared joints')
    best_shared = evaluate_best_pt(
        model,
        checkpoint_path,
        test_samples,
        device,
        batch_size,
        joint_indices=local_ids,
        load_weights=False,
    )
    print(format_metrics_summary('best_shared', best_shared))

    print('\n[3/6] best.pt — feet only')
    best_feet = evaluate_best_pt(
        model,
        checkpoint_path,
        test_samples,
        device,
        batch_size,
        joint_indices=feet_local_ids,
        load_weights=False,
    )
    print(format_metrics_summary('best_feet', best_feet))

    vitpose_shared = skipped_metrics()
    vitpose_feet = skipped_metrics()
    vitpose_shared_jittered = skipped_metrics()
    vitpose_feet_jittered = skipped_metrics()
    jitter_pred_delta: dict[str, float] | None = None
    jitter_meta: dict[str, Any] | None = None

    if args.skip_vitpose:
        print('\n[4–6/6] VitPose++ skipped (--skip-vitpose)')
    else:
        function_url = resolve_function_url(args.function_url, args.function_name)
        print(f'\n[4/6] VitPose++ — exact GT bbox via {function_url}')
        exact_preds = collect_vitpose_predictions(
            test_samples=test_samples,
            function_url=function_url,
            vitpose_ids=vitpose_ids,
            timeout=args.timeout,
            label='exact',
        )
        vitpose_shared = score_vitpose_predictions(
            exact_preds, local_ids, subset_rows=None, sigmas_full=sigmas_full
        )
        print(format_metrics_summary('vitpose_shared', vitpose_shared))

        print('\n[5/6] VitPose++ — feet only (same exact-bbox preds)')
        vitpose_feet = score_vitpose_predictions(
            exact_preds, local_ids, subset_rows=feet_rows, sigmas_full=sigmas_full
        )
        print(format_metrics_summary('vitpose_feet', vitpose_feet))

        if args.no_jitter:
            print('\n[6/6] VitPose++ jitter side metric skipped (--no-jitter)')
        else:
            jitter_meta = {
                'shift_frac': args.jitter_frac,
                'scale_frac': args.jitter_scale_frac,
                'seed': args.jitter_seed,
            }
            print(
                f'\n[6/6] VitPose++ — jittered bbox '
                f'(shift±{args.jitter_frac:.0%}, scale±{args.jitter_scale_frac:.0%}, '
                f'seed={args.jitter_seed})'
            )
            jitter_preds = collect_vitpose_predictions(
                test_samples=test_samples,
                function_url=function_url,
                vitpose_ids=vitpose_ids,
                timeout=args.timeout,
                label='jitter',
                jitter_shift_frac=args.jitter_frac,
                jitter_scale_frac=args.jitter_scale_frac,
                jitter_seed=args.jitter_seed,
            )
            vitpose_shared_jittered = score_vitpose_predictions(
                jitter_preds, local_ids, subset_rows=None, sigmas_full=sigmas_full
            )
            vitpose_feet_jittered = score_vitpose_predictions(
                jitter_preds, local_ids, subset_rows=feet_rows, sigmas_full=sigmas_full
            )
            jitter_pred_delta = mean_pred_jitter(exact_preds, jitter_preds)
            print(format_metrics_summary('vitpose_shared_jittered', vitpose_shared_jittered))
            print(format_metrics_summary('vitpose_feet_jittered', vitpose_feet_jittered))
            print(
                f'  pred L2 under jitter: mean={jitter_pred_delta["mean_l2_px"]:.2f}px '
                f'median={jitter_pred_delta["median_l2_px"]:.2f}px'
            )

    results = {
        'run': str(run_dir),
        'annotation_version': dataset_cfg.annotation_version,
        'split_mode': training_cfg.split_mode,
        'train_test_split': training_cfg.train_test_split,
        'seed': training_cfg.seed,
        'test_samples': len(test_samples),
        'shared_joints': shared_names,
        'feet_joints': feet_names,
        'excluded_joints': excluded,
        'jitter': jitter_meta,
        'best_full': best_full,
        'best_shared': best_shared,
        'best_feet': best_feet,
        'vitpose_shared': vitpose_shared,
        'vitpose_feet': vitpose_feet,
        'vitpose_shared_jittered': vitpose_shared_jittered,
        'vitpose_feet_jittered': vitpose_feet_jittered,
        'jitter_pred_delta': jitter_pred_delta,
    }

    out_dir = run_dir / 'vitpose_compare'
    out_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_json(results, out_dir / 'metrics.json')
    print_comparison(results)
    print(f'\nWrote {out_dir / "metrics.json"}')


if __name__ == '__main__':
    main()
