"""OKS-based Average Precision and related pose evaluation helpers."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


OKS_THRESHOLDS = np.arange(0.50, 1.00, 0.05)

# Local joint-name groups for dance_23-style formats (resolved via keypoint map).
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


def resolve_body_part_indices(keypoints_by_name: dict) -> dict[str, list[int]]:
    """Map body-part groups to local joint ids present in the active format."""
    resolved: dict[str, list[int]] = {}
    for part, names in BODY_PART_JOINT_NAMES.items():
        ids = [
            int(keypoints_by_name[name]['id'])
            for name in names
            if name in keypoints_by_name
        ]
        resolved[part] = ids
    return resolved


def decode_simcc(
    pred_x: torch.Tensor,
    pred_y: torch.Tensor,
    split_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode SimCC logits to crop coordinates and per-joint confidence.

    Returns:
        coords: [B, K, 2] in crop pixel space
        joint_conf: [B, K] confidence in [0, 1]
    """
    prob_x = F.softmax(pred_x.float(), dim=-1)
    prob_y = F.softmax(pred_y.float(), dim=-1)
    conf_x, idx_x = prob_x.max(dim=-1)
    conf_y, idx_y = prob_y.max(dim=-1)
    coords = torch.stack(
        (idx_x.float() / split_ratio, idx_y.float() / split_ratio),
        dim=-1,
    )
    joint_conf = torch.sqrt(conf_x * conf_y)
    return coords, joint_conf


def person_scale_in_crop(
    bbox: tuple[float, float, float, float],
    crop_width: float,
    crop_height: float,
    image_width: int,
    image_height: int,
) -> float:
    """Person scale s = sqrt(bbox area) after mapping the GT bbox into crop space."""
    x1, y1, x2, y2 = bbox
    scale_x = image_width / max(crop_width, 1e-6)
    scale_y = image_height / max(crop_height, 1e-6)
    box_w = max((x2 - x1) * scale_x, 1e-6)
    box_h = max((y2 - y1) * scale_y, 1e-6)
    return float(np.sqrt(box_w * box_h))


def compute_oks(
    pred_xy: np.ndarray,
    gt_xyv: np.ndarray,
    sigmas: np.ndarray,
    scale: float,
    visibility_eq: int | None = None,
) -> float | None:
    """Object Keypoint Similarity for one person.

    Args:
        visibility_eq: If None, use all labeled joints (v > 0). If 1 or 2, restrict
            to occluded or visible joints only. Returns None when the person has no
            joints in the selected class (caller should omit from AP).
    """
    pred_xy = np.asarray(pred_xy)
    gt_xyv = np.asarray(gt_xyv)
    sigmas = np.asarray(sigmas)
    if pred_xy.shape[0] != gt_xyv.shape[0] or sigmas.shape[0] != gt_xyv.shape[0]:
        raise ValueError(
            f'OKS shape mismatch: pred={pred_xy.shape[0]} gt={gt_xyv.shape[0]} '
            f'sigmas={sigmas.shape[0]}'
        )
    if visibility_eq is None:
        mask = gt_xyv[:, 2] > 0
    else:
        mask = gt_xyv[:, 2] == visibility_eq
    if not np.any(mask):
        return None if visibility_eq is not None else 0.0
    delta = pred_xy[mask] - gt_xyv[mask, :2]
    dist_sq = np.sum(delta * delta, axis=-1)
    denom = 2.0 * (scale ** 2) * (sigmas[mask] ** 2) + 1e-9
    return float(np.exp(-dist_sq / denom).mean())


def align_keypoints_to_num_joints(keypoints: np.ndarray, num_joints: int) -> np.ndarray:
    """Pad (v=0) or truncate keypoint rows to ``num_joints``.

    dance_20 is a prefix of dance_23 (body+feet, then head), so padding shorter
    arrays with invisible head joints is safe for that layout.
    """
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.ndim != 2 or keypoints.shape[1] < 3:
        raise ValueError(f'Expected [K, 3+] keypoints, got {keypoints.shape}')
    if keypoints.shape[0] == num_joints:
        return keypoints
    out = np.zeros((num_joints, keypoints.shape[1]), dtype=np.float32)
    n = min(num_joints, keypoints.shape[0])
    out[:n] = keypoints[:n]
    return out


def joint_score_for_visibility(
    joint_conf: np.ndarray,
    gt_xyv: np.ndarray,
    visibility_eq: int | None = None,
) -> float:
    """Mean SimCC confidence over joints in the visibility class (fallback: all)."""
    if visibility_eq is None:
        mask = gt_xyv[:, 2] > 0
    else:
        mask = gt_xyv[:, 2] == visibility_eq
    if np.any(mask):
        return float(joint_conf[mask].mean())
    return float(joint_conf.mean())


def body_part_oks_fields(
    pred_xy: np.ndarray,
    gt_xyv: np.ndarray,
    joint_conf: np.ndarray,
    sigmas: np.ndarray,
    scale: float,
    part_indices: dict[str, list[int]],
) -> dict[str, dict[str, Any]]:
    """Per-part OKS / score / labeled-joint count (oks=None if no labeled joints)."""
    parts: dict[str, dict[str, Any]] = {}
    for part, indices in part_indices.items():
        if not indices:
            parts[part] = {'oks': None, 'score': 0.0, 'n': 0}
            continue
        idx = np.asarray(indices, dtype=np.int64)
        gt = gt_xyv[idx]
        n = int(np.sum(gt[:, 2] > 0))
        if n == 0:
            parts[part] = {
                'oks': None,
                'score': float(joint_conf[idx].mean()),
                'n': 0,
            }
            continue
        parts[part] = {
            'oks': compute_oks(pred_xy[idx], gt, sigmas[idx], scale),
            'score': joint_score_for_visibility(joint_conf[idx], gt),
            'n': n,
        }
    return parts


def average_precision_at_threshold(scores: np.ndarray, oks_values: np.ndarray, threshold: float) -> float:
    """COCO-style AP at a single OKS threshold using score-ranked precision-recall."""
    n_gt = len(oks_values)
    if n_gt == 0:
        return 0.0

    order = np.argsort(-scores)
    oks_sorted = oks_values[order]
    tps = (oks_sorted >= threshold).astype(np.float64)
    fps = 1.0 - tps
    tp_cum = np.cumsum(tps)
    fp_cum = np.cumsum(fps)
    recalls = tp_cum / n_gt
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    # COCO / VOC 2010 precision envelope + 101-point interpolation
    for index in range(len(precisions) - 2, -1, -1):
        precisions[index] = max(precisions[index], precisions[index + 1])
    recall_points = np.linspace(0.0, 1.0, 101)
    interpolated = np.zeros(101, dtype=np.float64)
    for index, recall in enumerate(recall_points):
        above = precisions[recalls >= recall]
        interpolated[index] = above.max() if len(above) else 0.0
    return float(interpolated.mean())


def average_precision_suite(scores: np.ndarray, oks_values: np.ndarray) -> dict[str, float]:
    if len(oks_values) == 0:
        return {'AP': 0.0, 'AP50': 0.0, 'AP75': 0.0}
    aps = [average_precision_at_threshold(scores, oks_values, thr) for thr in OKS_THRESHOLDS]
    return {
        'AP': float(np.mean(aps)),
        'AP50': float(average_precision_at_threshold(scores, oks_values, 0.50)),
        'AP75': float(average_precision_at_threshold(scores, oks_values, 0.75)),
    }


def _safe_std(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1))


def _safe_mean(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.mean(values))


def _subset_metrics(scores: np.ndarray, oks_values: np.ndarray, losses: np.ndarray) -> dict[str, float]:
    ap_suite = average_precision_suite(scores, oks_values)
    return {
        **ap_suite,
        'mean_OKS': _safe_mean(oks_values),
        'std_OKS': _safe_std(oks_values),
        'mean_loss': _safe_mean(losses),
        'std_loss': _safe_std(losses),
        'num_samples': int(len(oks_values)),
    }


def _visibility_class_metrics(
    samples: list[dict[str, Any]],
    oks_key: str,
    score_key: str,
    n_joints_key: str,
) -> dict[str, float]:
    """OKS-AP over people that have ≥1 joint in the class; score ranks by class conf."""
    subset = [s for s in samples if s.get(oks_key) is not None]
    if not subset:
        empty = _subset_metrics(np.array([]), np.array([]), np.array([]))
        return {
            **empty,
            'num_people_with_class': 0,
            'num_joints': 0,
            'mean_joints_per_person': 0.0,
        }
    scores = np.asarray([s[score_key] for s in subset], dtype=np.float64)
    oks_values = np.asarray([s[oks_key] for s in subset], dtype=np.float64)
    # Loss is person-level (not visibility-split); keep for context only.
    losses = np.asarray([s['loss'] for s in subset], dtype=np.float64)
    joint_counts = np.asarray([s[n_joints_key] for s in subset], dtype=np.float64)
    metrics = _subset_metrics(scores, oks_values, losses)
    metrics['num_people_with_class'] = int(len(subset))
    metrics['num_joints'] = int(joint_counts.sum())
    metrics['mean_joints_per_person'] = _safe_mean(joint_counts)
    return metrics


def _body_part_class_metrics(
    samples: list[dict[str, Any]],
    part: str,
) -> dict[str, float]:
    subset = [
        s
        for s in samples
        if s.get('parts') and s['parts'].get(part) and s['parts'][part]['oks'] is not None
    ]
    if not subset:
        empty = _subset_metrics(np.array([]), np.array([]), np.array([]))
        return {
            **empty,
            'num_people_with_class': 0,
            'num_joints': 0,
            'mean_joints_per_person': 0.0,
        }
    scores = np.asarray([s['parts'][part]['score'] for s in subset], dtype=np.float64)
    oks_values = np.asarray([s['parts'][part]['oks'] for s in subset], dtype=np.float64)
    losses = np.asarray([s['loss'] for s in subset], dtype=np.float64)
    joint_counts = np.asarray([s['parts'][part]['n'] for s in subset], dtype=np.float64)
    metrics = _subset_metrics(scores, oks_values, losses)
    metrics['num_people_with_class'] = int(len(subset))
    metrics['num_joints'] = int(joint_counts.sum())
    metrics['mean_joints_per_person'] = _safe_mean(joint_counts)
    return metrics


def summarize_body_part_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Global (not per-task) OKS-AP for head / arms / legs / feet."""
    part_names = list(BODY_PART_JOINT_NAMES.keys())
    if samples and samples[0].get('parts'):
        part_names = list(samples[0]['parts'].keys())
    return {
        'description': (
            'Person-level OKS-AP restricted to body-part joint subsets. '
            'Same OKS/AP thresholds as primary; people with no labeled joints in '
            'the part are omitted. No per-task breakdown.'
        ),
        'parts': {part: _body_part_class_metrics(samples, part) for part in part_names},
    }


def summarize_custom_metrics(
    samples: list[dict[str, Any]],
    ownership: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extra eval suites that do not replace primary OKS-AP.

    Expects optional per-sample keys from evaluate:
      oks_visible / score_visible / n_visible
      oks_occluded / score_occluded / n_occluded
      parts: {part: {oks, score, n}}
    Optional ownership summary (swap / collapse / xy-mix) is attached as-is.
    """
    custom: dict[str, Any] = {
        'visibility_split': {
            'description': (
                'Person-level OKS-AP restricted to GT visibility: '
                'visible=v==2, occluded=v==1. Same OKS/AP thresholds as primary; '
                'people with no joints in the class are omitted. Ranking score is '
                'mean SimCC confidence over that class.'
            ),
            'visible': _visibility_class_metrics(
                samples, 'oks_visible', 'score_visible', 'n_visible'
            ),
            'occluded': _visibility_class_metrics(
                samples, 'oks_occluded', 'score_occluded', 'n_occluded'
            ),
        },
        'body_parts': summarize_body_part_metrics(samples),
    }
    if ownership is not None:
        custom['ownership'] = {
            'description': (
                'Partner swap / collapse / SimCC x-y mix rates (global + IoU buckets). '
                'No per-task breakdown. overall = samples with a partner on the frame.'
            ),
            **ownership,
        }
    return custom


def summarize_pose_metrics(
    samples: list[dict[str, Any]],
    ownership: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate global and per-video OKS-AP / loss metrics.

    Each sample dict must include: video_id, oks, loss, score.
    Optional visibility-split / body-part keys and ownership feed metrics['custom'].
    """
    if not samples:
        empty = _subset_metrics(np.array([]), np.array([]), np.array([]))
        return {
            **empty,
            'std_AP_across_videos': 0.0,
            'per_video': {},
            'custom': summarize_custom_metrics([], ownership=ownership),
        }

    scores = np.asarray([sample['score'] for sample in samples], dtype=np.float64)
    oks_values = np.asarray([sample['oks'] for sample in samples], dtype=np.float64)
    losses = np.asarray([sample['loss'] for sample in samples], dtype=np.float64)
    global_metrics = _subset_metrics(scores, oks_values, losses)

    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_video[sample['video_id']].append(sample)

    per_video: dict[str, dict[str, float]] = {}
    video_aps: list[float] = []
    for video_id in sorted(by_video):
        video_samples = by_video[video_id]
        video_scores = np.asarray([s['score'] for s in video_samples], dtype=np.float64)
        video_oks = np.asarray([s['oks'] for s in video_samples], dtype=np.float64)
        video_losses = np.asarray([s['loss'] for s in video_samples], dtype=np.float64)
        video_metrics = _subset_metrics(video_scores, video_oks, video_losses)
        per_video[video_id] = video_metrics
        video_aps.append(video_metrics['AP'])

    return {
        **global_metrics,
        'std_AP_across_videos': _safe_std(np.asarray(video_aps, dtype=np.float64)),
        'per_video': per_video,
        'custom': summarize_custom_metrics(samples, ownership=ownership),
    }


def format_metrics_summary(checkpoint_name: str, metrics: dict[str, Any]) -> str:
    lines = [
        (
            f'{checkpoint_name} | AP={metrics["AP"]:.3f} | AP50={metrics["AP50"]:.3f} | '
            f'AP75={metrics["AP75"]:.3f} | mean_OKS={metrics["mean_OKS"]:.3f} | '
            f'std_OKS={metrics["std_OKS"]:.3f} | std_AP_across_videos={metrics["std_AP_across_videos"]:.3f} | '
            f'mean_loss={metrics["mean_loss"]:.4f} | std_loss={metrics["std_loss"]:.4f} | '
            f'n={metrics["num_samples"]}'
        )
    ]
    for video_id, video_metrics in metrics.get('per_video', {}).items():
        lines.append(
            f'  {video_id:<12} AP={video_metrics["AP"]:.3f}  '
            f'AP50={video_metrics["AP50"]:.3f}  AP75={video_metrics["AP75"]:.3f}  '
            f'mean_OKS={video_metrics["mean_OKS"]:.3f}  std_OKS={video_metrics["std_OKS"]:.3f}  '
            f'mean_loss={video_metrics["mean_loss"]:.4f}  std_loss={video_metrics["std_loss"]:.4f}  '
            f'n={video_metrics["num_samples"]}'
        )

    custom = metrics.get('custom') or {}
    vis = custom.get('visibility_split')
    if vis:
        lines.append('  [custom] visibility_split (OKS-AP on v==2 vs v==1 only)')
        for label in ('visible', 'occluded'):
            m = vis[label]
            lines.append(
                f'    {label:<10} AP={m["AP"]:.3f}  AP50={m["AP50"]:.3f}  AP75={m["AP75"]:.3f}  '
                f'mean_OKS={m["mean_OKS"]:.3f}  people={m["num_people_with_class"]}  '
                f'joints={m["num_joints"]}  mean_j/person={m["mean_joints_per_person"]:.2f}'
            )

    body = custom.get('body_parts')
    if body and body.get('parts'):
        lines.append('  [custom] body_parts (OKS-AP; global only)')
        for part, m in body['parts'].items():
            lines.append(
                f'    {part:<10} AP={m["AP"]:.3f}  AP50={m["AP50"]:.3f}  AP75={m["AP75"]:.3f}  '
                f'mean_OKS={m["mean_OKS"]:.3f}  people={m["num_people_with_class"]}  '
                f'joints={m["num_joints"]}'
            )

    ownership = custom.get('ownership')
    if ownership:
        lines.append(
            '  [custom] ownership '
            f"(samples={ownership.get('num_samples', 0)} "
            f"with_partner={ownership.get('num_with_partner', 0)})"
        )
        overall = ownership.get('overall')
        if overall is not None:
            lines.append(_format_ownership_metric_line('overall', overall))
        for bucket_name, b in (ownership.get('buckets') or {}).items():
            lines.append(_format_ownership_metric_line(bucket_name, b))
    return '\n'.join(lines)


def _format_ownership_metric_line(label: str, b: dict[str, Any]) -> str:
    n = int(b.get('n') or 0)
    if n == 0 or b.get('swap_rate') is None:
        return f'    {label:<10} n={n}  —'
    return (
        f'    {label:<10} n={n}  '
        f'swap={b["swap_rate"]:.3f}±{b["swap_rate_se"]:.3f}  '
        f'collapse={b["collapse_rate"]:.3f}±{b["collapse_rate_se"]:.3f}  '
        f'xy_mix={b["xy_mix_rate"]:.3f}±{b["xy_mix_rate_se"]:.3f}  '
        f'oks_self={b["mean_oks_self"]:.3f}  oks_partner={b["mean_oks_partner"]:.3f}'
    )


def write_metrics_json(metrics: dict[str, Any], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')


def evaluate_run_folder(
    run_dir: Path,
    checkpoints: tuple[str, ...] = ('best', 'final'),
    batch_size: int | None = None,
    device: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate OKS-AP for checkpoints in an existing training run folder."""
    import yaml

    import src.training.train as train_mod

    run_dir = Path(run_dir)
    config_path = run_dir / 'config.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f'Run config not found: {config_path}')

    with open(config_path, encoding='utf-8') as config_file:
        config = train_mod.parse_config(yaml.safe_load(config_file))
    train_mod.setup_from_config(config)
    train_mod.set_seed(config['training'].seed)

    if train_mod.OKS_SIGMAS.numel() == 0:
        raise ValueError(
            'oks_sigmas missing from keypoint format; cannot compute AP. '
            'Ensure mapping/<format>.yaml defines oks_sigmas.'
        )

    torch_device = torch.device(
        device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    model = train_mod.build_model(config, torch_device)
    test_samples = train_mod.load_test_samples(config)
    eval_batch_size = batch_size or config['training'].batch_size

    print(
        f'Evaluating run {run_dir} on {torch_device} '
        f'({len(test_samples)} test samples)...'
    )

    results: dict[str, dict[str, Any]] = {}
    for checkpoint_name in checkpoints:
        checkpoint_path = run_dir / f'{checkpoint_name}.pt'
        if not checkpoint_path.exists():
            print(f'Skipping {checkpoint_name}: missing {checkpoint_path}')
            continue
        results[checkpoint_name] = train_mod.export_checkpoint_metrics(
            model=model,
            checkpoint_path=checkpoint_path,
            test_samples=test_samples,
            output_dir=run_dir / checkpoint_name,
            device=torch_device,
            batch_size=eval_batch_size,
            annotation_version=config['dataset'].annotation_version,
        )
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Evaluate OKS-AP metrics for an existing training run folder.'
    )
    parser.add_argument(
        '--run',
        required=True,
        type=Path,
        help='Path to a run directory (e.g. training/runs/2026_09_20_017)',
    )
    parser.add_argument(
        '--checkpoints',
        nargs='+',
        default=['best', 'final'],
        help='Checkpoint stems to evaluate (default: best final)',
    )
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--device', default=None, help='cuda | cpu (default: auto)')
    args = parser.parse_args()

    evaluate_run_folder(
        run_dir=args.run,
        checkpoints=tuple(args.checkpoints),
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == '__main__':
    main()
