#!/usr/bin/env python3
"""Evaluate swap / collapse / x-y mixing on the test split."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

import train as t
from metrics import compute_oks, decode_simcc, person_scale_in_crop
from ownership_metrics import (
    format_ownership_summary,
    iou_bucket,
    joint_xy_mix_fraction,
    pick_partner,
    summarize_ownership_records,
    write_ownership_json,
)

SWAP_PARTNER_OKS_FLOOR = 0.25
COLLAPSE_SELF_OKS_TAU = 0.5


def build_frame_index(annotation_file: Path, task_ids: set[str]):
    """Map image_path -> list of (keypoints, bbox, role) in full image coords."""
    root = ET.parse(annotation_file).getroot()
    image_root = annotation_file.parent
    index: dict[Path, list] = defaultdict(list)
    for video in root.findall('./project/videos/video'):
        if video.get('id') not in task_ids:
            continue
        for frame in video.findall('frame'):
            image_path = (image_root / frame.get('image')).resolve()
            for person in frame.findall('person'):
                kps, bbox, role = t.parse_person(person)
                index[image_path].append((kps, bbox, role))
    return index


def is_same_person(
    bbox_a: tuple[float, float, float, float],
    bbox_b: tuple[float, float, float, float],
) -> bool:
    return t.bbox_iou(bbox_a, bbox_b) >= 0.99


def forward_model(model, pixels_tensor, role_tensor, spatial_tensor, device):
    with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
        out = model(
            pixels_tensor,
            role=role_tensor if t.USE_ROLE else None,
            spatial=spatial_tensor if t.USE_SPATIAL_PROMPT else None,
        )
    if isinstance(out, tuple) and len(out) == 3:
        return out[0], out[1]
    return out[0], out[1]


def eval_ownership(
    run_dir: Path,
    checkpoint_name: str = 'best.pt',
    output_subdir: str = 'ownership_metrics',
    swap_partner_oks_floor: float = SWAP_PARTNER_OKS_FLOOR,
    collapse_self_tau: float = COLLAPSE_SELF_OKS_TAU,
) -> dict:
    config_path = run_dir / 'config.yaml'
    with open(config_path, encoding='utf-8') as f:
        config = t.parse_config(yaml.safe_load(f))
    t.setup_from_config(config)
    training_cfg = config['training']
    t.set_seed(training_cfg.seed)

    if t.COUPLE_MODE:
        raise ValueError('Ownership eval is for single-person checkpoints only.')

    ckpt_path = run_dir / checkpoint_name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = t.build_model(config, device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    annotation_file = t.get_annotation_file_path(config['dataset'].annotation_version)
    root = ET.parse(annotation_file).getroot()
    task_ids = {video.get('id') for video in root.findall('./project/videos/video')}
    frame_index = build_frame_index(annotation_file, task_ids)
    test_samples = t.load_test_samples(config)
    sigmas = t.OKS_SIGMAS.detach().cpu().numpy()
    batch_size = training_cfg.batch_size

    records = []

    for start in range(0, len(test_samples), batch_size):
        batch = test_samples[start : start + batch_size]
        pixels_batch = []
        role_batch = []
        spatial_batch = []
        meta = []

        for image_path, keypoints, bbox, role in batch:
            image = Image.open(image_path).convert('RGB')
            crop_x1, crop_y1, crop_width, crop_height = t.compute_padded_crop(
                bbox, image.width, image.height
            )
            crop = image.crop((crop_x1, crop_y1, crop_x1 + crop_width, crop_y1 + crop_height))
            crop = crop.resize((t.IMAGE_SIZE.x, t.IMAGE_SIZE.y), Image.Resampling.BILINEAR)
            pixels = torch.from_numpy(np.asarray(crop, dtype=np.float32)).permute(2, 0, 1) / 255.0
            pixels = (pixels - t.IMAGE_MEAN) / t.IMAGE_STD
            crop_kps = t.transform_keypoints_to_crop(
                keypoints, crop_x1, crop_y1, crop_width, crop_height
            )
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            spatial = t.crop_normalized_bbox_center(
                cx, cy, crop_x1, crop_y1, crop_width, crop_height
            )
            pixels_batch.append(pixels)
            role_batch.append(t.encode_role(role))
            spatial_batch.append(spatial)
            meta.append(
                {
                    'image_path': Path(image_path),
                    'bbox': bbox,
                    'role': role,
                    'crop_kps': crop_kps,
                    'crop_geom': (crop_x1, crop_y1, crop_width, crop_height),
                    'scale': person_scale_in_crop(
                        bbox, crop_width, crop_height, t.IMAGE_SIZE.x, t.IMAGE_SIZE.y
                    ),
                }
            )

        pixels_tensor = torch.stack(pixels_batch).to(device)
        role_tensor = torch.stack(role_batch).to(device)
        spatial_tensor = torch.stack(spatial_batch).to(device)
        pred_x, pred_y = forward_model(model, pixels_tensor, role_tensor, spatial_tensor, device)
        pred_x_np = pred_x.detach().float().cpu().numpy()
        pred_y_np = pred_y.detach().float().cpu().numpy()
        coords, _conf = decode_simcc(pred_x, pred_y, t.SIMCC_SPLIT_RATIO)
        coords_np = coords.detach().cpu().numpy()

        for offset, item in enumerate(meta):
            image_path = item['image_path']
            people = frame_index.get(image_path, [])
            self_bbox = item['bbox']
            others = [
                (kps, bbox, r)
                for kps, bbox, r in people
                if not is_same_person(bbox, self_bbox)
            ]

            partner_ctx = pick_partner(self_bbox, others)
            crop_kps = item['crop_kps']
            pred_xy = coords_np[offset]
            oks_self = compute_oks(pred_xy, crop_kps, sigmas, item['scale'])
            cx1, cy1, cw, ch = item['crop_geom']

            if partner_ctx is None:
                records.append(
                    {
                        'image_path': str(image_path),
                        'role': item['role'],
                        'iou_bucket': iou_bucket(0.0),
                        'max_partner_iou': 0.0,
                        'has_partner': False,
                        'swap': False,
                        'collapse': False,
                        'oks_self': float(oks_self),
                        'oks_partner': 0.0,
                        'xy_mix_rate': 0.0,
                        'xy_mix_joints': 0,
                    }
                )
                continue

            partner_crop = t.transform_keypoints_to_crop(
                partner_ctx.partner_kps, cx1, cy1, cw, ch
            )
            partner_scale = person_scale_in_crop(
                partner_ctx.partner_bbox, cw, ch, t.IMAGE_SIZE.x, t.IMAGE_SIZE.y
            )
            oks_partner = compute_oks(pred_xy, partner_crop, sigmas, partner_scale)
            swap = oks_partner > oks_self and oks_partner >= swap_partner_oks_floor
            collapse = oks_self < collapse_self_tau and oks_partner > oks_self
            xy_mix_rate, xy_n = joint_xy_mix_fraction(
                pred_x_np[offset],
                pred_y_np[offset],
                crop_kps,
                partner_crop,
                t.SIMCC_SPLIT_RATIO,
            )
            records.append(
                {
                    'image_path': str(image_path),
                    'role': item['role'],
                    'iou_bucket': iou_bucket(partner_ctx.max_partner_iou),
                    'max_partner_iou': partner_ctx.max_partner_iou,
                    'has_partner': True,
                    'swap': bool(swap),
                    'collapse': bool(collapse),
                    'oks_self': float(oks_self),
                    'oks_partner': float(oks_partner),
                    'xy_mix_rate': float(xy_mix_rate),
                    'xy_mix_joints': int(xy_n),
                }
            )

    summary = summarize_ownership_records(records)
    summary['checkpoint'] = str(ckpt_path.resolve())
    summary['run_dir'] = str(run_dir.resolve())
    summary['swap_partner_oks_floor'] = swap_partner_oks_floor
    summary['collapse_self_oks_tau'] = collapse_self_tau

    out_dir = run_dir / output_subdir
    write_ownership_json(summary, out_dir / 'ownership_summary.json')
    write_ownership_json({'records': records}, out_dir / 'ownership_records.json')
    (out_dir / 'ownership_summary.txt').write_text(
        format_ownership_summary(summary) + '\n', encoding='utf-8'
    )
    print(format_ownership_summary(summary))
    print(f'Wrote {out_dir / "ownership_summary.json"}')
    return summary


def main():
    parser = argparse.ArgumentParser(description='Ownership metrics on test split.')
    parser.add_argument('--run', type=Path, required=True, help='Training run directory.')
    parser.add_argument('--checkpoint', default='best.pt', help='Checkpoint filename in run dir.')
    parser.add_argument('--output-subdir', default='ownership_metrics')
    args = parser.parse_args()
    eval_ownership(args.run.resolve(), checkpoint_name=args.checkpoint, output_subdir=args.output_subdir)


if __name__ == '__main__':
    main()
