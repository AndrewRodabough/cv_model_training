#!/usr/bin/env python3
"""Break feet OKS-AP into ankles / toes / heels for model vs VitPose++."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

import compare_vitpose_ap as cmp
import train as train_mod
from metrics import compute_oks, decode_simcc, person_scale_in_crop

ANKLES = ('ankle_l', 'ankle_r')
TOES = ('toe_b_l', 'toe_s_l', 'toe_b_r', 'toe_s_r')
HEELS = ('heel_l', 'heel_r')
SUBSETS = {
    'ankles': ANKLES,
    'toes': TOES,
    'heels': HEELS,
    'feet_all': ANKLES + TOES + HEELS,
}


def slim(m: dict) -> dict:
    return {
        'AP': m['AP'],
        'AP50': m['AP50'],
        'AP75': m['AP75'],
        'mean_OKS': m['mean_OKS'],
        'num_samples': m['num_samples'],
    }


def cache_model_preds(model, checkpoint, test_samples, device, batch_size):
    checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint_data['model_state_dict'])
    model.eval()
    out = []
    for start in range(0, len(test_samples), batch_size):
        batch = test_samples[start:start + batch_size]
        pixels_batch = []
        kps_batch = []
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
            crop_kps = train_mod.transform_keypoints_to_crop(
                keypoints, crop_x1, crop_y1, crop_width, crop_height
            )
            pixels_batch.append(pixels)
            kps_batch.append(torch.from_numpy(crop_kps))
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
        role_tensor = torch.stack(role_batch).to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            pred_x, pred_y = model(
                pixels_tensor,
                role=role_tensor if train_mod.USE_ROLE else None,
            )
            coords, _ = decode_simcc(pred_x, pred_y, train_mod.SIMCC_SPLIT_RATIO)
        coords_np = coords.detach().cpu().numpy()
        kps_np = torch.stack(kps_batch).numpy()
        for i in range(len(batch)):
            out.append((coords_np[i], kps_np[i], scales[i]))
    return out


def per_joint_mean_oks(model_preds, vit_preds, names, name_to_row, sigmas_full):
    rows = {}
    for name in names:
        lid = train_mod.KEYPOINTS_BY_NAME[name]['id']
        row = name_to_row[name]
        k = float(sigmas_full[lid])
        m_vals = []
        v_vals = []
        for (pred, gt, scale), item in zip(model_preds, vit_preds):
            if gt[lid, 2] <= 0:
                continue
            m_vals.append(
                compute_oks(pred[lid : lid + 1, :2], gt[lid : lid + 1], np.array([k]), scale)
            )
            vpred = item['pred_xy'][row : row + 1]
            if np.isnan(vpred).any():
                continue
            vgt = item['keypoints'][lid : lid + 1]
            v_scale = cmp.person_scale_image(item['gt_bbox'])
            v_vals.append(compute_oks(vpred, vgt, np.array([k]), v_scale))
        rows[name] = {
            'model_mean_OKS': float(np.mean(m_vals)) if m_vals else float('nan'),
            'vitpose_mean_OKS': float(np.mean(v_vals)) if v_vals else float('nan'),
            'n_model': len(m_vals),
            'n_vitpose': len(v_vals),
        }
    return rows


def run_one(run_dir: Path, checkpoint: Path, function_url: str) -> dict:
    with open(run_dir / 'config.yaml', encoding='utf-8') as f:
        config = train_mod.parse_config(yaml.safe_load(f))
    train_mod.setup_from_config(config)
    train_mod.set_seed(config['training'].seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_samples = train_mod.load_test_samples(config)
    model = train_mod.build_model(config, device)
    sigmas_full = train_mod.OKS_SIGMAS.detach().cpu().numpy()
    batch_size = config['training'].batch_size

    local_ids, vitpose_ids, shared_names = cmp.joint_subset_from_names(
        train_mod.KEYPOINTS_BY_NAME,
        list(cmp.VITPOSE_SHARED_BY_NAME.keys()),
        cmp.VITPOSE_SHARED_BY_NAME,
    )
    name_to_row = {name: idx for idx, name in enumerate(shared_names)}

    print(
        f'\n======== {run_dir.name}/{checkpoint.name} '
        f'ver={config["dataset"].annotation_version} n={len(test_samples)} ========'
    )

    model_sub = {}
    first = True
    for sub_name, names in SUBSETS.items():
        ids = [train_mod.KEYPOINTS_BY_NAME[n]['id'] for n in names]
        model_sub[sub_name] = slim(
            cmp.evaluate_best_pt(
                model,
                checkpoint,
                test_samples,
                device,
                batch_size,
                joint_indices=ids,
                load_weights=first,
            )
        )
        first = False
        m = model_sub[sub_name]
        print(
            f"  model  {sub_name:<10} AP={m['AP']:.3f} AP50={m['AP50']:.3f} "
            f"AP75={m['AP75']:.3f} OKS={m['mean_OKS']:.3f}"
        )

    model_preds = cache_model_preds(model, checkpoint, test_samples, device, batch_size)

    print('  calling VitPose…')
    vit_preds = cmp.collect_vitpose_predictions(
        test_samples=test_samples,
        function_url=function_url,
        vitpose_ids=vitpose_ids,
        timeout=180.0,
        label='exact',
    )

    vit_sub = {}
    for sub_name, names in SUBSETS.items():
        rows = [name_to_row[n] for n in names]
        vit_sub[sub_name] = slim(
            cmp.score_vitpose_predictions(
                vit_preds, local_ids, subset_rows=rows, sigmas_full=sigmas_full
            )
        )
        m = vit_sub[sub_name]
        print(
            f"  vitpose {sub_name:<10} AP={m['AP']:.3f} AP50={m['AP50']:.3f} "
            f"AP75={m['AP75']:.3f} OKS={m['mean_OKS']:.3f}"
        )

    joint_names = list(ANKLES + TOES + HEELS)
    per_joint = per_joint_mean_oks(model_preds, vit_preds, joint_names, name_to_row, sigmas_full)
    print(f"  {'joint':<12} {'model':>7} {'vitpose':>7} {'Δ':>7}")
    for name in joint_names:
        row = per_joint[name]
        d = row['model_mean_OKS'] - row['vitpose_mean_OKS']
        print(
            f"  {name:<12} {row['model_mean_OKS']:7.3f} {row['vitpose_mean_OKS']:7.3f} {d:+7.3f}"
        )

    return {
        'run': str(run_dir),
        'checkpoint': checkpoint.name,
        'annotation_version': config['dataset'].annotation_version,
        'test_samples': len(test_samples),
        'model': model_sub,
        'vitpose': vit_sub,
        'per_joint_mean_OKS': per_joint,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--function-url', required=True)
    parser.add_argument(
        '--jobs',
        nargs='+',
        default=[
            'training/runs/2026_09_27_005:best.pt',
            'training/runs/2026_09_29_006:best.pt',
            'training/runs/2026_10_02_002:best.pt',
            'training/runs/2026_10_02_003:final.pt',
        ],
    )
    parser.add_argument('--out', type=Path, default=Path('training/feet_vs_vitpose_breakdown.json'))
    args = parser.parse_args()

    all_results = []
    for job in args.jobs:
        run_s, ckpt_name = job.split(':')
        run_dir = Path(run_s)
        all_results.append(run_one(run_dir, run_dir / ckpt_name, args.function_url))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(all_results, indent=2))
    print(f'\nWrote {args.out}')


if __name__ == '__main__':
    main()
