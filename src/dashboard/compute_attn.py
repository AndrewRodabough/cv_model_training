"""On-demand joint-query cross-attn (torch / dino env).

Examples::

    PYTHONPATH=. python -m src.dashboard.compute_attn \\
      --run-dir runs/pose_estimator/2026_10_25_010 \\
      --checkpoint best \\
      --stem task_45_frame_000384 \\
      --joint ankle_l \\
      --out /tmp/attn_ankle_l.jpg

    # Backfill worst+watchlist attn npz for an existing run checkpoint
    PYTHONPATH=. python -m src.dashboard.compute_attn \\
      --run-dir runs/pose_estimator/2026_10_25_008 \\
      --checkpoint best --backfill
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_run_bundle(run_dir: Path):
    """Build model + globals from a finished run's config.yaml."""
    # Late imports so dashboard venv can import this module's helpers without torch
    # when only rendering; this CLI always runs under TRAIN_PYTHON.
    from src.training import train as train_mod
    from src.util.dir import POSE_ESTIMATOR_RUN_DIR

    run_dir = Path(run_dir)
    if not run_dir.is_absolute():
        run_dir = (REPO_ROOT / run_dir).resolve()
    cfg_path = run_dir / 'config.yaml'
    if not cfg_path.is_file():
        raise FileNotFoundError(f'Missing {cfg_path}')

    raw = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    config = train_mod.parse_config(raw)
    train_mod.setup_from_config(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = train_mod.build_model(config, device)
    return train_mod, config, model, device, run_dir


def _stem_meta_from_val_preds(train_mod, run_dir: Path, stem: str):
    from src.dashboard.sample_id import find_sample_index, parse_sample_choice
    from src.training import run_artifacts

    del train_mod  # lookup is from val_preds only
    epochs = run_artifacts.list_val_pred_epochs(run_dir)
    if not epochs:
        raise FileNotFoundError(f'No val_preds under {run_dir}')
    data = run_artifacts.load_val_preds_npz(run_dir, epochs[-1])
    if data is None:
        raise FileNotFoundError('Failed to load latest val_preds')
    idx = find_sample_index(data, stem)
    if idx is None:
        plain, role, _ = parse_sample_choice(stem)
        raise KeyError(
            f'{stem!r} not in latest val_preds '
            f'(looked up stem={plain!r} role={role!r}; {len(data["image_stem"])} rows)'
        )
    role_raw = str(data['role'][idx]) if 'role' in data else None
    return {
        'image_path': str(data['image_path'][idx]),
        'bbox': data['bbox'][idx].tolist(),
        'role': None if role_raw in (None, '', 'None') else role_raw,
    }


def _forward_attn(train_mod, model, device, image_path, bbox, role):
    image = Image.open(image_path).convert('RGB')
    crop_x1, crop_y1, crop_width, crop_height = train_mod.compute_padded_crop(
        bbox, image.width, image.height
    )
    crop = image.crop((crop_x1, crop_y1, crop_x1 + crop_width, crop_y1 + crop_height))
    crop = crop.resize(
        (train_mod.IMAGE_SIZE.x, train_mod.IMAGE_SIZE.y), Image.Resampling.BILINEAR
    )
    pixels = torch.from_numpy(np.asarray(crop, dtype=np.float32)).permute(2, 0, 1) / 255.0
    pixels = ((pixels - train_mod.IMAGE_MEAN) / train_mod.IMAGE_STD).unsqueeze(0).to(device)
    role_tensor = train_mod.encode_role(role).unsqueeze(0).to(device)
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    spatial = train_mod.crop_normalized_bbox_center(
        cx, cy, crop_x1, crop_y1, crop_width, crop_height
    ).unsqueeze(0).to(device)

    model.eval()
    with torch.inference_mode():
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            out = model(
                pixels,
                role=role_tensor if train_mod.USE_ROLE else None,
                spatial=spatial if train_mod.USE_SPATIAL_PROMPT else None,
                return_cross_attn=True,
            )
    if len(out) == 4:
        pred_x, pred_y, _aux, cross_w = out
    else:
        pred_x, pred_y, cross_w = out
    grid_w = int(model.grid_size.x)
    grid_h = int(model.grid_size.y)
    attn = cross_w[0].detach().float().cpu().numpy().reshape(-1, grid_h, grid_w)
    pred_xy = (
        torch.stack(
            (
                pred_x.argmax(dim=-1).float() / train_mod.SIMCC_SPLIT_RATIO,
                pred_y.argmax(dim=-1).float() / train_mod.SIMCC_SPLIT_RATIO,
            ),
            dim=-1,
        )[0]
        .cpu()
        .numpy()
    )
    joint_names = [
        name
        for name, _ in sorted(
            train_mod.KEYPOINTS_BY_NAME.items(), key=lambda item: item[1]['id']
        )
    ]
    return attn, pred_xy, joint_names, (grid_w, grid_h)


def compute_one(
    run_dir: Path,
    checkpoint_name: str,
    stem: str,
    *,
    save_npz: bool = True,
    out_image: Path | None = None,
    joint: str | None = None,
) -> dict:
    from src.training import run_artifacts
    from src.dashboard.attn_overlay import attn_joint_index, render_attn_on_crop

    train_mod, config, model, device, run_dir = _load_run_bundle(run_dir)
    ckpt = run_dir / f'{checkpoint_name}.pt'
    if not ckpt.is_file():
        # allow best/final nested? usually best.pt at run root
        raise FileNotFoundError(f'Missing checkpoint {ckpt}')
    bundle = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(bundle['model_state_dict'])

    from src.dashboard.sample_id import attn_cache_key

    meta = _stem_meta_from_val_preds(train_mod, run_dir, stem)
    role = meta['role'] if meta['role'] not in ('', 'None', None) else None
    cache_stem = attn_cache_key(stem)
    attn, pred_xy, joint_names, grid_wh = _forward_attn(
        train_mod, model, device, meta['image_path'], meta['bbox'], role
    )

    npz_path = None
    if save_npz:
        # Cache under ondemand/ or the requested checkpoint folder
        cache_name = checkpoint_name if checkpoint_name in ('best', 'final') else 'ondemand'
        npz_path = run_artifacts.save_attn_npz(
            run_dir,
            cache_name,
            stem=cache_stem,
            attn_khw=attn,
            joint_names=joint_names,
            image_path=meta['image_path'],
            bbox=meta['bbox'],
            pred_xy=pred_xy,
            grid_wh=grid_wh,
        )

    image_path = None
    if out_image is not None and joint:
        jid = attn_joint_index(joint_names, joint)
        image_cfg = config['image']
        img = render_attn_on_crop(
            meta['image_path'],
            meta['bbox'],
            attn[jid],
            image_size=(int(image_cfg.width), int(image_cfg.height)),
            bbox_padding=float(image_cfg.bbox_padding),
            title=f'{joint}  ({checkpoint_name})',
        )
        out_image = Path(out_image)
        out_image.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_image, format='JPEG', quality=90)
        image_path = str(out_image)

    return {
        'npz': str(npz_path) if npz_path else None,
        'image': image_path,
        'joints': joint_names,
        'grid_wh': list(grid_wh),
    }


def backfill(run_dir: Path, checkpoint_name: str, top_n: int = 15) -> int:
    train_mod, config, model, device, run_dir = _load_run_bundle(run_dir)
    ckpt = run_dir / f'{checkpoint_name}.pt'
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    # Rebuild test split samples the same way training did.
    test_samples = train_mod.load_test_samples(config)
    watch = list(
        getattr(config['training'], 'watchlist_frames', None) or []
    )
    paths = train_mod.export_query_attn_maps(
        model=model,
        checkpoint_path=ckpt,
        test_samples=test_samples,
        run_dir=run_dir,
        checkpoint_name=checkpoint_name,
        top_n=top_n,
        watchlist_frames=watch,
        device=device,
        batch_size=int(config['training'].batch_size),
    )
    return len(paths)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--checkpoint', default='best', choices=['best', 'final'])
    parser.add_argument('--stem', default=None)
    parser.add_argument('--joint', default=None)
    parser.add_argument('--out', default=None, help='Optional JPEG heatmap path')
    parser.add_argument(
        '--backfill',
        action='store_true',
        help='Export worst+watchlist attn npz for this checkpoint',
    )
    parser.add_argument('--top-n', type=int, default=15)
    args = parser.parse_args(argv)

    if args.backfill:
        n = backfill(Path(args.run_dir), args.checkpoint, top_n=args.top_n)
        print(f'Wrote {n} attn npz files')
        return 0

    if not args.stem:
        print('--stem is required unless --backfill', file=sys.stderr)
        return 2

    result = compute_one(
        Path(args.run_dir),
        args.checkpoint,
        args.stem,
        save_npz=True,
        out_image=Path(args.out) if args.out else None,
        joint=args.joint,
    )
    print(result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
