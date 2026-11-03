#!/usr/bin/env python3
"""Extract L/R swap cases from test set for weight-space programming.

Identifies instances where the model confidently swaps left and right keypoints,
providing concrete examples for LLM-based weight editing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

import src.training.train as train_mod
from src.model_review.metrics import decode_simcc


# Keypoint pairs that should maintain left/right distinction
LR_KEYPOINT_PAIRS = [
    ('ankle_l', 'ankle_r'),
    ('heel_l', 'heel_r'),
    ('toe_b_l', 'toe_b_r'),
    ('toe_s_l', 'toe_s_r'),
    ('knee_l', 'knee_r'),
    ('hip_l', 'hip_r'),
]


def compute_swap_score(pred_xy: np.ndarray, gt_keypoints: np.ndarray, left_idx: int, right_idx: int) -> dict[str, float]:
    """Compute how much a prediction looks like a L/R swap.
    
    Returns dict with:
    - correct_dist: L2 distance for correct assignment (pred_L to GT_L, pred_R to GT_R)
    - swapped_dist: L2 distance for swapped assignment (pred_L to GT_R, pred_R to GT_L)
    - swap_advantage: how much better the swap fits (positive means swap is better)
    - confidence: how clear the swap is (higher = more confident swap)
    """
    # Check visibility
    if gt_keypoints[left_idx, 2] <= 0 or gt_keypoints[right_idx, 2] <= 0:
        return {'valid': False}
    
    pred_l = pred_xy[left_idx]
    pred_r = pred_xy[right_idx]
    gt_l = gt_keypoints[left_idx, :2]
    gt_r = gt_keypoints[right_idx, :2]
    
    # Correct assignment distances
    correct_dist = (
        np.linalg.norm(pred_l - gt_l) +
        np.linalg.norm(pred_r - gt_r)
    )
    
    # Swapped assignment distances
    swapped_dist = (
        np.linalg.norm(pred_l - gt_r) +
        np.linalg.norm(pred_r - gt_l)
    )
    
    swap_advantage = correct_dist - swapped_dist
    
    # Confidence: how much better is the swap (normalized by separation)
    gt_separation = np.linalg.norm(gt_l - gt_r)
    confidence = swap_advantage / max(gt_separation, 1.0)
    
    return {
        'valid': True,
        'correct_dist': float(correct_dist),
        'swapped_dist': float(swapped_dist),
        'swap_advantage': float(swap_advantage),
        'confidence': float(confidence),
        'gt_separation': float(gt_separation),
    }


def extract_swaps(
    model: torch.nn.Module,
    checkpoint_path: Path,
    test_samples: list,
    device: torch.device,
    batch_size: int,
    confidence_threshold: float = 0.15,
) -> list[dict[str, Any]]:
    """Run inference and extract cases where L/R swaps occur."""
    
    # Load checkpoint
    checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint_data['model_state_dict'])
    model.eval()
    
    swap_cases = []
    
    print(f"Processing {len(test_samples)} test samples...")
    
    for start in range(0, len(test_samples), batch_size):
        batch = test_samples[start:start + batch_size]
        pixels_batch = []
        kps_batch = []
        role_batch = []
        metadata_batch = []
        
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
            metadata_batch.append({
                'image_path': str(image_path),
                'bbox': bbox.tolist(),
                'crop': [crop_x1, crop_y1, crop_width, crop_height],
                'role': role,
            })
        
        pixels_tensor = torch.stack(pixels_batch).to(device)
        role_tensor = torch.stack(role_batch).to(device)
        
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            pred_x, pred_y = model(
                pixels_tensor,
                role=role_tensor if train_mod.USE_ROLE else None,
            )
            coords, _ = decode_simcc(pred_x, pred_y, train_mod.SIMCC_SPLIT_RATIO)
        
        coords_np = coords.cpu().numpy()
        kps_np = torch.stack(kps_batch).numpy()
        
        # Analyze each sample for swaps
        for i, (pred_xy, gt_kps, metadata) in enumerate(zip(coords_np, kps_np, metadata_batch)):
            sample_swaps = {}
            has_significant_swap = False
            
            for pair_name, (left_name, right_name) in [(name, pair) for name, pair in LR_KEYPOINT_PAIRS]:
                left_idx = train_mod.KEYPOINTS_BY_NAME[left_name]['id']
                right_idx = train_mod.KEYPOINTS_BY_NAME[right_name]['id']
                
                swap_info = compute_swap_score(pred_xy, gt_kps, left_idx, right_idx)
                
                if swap_info.get('valid') and swap_info['confidence'] > confidence_threshold:
                    sample_swaps[f"{left_name}_{right_name}"] = swap_info
                    has_significant_swap = True
            
            if has_significant_swap:
                swap_cases.append({
                    'metadata': metadata,
                    'gt_keypoints': gt_kps.tolist(),
                    'pred_keypoints': pred_xy.tolist(),
                    'swaps': sample_swaps,
                    'max_confidence': max(s['confidence'] for s in sample_swaps.values()),
                })
        
        if (start // batch_size + 1) % 10 == 0:
            print(f"  Processed {start + len(batch)}/{len(test_samples)}, found {len(swap_cases)} swap cases so far")
    
    # Sort by confidence (most confident swaps first)
    swap_cases.sort(key=lambda x: x['max_confidence'], reverse=True)
    
    return swap_cases


def main() -> None:
    parser = argparse.ArgumentParser(description='Extract L/R swap cases for weight-space programming')
    parser.add_argument('--run-dir', type=Path, required=True, help='Training run directory')
    parser.add_argument('--checkpoint', type=str, default='best.pt', help='Checkpoint filename')
    parser.add_argument('--output', type=Path, default=Path('weight_space_data/lr_swaps.json'))
    parser.add_argument('--confidence-threshold', type=float, default=0.15,
                        help='Minimum confidence for swap detection (default: 0.15)')
    parser.add_argument('--max-cases', type=int, default=200,
                        help='Maximum number of swap cases to save (default: 200)')
    args = parser.parse_args()
    
    # Load config
    config_path = args.run_dir / 'config.yaml'
    with open(config_path, encoding='utf-8') as f:
        config = train_mod.parse_config(yaml.safe_load(f))
    
    train_mod.setup_from_config(config)
    train_mod.set_seed(config['training'].seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and test samples
    test_samples = train_mod.load_test_samples(config)
    model = train_mod.build_model(config, device)
    checkpoint_path = args.run_dir / args.checkpoint
    
    print(f"\nExtracting L/R swaps from {args.run_dir.name}/{args.checkpoint}")
    print(f"Test set size: {len(test_samples)}")
    print(f"Confidence threshold: {args.confidence_threshold}")
    
    # Extract swaps
    swap_cases = extract_swaps(
        model,
        checkpoint_path,
        test_samples,
        device,
        config['training'].batch_size,
        args.confidence_threshold,
    )
    
    # Limit to max cases
    if len(swap_cases) > args.max_cases:
        print(f"\nLimiting to top {args.max_cases} most confident swap cases")
        swap_cases = swap_cases[:args.max_cases]
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'run_dir': str(args.run_dir),
        'checkpoint': args.checkpoint,
        'config': {
            'annotation_version': config['dataset'].annotation_version,
            'test_samples': len(test_samples),
            'confidence_threshold': args.confidence_threshold,
        },
        'swap_cases': swap_cases,
        'summary': {
            'total_cases': len(swap_cases),
            'avg_confidence': float(np.mean([c['max_confidence'] for c in swap_cases])) if swap_cases else 0.0,
            'pairs_affected': {},
        }
    }
    
    # Count how many times each pair is swapped
    for case in swap_cases:
        for pair_name in case['swaps'].keys():
            output_data['summary']['pairs_affected'][pair_name] = \
                output_data['summary']['pairs_affected'].get(pair_name, 0) + 1
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Extraction complete!")
    print(f"{'='*60}")
    print(f"Total swap cases found: {len(swap_cases)}")
    print(f"Average confidence: {output_data['summary']['avg_confidence']:.3f}")
    print(f"\nSwap frequency by pair:")
    for pair_name, count in sorted(output_data['summary']['pairs_affected'].items(), 
                                     key=lambda x: x[1], reverse=True):
        print(f"  {pair_name}: {count} cases")
    print(f"\nSaved to: {args.output}")


if __name__ == '__main__':
    main()
