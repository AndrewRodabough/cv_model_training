#!/usr/bin/env python3
"""Weight-space programming harness for LLM-based weight editing.

Provides a restricted Python environment where an LLM can:
- Inspect head weights
- Run forward passes on swap examples
- Edit weights directly (no gradients/optimizers)
- Measure improvement

This follows the methodology from Takehi et al. 2026.
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


class WeightEditHarness:
    """Harness for weight-space programming on pose head."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        checkpoint_path: Path,
        swap_cases: list[dict],
        device: torch.device,
        forward_pass_budget: int = 50,
    ):
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.swap_cases = swap_cases
        self.device = device
        self.forward_pass_budget = forward_pass_budget
        self.forward_passes_used = 0
        
        # Load initial weights
        checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.initial_state = {k: v.clone() for k, v in checkpoint_data['model_state_dict'].items()}
        self.model.load_state_dict(self.initial_state)
        self.model.eval()
        
        # Track edits
        self.edit_history = []
        
        # Disable gradients globally
        torch.set_grad_enabled(False)
    
    def get_head_weights(self) -> dict[str, torch.Tensor]:
        """Get current head weights (not backbone)."""
        head_weights = {}
        for name, param in self.model.named_parameters():
            if name.startswith('head.'):
                head_weights[name] = param.detach().cpu()
        return head_weights
    
    def set_head_weights(self, weights: dict[str, torch.Tensor], description: str = ""):
        """Set head weights from a dictionary.
        
        Args:
            weights: Dict mapping parameter names to new values
            description: Human-readable description of the edit
        """
        state_dict = self.model.state_dict()
        
        num_changed = 0
        total_params_changed = 0
        
        for name, new_value in weights.items():
            if name not in state_dict:
                print(f"Warning: {name} not found in model")
                continue
            
            old_value = state_dict[name]
            if old_value.shape != new_value.shape:
                print(f"Warning: Shape mismatch for {name}: {old_value.shape} vs {new_value.shape}")
                continue
            
            # Count changed parameters
            changed_mask = (old_value.cpu() != new_value.cpu())
            params_changed = changed_mask.sum().item()
            total_params_changed += params_changed
            
            state_dict[name] = new_value.to(self.device)
            num_changed += 1
        
        self.model.load_state_dict(state_dict)
        
        # Record edit
        self.edit_history.append({
            'description': description,
            'tensors_modified': num_changed,
            'parameters_changed': total_params_changed,
        })
        
        print(f"Applied edit: {description}")
        print(f"  Modified {num_changed} tensors, {total_params_changed:,} parameters")
        
        return num_changed, total_params_changed
    
    def run_forward_pass(
        self,
        case_indices: list[int] | None = None,
        batch_size: int = 16,
    ) -> list[dict[str, Any]]:
        """Run forward pass on swap cases.
        
        Args:
            case_indices: Specific cases to test (None = all)
            batch_size: Batch size for inference
            
        Returns:
            List of results with predictions and swap scores
        """
        if self.forward_passes_used >= self.forward_pass_budget:
            raise RuntimeError(
                f"Forward pass budget exhausted ({self.forward_pass_budget}). "
                "No more forward passes allowed."
            )
        
        self.forward_passes_used += 1
        print(f"Forward pass {self.forward_passes_used}/{self.forward_pass_budget}")
        
        if case_indices is None:
            cases = self.swap_cases
        else:
            cases = [self.swap_cases[i] for i in case_indices]
        
        results = []
        
        for start in range(0, len(cases), batch_size):
            batch_cases = cases[start:start + batch_size]
            pixels_batch = []
            role_batch = []
            
            for case in batch_cases:
                metadata = case['metadata']
                image_path = Path(metadata['image_path'])
                crop_info = metadata['crop']
                
                image = Image.open(image_path).convert('RGB')
                crop_x1, crop_y1, crop_width, crop_height = crop_info
                crop = image.crop((crop_x1, crop_y1, crop_x1 + crop_width, crop_y1 + crop_height))
                crop = crop.resize(
                    (train_mod.IMAGE_SIZE.x, train_mod.IMAGE_SIZE.y),
                    Image.Resampling.BILINEAR,
                )
                pixels = torch.from_numpy(np.asarray(crop, dtype=np.float32)).permute(2, 0, 1) / 255.0
                pixels = (pixels - train_mod.IMAGE_MEAN) / train_mod.IMAGE_STD
                
                pixels_batch.append(pixels)
                role_batch.append(train_mod.encode_role(metadata.get('role')))
            
            pixels_tensor = torch.stack(pixels_batch).to(self.device)
            role_tensor = torch.stack(role_batch).to(self.device)
            
            with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.device.type == 'cuda'):
                pred_x, pred_y = self.model(
                    pixels_tensor,
                    role=role_tensor if train_mod.USE_ROLE else None,
                )
                coords, _ = decode_simcc(pred_x, pred_y, train_mod.SIMCC_SPLIT_RATIO)
            
            coords_np = coords.cpu().numpy()
            
            for i, (pred_xy, case) in enumerate(zip(coords_np, batch_cases)):
                gt_kps = np.array(case['gt_keypoints'])
                
                # Recompute swap scores with new predictions
                swap_scores = {}
                for pair_name, original_swap_info in case['swaps'].items():
                    # Extract keypoint names
                    left_name, right_name = pair_name.split('_')[:-1], pair_name.split('_')[-1:]
                    left_name = '_'.join(left_name)
                    right_name = left_name.replace('_l', '_r')
                    
                    left_idx = train_mod.KEYPOINTS_BY_NAME[left_name]['id']
                    right_idx = train_mod.KEYPOINTS_BY_NAME[right_name]['id']
                    
                    # Recompute
                    pred_l = pred_xy[left_idx]
                    pred_r = pred_xy[right_idx]
                    gt_l = gt_kps[left_idx, :2]
                    gt_r = gt_kps[right_idx, :2]
                    
                    correct_dist = (
                        np.linalg.norm(pred_l - gt_l) +
                        np.linalg.norm(pred_r - gt_r)
                    )
                    swapped_dist = (
                        np.linalg.norm(pred_l - gt_r) +
                        np.linalg.norm(pred_r - gt_l)
                    )
                    swap_advantage = correct_dist - swapped_dist
                    gt_separation = np.linalg.norm(gt_l - gt_r)
                    confidence = swap_advantage / max(gt_separation, 1.0)
                    
                    swap_scores[pair_name] = {
                        'correct_dist': float(correct_dist),
                        'swapped_dist': float(swapped_dist),
                        'swap_advantage': float(swap_advantage),
                        'confidence': float(confidence),
                        'is_swapped': confidence > 0.1,  # Still swapped
                        'original_confidence': original_swap_info['confidence'],
                        'improvement': original_swap_info['confidence'] - confidence,
                    }
                
                results.append({
                    'case_metadata': case['metadata'],
                    'predictions': pred_xy.tolist(),
                    'swap_scores': swap_scores,
                    'still_swapped': any(s['is_swapped'] for s in swap_scores.values()),
                })
        
        return results
    
    def evaluate_swaps(self, results: list[dict]) -> dict[str, Any]:
        """Evaluate results from forward pass."""
        total_cases = len(results)
        still_swapped = sum(1 for r in results if r['still_swapped'])
        fixed = total_cases - still_swapped
        
        improvements = []
        for result in results:
            for pair_scores in result['swap_scores'].values():
                improvements.append(pair_scores['improvement'])
        
        return {
            'total_cases': total_cases,
            'fixed': fixed,
            'still_swapped': still_swapped,
            'fix_rate': fixed / total_cases if total_cases > 0 else 0.0,
            'avg_improvement': float(np.mean(improvements)) if improvements else 0.0,
            'improvements': improvements,
        }
    
    def save_checkpoint(self, output_path: Path, description: str = ""):
        """Save current model state as a new checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'edit_history': self.edit_history,
            'description': description,
            'forward_passes_used': self.forward_passes_used,
        }
        torch.save(checkpoint, output_path)
        print(f"Saved edited checkpoint to: {output_path}")
    
    def get_stats(self) -> dict:
        """Get current harness statistics."""
        total_head_params = sum(
            p.numel() for name, p in self.model.named_parameters()
            if name.startswith('head.')
        )
        
        total_edited_params = sum(e['parameters_changed'] for e in self.edit_history)
        
        return {
            'forward_passes_used': self.forward_passes_used,
            'forward_passes_remaining': self.forward_pass_budget - self.forward_passes_used,
            'num_edits': len(self.edit_history),
            'total_head_parameters': total_head_params,
            'total_edited_parameters': total_edited_params,
            'edit_percentage': 100 * total_edited_params / total_head_params if total_head_params > 0 else 0.0,
            'edit_history': self.edit_history,
        }


def interactive_session(harness: WeightEditHarness):
    """Run an interactive session for manual testing."""
    print("\n" + "=" * 80)
    print("WEIGHT EDITING HARNESS - Interactive Mode")
    print("=" * 80)
    print("\nAvailable commands:")
    print("  stats               - Show harness statistics")
    print("  test [indices]      - Run forward pass on cases (default: first 10)")
    print("  test_all            - Run forward pass on all swap cases")
    print("  save <path>         - Save current checkpoint")
    print("  exit                - Exit interactive mode")
    print()
    
    # Initial test
    print("Running initial evaluation on first 10 swap cases...")
    initial_results = harness.run_forward_pass(case_indices=list(range(min(10, len(harness.swap_cases)))))
    initial_eval = harness.evaluate_swaps(initial_results)
    print(f"\nInitial performance:")
    print(f"  Fix rate: {initial_eval['fix_rate']:.1%}")
    print(f"  Fixed: {initial_eval['fixed']}/{initial_eval['total_cases']}")
    print(f"  Avg improvement: {initial_eval['avg_improvement']:.3f}")
    
    print("\n" + "=" * 80)
    print("The harness is now ready for LLM-based weight editing.")
    print("An LLM (e.g., Claude Opus 5.5) should use the harness methods to:")
    print("1. Inspect weights with harness.get_head_weights()")
    print("2. Analyze swap patterns")
    print("3. Edit weights with harness.set_head_weights()")
    print("4. Test with harness.run_forward_pass()")
    print("5. Iterate until swaps are fixed")
    print("=" * 80)
    
    while True:
        try:
            cmd = input("\n> ").strip()
            
            if not cmd:
                continue
            
            if cmd == 'exit':
                break
            
            elif cmd == 'stats':
                stats = harness.get_stats()
                print(f"\nHarness Statistics:")
                print(f"  Forward passes: {stats['forward_passes_used']}/{harness.forward_pass_budget}")
                print(f"  Edits made: {stats['num_edits']}")
                print(f"  Parameters edited: {stats['total_edited_parameters']:,} / {stats['total_head_parameters']:,}")
                print(f"  Edit percentage: {stats['edit_percentage']:.4f}%")
            
            elif cmd.startswith('test'):
                parts = cmd.split()
                if len(parts) == 1 or parts[1] == 'all':
                    indices = None
                    print("Testing all swap cases...")
                else:
                    indices = [int(x) for x in parts[1].split(',')]
                    print(f"Testing cases: {indices}")
                
                results = harness.run_forward_pass(case_indices=indices)
                eval_result = harness.evaluate_swaps(results)
                print(f"\nResults:")
                print(f"  Fix rate: {eval_result['fix_rate']:.1%}")
                print(f"  Fixed: {eval_result['fixed']}/{eval_result['total_cases']}")
                print(f"  Avg improvement: {eval_result['avg_improvement']:.3f}")
            
            elif cmd.startswith('save'):
                parts = cmd.split(maxsplit=1)
                if len(parts) < 2:
                    print("Usage: save <path>")
                else:
                    harness.save_checkpoint(Path(parts[1]), description="Interactive save")
            
            else:
                print(f"Unknown command: {cmd}")
        
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Weight-space programming harness')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=str, default='best.pt')
    parser.add_argument('--swap-data', type=Path, required=True, 
                       help='JSON file with L/R swap cases')
    parser.add_argument('--forward-pass-budget', type=int, default=50,
                       help='Maximum number of forward passes allowed')
    parser.add_argument('--interactive', action='store_true',
                       help='Run interactive session for manual testing')
    parser.add_argument('--test-only', action='store_true',
                       help='Just run initial test and exit')
    args = parser.parse_args()
    
    # Load config
    config_path = args.run_dir / 'config.yaml'
    with open(config_path, encoding='utf-8') as f:
        config = train_mod.parse_config(yaml.safe_load(f))
    
    train_mod.setup_from_config(config)
    train_mod.set_seed(config['training'].seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = train_mod.build_model(config, device)
    checkpoint_path = args.run_dir / args.checkpoint
    
    # Load swap data
    with open(args.swap_data) as f:
        swap_data = json.load(f)
    swap_cases = swap_data['swap_cases']
    
    print(f"\nInitializing harness:")
    print(f"  Model: {args.run_dir.name}/{args.checkpoint}")
    print(f"  Swap cases: {len(swap_cases)}")
    print(f"  Forward pass budget: {args.forward_pass_budget}")
    
    # Create harness
    harness = WeightEditHarness(
        model=model,
        checkpoint_path=checkpoint_path,
        swap_cases=swap_cases,
        device=device,
        forward_pass_budget=args.forward_pass_budget,
    )
    
    if args.test_only:
        # Just run a quick test
        print("\nRunning test on first 20 swap cases...")
        results = harness.run_forward_pass(case_indices=list(range(min(20, len(swap_cases)))))
        eval_result = harness.evaluate_swaps(results)
        print(f"\nBaseline performance:")
        print(f"  Fix rate: {eval_result['fix_rate']:.1%}")
        print(f"  Fixed: {eval_result['fixed']}/{eval_result['total_cases']}")
        print(f"  Still swapped: {eval_result['still_swapped']}")
        print(f"  Avg improvement needed: {-eval_result['avg_improvement']:.3f}")
        return
    
    if args.interactive:
        interactive_session(harness)
    else:
        print("\nHarness initialized. Use --interactive for manual testing.")
        print("For LLM-based editing, import this module and use the WeightEditHarness class.")


if __name__ == '__main__':
    main()
