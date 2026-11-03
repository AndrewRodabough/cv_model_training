#!/usr/bin/env python3
"""Analyze pose head weights for weight-space programming.

Extracts and summarizes the head weights to help LLMs understand the architecture
and identify potential sources of L/R confusion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import src.training.train as train_mod


def extract_head_weights(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Extract just the head weights (not backbone)."""
    head_weights = {}
    
    for name, param in model.named_parameters():
        if name.startswith('head.'):
            head_weights[name] = param.detach().cpu()
    
    return head_weights


def analyze_attention_patterns(
    head_weights: dict[str, torch.Tensor],
    left_right_pairs: list[tuple[int, int]],
) -> dict[str, Any]:
    """Analyze attention-related weights for L/R symmetry."""
    
    analysis = {
        'has_self_attention': False,
        'has_cross_attention': False,
        'has_role_conditioning': False,
        'attention_heads': [],
    }
    
    # Check for attention components
    for name in head_weights.keys():
        if 'self_attn' in name:
            analysis['has_self_attention'] = True
        if 'cross_attn' in name:
            analysis['has_cross_attention'] = True
        if 'role_' in name or 'film' in name:
            analysis['has_role_conditioning'] = True
    
    # Analyze query embeddings if they exist
    query_embed = head_weights.get('head.query_embed')
    if query_embed is not None:
        # query_embed shape: [num_joints, dim]
        num_joints, dim = query_embed.shape
        
        # Check L/R pair similarity in query space
        pair_similarities = []
        for left_idx, right_idx in left_right_pairs:
            if left_idx < num_joints and right_idx < num_joints:
                left_query = query_embed[left_idx]
                right_query = query_embed[right_idx]
                similarity = torch.cosine_similarity(
                    left_query.unsqueeze(0),
                    right_query.unsqueeze(0)
                ).item()
                pair_similarities.append({
                    'left_idx': left_idx,
                    'right_idx': right_idx,
                    'cosine_sim': float(similarity),
                })
        
        analysis['query_embed'] = {
            'shape': list(query_embed.shape),
            'pair_similarities': pair_similarities,
            'avg_pair_similarity': float(np.mean([p['cosine_sim'] for p in pair_similarities])) if pair_similarities else None,
        }
    
    # Analyze pair bias if present (shared skeleton structure)
    pair_bias = head_weights.get('head.pair_bias')
    if pair_bias is not None:
        analysis['pair_bias'] = {
            'shape': list(pair_bias.shape),
            'mean': float(pair_bias.mean()),
            'std': float(pair_bias.std()),
            'has_lr_structure': True,  # If present, it encodes skeleton structure
        }
    
    return analysis


def compute_weight_statistics(head_weights: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Compute overall statistics about head weights."""
    
    stats = {
        'total_parameters': 0,
        'parameters_by_component': {},
        'layer_info': [],
    }
    
    # Group by component
    components = {}
    for name, weight in head_weights.items():
        # Extract component name (e.g., 'head.decoder.0.self_attn')
        parts = name.replace('head.', '').split('.')
        if len(parts) > 1:
            component = parts[0]
        else:
            component = 'other'
        
        if component not in components:
            components[component] = []
        components[component].append((name, weight))
    
    # Compute stats per component
    for component, weights in components.items():
        total_params = sum(w.numel() for _, w in weights)
        stats['parameters_by_component'][component] = {
            'count': total_params,
            'tensors': len(weights),
        }
        stats['total_parameters'] += total_params
    
    # Identify decoder layers
    decoder_layers = {}
    for name, weight in head_weights.items():
        if 'decoder' in name or 'layers' in name:
            # Try to extract layer number
            parts = name.split('.')
            for i, part in enumerate(parts):
                if part.isdigit():
                    layer_num = int(part)
                    if layer_num not in decoder_layers:
                        decoder_layers[layer_num] = []
                    decoder_layers[layer_num].append((name, weight))
                    break
    
    for layer_num in sorted(decoder_layers.keys()):
        layer_weights = decoder_layers[layer_num]
        layer_params = sum(w.numel() for _, w in layer_weights)
        stats['layer_info'].append({
            'layer': layer_num,
            'parameters': layer_params,
            'components': list(set(name.split('.')[2] if len(name.split('.')) > 2 else 'unknown' 
                                 for name, _ in layer_weights)),
        })
    
    return stats


def generate_weight_summary(
    head_weights: dict[str, torch.Tensor],
    left_right_pairs: list[tuple[int, int]],
    keypoints_by_name: dict,
) -> str:
    """Generate a human-readable summary of head weights for LLM."""
    
    lines = []
    lines.append("=" * 80)
    lines.append("POSE HEAD WEIGHT SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    
    # Overall stats
    stats = compute_weight_statistics(head_weights)
    lines.append(f"Total parameters: {stats['total_parameters']:,}")
    lines.append("")
    lines.append("Parameters by component:")
    for component, info in sorted(stats['parameters_by_component'].items(), 
                                   key=lambda x: x[1]['count'], reverse=True):
        pct = 100 * info['count'] / stats['total_parameters']
        lines.append(f"  {component:20} {info['count']:10,} params ({pct:5.1f}%)")
    lines.append("")
    
    # Layer info
    if stats['layer_info']:
        lines.append("Decoder layers:")
        for layer_info in stats['layer_info']:
            lines.append(f"  Layer {layer_info['layer']}: {layer_info['parameters']:,} params")
            lines.append(f"    Components: {', '.join(layer_info['components'])}")
        lines.append("")
    
    # L/R structure
    lines.append("Left-Right Keypoint Pairs:")
    for left_idx, right_idx in left_right_pairs:
        left_names = [n for n, info in keypoints_by_name.items() if info['id'] == left_idx]
        right_names = [n for n, info in keypoints_by_name.items() if info['id'] == right_idx]
        left_name = left_names[0] if left_names else f"joint_{left_idx}"
        right_name = right_names[0] if right_names else f"joint_{right_idx}"
        lines.append(f"  {left_name:15} ({left_idx:2d}) <-> {right_name:15} ({right_idx:2d})")
    lines.append("")
    
    # Attention analysis
    attn_analysis = analyze_attention_patterns(head_weights, left_right_pairs)
    lines.append("Architecture features:")
    lines.append(f"  Self-attention: {'Yes' if attn_analysis['has_self_attention'] else 'No'}")
    lines.append(f"  Cross-attention: {'Yes' if attn_analysis['has_cross_attention'] else 'No'}")
    lines.append(f"  Role conditioning: {'Yes' if attn_analysis['has_role_conditioning'] else 'No'}")
    lines.append("")
    
    if 'query_embed' in attn_analysis:
        qe = attn_analysis['query_embed']
        lines.append(f"Query embeddings: shape {qe['shape']}")
        if qe['avg_pair_similarity'] is not None:
            lines.append(f"  Avg L/R pair cosine similarity: {qe['avg_pair_similarity']:.3f}")
            lines.append("  (High similarity = queries are very similar for L/R pairs)")
            lines.append("  (This could make L/R swaps more likely)")
            lines.append("")
            lines.append("  Individual pair similarities:")
            for pair in qe['pair_similarities']:
                left_names = [n for n, info in keypoints_by_name.items() if info['id'] == pair['left_idx']]
                right_names = [n for n, info in keypoints_by_name.items() if info['id'] == pair['right_idx']]
                left_name = left_names[0] if left_names else f"joint_{pair['left_idx']}"
                right_name = right_names[0] if right_names else f"joint_{pair['right_idx']}"
                lines.append(f"    {left_name} <-> {right_name}: {pair['cosine_sim']:.3f}")
        lines.append("")
    
    if 'pair_bias' in attn_analysis:
        pb = attn_analysis['pair_bias']
        lines.append(f"Pair bias (skeleton structure): shape {pb['shape']}")
        lines.append(f"  Mean: {pb['mean']:.4f}, Std: {pb['std']:.4f}")
        lines.append("  (Encodes which joints are connected)")
        lines.append("")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description='Analyze pose head weights')
    parser.add_argument('--run-dir', type=Path, required=True, help='Training run directory')
    parser.add_argument('--checkpoint', type=str, default='best.pt', help='Checkpoint filename')
    parser.add_argument('--output', type=Path, default=Path('weight_space_data/head_analysis'))
    parser.add_argument('--swap-data', type=Path, help='Optional: L/R swap data for context')
    args = parser.parse_args()
    
    # Load config
    config_path = args.run_dir / 'config.yaml'
    with open(config_path, encoding='utf-8') as f:
        config = train_mod.parse_config(yaml.safe_load(f))
    
    train_mod.setup_from_config(config)
    
    device = torch.device('cpu')  # Analysis doesn't need GPU
    model = train_mod.build_model(config, device)
    
    # Load checkpoint
    checkpoint_path = args.run_dir / args.checkpoint
    checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint_data['model_state_dict'])
    
    print(f"Analyzing {args.run_dir.name}/{args.checkpoint}")
    
    # Extract head weights
    head_weights = extract_head_weights(model)
    print(f"Extracted {len(head_weights)} weight tensors from head")
    
    # Generate analysis
    stats = compute_weight_statistics(head_weights)
    attn_analysis = analyze_attention_patterns(head_weights, train_mod.LEFT_RIGHT_PAIRS)
    
    # Generate summary
    summary_text = generate_weight_summary(
        head_weights,
        train_mod.LEFT_RIGHT_PAIRS,
        train_mod.KEYPOINTS_BY_NAME,
    )
    
    # Save outputs
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Save weights as tensors
    weights_path = args.output / 'head_weights.pt'
    torch.save(head_weights, weights_path)
    print(f"Saved head weights to: {weights_path}")
    
    # Save analysis JSON
    analysis_data = {
        'run_dir': str(args.run_dir),
        'checkpoint': args.checkpoint,
        'statistics': stats,
        'attention_analysis': attn_analysis,
        'left_right_pairs': [
            {'left': l, 'right': r} for l, r in train_mod.LEFT_RIGHT_PAIRS
        ],
    }
    
    analysis_path = args.output / 'analysis.json'
    with open(analysis_path, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    print(f"Saved analysis to: {analysis_path}")
    
    # Save human-readable summary
    summary_path = args.output / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    print(f"Saved summary to: {summary_path}")
    
    # Print summary
    print("\n" + summary_text)
    
    # If swap data provided, add context
    if args.swap_data and args.swap_data.exists():
        with open(args.swap_data) as f:
            swap_data = json.load(f)
        
        print("\n" + "=" * 80)
        print("SWAP FAILURE CONTEXT")
        print("=" * 80)
        print(f"Total swap cases: {swap_data['summary']['total_cases']}")
        print(f"Avg confidence: {swap_data['summary']['avg_confidence']:.3f}")
        print("\nMost commonly swapped pairs:")
        for pair_name, count in sorted(swap_data['summary']['pairs_affected'].items(),
                                       key=lambda x: x[1], reverse=True)[:5]:
            pct = 100 * count / len(swap_data['swap_cases'])
            print(f"  {pair_name}: {count} cases ({pct:.1f}%)")


if __name__ == '__main__':
    main()
