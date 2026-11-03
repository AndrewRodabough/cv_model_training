# Weight-Space Programming for L/R Swap Fixes

This directory contains infrastructure for applying weight-space programming (following [Takehi et al. 2026](https://rikiyat.github.io/blog/weight-space-programming/)) to fix left-right leg/foot swap errors in the pose estimation head.

## Overview

The pose model has a persistent issue where it swaps left and right legs/feet (ankles, heels, toes). Traditional approaches would require:
- More labeled data
- Retraining with higher `lr_swap_loss_weight`
- Architecture changes

Weight-space programming offers a different approach: directly analyze and edit the model head weights to fix the specific failure mode.

## Why This Might Work

1. **Specific failure mode**: L/R leg/foot swaps - exactly the kind of identifiable issue the paper showed success with (like fixing reasoning loops in Qwen3.5-0.8B)
2. **Small head**: The SimCC head (~10-50M parameters) is much smaller than the 0.8B parameter model they successfully edited
3. **Existing structure**: The model already has `left_right_pairs` defined and `lr_swap_loss` infrastructure - the structure is there, the weights just haven't learned it properly
4. **Concrete examples**: We can provide specific instances where the model makes the swap mistake
5. **Frozen backbone**: Only targeting the head (not DINOv3 backbone) keeps the weight space constrained

## Pipeline

### 1. Extract L/R Swap Cases (`extract_lr_swaps.py`)

Runs inference on the test set and identifies cases where the model swaps left/right keypoints:
- Detects when predicted left is closer to GT right (and vice versa)
- Filters for high-confidence swaps (not just ambiguous cases)
- Saves examples with images, GT, predictions, and swap metrics

```bash
python -m src.weight_space.extract_lr_swaps \
  --run-dir training/runs/2026_10_02_003 \
  --checkpoint final.pt \
  --output weight_space_data/lr_swaps.json
```

### 2. Analyze Head Weights (`analyze_head.py`)

Extracts and analyzes the head weights to help understand the architecture:
- Separates head weights from backbone
- Computes parameter counts per component
- Identifies attention patterns related to L/R pairs
- Creates weight summaries for LLM analysis

```bash
python -m src.weight_space.analyze_head \
  --run-dir training/runs/2026_10_02_003 \
  --checkpoint final.pt \
  --swap-data weight_space_data/lr_swaps.json \
  --output weight_space_data/head_analysis
```

### 3. Weight-Space Programming Harness (`weight_edit_harness.py`)

A Python harness that an LLM (Claude Opus 5.5 recommended) can use to:
- Load and inspect head weights
- Run forward passes on swap examples
- Edit specific weight tensors
- Test changes and measure improvement

The harness restricts:
- No gradient computation or backprop
- No optimizer usage
- Direct weight manipulation only
- Limited forward passes (budget-based like the paper)

```bash
# Interactive mode for LLM use
python -m src.weight_space.weight_edit_harness \
  --run-dir training/runs/2026_10_02_003 \
  --checkpoint final.pt \
  --swap-data weight_space_data/lr_swaps.json \
  --forward-pass-budget 50 \
  --interactive
```

### 4. Prompt Template (`llm_prompt.md`)

A carefully structured prompt for the LLM that includes:
- Problem description (L/R swaps)
- Architecture details
- Example failures
- Available operations
- Success criteria

## Expected Workflow

1. **Extract swaps**: Identify 50-200 clear L/R swap cases from test set
2. **Analyze weights**: Generate human-readable summaries of head architecture
3. **LLM editing session**: 
   - Provide LLM with swap examples and head weights
   - LLM analyzes attention patterns and weight structures
   - LLM hypothesizes root cause (e.g., certain attention heads confusing L/R)
   - LLM edits weights (e.g., adjusting attention biases for `left_right_pairs`)
   - Test on held-out swap cases
4. **Validation**: Run full evaluation to ensure no regression on non-swap cases

## Results to Expect

Based on the paper's Qwen3.5-0.8B math improvement (7.9% → 48.1% by editing 0.016% of weights):
- We might fix 30-80% of clear swap cases
- Edits likely affect <1% of head weights
- Should preserve (or improve) overall AP on non-problematic cases
- Changes might be in attention biases, role conditioning, or query initialization

## Hardware Requirements

- **Extraction**: Light (CPU ok, ~10-30 min on test set)
- **Analysis**: Light (CPU ok, <5 min)
- **LLM Harness**: 
  - Need GPU for forward passes (CUDA recommended)
  - LLM reasoning runs on API (Claude Opus 5.5)
  - Budget 1-2 hours for 25-50 forward pass iterations

## Notes

- Start with a good checkpoint (not early in training)
- The backbone stays frozen - we only edit the head
- Each edit iteration should be small and targeted
- Document all changes for potential paper/blog post
- If this works, it's highly publishable (novel application domain)
