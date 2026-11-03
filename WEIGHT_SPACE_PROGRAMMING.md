# Weight-Space Programming for L/R Swap Fixes

This repository now includes infrastructure for **weight-space programming** - a novel technique from [Takehi et al. 2026](https://rikiyat.github.io/blog/weight-space-programming/) where LLMs directly read and edit neural network weights to fix specific failures.

## Quick Start

### 1. Run the Automated Pipeline

```bash
# Extract swap cases, analyze weights, and test baseline
./src/weight_space/run_pipeline.sh training/runs/2026_10_02_003 best.pt 50
```

This will:
- Identify L/R swap cases in your test set
- Analyze the head architecture and weights
- Measure baseline swap performance
- Save results to `weight_space_data/`

### 2. Review the Analysis

```bash
# Human-readable weight summary
cat weight_space_data/head_analysis/summary.txt

# Swap case details
cat weight_space_data/lr_swaps.json | jq '.summary'
```

### 3. Run Interactive Weight Editing (or provide to LLM)

#### Option A: Interactive Manual Testing
```bash
python -m src.weight_space.weight_edit_harness \
  --run-dir training/runs/2026_10_02_003 \
  --checkpoint best.pt \
  --swap-data weight_space_data/lr_swaps.json \
  --forward-pass-budget 50 \
  --interactive
```

#### Option B: LLM-Based Weight Editing (Recommended)

Use **Claude Opus 5.5** or similar frontier model with the harness:

1. Start a chat with the model
2. Provide the prompt from `src/weight_space/llm_prompt.md`
3. Give access to:
   - `weight_space_data/head_analysis/` (weight analysis)
   - `weight_space_data/lr_swaps.json` (swap examples)
   - `WeightEditHarness` Python class (for editing)

The LLM will:
- Analyze why L/R swaps occur (query embeddings, attention patterns, etc.)
- Design targeted weight edits (e.g., separate query embeddings for L/R pairs)
- Test edits using the forward pass budget
- Iterate until swaps are fixed

## What Problem Does This Solve?

The pose model **swaps left and right legs/feet** in some cases:
- `ankle_l` ↔ `ankle_r`
- `heel_l` ↔ `heel_r`
- `toe_b_l` ↔ `toe_b_r`
- etc.

Traditional solutions require:
- ❌ More labeled data
- ❌ Retraining with different hyperparameters
- ❌ Architecture changes

**Weight-space programming** offers a different approach:
- ✅ Directly edit the head weights to fix the specific failure
- ✅ No retraining, no gradients, no optimizers
- ✅ Surgical fix targeting the root cause

## Expected Results

Based on the paper's Qwen3.5-0.8B result (MATH-500: 7.9% → 48.1% by editing 0.016% of weights):

- **Realistic**: Fix 30-60% of clear swap cases
- **Optimistic**: Fix 60-80% of swap cases
- **Edit size**: <1% of head parameters
- **Side effects**: None or positive (better overall performance)

## How It Works

### The Paper's Key Insight

Frontier LLMs (GPT-6 Astra, Claude Opus 5.5) can:
1. **Read** neural network weights and understand what they compute
2. **Write** weights from scratch or edit existing ones
3. **Fix** specific failures without training

Examples from the paper:
- Fixed Qwen3.5-0.8B's reasoning loops (MATH-500: 7.9% → 48.1%)
- Wrote Go-playing networks from scratch that beat PPO
- Identified sycophantic behavior from attention patterns alone

### Our Application

1. **Identify failure**: Extract L/R swap cases from test set
2. **Analyze weights**: Find root cause (e.g., query embeddings too similar)
3. **Design edit**: Change specific weights to encode L/R distinction
4. **Test & iterate**: Use forward pass budget to validate
5. **Deploy**: Save successful checkpoint

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  DINOv3 Backbone (768-dim)                              │
│  (FROZEN - do not edit)                                 │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Pose Head (~10-50M params) ← EDIT THIS                 │
│                                                           │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Query Embeddings [23, 256]                       │   │
│  │ (One per keypoint - might be too similar for L/R)│   │
│  └─────────────────────────────────────────────────┘   │
│                         │                                │
│                         ▼                                │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Decoder Layer 0                                  │   │
│  │ ├─ Self-attention (8 heads)                      │   │
│  │ ├─ Cross-attention (8 heads) to image patches    │   │
│  │ └─ FFN                                           │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                                │
│                         ▼                                │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Decoder Layer 1                                  │   │
│  │ ├─ Self-attention (8 heads)                      │   │
│  │ ├─ Cross-attention (8 heads)                     │   │
│  │ └─ FFN                                           │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                                │
│                         ▼                                │
│  ┌─────────────────────────────────────────────────┐   │
│  │ SimCC Output Heads                               │   │
│  │ (Spatial classification for x, y coords)         │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Directory Structure

```
src/weight_space/
├── README.md                  # Detailed documentation
├── extract_lr_swaps.py        # Find swap cases in test set
├── analyze_head.py            # Analyze head weights
├── weight_edit_harness.py     # LLM editing environment
├── llm_prompt.md              # Prompt template for LLM
└── run_pipeline.sh            # Automated pipeline script

weight_space_data/             # Generated outputs (gitignored)
├── lr_swaps.json              # Extracted swap cases
└── head_analysis/
    ├── head_weights.pt        # Extracted head weights
    ├── analysis.json          # Structured analysis
    └── summary.txt            # Human-readable summary
```

## Advanced Usage

### Custom Swap Detection Threshold

```bash
python -m src.weight_space.extract_lr_swaps \
  --run-dir training/runs/2026_10_02_003 \
  --checkpoint best.pt \
  --confidence-threshold 0.20  # Higher = only clear swaps
```

### Analyze Different Checkpoint

```bash
python -m src.weight_space.analyze_head \
  --run-dir training/runs/2026_10_02_003 \
  --checkpoint final.pt  # Instead of best.pt
```

### Save Edited Checkpoint

```python
# In harness or LLM session
harness.save_checkpoint(
    Path('training/runs/2026_10_02_003/edited_lr_fixed.pt'),
    description="Fixed L/R swaps by separating query embeddings"
)
```

## Research Notes

If this works well, it's **highly publishable**:
- Novel application of weight-space programming
- Computer vision domain (paper focused on RL and language)
- Practical impact (fixing real model failures)
- Interpretable edits (not black-box optimization)

Document everything:
- Initial swap statistics
- Hypothesis about root cause
- Exact weight edits made
- Before/after performance
- Generalization to other checkpoints

## References

- **Paper**: Takehi et al. 2026, "Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"
- **URL**: https://rikiyat.github.io/blog/weight-space-programming/
- **Key Result**: Fixed Qwen3.5-0.8B math reasoning (7.9% → 48.1% on MATH-500) by editing 0.016% of weights

## Troubleshooting

### No checkpoint files found
Make sure you have a trained model checkpoint in your run directory.

### CUDA out of memory
The harness needs GPU for forward passes. Reduce `--forward-pass-budget` or use smaller `batch_size`.

### Too few swap cases
Lower `--confidence-threshold` to find more (possibly less confident) swaps.

### Harness shows budget exhausted
You've used all forward passes. Save your work and create a new harness instance if needed.

## Next Steps

1. ✅ Run pipeline to establish baseline
2. ⏳ Provide data to LLM (Claude Opus 5.5 recommended)
3. ⏳ LLM analyzes and edits weights
4. ⏳ Test edited checkpoint on full evaluation
5. ⏳ Document results and potentially publish!

---

**Built on**: Branch `cursor/weight-space-lr-swap-fix-0ec6`
**Status**: Ready for testing (infrastructure complete)
**Best Model**: Claude Opus 5.5 or similar frontier model with strong reasoning
